# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implements HSTU (Hierarchical Sequential Transduction Unit) in
Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
(https://arxiv.org/abs/2402.17152).
"""

import abc
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from logging import getLogger
import fbgemm_gpu

from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, l2_norm, all_gather
from REC.model.HLLM.modeling_llama import LlamaForCausalLM
from REC.model.HLLM.modeling_mistral import MistralForCausalLM
from REC.model.HLLM.modeling_bert import BertModel
from REC.model.HLLM.baichuan.modeling_baichuan import BaichuanForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
import transformers

def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x


TIMESTAMPS_KEY = "timestamps"


class RelativeAttentionBiasModule(torch.nn.Module):

    @abc.abstractmethod
    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: [B, N] x int64
        Returns:
            torch.float tensor broadcastable to [B, N, N]
        """
        pass


class RelativePositionalBias(RelativeAttentionBiasModule):
    def __init__(self, max_seq_len: int) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        del all_timestamps
        n: int = self._max_seq_len
        t = F.pad(self._w[: 2 * n - 1], [0, n]).repeat(n)
        t = t[..., :-n].reshape(1, n, 3 * n - 2)
        r = (2 * n - 1) // 2
        return t[..., r:-r]


class RelativeBucketedTimeAndPositionBasedBias(RelativeAttentionBiasModule):
    """
    Bucketizes timespans based on ts(next-item) - ts(current-item).
    """

    def __init__(
        self,
        max_seq_len: int,
        num_buckets: int,
        bucketization_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()

        self._max_seq_len: int = max_seq_len
        self._ts_w = torch.nn.Parameter(
            torch.empty(num_buckets + 1).normal_(mean=0, std=0.02),
        )
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * max_seq_len - 1).normal_(mean=0, std=0.02),
        )
        self._num_buckets: int = num_buckets
        self._bucketization_fn: Callable[[torch.Tensor], torch.Tensor] = (
            bucketization_fn
        )

    def forward(
        self,
        all_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            all_timestamps: (B, N).
        Returns:
            (B, N, N).
        """
        B = all_timestamps.size(0)
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N]).repeat(N)
        t = t[..., :-N].reshape(1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat(
            [all_timestamps, all_timestamps[:, N - 1: N]], dim=1
        )
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                ext_timestamps[:, 1:].unsqueeze(2) - ext_timestamps[:, :-1].unsqueeze(1)
            ),
            min=0,
            max=self._num_buckets,
        ).detach()
        rel_pos_bias = t[:, :, r:-r]
        rel_ts_bias = torch.index_select(
            self._ts_w, dim=0, index=bucketed_timestamps.view(-1)
        ).view(B, N, N)
        return rel_pos_bias + rel_ts_bias


HSTUCacheState = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor  # [bs, 1, n, n]
):
    B, _, n, _ = attention_mask.size()

    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        q.view(B, n, num_heads, attention_dim),
        k.view(B, n, num_heads, attention_dim),
    )
    qk_attn = F.silu(qk_attn) / n
    qk_attn = qk_attn * attention_mask
    # print(f"{qk_attn.size() = } {v.size() = }")
    attn_output = torch.einsum(
        "bhnm,bmhd->bnhd",
        qk_attn,
        v.reshape(B, n, num_heads, linear_dim),
    ).reshape(B, n, num_heads * linear_dim)
    return attn_output


class SequentialTransductionUnitJagged(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        attn_dropout_ratio: float,
        num_heads: int,
        linear_activation: str,
        relative_attention_bias_module: Optional[RelativeAttentionBiasModule] = None,
        normalization: str = "rel_bias",
        linear_config: str = "uvqk",
        concat_ua: bool = False,
        epsilon: float = 1e-6,
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._embedding_dim: int = embedding_dim
        self._linear_dim: int = linear_hidden_dim
        self._attention_dim: int = attention_dim
        self._dropout_ratio: float = dropout_ratio
        self._attn_dropout_ratio: float = attn_dropout_ratio
        self._num_heads: int = num_heads
        self._rel_attn_bias: Optional[RelativeAttentionBiasModule] = (
            relative_attention_bias_module
        )
        self._normalization: str = normalization
        self._linear_config: str = linear_config
        if self._linear_config == "uvqk":
            self._uvqk = torch.nn.Parameter(
                torch.empty(
                    (
                        embedding_dim,
                        linear_hidden_dim * 2 * num_heads
                        + attention_dim * num_heads * 2,
                    )
                ).normal_(mean=0, std=0.02),
            )
        else:
            raise ValueError(f"Unknown linear_config {self._linear_config}")
        self._linear_activation: str = linear_activation
        self._concat_ua: bool = concat_ua
        self._o = torch.nn.Linear(
            in_features=linear_hidden_dim * num_heads * (3 if concat_ua else 1),
            out_features=embedding_dim,
        )
        torch.nn.init.xavier_uniform_(self._o.weight)
        self._eps: float = epsilon

    def _norm_input(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, normalized_shape=[self._embedding_dim], eps=self._eps)

    def _norm_attn_output(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=[self._linear_dim * self._num_heads], eps=self._eps
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (\sum_i N_i, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: optional (B, N) x int64.
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
            delta_x_offsets: optional 2-tuple ((B,) x int32, (B,) x int32).
                For the 1st element in the tuple, each element is in [0, x_offsets[-1]). For the
                2nd element in the tuple, each element is in [0, N).
            cache: Optional 4-tuple of (v, padded_q, padded_k, output) from prior runs,
                where all except padded_q, padded_k are jagged.
        Returns:
            x' = f(x), (\sum_i N_i, D) x float.
        """

        normed_x = self._norm_input(x)
        if self._linear_config == "uvqk":
            batched_mm_output = torch.matmul(normed_x, self._uvqk)
            if self._linear_activation == "silu":
                batched_mm_output = F.silu(batched_mm_output)
            elif self._linear_activation == "none":
                batched_mm_output = batched_mm_output
            u, v, q, k = torch.split(
                batched_mm_output,
                [
                    self._linear_dim * self._num_heads,
                    self._linear_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                    self._attention_dim * self._num_heads,
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown self._linear_config {self._linear_config}")

        B: int = attention_mask.size(0)
        if self._normalization == "rel_bias" or self._normalization == "hstu_rel_bias":
            attn_output = _hstu_attention_maybe_from_cache(
                num_heads=self._num_heads,
                attention_dim=self._attention_dim,
                linear_dim=self._linear_dim,
                q=q,
                k=k,
                v=v,
                attention_mask=attention_mask
            )

        if self._concat_ua:
            a = self._norm_attn_output(attn_output)
            o_input = torch.cat([u, a, u * a], dim=-1)
        else:
            o_input = u * self._norm_attn_output(attn_output)

        new_outputs = (
            self._o(
                F.dropout(
                    o_input,
                    p=self._dropout_ratio,
                    training=self.training,
                )
            )
            + x
        )

        return new_outputs


class HSTUJagged(torch.nn.Module):

    def __init__(
        self,
        modules: List[SequentialTransductionUnitJagged],
        autocast_dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self._attention_layers: torch.nn.ModuleList = torch.nn.ModuleList(
            modules=modules
        )
        self._autocast_dtype: torch.dtype = autocast_dtype

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[HSTUCacheState]]:
        """
        Args:
            x: (B, N, D) x float.
            x_offsets: (B + 1) x int32.
            all_timestamps: (B, 1 + N) x int64
            invalid_attn_mask: (B, N, N) x float, each element in {0, 1}.
        Returns:
            x' = f(x), (B, N, D) x float
        """

        for i, layer in enumerate(self._attention_layers):
            x = layer(
                x=x,
                attention_mask=attention_mask
            )

        return x


class HSTU(BaseModel):
    """
    Implements HSTU (Hierarchical Sequential Transduction Unit) in
    Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations,
    https://arxiv.org/abs/2402.17152.

    Note that this implementation is intended for reproducing experiments in
    the traditional sequential recommender setting (Section 4.1.1), and does
    not yet use optimized kernels discussed in the paper.
    """
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super().__init__()
        self.logger = getLogger()
        self.item_num = dataload.item_num
        self._item_embedding_dim: int = config['item_embedding_size']
        self._hstu_embedding_dim: int = config['hstu_embedding_size']
        self._max_sequence_length: int = config['MAX_ITEM_LIST_LENGTH']
        self._num_blocks: int = config['n_layers']
        self._num_heads: int = config['n_heads']
        self._dqk: int = config['hstu_embedding_size'] // config['n_heads']
        self._dv: int = config['hstu_embedding_size'] // config['n_heads']
        self._linear_activation: str = config['hidden_act'] if config['hidden_act'] else "silu"
        self._linear_dropout_rate: float = config['hidden_dropout_prob']
        self._attn_dropout_rate: float = config['attn_dropout_prob']
        self._enable_relative_attention_bias: bool = config['enable_relative_attention_bias'] if config['enable_relative_attention_bias'] else False
        self._linear_config = 'uvqk'
        self._normalization = 'rel_bias'
        self.position_embedding = nn.Embedding(self._max_sequence_length+1, self._hstu_embedding_dim)
        self._hstu = HSTUJagged(
            modules=[
                SequentialTransductionUnitJagged(
                    embedding_dim=self._hstu_embedding_dim,
                    linear_hidden_dim=self._dv,
                    attention_dim=self._dqk,
                    normalization=self._normalization,
                    linear_config=self._linear_config,
                    linear_activation=self._linear_activation,
                    num_heads=self._num_heads,
                    # TODO: change to lambda x.
                    relative_attention_bias_module=(
                        RelativeBucketedTimeAndPositionBasedBias(
                            max_seq_len=self._max_sequence_length
                            + self._max_sequence_length,  # accounts for next item.
                            num_buckets=128,
                            bucketization_fn=lambda x: (
                                torch.log(torch.abs(x).clamp(min=1)) / 0.301
                            ).long(),
                        )
                        if self._enable_relative_attention_bias
                        else None
                    ),
                    dropout_ratio=self._linear_dropout_rate,
                    attn_dropout_ratio=self._attn_dropout_rate,
                    concat_ua=False,
                )
                for _ in range(self._num_blocks)
            ],
            autocast_dtype=None,
        )
        self.id_emb = config['id_emb'] 
        if self.id_emb == 'id':
            self.item_embedding = nn.Embedding(self.item_num, self._item_embedding_dim, padding_idx=0)
        elif self.id_emb == 'text':
            self.item_pretrain_dir = config['item_pretrain_dir']
            self.gradient_checkpointing = config['gradient_checkpointing']
            self.use_ft_flash_attn = True
            self.logger.info(f"create item llm")
            self.item_llm = self.create_llm(self.item_pretrain_dir, True)
            self.item_emb_token_n = config['item_emb_token_n']
            if self.item_emb_token_n > 1:
                raise NotImplementedError(f"Not support item_emb_token_n {self.item_emb_token_n} > 1")
            if self.item_emb_token_n > 0:
                self.item_emb_tokens = nn.Parameter(
                    torch.zeros(1, self.item_emb_token_n, self.item_llm.config.hidden_size)
                )
                self.item_emb_tokens.data.normal_(mean=0.0, std=0.02)
                if config['item_emb_pretrain']:
                    ckpt = torch.load(config['item_emb_pretrain'], map_location='cpu')
                    self.logger.info(f"load item_emb_token from {config['item_emb_pretrain']} with {ckpt.size()}")
                    self.item_emb_tokens.data = nn.Parameter(ckpt)
            else:  # mean pooling
                self.item_emb_tokens = None

        self.item_id_proj_tower = nn.Identity() if config['item_embedding_size'] == config['hstu_embedding_size'] else nn.Linear(config['item_embedding_size'], config['hstu_embedding_size'], bias=False)
        self.loss = config['loss']
        if self.loss == 'nce':
            if config['fix_temp']:
                self.logger.info(f"Fixed logit_scale 20")
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.05), requires_grad=False)
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.nce_thres = config['nce_thres'] if config['nce_thres'] else 0.99
            self.num_negatives = config['num_negatives']
            self.logger.info(f"nce thres setting to {self.nce_thres}")
        else:
            raise NotImplementedError(f"Only nce is supported")

        # causal forward, w/ +1 for padding.
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones(
                    (
                        self._max_sequence_length,
                        self._max_sequence_length,
                    ),
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        self._verbose: bool = True
        self.reset_params()

    def create_llm(self, pretrain_dir, init=True):
        self.logger.info(f"******* create LLM {pretrain_dir} *******")
        hf_config = AutoConfig.from_pretrained(pretrain_dir, trust_remote_code=True)
        self.logger.info(f"hf_config: {hf_config}")
        hf_config.gradient_checkpointing = self.gradient_checkpointing
        hf_config.use_cache = False
        hf_config.output_hidden_states = True
        hf_config.return_dict = True

        self.logger.info("xxxxx starting loading checkpoint")
        if isinstance(hf_config, transformers.LlamaConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for llama')
            self.logger.info(f'Init {init} for llama')
            if init:
                return LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return LlamaForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.MistralConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for mistral')
            self.logger.info(f'Init {init} for mistral')
            if init:
                return MistralForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return MistralForCausalLM(config=hf_config).cuda()
        elif isinstance(hf_config, transformers.BertConfig):
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for bert')
            self.logger.info(f'Init {init} for bert')
            if init:
                return BertModel.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BertModel(config=hf_config).cuda()
        elif getattr(hf_config, "model_type", None) == "baichuan":
            hf_config.use_ft_flash_attn = self.use_ft_flash_attn
            self.logger.info(f'Using flash attention {hf_config.use_ft_flash_attn} for baichuan')
            self.logger.info(f'Init {init} for baichuan')
            if init:
                return BaichuanForCausalLM.from_pretrained(pretrain_dir, config=hf_config)
            else:
                return BaichuanForCausalLM(config=hf_config).cuda()
        else:
            return AutoModelForCausalLM.from_pretrained(
                self.local_dir, config=hf_config
            )

    def forward_item_emb(
        self,
        input_ids,
        position_ids,
        cu_input_lens,
        emb_token_n,
        emb_tokens,
        llm
    ):
        inputs_embeds = llm.get_input_embeddings()(input_ids)
        # print("inputs_embeds", inputs_embeds.shape)
        emb_pos = cu_input_lens.cumsum(dim=0, dtype=torch.int32)
        if emb_token_n > 0:
            inputs_embeds[emb_pos - 1] = emb_tokens
        model_out = llm(inputs_embeds=inputs_embeds.unsqueeze(0), cu_input_lens=cu_input_lens, position_ids=position_ids.unsqueeze(0))
        model_out = model_out.hidden_states[-1].squeeze(0)

        if emb_token_n > 0:
            emb = model_out[emb_pos - 1]
        else:
            max_len = cu_input_lens.max().item()
            cu_seqlens = F.pad(cu_input_lens.cumsum(dim=0, dtype=torch.int32), (1, 0))
            seqs = [model_out[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
            padded_seqs = [
                F.pad(
                    seqs[i],
                    (0, 0) * (seqs[i].dim() - 1) + (0, max_len - cu_input_lens[i]),
                    value=0.0,
                )
                for i in range(cu_input_lens.size(0))
            ]
            out = torch.stack(padded_seqs)
            emb = out.sum(dim=1) / cu_input_lens.unsqueeze(1)
        return emb

    def reset_params(self):
        for name, params in self.named_parameters():
            if ("_hstu" in name) or ("_embedding_module" in name) or ('logit_scale' in name):
                if self._verbose:
                    print(f"Skipping init for {name}")
                continue
            try:
                truncated_normal(params.data, mean=0.0, std=0.02)
                if self._verbose:
                    print(
                        f"Initialize {name} as trunc normal: {params.data.size()} params"
                    )
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def debug_str(self) -> str:
        debug_str = (
            f"HSTU-b{self._num_blocks}-h{self._num_heads}-dqk{self._dqk}-dv{self._dv}"
            + f"-l{self._linear_activation}d{self._linear_dropout_rate}"
            + f"-ad{self._attn_dropout_rate}"
        )
        if not self._enable_relative_attention_bias:
            debug_str += "-norab"
        return debug_str

    def forward(self, interaction, mode='train'):
        if mode == 'compute_item':
            return self.compute_item(interaction)
        if self.id_emb == 'id':
            items, neg_items, masked_index = interaction  # [batch, 2, seq_len]    #[batch, max_seq_len-1]
            if self.num_negatives:
                neg_items = torch.randint(
                    low=1,
                    high=self.item_num,
                    size=(items.size(0), items.size(1) - 1, self.num_negatives),
                    dtype=items.dtype,
                    device=items.device,
                )
            # print(self.item_embedding(items).shape) = [batch, max_seq_len+1, dim]
            pos_items_embs = self.item_id_proj_tower(self.item_embedding(items))  # [batch, 2, max_seq_len+1, dim]
            neg_items_embs = self.item_id_proj_tower(self.item_embedding(neg_items))  # [128, 200, 1024, 50]
            # neg_items torch.Size([8, 50, 512]) neg_items_embs torch.Size([8, 50, 512, 2048])

            input_emb = pos_items_embs[:, :-1, :]  # [batch, max_seq_len, dim]

            position_ids = torch.arange(masked_index.size(1), dtype=torch.long, device=masked_index.device)
            position_ids = position_ids.unsqueeze(0).expand_as(masked_index)
            position_embedding = self.position_embedding(position_ids)
            input_emb = input_emb + position_embedding

            attention_mask = self.get_attention_mask(masked_index)
            output_embs = self._hstu(
                x=input_emb,
                attention_mask=attention_mask
            )

            target_pos_embs = pos_items_embs[:, 1:, :]  # [batch, max_seq_len, dim]
            neg_embedding_all = neg_items_embs  # [batch, max_seq_len, dim]

            with torch.no_grad():
                self.logit_scale.clamp_(0, np.log(100))
            logit_scale = self.logit_scale.exp()
            output_embs = output_embs / output_embs.norm(dim=-1, keepdim=True)
            target_pos_embs = target_pos_embs / target_pos_embs.norm(dim=-1, keepdim=True)
            neg_embedding_all = neg_embedding_all / neg_embedding_all.norm(dim=-1, keepdim=True)
            pos_logits = F.cosine_similarity(output_embs, target_pos_embs, dim=-1).unsqueeze(-1)
            if self.num_negatives and self.id_emb == "id":
                # print("output_embs", output_embs.shape, "neg_embedding_all", neg_embedding_all.shape)
                neg_logits = F.cosine_similarity(output_embs.unsqueeze(2), neg_embedding_all, dim=-1)
                fix_logits = F.cosine_similarity(target_pos_embs.unsqueeze(2), neg_embedding_all, dim=-1)
            else:
                D = neg_embedding_all.size(-1)
                neg_embedding_all = all_gather(neg_embedding_all, sync_grads=True).reshape(-1, D)  # [num, dim]
                neg_embedding_all = neg_embedding_all.transpose(-1, -2)
                neg_logits = torch.matmul(output_embs, neg_embedding_all)
                fix_logits = torch.matmul(target_pos_embs, neg_embedding_all)

            neg_logits[fix_logits > self.nce_thres] = torch.finfo(neg_logits.dtype).min
            # print("neg_logits min:", neg_logits.min().item(), "max:", neg_logits.max().item())
            # print("fix_logits min:", fix_logits.min().item(), "max:", fix_logits.max().item())

            logits = torch.cat([pos_logits, neg_logits], dim=-1)
            # print("logits", logits.shape, masked_index.shape, logit_scale.shape)
            # [8,50,513] [8,50] []
            logits = logits[masked_index.bool()] * logit_scale
            labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        elif self.id_emb == 'text':
            masked_index = interaction['attention_mask']
            N, S = masked_index.shape
            pos_input_ids, pos_cu_input_lens, pos_position_ids = interaction['pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids']
            neg_input_ids, neg_cu_input_lens, neg_position_ids = interaction['neg_input_ids'], interaction['neg_cu_input_lens'], interaction['neg_position_ids']
            print("pos_input_ids", pos_input_ids)
            print("neg_input_ids", neg_input_ids)
            print("pos_cu_input_lens", pos_cu_input_lens)
            print("neg_cu_input_lens", neg_cu_input_lens)
            pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
            pos_embedding = pos_embedding.reshape(N, S+1, -1)
            neg_embedding = self.forward_item_emb(neg_input_ids, neg_position_ids, neg_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
            neg_embedding = neg_embedding.reshape(N, -1, self.item_llm.config.hidden_size)
            # pos_items_embs = self.item_id_proj_tower(pos_embedding)  # [batch, 2, max_seq_len+1, dim]
            # neg_items_embs = self.item_id_proj_tower(neg_embedding)  # [128, 200, 1024, 50]
            target_pos_embs = pos_embedding[:, 1:]
            target_neg_embs = neg_embedding
            # print("target_pos_embs", target_pos_embs)
            # print("target_neg_embs", target_neg_embs)
            
            input_emb = pos_embedding[:, :-1, :]
            position_ids = torch.arange(masked_index.size(1), dtype=torch.long, device=masked_index.device)
            position_ids = position_ids.unsqueeze(0).expand_as(masked_index)
            position_embedding = self.position_embedding(position_ids)
            input_emb = input_emb + position_embedding

            attention_mask = self.get_attention_mask(masked_index)
            user_embedding = self._hstu(
                x=input_emb,
                attention_mask=attention_mask
            )

            logits, labels = self.nce_loss(user_embedding, target_pos_embs, target_neg_embs, masked_index)
        
        model_out = {}
        model_out['loss'] = F.cross_entropy(logits, labels)
        model_out['nce_samples'] = (logits > torch.finfo(logits.dtype).min/100).sum(dim=1).float().mean()
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(1):
                break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = labels.view(-1, 1).eq(indices).any(dim=1).float().mean()
        return model_out

    def nce_loss(self, cur_embs, target_pos, target_neg, user_attention_mask):
        with torch.no_grad():
            self.logit_scale.clamp_(0, np.log(100))
        logit_scale = self.logit_scale.exp()
        D = target_neg.size(-1)
        output_embs = cur_embs / cur_embs.norm(dim=-1, keepdim=True)
        target_pos_embs = target_pos / target_pos.norm(dim=-1, keepdim=True)
        pos_logits = F.cosine_similarity(output_embs, target_pos_embs, dim=-1).unsqueeze(-1)

        target_neg = target_neg / target_neg.norm(dim=-1, keepdim=True)

        neg_embedding_all = all_gather(target_neg, sync_grads=True).reshape(-1, D)  # [num, dim]
        neg_embedding_all = neg_embedding_all.transpose(-1, -2)
        neg_logits = torch.matmul(output_embs, neg_embedding_all)
        fix_logits = torch.matmul(target_pos_embs, neg_embedding_all)
        neg_logits[fix_logits > self.nce_thres] = torch.finfo(neg_logits.dtype).min

        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        logits = logits[user_attention_mask.bool()] * logit_scale
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)
        return logits, labels

    @torch.no_grad()
    def predict(self, item_seq, time_seq, item_feature):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        if self.id_emb == 'id':
            item_emb = self.item_id_proj_tower(self.item_embedding(item_seq))
        elif self.id_emb == 'text':
            item_emb = item_feature[item_seq]
        item_emb = item_emb + position_embedding
        attention_mask = self.get_attention_mask(item_seq)
        output_embs = self._hstu(
            x=item_emb,
            attention_mask=attention_mask
        )
        seq_output = output_embs[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)

        scores = torch.matmul(seq_output, item_feature.t())
        return scores

    @torch.no_grad()
    def compute_item_all(self):
        weight = self.item_id_proj_tower(self.item_embedding.weight)
        return weight / weight.norm(dim=-1, keepdim=True)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        # extended_attention_mask = torch.where(extended_attention_mask, 0., -1e9)
        return extended_attention_mask
    def connect_wandblogger(self, wandblogger):
        self.wandblogger = wandblogger

    @torch.no_grad()
    def compute_item(self, interaction):
        pos_input_ids, pos_cu_input_lens, pos_position_ids = interaction['pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids']
        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        N = pos_cu_input_lens.size(0)
        pos_embedding = pos_embedding.view(N, -1)

        return pos_embedding