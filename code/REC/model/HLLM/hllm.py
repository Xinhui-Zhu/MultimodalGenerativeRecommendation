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

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from logging import getLogger

from REC.utils.enum_type import InputType
from REC.model.basemodel import BaseModel, all_gather
from REC.model.HLLM.modeling_llama import LlamaForCausalLM
from REC.model.HLLM.modeling_mistral import MistralForCausalLM
from REC.model.HLLM.modeling_bert import BertModel
from REC.model.HLLM.baichuan.modeling_baichuan import BaichuanForCausalLM
from REC.model.IDNet.hstu import *

from peft import LoraConfig, get_peft_model

class MoE(nn.Module):
    def __init__(self, emb_size, num_experts, top_k):
        """
        emb_size: 文本嵌入的维度
        num_experts: 专家数量 K
        top_k: 每个位置激活的专家数 k
        """
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 定义 K 个专家，每个专家为一个线性变换
        self.experts = nn.ModuleList([nn.Linear(emb_size, emb_size) for _ in range(num_experts)])
        # gating 网络将输入映射到每个专家的得分
        self.gate = nn.Linear(emb_size, num_experts)
    
    def forward(self, x):
        # 判断输入是二维还是三维
        original_dim = x.dim()
        if original_dim == 2:
            # [batch_size, emb_size] -> [batch_size, 1, emb_size]
            x = x.unsqueeze(1)
        
        # 现在 x 的形状为 [batch_size, seq_len, emb_size]
        # 计算 gating 得分，shape: [batch_size, seq_len, num_experts]
        gate_scores = self.gate(x)
        # 对每个位置选出 top_k 个专家，将其它专家得分置为 -inf
        topk_values, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        mask = torch.full_like(gate_scores, float('-inf'))
        mask.scatter_(-1, topk_indices, topk_values)
        # softmax 得到每个专家的权重，shape: [batch_size, seq_len, num_experts]
        gate_weights = F.softmax(mask, dim=-1)
        
        # 计算每个专家的输出，列表中每个张量形状为 [batch_size, seq_len, emb_size]
        expert_outputs = [expert(x) for expert in self.experts]
        # 堆叠后得到 shape: [batch_size, seq_len, num_experts, emb_size]
        expert_outputs = torch.stack(expert_outputs, dim=2)
        # 扩展 gate_weights 为 [batch_size, seq_len, num_experts, 1] 便于加权求和
        gate_weights = gate_weights.unsqueeze(-1)
        # 对专家维度加权求和，得到融合后的语义表示，形状为 [batch_size, seq_len, emb_size]
        fused_output = torch.sum(gate_weights * expert_outputs, dim=2)
        
        # 如果原始输入是二维，则 squeeze 掉序列维度，输出 [batch_size, emb_size]
        if original_dim == 2:
            fused_output = fused_output.squeeze(1)
        return fused_output

class HLLM(BaseModel):
    input_type = InputType.SEQ

    def __init__(self, config, dataload):
        super(HLLM, self).__init__()
        self.logger = getLogger()

        self.item_pretrain_dir = config['item_pretrain_dir']
        self.user_pretrain_dir = config['user_pretrain_dir']
        self.gradient_checkpointing = config['gradient_checkpointing']
        self.use_ft_flash_attn = config['use_ft_flash_attn']
        self.logger.info(f"create item llm")
        self.item_llm = self.create_llm(self.item_pretrain_dir, config['item_llm_init'])

        # >>> 在这里进行冻结 <<<
        self.freeze = config.get('freeze', None)
        if self.freeze == "item_llm":
            for param in self.item_llm.parameters():
                param.requires_grad = False

        # >>> 在这里使用lora <<<
        self.use_lora = config.get('use_lora', False)
        if self.use_lora in ["both", "item_llm", "user_llm"]:
            self.logger.info("** LoRA is enabled. **")
            torch.set_float32_matmul_precision('medium')  # or 'high'
            
            # 你可以在 config 中再加一个 'lora_args' 里含 r, alpha, target_modules 等
            lora_args = config.get('lora_args', {})
            # 示例：从 config 中获取一些参数，如果取不到就用默认
            r_val = lora_args.get('r', 8)
            alpha_val = lora_args.get('lora_alpha', 32)
            dropout_val = lora_args.get('lora_dropout', 0.1)
            target_modules = lora_args.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])

            self.lora_config_item = LoraConfig(
                r=r_val,
                lora_alpha=alpha_val,
                lora_dropout=dropout_val,
                bias="none",
                target_modules=target_modules,
                task_type="CAUSAL_LM"
            )
            if self.use_lora in ["both", "item_llm"]:
                self.item_llm = get_peft_model(self.item_llm, self.lora_config_item)
        else:
            self.logger.info("** LoRA is disabled. **")

        # >>> 在这里选择推荐模型 <<<
        self.transformer_type = config.get('transformer_type', "user_llm")
        if self.transformer_type == "user_llm":
            self.logger.info(f"create user llm")
            self.user_llm = self.create_llm(self.user_pretrain_dir, config['user_llm_init'])
            self.GR_embedding_size = self.user_llm.config.hidden_size
            if self.use_lora in ["both", "user_llm"]:
                self.user_llm = get_peft_model(self.user_llm, self.lora_config_item)
        elif self.transformer_type == "HSTU":
            self.position_embedding = config['position_embedding']
            self._hstu_embedding_dim: int = config['hstu_embedding_size']
            self.GR_embedding_size = self._hstu_embedding_dim
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
       
        # >>> 在这里加入融合ID嵌入所需的参数 <<<
        self.item_emb_token_n = config['item_emb_token_n']
        self.id_emb = config['id_emb']
        if self.id_emb != "text":
            self.by_case = config['by_case']
            self.weights_cold = nn.Parameter(torch.randn(1).float(), requires_grad=True)
            self.weights_warm = nn.Parameter(torch.randn(1).float(), requires_grad=True)
            self.ffn = nn.Linear(self.item_llm.config.hidden_size*2, self.GR_embedding_size)
            self.ffn_cold = nn.Linear(self.item_llm.config.hidden_size*2, self.GR_embedding_size)
            self.ffn_warm = nn.Linear(self.item_llm.config.hidden_size*2, self.GR_embedding_size)
            self.ffn_for_ca = nn.Sequential(
                nn.Linear(self.item_llm.config.hidden_size, self.item_llm.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.item_llm.config.hidden_size, self.GR_embedding_size),
            )
            self.ffn_for_map = nn.Sequential(
                nn.Linear(self.item_llm.config.hidden_size, self.item_llm.config.hidden_size//2),
                nn.ReLU(),
                nn.Linear(self.item_llm.config.hidden_size//2, self.GR_embedding_size),
            )
            self.weight_emb_cold = nn.Parameter(torch.randn(self.item_llm.config.hidden_size), requires_grad=True)
            self.weight_emb_warm = nn.Parameter(torch.randn(self.item_llm.config.hidden_size), requires_grad=True)
            
            self.attention_layer1 = nn.MultiheadAttention(embed_dim=self.item_llm.config.hidden_size, num_heads=16, dropout=0.1, batch_first=True)
            self.attention_layer2 = nn.MultiheadAttention(embed_dim=self.item_llm.config.hidden_size, num_heads=16, dropout=0.1, batch_first=True)
            self.attention_layer3 = nn.MultiheadAttention(embed_dim=self.item_llm.config.hidden_size*2, num_heads=32, dropout=0.1, batch_first=True)
            self.layernorm3 = nn.LayerNorm(self.item_llm.config.hidden_size) 
            self.layernorm4 = nn.LayerNorm(self.item_llm.config.hidden_size)
        self.item_num = dataload.item_num
        if self.id_emb == "moe":
            self.moe = MoE(emb_size=self.item_llm.config.hidden_size, num_experts=8, top_k=2)
 
        self.layernorm1 = nn.LayerNorm(self.item_llm.config.hidden_size) 
        self.layernorm2 = nn.LayerNorm(self.item_llm.config.hidden_size) 
        
        # >>> 在这里加入分层学习率 <<<
        if config['lr_mult_prefix'] or (self.transformer_type == "HSTU" and self.id_emb != "text"):
            print("Using hierarchical lr")
            self.hierarchical_lr = True
            self.item_embedding = nn.Embedding(self.item_num, self.item_llm.config.hidden_size, padding_idx=0)
            self.item_embedding.weight.data.normal_(mean=0.0, std=0.02)
        else:
            self.hierarchical_lr = False
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

        self.loss = config['loss']
        if self.loss == 'nce':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.nce_thres = config['nce_thres'] if config['nce_thres'] else 0.99
            self.num_negatives = config['num_negatives']
            self.logger.info(f"nce thres setting to {self.nce_thres}")
        else:
            raise NotImplementedError(f"Only nce is supported")

        if config['load_pretrain']:
            state_dict = torch.load(config['load_pretrain'], map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            self.logger.info(f"{msg.missing_keys = }")
            self.logger.info(f"{msg.unexpected_keys = }")

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
                return LlamaForCausalLM.from_pretrained(pretrain_dir, config=hf_config, cache_dir="/proj/arise/arise/xz3276/model")
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

    def emb_fusion(self, item_ids, emb, is_cold=None):
        # if only text emb
        if self.id_emb == "text_wo_ln":
            return emb
        emb = self.layernorm1(emb)
        if self.id_emb == "text":
            return emb

        # get id emb
        if self.hierarchical_lr or self.transformer_type == "HSTU":
            id_emb = self.item_embedding(item_ids)
        else:
            embedding_layer = self.user_llm.get_input_embeddings()
            id_emb = embedding_layer(item_ids)
        # if id_emb.shape != emb.shape:
        # print("emb.shape", id_emb.shape, emb.shape)

        # norm id emb
        id_emb = self.layernorm2(id_emb)

        # if only id emb
        if self.id_emb == "id":
            return id_emb

        # fuse id & text emb
        if self.by_case is None or self.by_case == False:
            B = id_emb.shape[0]
            is_cold = torch.zeros((B, 1), dtype=torch.bool, device=id_emb.device)
        if self.id_emb == "by_case":
            # print("is_cold",is_cold.shape)
            # print("emb",emb.shape)
            # print("id_emb",id_emb.shape)
            return torch.where(is_cold.unsqueeze(-1) == 1, emb, id_emb)
        if self.id_emb == "simple_concat":
            concated_emb = torch.concat([emb, id_emb], dim = -1)
            return self.ffn(concated_emb)
        if self.id_emb == "concat":
            id_emb = self.ffn_for_map(id_emb)
            # print("id_emb", id_emb.shape)   
            concated_emb = torch.concat([emb, id_emb], dim = -1)
            # print("concated_emb", concated_emb.shape)   
            fused_emb_warm = self.ffn_warm(concated_emb)
            fused_emb_cold = self.ffn_cold(concated_emb)
            fused_emb = torch.where(is_cold.unsqueeze(-1) == 1, fused_emb_cold, fused_emb_warm)
            return fused_emb
        if self.id_emb == "gating_concat":
            # mapping id emb to text space
            id_emb = self.ffn_for_map(id_emb)
            concated_emb = torch.cat([emb, id_emb], dim=-1)  
            # 线性变换 + sigmoid 得到门控向量
            w_cold = torch.sigmoid(self.ffn_cold(concated_emb))    
            w_warm = torch.sigmoid(self.ffn_warm(concated_emb)) 
            concated_emb_cold = torch.concat([w_cold*emb, (1-w_cold)*id_emb], dim = -1)
            concated_emb_warm = torch.concat([w_warm*emb, (1-w_warm)*id_emb], dim = -1)
            fused_emb = torch.where(is_cold.unsqueeze(-1) == 1, concated_emb_cold, concated_emb_warm)
            return self.ffn(fused_emb)
        
        elif self.id_emb == "mean":
            return (id_emb + emb)/2
        elif self.id_emb == "fix_weighted":
            return 0.001 * id_emb + 0.999 * emb
        elif self.id_emb == "simple_weighted_avg":
            normalized_weights = F.sigmoid(self.weights_warm)
            return normalized_weights * id_emb + (1 - normalized_weights) * emb
        elif self.id_emb == "weighted_avg":
            normalized_weights_warm = F.sigmoid(self.weights_warm)
            normalized_weights_cold = F.sigmoid(self.weights_cold)
            warm_result = normalized_weights_warm * id_emb + (1 - normalized_weights_warm) * emb
            cold_result = normalized_weights_cold * id_emb + (1 - normalized_weights_cold) * emb
            # 利用 is_cold 掩码选择结果：0为暖物品，1为冷物品
            result = torch.where(is_cold.bool().unsqueeze(-1), cold_result, warm_result)
            return result
        elif self.id_emb == "weighted_sum":
            normalized_weights_warm = F.sigmoid(self.weights_warm)
            normalized_weights_cold = F.sigmoid(self.weights_cold)
            warm_result = normalized_weights_warm * id_emb + emb
            cold_result = normalized_weights_cold * id_emb + emb
            result = torch.where(is_cold.bool().unsqueeze(-1), cold_result, warm_result)
            return result

        elif self.id_emb == "moe":
            # 利用 MoE 模块聚合文本嵌入，得到语义表示
            semantic_emb = self.moe(emb)  # 形状为 [batch_size, seq_len, emb_size]
            # 将 ID 嵌入和语义表示在最后一维上拼接
            fused_emb = torch.cat([id_emb, semantic_emb], dim=-1)  # [batch_size, seq_len, 2 * emb_size]
            if self.item_llm.config.hidden_size * 2 == self.GR_embedding_size:
                return fused_emb
            else:
                return self.ffn(fused_emb)
        
        elif self.id_emb == "cross-attention-id-as-q":
            attn_output, _ = self.attention_layer2(id_emb, emb, emb)
            # return attn_output
            # 3) 做残差连接 + LayerNorm
            x = self.layernorm3(attn_output + id_emb)
            # 4) 前向网络(FFN) + 残差 + LayerNorm
            ffn_out = self.ffn_for_ca(x) 
            fused = self.layernorm4(ffn_out + x)  
            return fused
        elif self.id_emb == "cross-attention-id-as-q-M3CSR":
            id_emb = self.ffn_for_map(id_emb)
            attn_emb, _ = self.attention_layer2(id_emb, emb, emb)
            concated_emb = torch.concat([attn_emb, id_emb], dim = -1)
            fused_emb_warm = self.ffn_warm(concated_emb)
            fused_emb = torch.where(is_cold.unsqueeze(-1) == 1, emb, fused_emb_warm)
            return fused_emb
        elif self.id_emb == "cross-attention-text-as-q-M3CSR":
            id_emb = self.ffn_for_map(id_emb)
            attn_id_emb, _ = self.attention_layer1(emb, id_emb, id_emb) 
            concated_emb = torch.concat([attn_id_emb, emb], dim = -1)
            fused_emb_warm = self.ffn_warm(concated_emb)
            fused_emb = torch.where(is_cold.unsqueeze(-1) == 1, emb, fused_emb_warm)
            return fused_emb
        elif self.id_emb == "cross-attention-text-as-q":
            attn_output, _ = self.attention_layer1(emb, id_emb, id_emb) 
            x = self.layernorm3(attn_output + emb)
            ffn_out = self.ffn_for_ca(x) 
            fused = self.layernorm4(ffn_out + x)  
            return fused
        elif self.id_emb == "cross-attention1":
            attention_output1, _ = self.attention_layer1(emb, id_emb, id_emb) 
            attention_output2, _ = self.attention_layer2(id_emb, emb, emb)
            return attention_output1 + attention_output2
        elif self.id_emb == "cross-attention2":
            g_input = torch.cat([emb, id_emb], dim=-1)  
            attention_output, _ = self.attention_layer3(g_input, g_input, g_input)
            return self.ffn(attention_output)
        
        elif self.id_emb == "gating":
            # 拼接
            g_input = torch.cat([emb, id_emb], dim=-1)  
            # 线性变换 + sigmoid 得到门控向量
            g_cold = torch.sigmoid(self.ffn_cold(g_input))    
            g_warm = torch.sigmoid(self.ffn_warm(g_input))    
            # 融合: g * user_emb + (1 - g) * item_emb
            fused_emb_warm = g_warm * emb + (1.0 - g_warm) * id_emb
            fused_emb_cold = g_cold * emb + (1.0 - g_cold) * id_emb
            fused_emb = torch.where(is_cold.unsqueeze(-1) == 1, fused_emb_cold, fused_emb_warm)
            return fused_emb
        elif self.id_emb == "domain_gating":
            embeddings = [id_emb, emb]
            weights_cold = torch.stack([
                torch.sum(self.weight_emb_cold * emb, dim=-1, keepdim=True) for emb in embeddings
            ], dim=-1)  # shape: (batch_size, 1, num_embeddings)
            # 使用 softmax 归一化
            normalized_weights_cold = torch.softmax(weights_cold.clamp(max=10), dim=-1)
            # 加权融合
            fused_emb_cold = sum(w * emb for w, emb in zip(normalized_weights_cold.unbind(dim=-1), embeddings))
            
            weights_warm = torch.stack([
                torch.sum(self.weight_emb_warm * emb, dim=-1, keepdim=True) for emb in embeddings
            ], dim=-1)  # shape: (batch_size, 1, num_embeddings)
            normalized_weights_warm = torch.softmax(weights_warm.clamp(max=10), dim=-1)
            fused_emb_warm = sum(w * emb for w, emb in zip(normalized_weights_warm.unbind(dim=-1), embeddings))
            fused_emb = torch.where(is_cold.unsqueeze(-1) == 1, fused_emb_cold, fused_emb_warm)
            
            return fused_emb

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

        # emb = self.layernorm(emb)
        return emb
            
    def forward(self, interaction, mode='train'):
        if mode == 'predict':
            return self.predict(interaction[0], interaction[1], interaction[2])
        if mode == 'compute_item':
            return self.compute_item(interaction)
        user_attention_mask = interaction['attention_mask']
        N, S = user_attention_mask.shape
        pos_input_ids, pos_cu_input_lens, pos_position_ids = interaction['pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids']
        neg_input_ids, neg_cu_input_lens, neg_position_ids = interaction['neg_input_ids'], interaction['neg_cu_input_lens'], interaction['neg_position_ids']
        
        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        pos_embedding = pos_embedding.reshape(N, S+1, -1)  # [batch_size * seq_len, 2048] -> [batch_size, seq_len, 2048]
        neg_embedding = self.forward_item_emb(neg_input_ids, neg_position_ids, neg_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        neg_embedding = neg_embedding.reshape(N, -1, self.item_llm.config.hidden_size)
        pos_item_ids = interaction['pos_item_ids']  # [batch_size, seq_len]
        neg_item_ids = interaction['neg_item_ids']  # [batch_size, seq_len]
        pos_is_cold = interaction['pos_cold_ids']
        neg_is_cold = interaction['neg_cold_ids']
        pos_embedding = self.emb_fusion(pos_item_ids, pos_embedding, pos_is_cold)
        neg_embedding = self.emb_fusion(neg_item_ids, neg_embedding, neg_is_cold)

        target_pos_embs = pos_embedding[:, 1:]
        target_neg_embs = neg_embedding
        # print("target_pos_embs", target_pos_embs)
        # print("target_neg_embs", target_neg_embs)
        if self.transformer_type=="user_llm":
            user_embedding = self.user_llm(inputs_embeds=pos_embedding[:, :-1], attention_mask=user_attention_mask).hidden_states[-1]
        elif self.transformer_type=="HSTU":
            input_emb = pos_embedding[:, :-1, :]
            if self.position_embedding:
                position_ids = torch.arange(user_attention_mask.size(1), dtype=torch.long, device=user_attention_mask.device)
                position_ids = position_ids.unsqueeze(0).expand_as(user_attention_mask)
                position_embedding = self.position_embedding(position_ids)
                input_emb = input_emb + position_embedding

            attention_mask = self.get_attention_mask(user_attention_mask)
            user_embedding = self._hstu(
                x=input_emb,
                attention_mask=attention_mask
            )
        model_out = {}
        logits, labels = self.nce_loss(user_embedding, target_pos_embs, target_neg_embs, user_attention_mask)
        model_out['loss'] = F.cross_entropy(logits, labels)
        model_out['nce_samples'] = (logits > torch.finfo(logits.dtype).min/100).sum(dim=1).float().mean()  # samples after filtering same negatives
        for k in [1, 5, 10, 50, 100]:
            if k > logits.size(1):
                break
            indices = logits.topk(k, dim=1).indices
            model_out[f"nce_top{k}_acc"] = labels.view(-1, 1).eq(indices).any(dim=1).float().mean()
        return model_out

    @torch.no_grad()
    def predict(self, item_seq, time_seq, item_feature):
        attention_mask = (item_seq > 0).int()

        pos_embedding = item_feature[item_seq]

        if self.transformer_type=="user_llm":
            user_embedding = self.user_llm(inputs_embeds=pos_embedding, attention_mask=attention_mask).hidden_states[-1]
        elif self.transformer_type == "HSTU":
            input_emb = pos_embedding
            if self.position_embedding:
                position_ids = torch.arange(attention_mask.size(1), dtype=torch.long, device=attention_mask.device)
                position_ids = position_ids.unsqueeze(0).expand_as(attention_mask)
                position_embedding = self.position_embedding(position_ids)
                input_emb = input_emb + position_embedding
            attention_mask = self.get_attention_mask(attention_mask)
            user_embedding = self._hstu(
                x=input_emb,
                attention_mask=attention_mask
            )
        seq_output = user_embedding[:, -1]
        seq_output = seq_output / seq_output.norm(dim=-1, keepdim=True)
        item_feature = item_feature / item_feature.norm(dim=-1, keepdim=True)

        return torch.matmul(seq_output, item_feature.t())

    @torch.no_grad()
    def compute_item_all(self):
        return self.item_embedding.weight

    @torch.no_grad()
    def compute_item(self, interaction):
        pos_input_ids, pos_cu_input_lens, pos_position_ids = interaction['pos_input_ids'], interaction['pos_cu_input_lens'], interaction['pos_position_ids']
        N = pos_cu_input_lens.size(0)
        pos_embedding = self.forward_item_emb(pos_input_ids, pos_position_ids, pos_cu_input_lens, self.item_emb_token_n, self.item_emb_tokens, self.item_llm)
        pos_item_ids = interaction['pos_item_ids']  # [batch_size, seq_len]
        pos_is_cold = interaction['pos_cold_ids']
        pos_embedding = self.emb_fusion(pos_item_ids, pos_embedding, pos_is_cold)
        # print(pos_embedding.shape)
        pos_embedding = pos_embedding.view(N, -1)

        return pos_embedding

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