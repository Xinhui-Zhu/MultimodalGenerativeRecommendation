#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# batch_size = 16GPUs * 8 = 128
cd code 
export NCCL_SOCKET_IFNAME=lo
export PATH=~/.conda/envs/hllm/bin:$PATH
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0        # 启用 InfiniBand（如果适用）
export NCCL_NET_GDR_LEVEL=PHB   # 优化 GPU 直接通信
export NCCL_TIMEOUT=30 
export NCCL_P2P_DISABLE=1

id_emb="id"

CUDA_VISIBLE_DEVICES="0,1,2,3" python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --MAX_ITEM_LIST_LENGTH 50 \
  --epochs 5 \
  --optim_args.learning_rate 1e-4 \
  --checkpoint_dir "saved_dir/HSTU_large/arts/${id_emb}" \
  --loss nce \
  --dataset arts \
  --hidden_dropout_prob 0.1 \
  --attn_dropout_prob 0.1 \
  --n_layers 2 \
  --n_heads 4 \
  --hstu_embedding_size 2048 \
  --num_negatives 512 \
  --train_batch_size 2 \
  --seed 42 \
  --gradient_checkpointing True \
  --item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --text_keys '[\"title\",\"description\"]' \
  --text_path ../information \
  --stage 3 \
  --id_emb "${id_emb}" \
  --split_data False \
  --coldrec item \
  --transformer_type HSTU \
  --enable_relative_attention_bias True \
  --MAX_TEXT_LENGTH 256 \
  --position_embedding False \
  --accumulation_steps 6 \
  --by_case True

id_emb="text"

CUDA_VISIBLE_DEVICES="0,1,2,3" python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --MAX_ITEM_LIST_LENGTH 50 \
  --epochs 5 \
  --optim_args.learning_rate 1e-4 \
  --checkpoint_dir "saved_dir/HSTU_large/arts/${id_emb}" \
  --loss nce \
  --dataset arts \
  --hidden_dropout_prob 0.1 \
  --attn_dropout_prob 0.1 \
  --n_layers 2 \
  --n_heads 4 \
  --hstu_embedding_size 2048 \
  --num_negatives 512 \
  --train_batch_size 2 \
  --seed 42 \
  --gradient_checkpointing True \
  --item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --text_keys '[\"title\",\"description\"]' \
  --text_path ../information \
  --stage 3 \
  --id_emb "${id_emb}" \
  --split_data False \
  --coldrec item \
  --transformer_type HSTU \
  --enable_relative_attention_bias True \
  --MAX_TEXT_LENGTH 256 \
  --position_embedding False \
  --accumulation_steps 6 \
  --by_case True