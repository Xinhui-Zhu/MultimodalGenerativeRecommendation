#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# Use 8GPUs for batch_size = 8x16 = 128
cd code 
export NCCL_SOCKET_IFNAME=lo
export PATH=~/.conda/envs/hllm/bin:$PATH
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0        # 启用 InfiniBand（如果适用）
export NCCL_NET_GDR_LEVEL=PHB   # 优化 GPU 直接通信
export NCCL_TIMEOUT=30 
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES="0,1,2,3" python3 main.py \
--config_file IDNet/sasrec.yaml overall/ID.yaml \
--optim_args.learning_rate 1e-3 \
--loss nce \
--train_batch_size 16 \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 101 \
--dataset arts \
--hidden_dropout_prob 0.5 \
--attn_dropout_prob 0.5 \
--num_negatives 512 \
--n_layers 4 \
--n_heads 4 \
--embedding_size 64 \
--inner_size 1 \
--show_progress True \
--update_interval 100 \
--optim_args.weight_decay 0.0 \
--stopping_step 10 \
--coldrec item \
--split_data False \
--by_case True \
--transformer_type sasrec \
--id_emb id