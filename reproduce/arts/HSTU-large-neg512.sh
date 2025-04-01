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

CUDA_VISIBLE_DEVICES="0,1,2,3" python3 main.py \
--config_file IDNet/hstu.yaml overall/ID_deepspeed.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 101 \
--optim_args.learning_rate 1e-3 \
--checkpoint_dir saved_dir/HSTU_large/arts \
--loss nce \
--dataset arts \
--hidden_dropout_prob 0.5 \
--attn_dropout_prob 0.5 \
--n_layers 22 \
--n_heads 32 \
--item_embedding_size 64 \
--hstu_embedding_size 64 \
--fix_temp True \
--num_negatives 512 \
--train_batch_size 16 \
--show_progress True \
--update_interval 100 \
--stopping_step 10 \
--id_emb id \
--split_data False \
--coldrec item \
--transformer_type 'HSTU'
