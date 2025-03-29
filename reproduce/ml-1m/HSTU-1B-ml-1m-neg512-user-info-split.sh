#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# 1B: 16 H100s for ≈ 2days
cd code 
export NCCL_SOCKET_IFNAME=lo
export PATH=~/.conda/envs/hllm/bin:$PATH
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0        # 启用 InfiniBand（如果适用）
export NCCL_NET_GDR_LEVEL=PHB   # 优化 GPU 直接通信
export NCCL_TIMEOUT=30 


CUDA_VISIBLE_DEVICES="4,5,6,7" python3 main.py \
--config_file IDNet/hstu.yaml overall/ID_deepspeed.yaml \
--optim_args.learning_rate 1e-3 \
--loss nce \
--train_batch_size 12 \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 100 \
--dataset ml-1m-user-info \
--hidden_dropout_prob 0.5 \
--attn_dropout_prob 0.5 \
--n_layers 22 \
--n_heads 32 \
--item_embedding_size 2048 \
--hstu_embedding_size 2048 \
--fix_temp True \
--gradient_checkpointing True \
--num_negatives 512 \
--show_progress True \
--update_interval 100 \
--seed 42 \
--stage 3 \
--split_data True \
--id_emb None
--checkpoint_dir saved_dir/hstu/ml-1m-user-info-split \
--stopping_step 10 
