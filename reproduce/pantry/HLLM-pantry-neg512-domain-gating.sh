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
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES="0,5,6,7" python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir/HLLM/prime_pantry/domain_gating \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--dataset prime_pantry \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--text_path ../information \
--item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--num_negatives 512 \
--train_batch_size 12 \
--seed 42 \
--stage 3 \
--split_data False \
--id_emb domain_gating \
--coldrec item \
--transformer_type user_llm \
--by_case True



CUDA_VISIBLE_DEVICES="0,5,6,7" python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir/HLLM/prime_pantry/gating \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--dataset prime_pantry \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--text_path ../information \
--item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--num_negatives 512 \
--train_batch_size 12 \
--seed 42 \
--stage 3 \
--split_data False \
--id_emb gating \
--coldrec item \
--transformer_type user_llm \
--by_case True


CUDA_VISIBLE_DEVICES="0,5,6,7" python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir/HLLM/prime_pantry/weighted_avg \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--dataset prime_pantry \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--text_path ../information \
--item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--num_negatives 512 \
--train_batch_size 12 \
--seed 42 \
--stage 3 \
--split_data False \
--id_emb weighted_avg \
--coldrec item \
--transformer_type user_llm \
--by_case True


CUDA_VISIBLE_DEVICES="0,5,6,7" python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir/HLLM/prime_pantry/weighted_sum \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--dataset prime_pantry \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--text_path ../information \
--item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--num_negatives 512 \
--train_batch_size 12 \
--seed 42 \
--stage 3 \
--split_data False \
--id_emb weighted_sum \
--coldrec item \
--transformer_type user_llm \
--by_case True


CUDA_VISIBLE_DEVICES="0,5,6,7" python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir/HLLM/prime_pantry/concat \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--dataset prime_pantry \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--text_path ../information \
--item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--num_negatives 512 \
--train_batch_size 12 \
--seed 42 \
--stage 3 \
--split_data False \
--id_emb concat \
--coldrec item \
--transformer_type user_llm \
--by_case True



CUDA_VISIBLE_DEVICES="0,5,6,7" python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--MAX_ITEM_LIST_LENGTH 50 \
--epochs 5 \
--optim_args.learning_rate 1e-4 \
--checkpoint_dir saved_dir/HLLM/prime_pantry/by_case \
--loss nce \
--MAX_TEXT_LENGTH 256 \
--dataset prime_pantry \
--gradient_checkpointing True \
--text_keys '[\"title\",\"description\"]' \
--text_path ../information \
--item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
--num_negatives 512 \
--train_batch_size 12 \
--seed 42 \
--stage 3 \
--split_data False \
--id_emb by_case \
--coldrec item \
--transformer_type user_llm \
--by_case True