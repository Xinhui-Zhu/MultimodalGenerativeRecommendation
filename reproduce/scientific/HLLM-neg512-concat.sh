cd code 
export NCCL_SOCKET_IFNAME=lo
export PATH=~/.conda/envs/hllm/bin:$PATH
export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0        # 启用 InfiniBand（如果适用）
export NCCL_NET_GDR_LEVEL=PHB   # 优化 GPU 直接通信
export NCCL_TIMEOUT=30 
export NCCL_P2P_DISABLE=1

CUDA_VISIBLE_DEVICES="0,1,2,3" python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --MAX_ITEM_LIST_LENGTH 50 \
  --epochs 5 \
  --optim_args.learning_rate 1e-4 \
  --checkpoint_dir saved_dir/HLLM/scientific/simple_concat \
  --loss nce \
  --MAX_TEXT_LENGTH 128 \
  --dataset scientific \
  --gradient_checkpointing True \
  --text_keys '[\"title\",\"description\"]' \
  --train_batch_size 2 \
  --text_path ../information \
  --item_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --user_pretrain_dir TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --num_negatives 512 \
  --seed 42 \
  --stage 3 \
  --split_data False \
  --id_emb simple_concat \
  --coldrec item \
  --transformer_type user_llm \
  --by_case True \
  --accumulation_steps 6