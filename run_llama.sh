#!/bin/bash
# ETEGRec LLaMA2-7B 训练脚本
# 使用方式: bash run_llama.sh [GPU_IDS]
# 示例: bash run_llama.sh 0,1  (使用 GPU 0 和 1)
#       bash run_llama.sh       (默认使用所有可用 GPU)

# === 配置 ===
DATASET=Instrument2018_5090
CONFIG_FILE=./config/llama_instrument2018.yaml

# GPU 配置
GPU_IDS=${1:-"all"}  # 默认使用所有 GPU
if [ "$GPU_IDS" != "all" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

echo "=========================================="
echo "ETEGRec LLaMA2-7B Training"
echo "Dataset: $DATASET"
echo "GPUs: $GPU_IDS (count: $NUM_GPUS)"
echo "Config: $CONFIG_FILE"
echo "=========================================="

# === 动态生成 accelerate 配置 ===
cat > accelerate_config_llama.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
main_process_port: 33212
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $NUM_GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# === 运行训练 ===
accelerate launch --config_file accelerate_config_llama.yaml main_llama.py \
    --config $CONFIG_FILE \
    --lr_rec=0.0001 \
    --lr_id=0.0001 \
    --cycle=2 \
    --eval_step=2 \
    --batch_size=16 \
    --gradient_accumulation_steps=1 \
    --eval_batch_size=1 \
    --num_beams=20 \
    --rec_kl_loss=0.0001 \
    --rec_dec_cl_loss=0.0003 \
    --id_kl_loss=0.0001 \
    --id_dec_cl_loss=0.0003
