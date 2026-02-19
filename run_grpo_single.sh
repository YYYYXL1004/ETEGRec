#!/bin/bash
# ============================================================
# GRPO Training Script (Single GPU)
# ============================================================

DATASET=Instrument2018_5090
SFT_CKPT=./myckpt/${DATASET}/Dec-23-2025_19-39-c9d613/2.pt

# 使用单 GPU (指定 GPU ID，避免与正在跑的任务冲突)
CUDA_VISIBLE_DEVICES=1 python train_grpo.py \
    --config ./config/${DATASET}.yaml \
    --sft_ckpt ${SFT_CKPT} \
    --grpo_group_size=16 \
    --grpo_kl_coeff=0.04 \
    --grpo_use_beam_search=True \
    --grpo_reward_type=mixed \
    --grpo_reward_alpha=0.5 \
    --grpo_lr=1e-5 \
    --grpo_epochs=20 \
    --grpo_early_stop=5 \
    --grpo_eval_step=1 \
    --grpo_batch_size=32 \
    --grpo_max_grad_norm=0.3 \
    --gradient_accumulation_steps=1
