#!/bin/bash
DATASET=Instrument2018

# SFT checkpoint 路径 (修改为你的实际路径)
SFT_CKPT=./myckpt/${DATASET}/Dec-04-2025_00-55-5925d6/11.pt

accelerate launch --config_file accelerate_config_ddp.yaml train_grpo.py \
    --config ./config/${DATASET}.yaml \
    --sft_ckpt ${SFT_CKPT} \
    --grpo_group_size=4 \
    --grpo_kl_coeff=0.1 \
    --grpo_temperature=1.0 \
    --grpo_reward_alpha=0.3 \
    --grpo_lr=5e-6 \
    --grpo_epochs=30 \
    --grpo_early_stop=10 \
    --grpo_batch_size=24 \
    --gradient_accumulation_steps=4
