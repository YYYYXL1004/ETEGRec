#!/bin/bash
DATASET=Instrument2018

# SFT checkpoint 路径 (修改为你的实际路径)
SFT_CKPT=./myckpt/${DATASET}/Dec-04-2025_00-55-5925d6/11.pt

accelerate launch --config_file accelerate_config_ddp.yaml train_grpo.py \
    --config ./config/${DATASET}.yaml \
    --sft_ckpt ${SFT_CKPT} \
    --grpo_group_size=2 \
    --grpo_kl_coeff=0.05 \
    --grpo_temperature=1.0 \
    --grpo_reward_alpha=0.5 \
    --grpo_lr=1e-5 \
    --grpo_epochs=20 \
    --grpo_early_stop=5 \
    --grpo_batch_size=32 \
    --gradient_accumulation_steps=4