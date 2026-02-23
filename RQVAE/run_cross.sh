#!/bin/bash

DEVICE="cuda:7"

# Paths
COLLAB_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_collab_emb_256.npy"
TEXT_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_sentence-transformer_text_768.npy"
IMAGE_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_clip_image_768.npy"

# 伪标签路径 (需要先运行 gen_pseudo_labels.py 生成)
LABEL_DIR="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM"
TEXT_CLASS_INFO="${LABEL_DIR}/text_class_info.json"
IMAGE_CLASS_INFO="${LABEL_DIR}/image_class_info.json"

CKPT_DIR="./rqvae_ckpt/CrossModal/"
mkdir -p $CKPT_DIR

# Step 0: 生成伪标签 (如果还没生成)
if [ ! -f "$TEXT_CLASS_INFO" ] || [ ! -f "$IMAGE_CLASS_INFO" ]; then
    echo "Generating pseudo-labels..."
    python -u gen_pseudo_labels.py \
        --text_path $TEXT_PATH \
        --image_path $IMAGE_PATH \
        --n_clusters 512 \
        --output_dir $LABEL_DIR
fi

# Step 1: CrossRQVAE 预训练
# 输入维度: collab(256) + text/image(768) = 1024 per route
# encoder: [1024, 512, 256, 128], decoder: reverse
# 3 层 RQ, 从第 2 层 (index=1) 开始 cross-modal contrastive

python -u cross_main.py \
    --lr 1e-3 \
    --epochs 10000 \
    --batch_size 2048 \
    --weight_decay 1e-4 \
    --lr_scheduler_type linear \
    --e_dim 128 \
    --quant_loss_weight 1.0 \
    --beta 0.25 \
    --num_emb_list 256 256 256 \
    --layers 512 256 \
    --vq_type vq \
    --loss_type mse \
    --dist l2 \
    --device $DEVICE \
    --kmeans_init True \
    --kmeans_iters 500 \
    --collab_path $COLLAB_PATH \
    --text_path $TEXT_PATH \
    --image_path $IMAGE_PATH \
    --text_class_info $TEXT_CLASS_INFO \
    --image_class_info $IMAGE_CLASS_INFO \
    --begin_cross_layer 1 \
    --text_contrast_weight 0.1 \
    --image_contrast_weight 0.1 \
    --recon_contrast_weight 0.001 \
    --ckpt_dir $CKPT_DIR
