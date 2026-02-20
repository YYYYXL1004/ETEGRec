
DEVICE="cuda:7"

# Paths for Dual SCID
COLLAB_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_collab_emb_256.npy"
SEMANTIC_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_sentence-transformer_text_768.npy"
CKPT_DIR="./rqvae_ckpt/DualSCID/"

# Create checkpoint directory if it doesn't exist
mkdir -p $CKPT_DIR

# 核心修改说明：
# 1. --kmeans_iters 500: 给 K-Means 更多时间找准中心
# 2. --sk_epsilons 0.003 0.003 0.003: 强制开启 Sinkhorn，强迫 ID 分布均匀，降低冲突
# 3. --sk_iters 100: 配合 Sinkhorn，保证平衡算法收敛
# 4. --batch_size 2048: 5090 显存大，大 Batch 聚类更准

python -u main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 2048 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --e_dim 128 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 \
  --sk_epsilons 0.003 0.003 0.003 \
  --kmeans_iters 500 \
  --sk_iters 100 \
  --layers 1024 512 256 \
  --vq_type vq \
  --loss_type mse \
  --dist l2 \
  --device $DEVICE \
  --kmeans_init True \
  --collab_path $COLLAB_PATH \
  --semantic_path $SEMANTIC_PATH \
  --ckpt_dir $CKPT_DIR
