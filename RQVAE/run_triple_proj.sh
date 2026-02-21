
DEVICE="cuda:7"

# Paths for Triple Modality with PCA Projection
# collab(256) + text_proj(128) + image_proj(128) = 512 dim
COLLAB_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_collab_emb_256.npy"
SEMANTIC_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_text_proj128.npy"
IMAGE_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_image_proj128.npy"
CKPT_DIR="./rqvae_ckpt/TripleProj/"

mkdir -p $CKPT_DIR

# layers: 512 -> 256 -> e_dim(128)
# 输入维度 512 = collab(256) + text_proj(128) + image_proj(128)

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
  --layers 256 \
  --vq_type vq \
  --loss_type mse \
  --dist l2 \
  --device $DEVICE \
  --kmeans_init True \
  --collab_path $COLLAB_PATH \
  --semantic_path $SEMANTIC_PATH \
  --image_path $IMAGE_PATH \
  --ckpt_dir $CKPT_DIR
