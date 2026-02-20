
DEVICE="cuda:7"

# Paths for Triple Modality (Collab + Text + Image)
COLLAB_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_collab_emb_256.npy"
SEMANTIC_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_sentence-transformer_text_768.npy"
IMAGE_PATH="/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018_MM/Instrument2018_MM_clip_image_768.npy"
CKPT_DIR="./rqvae_ckpt/TripleMM/"

mkdir -p $CKPT_DIR

# layers: 1792 -> 1024 -> 512 -> 256 -> e_dim(128)
# 输入维度 1792 = collab(256) + text(768) + image(768)

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
  --image_path $IMAGE_PATH \
  --ckpt_dir $CKPT_DIR
