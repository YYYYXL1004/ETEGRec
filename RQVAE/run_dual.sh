
DEVICE="cuda:0"

# Paths for Dual SCID
COLLAB_PATH="/sda/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018/Instrument2018_emb_256.npy"
SEMANTIC_PATH="/sda/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018/Instrument2018_text_768.npy"
CKPT_DIR="./rqvae_ckpt/DualSCID/"

# Create checkpoint directory if it doesn't exist
mkdir -p $CKPT_DIR

python -u main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 1024 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --e_dim 128 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 \
  --layers 1024 512 256 \
  --vq_type vq \
  --loss_type mse \
  --dist l2 \
  --device $DEVICE \
  --kmeans_init True \
  --collab_path $COLLAB_PATH \
  --semantic_path $SEMANTIC_PATH \
  --ckpt_dir $CKPT_DIR
