"""
CrossRQVAE 预训练入口

用法:
python cross_main.py \
    --collab_path /path/to/collab_256.npy \
    --text_path /path/to/text_768.npy \
    --image_path /path/to/image_768.npy \
    --text_class_info /path/to/text_class_info.json \
    --image_class_info /path/to/image_class_info.json \
    --ckpt_dir ./rqvae_ckpt/CrossModal/
"""

import argparse
import os
import random
import json
import torch
import numpy as np
from time import time
import logging
from torch.utils.data import DataLoader

# 5090 补丁
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from datasets import CrossModalEmbDataset
from cross_vq import CrossRQVAE
from cross_trainer import CrossTrainer


def load_class_info(path):
    """
    加载伪标签文件，将 JSON key 从 str 转回 int。
    返回 {item_idx(int): [item_indices(int)]}
    """
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        raw = json.load(f)
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="CrossRQVAE Pre-training")

    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--eval_step', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--learner', type=str, default="AdamW")
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler_type', type=str, default="linear")
    parser.add_argument('--warmup_epochs', type=int, default=50)

    # Data
    parser.add_argument('--collab_path', type=str, required=True)
    parser.add_argument('--text_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--normalize', type=bool, default=False)

    # Cross-modal
    parser.add_argument('--text_class_info', type=str, default="",
                        help="Path to text_class_info.json (image pseudo-labels)")
    parser.add_argument('--image_class_info', type=str, default="",
                        help="Path to image_class_info.json (text pseudo-labels)")
    parser.add_argument('--begin_cross_layer', type=int, default=1,
                        help="从第几层 RQ 开始做 cross-modal contrastive (0-indexed)")
    parser.add_argument('--text_contrast_weight', type=float, default=0.1)
    parser.add_argument('--image_contrast_weight', type=float, default=0.1)
    parser.add_argument('--recon_contrast_weight', type=float, default=0.001)

    # Model
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument('--e_dim', type=int, default=128)
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.25)
    parser.add_argument('--layers', type=int, nargs='+', default=[1024, 512, 256])
    parser.add_argument('--vq_type', type=str, default="vq")
    parser.add_argument('--loss_type', type=str, default="mse")
    parser.add_argument('--dist', type=str, default="l2")
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--bn', type=bool, default=False)
    parser.add_argument('--kmeans_init', type=bool, default=False)
    parser.add_argument('--kmeans_iters', type=int, default=100)

    # Save
    parser.add_argument('--save_limit', type=int, default=3)
    parser.add_argument('--ckpt_dir', type=str, default="./rqvae_ckpt/CrossModal/")
    parser.add_argument('--device', type=str, default="cuda:0")

    return parser.parse_args()


if __name__ == '__main__':
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=" * 60)
    print(args)
    print("=" * 60)

    logging.basicConfig(level=logging.DEBUG)

    # 加载伪标签
    text_class_info = load_class_info(args.text_class_info)
    image_class_info = load_class_info(args.image_class_info)
    if text_class_info:
        print(f"Loaded text_class_info: {len(text_class_info)} items")
    if image_class_info:
        print(f"Loaded image_class_info: {len(image_class_info)} items")

    # 构建 Dataset
    data = CrossModalEmbDataset(
        args.collab_path, args.text_path, args.image_path,
        normalize=args.normalize
    )

    # 构建 CrossRQVAE
    config = vars(args)
    model = CrossRQVAE(
        config=config,
        in_dim=data.dim,
        text_class_info=text_class_info,
        image_class_info=image_class_info,
    )
    print(model)

    data_loader = DataLoader(
        data, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    trainer = CrossTrainer(args, model, len(data_loader))
    best_loss, best_text_col, best_image_col = trainer.fit(data_loader)

    print(f"Best Loss: {best_loss:.4f}")
    print(f"Best Text Collision: {best_text_col:.4f}")
    print(f"Best Image Collision: {best_image_col:.4f}")
