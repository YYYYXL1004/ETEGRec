import argparse
import os
import random
import torch
# ================= 5090 迁移专用补丁 (完整版) =================
# 1. 强制关闭 TF32，防止精度损失 (最重要)
torch.backends.cuda.matmul.allow_tf32 = False 
torch.backends.cudnn.allow_tf32 = False

# 2. 开启全局确定性算法，防止并行计算带来的随机误差
# warn_only=True 表示如果某个算子没有确定性实现，只报错警告但不中断程序
torch.use_deterministic_algorithms(True, warn_only=True)
# 3. 设置环境变量，强制 CUDA 算子确定性 (可选，加上更稳)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
from time import time
import logging
os.environ["TORCH_USE_CUDA_DSA"]="1"
from torch.utils.data import DataLoader

from datasets import EmbDataset, DualEmbDataset
from vq import RQVAE
from trainer import  Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    #training
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument('--lr_scheduler_type', type=str, default="linear", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="/media/zhengbowen/data/Amazon2018/Processed/Instruments/Instruments.emb-llama2-td.npy",
                        help="Input data path.")
    parser.add_argument("--collab_path", type=str, default=None, help="Collaborative embeddings path.")
    parser.add_argument("--semantic_path", type=str, default=None, help="Semantic embeddings path.")
    parser.add_argument("--device", type=str, default="cuda:1", help="gpu or cpu")

    # model
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256, 256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=2048, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    # parser.add_argument('--layers', type=int, nargs='+', default=[2048, 1024, 512, 256, 128, 64],
    #                     help='hidden sizes of every layer')
    parser.add_argument('--layers', type=int, nargs='+', default=[3072],
                        help='hidden sizes of every layer')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type, l1/mse/infonce")
    parser.add_argument("--dist", type=str, default="l2", help="distance measure, dot/l2/cos")
    parser.add_argument("--tau", type=int, default=0.1, help="temperature")
    parser.add_argument("--vq_type", type=str, default="vq", help="vector quantizer type, vq, ema, gumbel")
    parser.add_argument('--h_dim', type=int, default=2048, help='hidden size for gumbel softmax')
    parser.add_argument('--temperature', type=float, default=0.9, help='temperature for gumbel softmax')

    parser.add_argument("--kmeans_init", type=bool, default=False, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--moving_avg_decay", type=int, default=0.99, help="moving_average decay")
    parser.add_argument("--normalize", type=bool, default=False, help="use normalize or not")

    #save
    parser.add_argument('--save_limit', type=int, default=3)
    parser.add_argument("--ckpt_dir", type=str,
                        default="/media/zhengbowen/rqvae_ckpt/LLaMA2/", help="output directory for model")

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    if args.collab_path and args.semantic_path:
        print(f"Loading DualEmbDataset with collab_path={args.collab_path} and semantic_path={args.semantic_path}, normalize={args.normalize}")
        data = DualEmbDataset(args.collab_path, args.semantic_path, normalize=args.normalize)
    else:
        print(f"Loading Single EmbDataset with data_path={args.data_path}")
        data = EmbDataset(args.data_path)

    model = RQVAE(config=vars(args), in_dim=data.dim)
    # model = RQVAE(args=args, in_dim=data.dim)
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args,model, len(data_loader))
    best_loss, best_collision_rate, best_gini = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)
    print("Best Gini", best_gini)

