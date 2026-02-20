#!/usr/bin/env python3
"""
协同过滤Embedding提取脚本 (get_collab_emb.py)

训练 SASRec 模型，提取物品协同嵌入 (.npy) 和 item2id 映射 (.json)。

流程:
    1. 读取带 split 标签的 .inter 文件
    2. 合并为 RecBole 格式 (RecBole 自动 leave-one-out 划分)
    3. 训练 SASRec (train 梯度更新, valid 早停, test 最终评估)
    4. 提取 item_embedding 保存为 .npy + emb_map.json

用法:
    python get_collab_emb.py --dataset Instrument2018_MM
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

# ================= 5090 精度补丁 =================
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# PyTorch 2.6+ 兼容: torch.load 默认 weights_only=True 会导致 RecBole 报错
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

from recbole.utils import init_seed
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="训练SASRec并提取协同嵌入")
    parser.add_argument("--dataset", type=str, default="Instrument2018_5090")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--stopping_step", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--gpu_id", type=str, default="1")
    return parser.parse_args()


def load_and_split_inter(inter_file):
    """加载 .inter 文件，按 split 标签分割为 train/valid/test DataFrame"""
    df = pd.read_csv(inter_file, sep='\t', header=0, dtype=str, keep_default_na=False)
    df.columns = [c.split(':')[0] for c in df.columns]
    df = df.replace({'': pd.NA})
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(1.0)
    df['split'] = df['split'].str.strip().str.lower()

    df_train = df[df['split'] == 'train'].copy()
    df_valid = df[df['split'] == 'valid'].copy()
    df_test = df[df['split'] == 'test'].copy()

    # 只保留在三个集合中都出现的用户
    valid_users = (set(df_train['user_id'].unique())
                   & set(df_valid['user_id'].unique())
                   & set(df_test['user_id'].unique()))
    df_train = df_train[df_train['user_id'].isin(valid_users)].sort_values(['user_id', 'timestamp'])
    df_valid = df_valid[df_valid['user_id'].isin(valid_users)].sort_values(['user_id', 'timestamp'])
    df_test = df_test[df_test['user_id'].isin(valid_users)].sort_values(['user_id', 'timestamp'])

    print(f"数据分布: train={len(df_train):,}, valid={len(df_valid):,}, test={len(df_test):,}")
    print(f"用户数: {len(valid_users):,}")
    return df_train, df_valid, df_test


def save_recbole_inter(df_train, df_valid, df_test, output_dir, dataset_name):
    """合并并保存为 RecBole 格式 .inter 文件 (RecBole 自动 leave-one-out)"""
    os.makedirs(output_dir, exist_ok=True)
    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    df_all = df_all.sort_values(['user_id', 'timestamp'])

    inter_file = os.path.join(output_dir, f'{dataset_name}.inter')
    df_all[['user_id', 'item_id', 'rating', 'timestamp']].to_csv(
        inter_file, sep='\t', index=False,
        header=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
    )
    print(f"已保存 RecBole 格式: {inter_file}")
    return inter_file


def train_sasrec(recbole_dataset_name, data_path, args):
    """训练 SASRec，返回 (model, dataset, test_result)"""
    config_dict = {
        'model': 'SASRec',
        'dataset': recbole_dataset_name,
        'data_path': data_path,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'seed': args.seed,
        'reproducibility': True,
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'group_by': 'user',
            'mode': 'full',
        },
        # 模型
        'hidden_size': args.hidden_size,
        'inner_size': args.hidden_size,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'loss_type': 'CE',
        'max_seq_length': args.max_seq_length,
        # 训练
        'train_neg_sample_args': None,
        'epochs': args.epochs,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.train_batch_size,
        'learner': 'adam',
        'learning_rate': args.learning_rate,
        'eval_step': 1,
        'stopping_step': args.stopping_step,
        # 评估
        'metrics': ['Recall', 'NDCG', 'Hit', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
        # 设备
        'gpu_id': args.gpu_id,
        'use_gpu': True,
        'checkpoint_dir': f'./saved/SASRec_{recbole_dataset_name}',
        'show_progress': True,
    }

    config = Config(model='SASRec', dataset=recbole_dataset_name, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = SASRec(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)

    print(f"\n训练配置: epochs={args.epochs}, early_stop={args.stopping_step}, "
          f"hidden={args.hidden_size}, layers={args.n_layers}")

    best_valid_score, _ = trainer.fit(train_data, valid_data=valid_data, saved=True, show_progress=True)
    print(f"最佳验证 NDCG@10: {best_valid_score:.4f}")

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
    print("\n测试集结果:")
    for metric, value in test_result.items():
        print(f"  {metric}: {value:.4f}")

    return model, dataset, test_result


def extract_embeddings(model, dataset, output_dir, dataset_name):
    """提取 item embedding 和 item2id 映射，保存到 output_dir"""
    # embedding: 去掉 index 0 (padding)
    item_emb = model.item_embedding.weight.data.cpu().numpy()
    item_emb_no_pad = item_emb[1:]

    emb_dim = item_emb_no_pad.shape[1]
    npy_path = os.path.join(output_dir, f'{dataset_name}_collab_emb_{emb_dim}.npy')
    np.save(npy_path, item_emb_no_pad)
    print(f"已保存嵌入: {npy_path} (shape={item_emb_no_pad.shape})")

    # item2id 映射 (含 [PAD]=0)
    item_token2id = dataset.field2token_id['item_id']
    item2id_map = {'[PAD]': 0}
    for token, idx in item_token2id.items():
        if idx > 0:
            item2id_map[str(token)] = int(idx)

    map_path = os.path.join(output_dir, f'{dataset_name}.emb_map.json')
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(item2id_map, f, indent=2, ensure_ascii=False)
    print(f"已保存映射: {map_path} (共 {len(item2id_map)} 个item, 含 [PAD])")

    assert len(item2id_map) == item_emb_no_pad.shape[0] + 1, "映射与嵌入数量不一致"
    return npy_path, map_path


def main():
    args = parse_args()

    base_dir = os.path.join(args.data_root, args.dataset)
    inter_file = os.path.join(base_dir, f'{args.dataset}.inter')

    if not os.path.exists(inter_file):
        print(f"❌ 文件不存在: {inter_file}")
        print(f"请先运行: python prepare_data_2018.py 或 python prepare_data_mm.py")
        return

    print("=" * 70)
    print(f"协同嵌入提取: {args.dataset}")
    print("=" * 70)

    # 1. 加载并分割数据
    df_train, df_valid, df_test = load_and_split_inter(inter_file)

    # 2. 保存 RecBole 格式
    recbole_name = f'{args.dataset}_recbole'
    recbole_dir = os.path.join(base_dir, recbole_name)
    save_recbole_inter(df_train, df_valid, df_test, recbole_dir, recbole_name)

    # 3. 训练 SASRec
    model, dataset, test_result = train_sasrec(recbole_name, base_dir, args)

    # 4. 提取嵌入
    npy_path, map_path = extract_embeddings(model, dataset, base_dir, args.dataset)

    print("\n" + "=" * 70)
    print(f"✅ 完成! NDCG@10 = {test_result.get('ndcg@10', 0):.4f}")
    print(f"   嵌入: {npy_path}")
    print(f"   映射: {map_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
