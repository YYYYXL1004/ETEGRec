#!/usr/bin/env python3
"""
数据准备脚本 - 从Amazon 2014原始数据生成RecBole .inter文件和ETEGRec所需的train/valid/test.jsonl

功能:
1. 读取Amazon 2014原始JSON数据
2. 数据清洗和过滤 (最少5次交互)
3. 统一划分train/valid/test (leave-one-out)
4. 生成RecBole格式的.inter文件 (带split标签)
5. 生成ETEGRec格式的train/valid/test.jsonl文件

数据划分策略:
- 每个用户最后一个交互 → test
- 每个用户倒数第二个交互 → valid
- 其余交互 → train
"""

import json
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm


def load_and_preprocess(review_file, min_interactions=5):
    """加载并预处理Amazon 2014评论数据"""
    print(f"📖 读取数据: {review_file}")
    
    # 读取JSON (2014版本是一行一个JSON对象，但不是标准JSONL)
    reviews = []
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(reviews)
    print(f"原始数据: {len(df):,} 条交互")
    
    # 2014版本字段映射
    df = df.rename(columns={
        'reviewerID': 'user_id',
        'asin': 'item_id',
        'overall': 'rating',
        'unixReviewTime': 'timestamp'
    })
    
    # 确保必要字段存在
    df = df.dropna(subset=['user_id', 'item_id', 'timestamp'])
    
    # 2014版本的timestamp已经是秒级，不需要除以1000
    df['timestamp'] = df['timestamp'].astype(float)
    
    # 迭代过滤 (保留至少min_interactions次交互的用户和物品)
    print(f"🔄 迭代过滤 (最少{min_interactions}次交互)...")
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        iteration += 1
        prev_len = len(df)
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= min_interactions].index)]
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= min_interactions].index)]
        print(f"  迭代{iteration}: {len(df):,} 条, {df['user_id'].nunique():,} 用户, {df['item_id'].nunique():,} 物品")
    
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    print(f"✅ 预处理完成: {df['user_id'].nunique():,} 用户, {df['item_id'].nunique():,} 物品, {len(df):,} 交互\n")
    return df


def split_data(df):
    """统一划分train/valid/test (leave-one-out)"""
    print("🔪 划分数据集 (leave-one-out)...")
    df['split'] = 'train'
    
    for user_id, group in df.groupby('user_id'):
        indices = group.index.tolist()
        if len(indices) >= 3:
            df.loc[indices[-1], 'split'] = 'test'
            df.loc[indices[-2], 'split'] = 'valid'
    
    train_cnt = len(df[df['split'] == 'train'])
    valid_cnt = len(df[df['split'] == 'valid'])
    test_cnt = len(df[df['split'] == 'test'])
    print(f"✅ 划分完成: train={train_cnt:,}, valid={valid_cnt:,}, test={test_cnt:,}\n")
    return df


def save_inter_file(df, output_dir, dataset_name):
    """保存RecBole格式的.inter文件 (带split标签)"""
    os.makedirs(output_dir, exist_ok=True)
    inter_file = os.path.join(output_dir, f'{dataset_name}.inter')
    
    with open(inter_file, 'w', encoding='utf-8') as f:
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\tsplit:token\n')
        for _, row in df.iterrows():
            f.write(f"{row['user_id']}\t{row['item_id']}\t{row['rating']}\t{row['timestamp']}\t{row['split']}\n")
    
    print(f"✅ 已保存: {inter_file}")
    return inter_file


def build_sequences(df, max_seq_length=50):
    """根据split标签构建序列数据"""
    print(f"🔨 构建序列 (max_length={max_seq_length})...")
    
    df = df.sort_values(['user_id', 'timestamp'])
    train_seqs, valid_seqs, test_seqs = [], [], []
    
    for user_id, group in tqdm(df.groupby('user_id'), desc="构建序列"):
        items = group['item_id'].tolist()
        splits = group['split'].tolist()
        
        if len(items) < 3:
            continue
        
        # 找到valid和test的位置
        valid_idx = next((i for i, s in enumerate(splits) if s == 'valid'), None)
        test_idx = next((i for i, s in enumerate(splits) if s == 'test'), None)
        
        if valid_idx is None or test_idx is None:
            continue
        
        # 训练集: 增量序列 (每个train位置生成一个样本)
        for i in range(1, valid_idx):
            history = items[:i][-max_seq_length:]
            train_seqs.append({
                'user_id': user_id,
                'target_id': items[i],
                'inter_history': history
            })
        
        # 验证集: valid位置的样本
        valid_history = items[:valid_idx][-max_seq_length:]
        valid_seqs.append({
            'user_id': user_id,
            'target_id': items[valid_idx],
            'inter_history': valid_history
        })
        
        # 测试集: test位置的样本
        test_history = items[:test_idx][-max_seq_length:]
        test_seqs.append({
            'user_id': user_id,
            'target_id': items[test_idx],
            'inter_history': test_history
        })
    
    print(f"✅ 序列构建完成: train={len(train_seqs):,}, valid={len(valid_seqs):,}, test={len(test_seqs):,}\n")
    return train_seqs, valid_seqs, test_seqs


def save_jsonl(data, output_file):
    """保存为JSONL格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ 已保存: {output_file}")


def main():
    print("=" * 70)
    print("🎵 数据准备 - Amazon Musical Instruments 2014")
    print("=" * 70)
    
    # 配置
    BASE_DIR = './dataset/Instrument2014'
    REVIEW_FILE = os.path.join(BASE_DIR, 'reviews_Musical_Instruments.json')
    DATASET_NAME = 'Instrument2014'
    MIN_INTERACTIONS = 5
    MAX_SEQ_LENGTH = 50
    
    if not os.path.exists(REVIEW_FILE):
        print(f"❌ 文件不存在: {REVIEW_FILE}")
        return
    
    # 步骤1: 加载和预处理
    df = load_and_preprocess(REVIEW_FILE, MIN_INTERACTIONS)
    
    # 步骤2: 划分数据集
    df = split_data(df)
    
    # 步骤3: 保存.inter文件
    save_inter_file(df, BASE_DIR, DATASET_NAME)
    
    # 步骤4: 构建序列
    train_seqs, valid_seqs, test_seqs = build_sequences(df, MAX_SEQ_LENGTH)
    
    # 步骤5: 保存JSONL文件
    print("💾 保存JSONL文件...")
    save_jsonl(train_seqs, os.path.join(BASE_DIR, f'{DATASET_NAME}.train.jsonl'))
    save_jsonl(valid_seqs, os.path.join(BASE_DIR, f'{DATASET_NAME}.valid.jsonl'))
    save_jsonl(test_seqs, os.path.join(BASE_DIR, f'{DATASET_NAME}.test.jsonl'))
    
    # 保存统计信息
    stats = {
        'num_users': int(df['user_id'].nunique()),
        'num_items': int(df['item_id'].nunique()),
        'num_interactions': int(len(df)),
        'train_interactions': int(len(df[df['split'] == 'train'])),
        'valid_interactions': int(len(df[df['split'] == 'valid'])),
        'test_interactions': int(len(df[df['split'] == 'test'])),
        'train_sequences': len(train_seqs),
        'valid_sequences': len(valid_seqs),
        'test_sequences': len(test_seqs),
    }
    stats_file = os.path.join(BASE_DIR, 'dataset_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ 已保存: {stats_file}")
    
    print("\n" + "=" * 70)
    print("🎉 数据准备完成!")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  1. {DATASET_NAME}.inter - RecBole格式 (带split标签)")
    print(f"  2. {DATASET_NAME}.train.jsonl - ETEGRec训练集")
    print(f"  3. {DATASET_NAME}.valid.jsonl - ETEGRec验证集")
    print(f"  4. {DATASET_NAME}.test.jsonl - ETEGRec测试集")
    print(f"  5. dataset_stats.json - 数据统计")


if __name__ == '__main__':
    main()
