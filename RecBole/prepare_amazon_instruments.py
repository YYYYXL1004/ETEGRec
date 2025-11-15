"""
统一数据预处理脚本
功能：
1. 读取Amazon原始数据
2. 过滤和清洗
3. 统一划分 train/valid/test (记录每条数据属于哪个集合)
4. 保存 RecBole 格式 + 划分信息
"""

import json
import pandas as pd
import os
from collections import defaultdict
from datetime import datetime
import numpy as np

def load_reviews(review_file):
    """加载评论数据"""
    print("📖 正在读取评论数据...")
    reviews = []
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                review = json.loads(line.strip())
                reviews.append(review)
            except json.JSONDecodeError as e:
                if line_num % 10000 == 0:
                    print(f"⚠️  警告: 第 {line_num} 行解析失败")
                continue
    
    print(f"✅ 读取了 {len(reviews)} 条评论")
    return reviews

def preprocess_reviews(reviews, min_user_interactions=5, min_item_interactions=5):
    """预处理评论数据"""
    print("\n🔧 开始预处理评论数据...")
    
    df = pd.DataFrame(reviews)
    print(f"原始数据: {len(df)} 条交互")
    
    # 使用 parent_asin 作为 item_id
    df['item_id'] = df['parent_asin']
    
    # 处理时间戳
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'] / 1000
    else:
        df['timestamp'] = range(len(df))
    
    # 过滤缺失值
    df = df.dropna(subset=['user_id', 'item_id', 'timestamp'])
    print(f"去除缺失值后: {len(df)} 条交互")
    
    # 迭代过滤
    print(f"\n🔄 开始迭代过滤...")
    prev_len = -1
    iteration = 0
    
    while len(df) != prev_len:
        iteration += 1
        prev_len = len(df)
        
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]
        
        print(f"   迭代 {iteration}: {len(df):,} 条交互, {df['user_id'].nunique():,} 用户, {df['item_id'].nunique():,} 物品")
    
    # 按用户和时间排序
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    print(f"\n✅ 预处理完成！")
    print(f"   最终用户数: {df['user_id'].nunique():,}")
    print(f"   最终物品数: {df['item_id'].nunique():,}")
    print(f"   最终交互数: {len(df):,}")
    
    return df

def split_data_by_user(df, max_seq_length=50):
    """
    🔑 核心函数：统一划分数据
    为每个用户的交互打上标签：train/valid/test
    """
    print(f"\n🔪 正在统一划分数据集...")
    print(f"   策略: Leave-one-out per user")
    print(f"   最大序列长度: {max_seq_length}")
    
    # 为每条交互添加 split 标签
    df['split'] = 'train'  # 默认都是训练集
    
    user_groups = df.groupby('user_id')
    print(f"   用户数: {len(user_groups)}")
    
    stats = {
        'total_users': 0,
        'valid_users': 0,
        'test_users': 0,
        'skipped_users': 0,
    }
    
    for user_id, group in user_groups:
        stats['total_users'] += 1
        indices = group.index.tolist()
        n = len(indices)
        
        if n < 3:
            # 交互太少，全部标记为训练集（但会被后续过滤掉）
            stats['skipped_users'] += 1
            continue
        
        # 🔑 统一标记策略：
        # - 最后一个交互 → test
        # - 倒数第二个交互 → valid
        # - 其余 → train
        
        df.loc[indices[-1], 'split'] = 'test'
        df.loc[indices[-2], 'split'] = 'valid'
        # indices[:-2] 自动保持为 'train'
        
        stats['valid_users'] += 1
        stats['test_users'] += 1
    
    print(f"\n✅ 数据划分完成:")
    print(f"   总用户数: {stats['total_users']:,}")
    print(f"   有效用户数: {stats['valid_users']:,}")
    print(f"   跳过用户: {stats['skipped_users']:,}")
    
    # 统计各个集合的大小
    train_count = len(df[df['split'] == 'train'])
    valid_count = len(df[df['split'] == 'valid'])
    test_count = len(df[df['split'] == 'test'])
    
    print(f"\n   交互分布:")
    print(f"   训练集: {train_count:,} 条")
    print(f"   验证集: {valid_count:,} 条")
    print(f"   测试集: {test_count:,} 条")
    
    return df

def save_unified_data(df, output_dir, dataset_name='Instruments2023'):
    """
    保存统一划分的数据
    1. RecBole 格式的 .inter 文件（带 split 标签）
    2. 划分信息 JSON
    """
    print(f"\n💾 正在保存统一格式数据...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存完整的 .inter 文件（包含 split 列）
    inter_file = os.path.join(output_dir, f'{dataset_name}.inter')
    
    with open(inter_file, 'w', encoding='utf-8') as f:
        # 表头（增加 split 字段）
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\tsplit:token\n')
        
        for _, row in df.iterrows():
            user_id = str(row['user_id'])
            item_id = str(row['item_id'])
            rating = row.get('rating', 5.0)
            timestamp = row['timestamp']
            split = row['split']
            f.write(f"{user_id}\t{item_id}\t{rating}\t{timestamp}\t{split}\n")
    
    print(f"✅ 已保存交互文件: {inter_file}")
    
    # 2. 保存划分信息（方便后续使用）
    split_info = {
        'train': df[df['split'] == 'train'].index.tolist(),
        'valid': df[df['split'] == 'valid'].index.tolist(),
        'test': df[df['split'] == 'test'].index.tolist(),
    }
    
    split_file = os.path.join(output_dir, f'{dataset_name}.split.json')
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f)
    print(f"✅ 已保存划分信息: {split_file}")
    
    # 3. 保存统计信息
    stats = {
        'dataset_name': dataset_name,
        'num_users': int(df['user_id'].nunique()),
        'num_items': int(df['item_id'].nunique()),
        'num_interactions': int(len(df)),
        'train_interactions': int(len(df[df['split'] == 'train'])),
        'valid_interactions': int(len(df[df['split'] == 'valid'])),
        'test_interactions': int(len(df[df['split'] == 'test'])),
        'sparsity': float(1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique())),
    }
    
    stats_file = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ 已保存统计信息: {stats_file}")
    
    return inter_file, split_file

def main():
    """主函数"""
    print("=" * 70)
    print("🎵 统一数据预处理工具 - Musical Instruments 2023")
    print("=" * 70)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"用户: YYYYXL1004")
    print("=" * 70)
    
    # 配置
    BASE_DIR = './dataset/Instruments2023'
    REVIEW_FILE = os.path.join(BASE_DIR, 'Musical_Instruments.jsonl')
    OUTPUT_DIR = BASE_DIR
    DATASET_NAME = 'Instruments2023'
    
    MIN_USER_INTERACTIONS = 5
    MIN_ITEM_INTERACTIONS = 5
    MAX_SEQ_LENGTH = 50
    
    print(f"\n📂 输入文件: {REVIEW_FILE}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"\n⚙️  参数:")
    print(f"   最少用户交互: {MIN_USER_INTERACTIONS}")
    print(f"   最少物品交互: {MIN_ITEM_INTERACTIONS}")
    print(f"   最大序列长度: {MAX_SEQ_LENGTH}")
    
    if not os.path.exists(REVIEW_FILE):
        print(f"\n❌ 错误: 找不到文件 {REVIEW_FILE}")
        return
    
    # 步骤 1: 加载数据
    reviews = load_reviews(REVIEW_FILE)
    
    # 步骤 2: 预处理
    df = preprocess_reviews(reviews, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
    
    # 步骤 3: 统一划分
    df = split_data_by_user(df, MAX_SEQ_LENGTH)
    
    # 步骤 4: 保存
    inter_file, split_file = save_unified_data(df, OUTPUT_DIR, DATASET_NAME)
    
    print("\n" + "=" * 70)
    print("🎉 统一数据预处理完成！")
    print("=" * 70)
    print(f"\n📁 生成的文件:")
    print(f"   1. {inter_file}")
    print(f"      - RecBole 格式交互文件（含 split 标签）")
    print(f"   2. {split_file}")
    print(f"      - 数据划分信息（train/valid/test 索引）")
    print(f"   3. {OUTPUT_DIR}/dataset_stats.json")
    print(f"      - 数据统计信息")
    
    print(f"\n✨ 下一步:")
    print(f"   运行: python train_sasrec_unified.py")
    print("=" * 70)

if __name__ == '__main__':
    main()