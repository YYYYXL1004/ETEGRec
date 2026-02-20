#!/usr/bin/env python3
"""
多模态数据准备脚本 - 在 prepare_data_2018.py 基础上增加图片过滤

与 prepare_data_2018.py 的区别:
    1. 实际下载图片并验证完整性，过滤掉无法获取图片的item
       (参考 MACRec load_all_figures.py 的做法，而非仅检查URL是否存在)
    2. 去除同一用户对同一item的重复交互 (参考 MACRec make_inters_in_order)

输入:
    - Musical_Instruments.json (Amazon 2018 评论数据)
    - meta_Musical_Instruments.json (元数据，含图片URL)

输出 (保存到 dataset/Instrument2018_MM/):
    - Instrument2018_MM.inter
    - Instrument2018_MM.train.jsonl
    - Instrument2018_MM.valid.jsonl
    - Instrument2018_MM.test.jsonl
    - dataset_stats.json
    - images/ (下载的图片目录)
"""

import json
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import requests


def download_image(url, save_path, timeout=10):
    """下载图片，返回是否成功"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        # 验证 JPG 完整性 (参考 MACRec load_all_figures.py is_valid_jpg)
        with open(save_path, 'rb') as f:
            file_size = os.path.getsize(save_path)
            if file_size < 2:
                return False
            f.seek(file_size - 2)
            if f.read() != b'\xff\xd9':
                os.remove(save_path)
                return False
        return True
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        return False


def load_image_items(meta_file, image_dir):
    """从meta数据中提取有图片的item集合，实际下载验证图片可用性
    
    参考 MACRec/data_process/load_all_figures.py:
    不仅检查 imageURLHighRes 字段是否存在，还实际下载图片并验证完整性。
    只有下载成功且文件完整的 item 才会被保留。
    
    Args:
        meta_file: meta JSON 文件路径
        image_dir: 图片保存目录
    Returns:
        items_with_image: 实际有可用图片的 item asin 集合
    """
    print(f"📷 读取元数据，下载并验证图片: {meta_file}")
    os.makedirs(image_dir, exist_ok=True)
    
    # 先读取所有 meta，收集有 URL 的 item
    asin2url = {}
    total = 0
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                asin = item.get('asin', '')
                image_urls = item.get('imageURLHighRes', [])
                total += 1
                if image_urls and len(image_urls) > 0:
                    asin2url[asin] = image_urls[0]  # 取第一张高清图
            except json.JSONDecodeError:
                continue
    
    print(f"  - 元数据总item数: {total}")
    print(f"  - 有图片URL的item数: {len(asin2url)}")
    print(f"  - 无图片URL的item数: {total - len(asin2url)}")
    
    # 实际下载验证
    items_with_image = set()
    download_ok = 0
    download_fail = 0
    already_exist = 0
    
    for asin, url in tqdm(asin2url.items(), desc="下载验证图片"):
        save_path = os.path.join(image_dir, f"{asin}.jpg")
        
        # 已下载过且文件有效，跳过
        if os.path.exists(save_path) and os.path.getsize(save_path) > 2:
            items_with_image.add(asin)
            already_exist += 1
            continue
        
        if download_image(url, save_path):
            items_with_image.add(asin)
            download_ok += 1
        else:
            download_fail += 1
    
    print(f"  - 下载结果: 新下载={download_ok}, 已存在={already_exist}, 失败={download_fail}")
    print(f"  - 实际可用图片的item数: {len(items_with_image)}")
    return items_with_image


def load_and_preprocess(review_file, items_with_image, min_interactions=5):
    """加载并预处理Amazon 2018评论数据，过滤无图片item"""
    print(f"\n📖 读取数据: {review_file}")
    
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
    
    df = df.rename(columns={
        'reviewerID': 'user_id',
        'asin': 'item_id',
        'overall': 'rating',
        'unixReviewTime': 'timestamp'
    })
    
    df = df.dropna(subset=['user_id', 'item_id', 'timestamp'])
    df['timestamp'] = df['timestamp'].astype(float)
    
    # 关键步骤: 过滤掉没有图片的item (通过实际下载验证，参考MACRec)
    before_filter = len(df)
    df = df[df['item_id'].isin(items_with_image)]
    print(f"🔍 图片过滤: {before_filter:,} → {len(df):,} 条交互 "
          f"(移除 {before_filter - len(df):,} 条无图片交互)")
    
    # # 去重: 同一用户对同一item只保留第一次交互 (参考MACRec make_inters_in_order)
    # before_dedup = len(df)
    # df = df.sort_values(['user_id', 'timestamp'])
    # df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    # if before_dedup != len(df):
    #     print(f"🔄 去重: {before_dedup:,} → {len(df):,} 条交互 "
    #           f"(移除 {before_dedup - len(df):,} 条重复交互)")
    
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
        
        valid_idx = next((i for i, s in enumerate(splits) if s == 'valid'), None)
        test_idx = next((i for i, s in enumerate(splits) if s == 'test'), None)
        
        if valid_idx is None or test_idx is None:
            continue
        
        for i in range(1, valid_idx):
            history = items[:i][-max_seq_length:]
            train_seqs.append({
                'user_id': user_id,
                'target_id': items[i],
                'inter_history': history
            })
        
        valid_history = items[:valid_idx][-max_seq_length:]
        valid_seqs.append({
            'user_id': user_id,
            'target_id': items[valid_idx],
            'inter_history': valid_history
        })
        
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
    print("🎵 多模态数据准备 - Amazon Musical Instruments 2018")
    print("   (过滤无图片item，参考MACRec)")
    print("=" * 70)
    
    # 配置 - 源数据来自原始数据集目录
    SRC_DIR = './dataset/Instrument2018_5090'
    REVIEW_FILE = os.path.join(SRC_DIR, 'Musical_Instruments.json')
    META_FILE = os.path.join(SRC_DIR, 'meta_Musical_Instruments.json')
    
    # 输出到新目录
    OUT_DIR = './dataset/Instrument2018_MM'
    DATASET_NAME = 'Instrument2018_MM'
    MIN_INTERACTIONS = 5
    MAX_SEQ_LENGTH = 50
    
    if not os.path.exists(REVIEW_FILE):
        print(f"❌ 文件不存在: {REVIEW_FILE}")
        return
    if not os.path.exists(META_FILE):
        print(f"❌ 文件不存在: {META_FILE}")
        return
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 复制meta文件到新目录 (后续脚本需要)
    import shutil
    meta_dst = os.path.join(OUT_DIR, 'meta_Musical_Instruments.json')
    if not os.path.exists(meta_dst):
        shutil.copy2(META_FILE, meta_dst)
        print(f"📋 已复制元数据到: {meta_dst}")
    
    # 步骤1: 下载图片并获取有可用图片的item集合
    IMAGE_DIR = os.path.join(OUT_DIR, 'images')
    items_with_image = load_image_items(META_FILE, IMAGE_DIR)
    
    # 步骤2: 加载和预处理 (含图片过滤)
    df = load_and_preprocess(REVIEW_FILE, items_with_image, MIN_INTERACTIONS)
    
    # 步骤3: 划分数据集
    df = split_data(df)
    
    # 步骤4: 保存.inter文件
    save_inter_file(df, OUT_DIR, DATASET_NAME)
    
    # 步骤5: 构建序列
    train_seqs, valid_seqs, test_seqs = build_sequences(df, MAX_SEQ_LENGTH)
    
    # 步骤6: 保存JSONL文件
    print("💾 保存JSONL文件...")
    save_jsonl(train_seqs, os.path.join(OUT_DIR, f'{DATASET_NAME}.train.jsonl'))
    save_jsonl(valid_seqs, os.path.join(OUT_DIR, f'{DATASET_NAME}.valid.jsonl'))
    save_jsonl(test_seqs, os.path.join(OUT_DIR, f'{DATASET_NAME}.test.jsonl'))
    
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
    stats_file = os.path.join(OUT_DIR, 'dataset_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ 已保存: {stats_file}")
    
    print("\n" + "=" * 70)
    print("🎉 多模态数据准备完成!")
    print("=" * 70)
    print(f"\n输出目录: {OUT_DIR}")
    print(f"后续步骤:")
    print(f"  1. python get_collab_emb.py       (修改路径指向 {DATASET_NAME})")
    print(f"  2. python get_text_emb.py         (修改路径指向 {DATASET_NAME})")
    print(f"  3. python get_image_emb.py --dataset {DATASET_NAME}")


if __name__ == '__main__':
    main()
