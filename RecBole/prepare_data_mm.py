#!/usr/bin/env python3
"""
多模态数据准备脚本 - 在 prepare_data_2018.py 基础上增加图片过滤

与 prepare_data_2018.py 的区别:
    1. 先用URL存在性初筛 + 5-core过滤，得到最终item集合
    2. 只下载最终item的图片并验证完整性 (避免下载大量无用图片)
    3. 踢掉下载失败的item，必要时补一轮5-core
    (参考 MACRec load_all_figures.py 的图片验证逻辑)

流程:
    URL初筛 → 5-core → 下载验证图片 → (补充5-core) → 划分 → 输出

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
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def load_image_items_url(meta_file):
    """从meta数据中提取有图片URL的item集合 (仅检查URL存在性，不下载)"""
    print(f"📷 读取元数据，筛选有图片URL的item: {meta_file}")
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
                    asin2url[asin] = image_urls[0]
            except json.JSONDecodeError:
                continue
    
    print(f"  - 元数据总item数: {total}")
    print(f"  - 有图片URL的item数: {len(asin2url)}")
    print(f"  - 无图片URL的item数: {total - len(asin2url)}")
    return asin2url


def download_and_verify(asin2url, target_asins, image_dir, max_workers=32):
    """只下载 target_asins 中的图片，返回实际下载成功的 asin 集合
    
    Args:
        asin2url: asin -> 图片URL 的完整映射
        target_asins: 需要下载的 asin 集合 (5-core 过滤后的最终 item)
        image_dir: 图片保存目录
        max_workers: 并发线程数
    Returns:
        verified_asins: 实际有可用图片的 asin 集合
    """
    os.makedirs(image_dir, exist_ok=True)
    
    verified = set()
    to_download = {}
    already_exist = 0
    no_url = 0
    
    for asin in target_asins:
        if asin not in asin2url:
            no_url += 1
            continue
        save_path = os.path.join(image_dir, f"{asin}.jpg")
        if os.path.exists(save_path) and os.path.getsize(save_path) > 2:
            verified.add(asin)
            already_exist += 1
        else:
            to_download[asin] = asin2url[asin]
    
    print(f"\n📥 下载图片 (仅 5-core 后的 {len(target_asins)} 个item)")
    print(f"  - 已存在: {already_exist}, 待下载: {len(to_download)}, 无URL: {no_url}")
    
    if to_download:
        download_ok = 0
        download_fail = 0
        
        def _download_one(asin_url):
            asin, url = asin_url
            save_path = os.path.join(image_dir, f"{asin}.jpg")
            return asin, download_image(url, save_path)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download_one, item): item 
                       for item in to_download.items()}
            with tqdm(total=len(futures), desc="下载验证图片") as pbar:
                for future in as_completed(futures):
                    asin, success = future.result()
                    if success:
                        verified.add(asin)
                        download_ok += 1
                    else:
                        download_fail += 1
                    pbar.update(1)
        
        print(f"  - 下载结果: 成功={download_ok}, 失败={download_fail}")
    
    print(f"  - 实际可用图片: {len(verified)} / {len(target_asins)}")
    return verified


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
    
    # 步骤1: 用URL存在性初筛 + 5-core过滤 (不下载图片，速度快)
    asin2url = load_image_items_url(META_FILE)
    items_with_url = set(asin2url.keys())
    df = load_and_preprocess(REVIEW_FILE, items_with_url, MIN_INTERACTIONS)
    
    # 步骤2: 只下载 5-core 后最终 item 的图片并验证
    IMAGE_DIR = os.path.join(OUT_DIR, 'images')
    final_items = set(df['item_id'].unique())
    verified_items = download_and_verify(asin2url, final_items, IMAGE_DIR)
    
    # 步骤3: 踢掉下载失败的item，如果有的话再补一轮 5-core
    failed_items = final_items - verified_items
    if failed_items:
        print(f"\n⚠️  {len(failed_items)} 个item图片下载失败，重新过滤...")
        df = df[df['item_id'].isin(verified_items)]
        # 补一轮 5-core (下载失败可能导致某些用户/item不满足5次)
        prev_len = -1
        iteration = 0
        while len(df) != prev_len:
            iteration += 1
            prev_len = len(df)
            user_counts = df['user_id'].value_counts()
            df = df[df['user_id'].isin(user_counts[user_counts >= MIN_INTERACTIONS].index)]
            item_counts = df['item_id'].value_counts()
            df = df[df['item_id'].isin(item_counts[item_counts >= MIN_INTERACTIONS].index)]
        print(f"  补充过滤后: {len(df):,} 条, {df['user_id'].nunique():,} 用户, {df['item_id'].nunique():,} 物品")
    else:
        print(f"\n✅ 所有 {len(final_items)} 个item图片均可用，无需补充过滤")
    
    # 步骤4: 划分数据集
    df = split_data(df)
    
    # 步骤5: 保存.inter文件
    save_inter_file(df, OUT_DIR, DATASET_NAME)
    
    # 步骤6: 构建序列
    train_seqs, valid_seqs, test_seqs = build_sequences(df, MAX_SEQ_LENGTH)
    
    # 步骤7: 保存JSONL文件
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
    print(f"  1. python get_collab_emb.py --dataset {DATASET_NAME}")
    print(f"  2. python get_text_emb.py --dataset {DATASET_NAME}")
    print(f"  3. python get_image_emb.py --dataset {DATASET_NAME}")


if __name__ == '__main__':
    main()
