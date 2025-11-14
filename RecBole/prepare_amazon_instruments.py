import json
import pandas as pd
import os
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

def load_reviews(review_file):
    """
    加载 Musical_Instruments.jsonl (用户评论数据)
    """
    print("📖 正在读取评论数据...")
    reviews = []
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                review = json.loads(line.strip())
                reviews.append(review)
            except json.JSONDecodeError as e:
                print(f"⚠️  警告: 第 {line_num} 行解析失败: {e}")
                continue
    
    print(f"✅ 读取了 {len(reviews)} 条评论")
    return reviews

def load_metadata(meta_file):
    """
    加载 meta_Musical_Instruments.jsonl (商品元数据)
    """
    print("📖 正在读取商品元数据...")
    metadata = {}
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                # 使用 parent_asin 作为主键
                asin = item.get('parent_asin') or item.get('asin')
                if asin:
                    metadata[asin] = item
            except json.JSONDecodeError as e:
                print(f"⚠️  警告: 第 {line_num} 行解析失败: {e}")
                continue
    
    print(f"✅ 读取了 {len(metadata)} 个商品的元数据")
    return metadata

def preprocess_reviews(reviews, min_user_interactions=5, min_item_interactions=5):
    """
    预处理评论数据 - 🔧 使用迭代过滤确保数据质量
    """
    print("\n🔧 开始预处理评论数据...")
    
    # 转换为 DataFrame
    df = pd.DataFrame(reviews)
    
    print(f"原始数据: {len(df)} 条交互")
    print(f"唯一用户数: {df['user_id'].nunique()}")
    print(f"唯一商品数: {df['parent_asin'].nunique()}")
    
    # 1. 使用 parent_asin 作为 item_id
    df['item_id'] = df['parent_asin']
    
    # 2. 处理时间戳 (毫秒 -> 秒)
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'] / 1000  # 转换为秒
    else:
        print("⚠️  警告: 没有找到 timestamp 字段，将按顺序生成")
        df['timestamp'] = range(len(df))
    
    # 3. 过滤缺失值
    df = df.dropna(subset=['user_id', 'item_id', 'timestamp'])
    print(f"去除缺失值后: {len(df)} 条交互")
    
    # 🔧 4. 迭代过滤 - 确保数据一致性
    print(f"\n🔄 开始迭代过滤...")
    prev_len = -1
    iteration = 0
    
    while len(df) != prev_len:
        iteration += 1
        prev_len = len(df)
        
        # 过滤低频用户
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(valid_users)]
        
        # 过滤低频物品
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(valid_items)]
        
        print(f"   迭代 {iteration}: {len(df):,} 条交互, {df['user_id'].nunique():,} 用户, {df['item_id'].nunique():,} 物品")
    
    # 5. 按用户和时间排序
    df = df.sort_values(['user_id', 'timestamp'])
    
    # 6. 重置索引
    df = df.reset_index(drop=True)
    
    print(f"\n✅ 预处理完成！")
    print(f"   最终用户数: {df['user_id'].nunique():,}")
    print(f"   最终物品数: {df['item_id'].nunique():,}")
    print(f"   最终交互数: {len(df):,}")
    print(f"   稀疏度: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4%}")
    
    return df

def save_recbole_format(df, output_dir, dataset_name='Instruments2023'):
    """
    保存为 RecBole 格式
    """
    print(f"\n💾 正在保存为 RecBole 格式...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存 .inter 文件
    inter_file = os.path.join(output_dir, f'{dataset_name}.inter')
    
    with open(inter_file, 'w', encoding='utf-8') as f:
        # 写入表头 (指定字段类型)
        f.write('user_id:token\titem_id:token\trating:float\ttimestamp:float\n')
        
        # 写入数据
        for _, row in df.iterrows():
            user_id = str(row['user_id'])
            item_id = str(row['item_id'])
            rating = row.get('rating', 5.0)
            timestamp = row['timestamp']
            f.write(f"{user_id}\t{item_id}\t{rating}\t{timestamp}\n")
    
    print(f"✅ 已保存交互文件: {inter_file}")
    
    # 2. 统计信息
    stats = {
        'dataset_name': dataset_name,
        'num_users': int(df['user_id'].nunique()),
        'num_items': int(df['item_id'].nunique()),
        'num_interactions': int(len(df)),
        'sparsity': float(1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique())),
        'avg_interactions_per_user': float(len(df) / df['user_id'].nunique()),
        'avg_interactions_per_item': float(len(df) / df['item_id'].nunique()),
        'timestamp_range': {
            'min': float(df['timestamp'].min()),
            'max': float(df['timestamp'].max()),
        }
    }
    
    stats_file = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ 已保存统计信息: {stats_file}")

def main():
    """
    主函数
    """
    print("=" * 70)
    print("🎵 Amazon 2023 Musical Instruments 数据集转换工具 (优化版)")
    print("=" * 70)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"用户: YYYYXL1004")
    print("=" * 70)
    
    # 配置
    BASE_DIR = './dataset/Instruments2023'
    REVIEW_FILE = os.path.join(BASE_DIR, 'Musical_Instruments.jsonl')
    META_FILE = os.path.join(BASE_DIR, 'meta_Musical_Instruments.jsonl')
    OUTPUT_DIR = BASE_DIR
    DATASET_NAME = 'Instruments2023'
    
    # 🔧 优化参数
    MIN_USER_INTERACTIONS = 5  # 用户最少交互次数
    MIN_ITEM_INTERACTIONS = 5  # 物品最少交互次数
    
    print(f"\n📂 输入文件:")
    print(f"   评论数据: {REVIEW_FILE}")
    print(f"   元数据: {META_FILE}")
    print(f"\n📂 输出目录: {OUTPUT_DIR}")
    print(f"\n⚙️  过滤参数:")
    print(f"   最少用户交互: {MIN_USER_INTERACTIONS}")
    print(f"   最少物品交互: {MIN_ITEM_INTERACTIONS}")
    
    # 检查文件是否存在
    if not os.path.exists(REVIEW_FILE):
        print(f"\n❌ 错误: 找不到评论文件 {REVIEW_FILE}")
        return
    
    # 步骤 1: 加载评论数据
    reviews = load_reviews(REVIEW_FILE)
    
    # 步骤 2: 预处理
    df = preprocess_reviews(reviews, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
    
    # 步骤 3: 保存 RecBole 格式
    save_recbole_format(df, OUTPUT_DIR, DATASET_NAME)
    
    print("\n" + "=" * 70)
    print("🎉 数据转换完成！")
    print("=" * 70)
    print(f"\n📁 生成的文件:")
    print(f"   1. {OUTPUT_DIR}/{DATASET_NAME}.inter - RecBole 交互文件 ✅")
    print(f"   2. {OUTPUT_DIR}/dataset_stats.json - 数据集统计 ✅")
    
    print(f"\n✨ 下一步:")
    print(f"   运行: python train_sasrec_instruments.py")
    print("=" * 70)

if __name__ == '__main__':
    main()