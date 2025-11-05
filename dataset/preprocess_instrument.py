import json
import gzip
import pandas as pd
from collections import defaultdict
import os
import random

def parse(path):
    # 为了向后兼容，同时支持 .json 和 .json.gz
    if path.endswith('.gz'):
        g = gzip.open(path, 'r')
    else:
        g = open(path, 'r')
    for l in g:
        yield json.loads(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def main():
    # 定义输入输出路径
    base_path = os.path.dirname(__file__)
    instrument_2014_path = os.path.join(base_path, 'instrument2014')
    output_path = os.path.join(base_path, 'instrument')
    
    os.makedirs(output_path, exist_ok=True)

    # 兼容两种文件名
    reviews_file = 'reviews_Musical_Instruments.json.gz'
    if not os.path.exists(os.path.join(instrument_2014_path, reviews_file)):
        reviews_file = 'reviews_Musical_Instruments.json'

    meta_file = 'meta_Musical_Instruments.json.gz'
    if not os.path.exists(os.path.join(instrument_2014_path, meta_file)):
        meta_file = 'meta_Musical_Instruments.json'

    reviews_path = os.path.join(instrument_2014_path, reviews_file)
    meta_path = os.path.join(instrument_2014_path, meta_file)

    print("Loading data...")
    reviews_df = get_df(reviews_path)
    meta_df = get_df(meta_path)
    
    # 仅保留有用的列
    reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
    meta_df = meta_df[['asin']]

    # 5-core 过滤
    print("Filtering data (5-core)...")
    while True:
        user_counts = reviews_df['reviewerID'].value_counts()
        item_counts = reviews_df['asin'].value_counts()
        
        # 找出少于5次交互的用户和物品
        weak_users = user_counts[user_counts < 5].index
        weak_items = item_counts[item_counts < 5].index
        
        if len(weak_users) == 0 and len(weak_items) == 0:
            break
            
        # 移除相关记录
        reviews_df = reviews_df[~reviews_df['reviewerID'].isin(weak_users)]
        reviews_df = reviews_df[~reviews_df['asin'].isin(weak_items)]

    # 确保所有交互的物品都在元数据中
    all_items = set(meta_df['asin'])
    reviews_df = reviews_df[reviews_df['asin'].isin(all_items)]

    # 按用户和时间排序
    print("Grouping and sorting interactions...")
    reviews_df = reviews_df.sort_values(by=['reviewerID', 'unixReviewTime'])
    
    interactions = defaultdict(list)
    for _, row in reviews_df.iterrows():
        interactions[row['reviewerID']].append(row['asin'])

    # 过滤掉交互历史少于3的用户（因为需要至少 train, valid, test 各一个）
    user_list = list(interactions.keys())
    for user_id in user_list:
        if len(interactions[user_id]) < 3:
            del interactions[user_id]

    # 创建用户和物品的映射
    print("Creating user and item maps...")
    all_users = sorted(list(interactions.keys()))
    all_items_in_reviews = sorted(list(reviews_df['asin'].unique()))
    
    user_map = {user: i for i, user in enumerate(all_users)}
    item_map = {item: i for i, item in enumerate(all_items_in_reviews)}

    emb_map = {
        "user_map": user_map,
        "item_map": item_map
    }
    with open(os.path.join(output_path, 'instrument.emb_map.json'), 'w') as f:
        json.dump(emb_map, f, indent=4)

    # 分割数据集
    print("Splitting data into train, valid, test...")
    random.seed(42)
    random.shuffle(all_users)
    
    num_users = len(all_users)
    train_split = int(num_users * 0.8)
    valid_split = int(num_users * 0.9)
    
    train_users = all_users[:train_split]
    valid_users = all_users[train_split:valid_split]
    test_users = all_users[valid_split:]

    def create_jsonl(users, dataset_type):
        output_file = os.path.join(output_path, f'instrument.{dataset_type}.jsonl')
        with open(output_file, 'w') as f:
            for user_id in users:
                history = interactions[user_id]
                if dataset_type == 'train':
                    # 训练集：使用到倒数第二个之前的所有交互来预测倒数第二个
                    if len(history) > 2:
                        record = {
                            "user_id": user_id,
                            "target_id": history[-2],
                            "inter_history": history[:-2]
                        }
                        f.write(json.dumps(record) + '\n')
                elif dataset_type == 'valid':
                    # 验证集：使用到倒数第二个之前的所有交互来预测倒数第二个
                    record = {
                        "user_id": user_id,
                        "target_id": history[-2],
                        "inter_history": history[:-2]
                    }
                    f.write(json.dumps(record) + '\n')
                elif dataset_type == 'test':
                    # 测试集：使用到最后一个之前的所有交互来预测最后一个
                    record = {
                        "user_id": user_id,
                        "target_id": history[-1],
                        "inter_history": history[:-1]
                    }
                    f.write(json.dumps(record) + '\n')

    print(f"Generating train set ({len(train_users)} users)...")
    create_jsonl(train_users, 'train')
    
    print(f"Generating valid set ({len(valid_users)} users)...")
    create_jsonl(valid_users, 'valid')

    print(f"Generating test set ({len(test_users)} users)...")
    create_jsonl(test_users, 'test')

    print("Preprocessing finished.")

if __name__ == '__main__':
    main()
