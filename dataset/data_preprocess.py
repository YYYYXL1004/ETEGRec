import json
import gzip
import pandas as pd
from collections import defaultdict
import os
import random
import ast

def parse(path):
    # 为了向后兼容，同时支持 .json 和 .json.gz
    if path.endswith('.gz'):
        g = gzip.open(path, 'rb') # 以二进制模式打开
    else:
        g = open(path, 'r', encoding='utf-8')
    for l in g:
        # 如果是二进制，先解码
        if isinstance(l, bytes):
            l = l.decode('utf-8')
        # 使用 ast.literal_eval 代替 json.loads 来处理非严格的JSON格式
        yield ast.literal_eval(l.strip())

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
    output_path = os.path.join(base_path, 'instrument2014')
    
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

    # 确保所有交互的物品都在元数据中
    all_items_in_meta = set(meta_df['asin'])
    reviews_df = reviews_df[reviews_df['asin'].isin(all_items_in_meta)]

    # 5-core 过滤，仅过滤用户
    print("Filtering data (5-core on users)...")
    while True:
        # 计算每个用户的交互次数
        user_counts = reviews_df['reviewerID'].value_counts()
        
        # 找出少于5次交互的用户
        weak_users = user_counts[user_counts < 5].index
        
        # 如果没有弱用户，则过滤完成
        if len(weak_users) == 0:
            break
            
        # 移除弱用户的记录
        reviews_df = reviews_df[~reviews_df['reviewerID'].isin(weak_users)]

    # 按用户和时间排序
    print("Grouping and sorting interactions...")
    reviews_df = reviews_df.sort_values(by=['reviewerID', 'unixReviewTime'])
    
    interactions = defaultdict(list)
    for _, row in reviews_df.iterrows():
        interactions[row['reviewerID']].append(row['asin'])

    # 再次过滤，确保每个用户在过滤后仍有至少3次交互
    print("Filtering users with less than 3 interactions after 5-core filtering...")
    user_list = list(interactions.keys())
    for user_id in user_list:
        if len(interactions[user_id]) < 3:
            del interactions[user_id]

    # 创建用户和物品的映射
    print("Creating user and item maps...")
    all_users = sorted(list(interactions.keys()))
    all_items_in_reviews = sorted(list(reviews_df['asin'].unique()))
    
    item_map = {"[PAD]": 0}
    for i, item in enumerate(all_items_in_reviews):
        item_map[item] = i + 1

    with open(os.path.join(output_path, 'instrument.emb_map.json'), 'w') as f:
        json.dump(item_map, f, indent=4)

    # 按留一法分割数据集
    print("Splitting data into train, valid, test using leave-one-out...")
    
    train_file = open(os.path.join(output_path, 'instrument.train.jsonl'), 'w')
    valid_file = open(os.path.join(output_path, 'instrument.valid.jsonl'), 'w')
    test_file = open(os.path.join(output_path, 'instrument.test.jsonl'), 'w')

    for user_id in all_users:
        history = interactions[user_id]
        
        # 1. 生成测试集: 用[:-1]预测[-1]
        test_record = {
            "user_id": user_id,
            "target_id": history[-1],
            "inter_history": history[:-1]
        }
        test_file.write(json.dumps(test_record) + '\n')
        
        # 2. 生成验证集: 用[:-2]预测[-2]
        valid_record = {
            "user_id": user_id,
            "target_id": history[-2],
            "inter_history": history[:-2]
        }
        valid_file.write(json.dumps(valid_record) + '\n')
        
        # 3. 生成训练集: 遍历历史记录，为每个交互创建一个样本
        # 至少需要2个item才能构成第一条训练数据 (history[0] -> history[1])
        if len(history) > 2:
            for i in range(1, len(history) - 2):
                train_record = {
                    "user_id": user_id,
                    "target_id": history[i],
                    "inter_history": history[:i]
                }
                train_file.write(json.dumps(train_record) + '\n')

    train_file.close()
    valid_file.close()
    test_file.close()

    print(f"Generated datasets for {len(all_users)} users.")
    print("Preprocessing finished.")

if __name__ == '__main__':
    main()
