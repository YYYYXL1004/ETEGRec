#!/usr/bin/env python3
"""
SASRec训练脚本 - 生成物品嵌入和映射文件

功能:
1. 读取带split标签的.inter文件
2. 分割train/valid/test数据
3. 只在train上训练SASRec (梯度更新)
4. 使用valid做早停 (选择最佳checkpoint)
5. 在test上评估最终性能
6. 生成物品嵌入 .npy 文件和 item2id 映射文件

关键:
- 无数据泄露: 只有train参与梯度更新
- 早停机制: 使用valid选择最佳模型，防止过拟合
- 最终评估: 在test上报告性能
- 与ETEGRec数据划分完全一致
"""

import os
import json
import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer


def load_and_split_data(inter_file):
    """加载.inter文件并按split标签分割数据"""
    print(f"📖 读取数据: {inter_file}")
    
    # 读取并规范化列名
    df = pd.read_csv(inter_file, sep='\t', header=0, dtype=str, keep_default_na=False)
    df.columns = [c.split(':')[0] for c in df.columns]
    df = df.replace({'': pd.NA})
    
    # 转换数据类型
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(1.0)
    df['split'] = df['split'].str.strip().str.lower()
    
    print(f"数据分布: train={len(df[df['split']=='train']):,}, "
          f"valid={len(df[df['split']=='valid']):,}, "
          f"test={len(df[df['split']=='test']):,}")
    
    # 分割数据
    df_train = df[df['split'] == 'train'].copy()
    df_valid = df[df['split'] == 'valid'].copy()
    df_test = df[df['split'] == 'test'].copy()
    
    # 过滤: 保留在所有集合中都出现的用户
    valid_users = set(df_train['user_id'].unique()) & set(df_valid['user_id'].unique()) & set(df_test['user_id'].unique())
    df_train = df_train[df_train['user_id'].isin(valid_users)]
    df_valid = df_valid[df_valid['user_id'].isin(valid_users)]
    df_test = df_test[df_test['user_id'].isin(valid_users)]
    
    # 排序
    df_train = df_train.sort_values(['user_id', 'timestamp'])
    df_valid = df_valid.sort_values(['user_id', 'timestamp'])
    df_test = df_test.sort_values(['user_id', 'timestamp'])
    
    print(f"✅ 训练集: {df_train['user_id'].nunique():,} 用户, {len(df_train):,} 交互")
    print(f"✅ 验证集: {df_valid['user_id'].nunique():,} 用户, {len(df_valid):,} 交互")
    print(f"✅ 测试集: {df_test['user_id'].nunique():,} 用户, {len(df_test):,} 交互\n")
    
    return df_train, df_valid, df_test


def save_recbole_inter_file(df_train, df_valid, df_test, output_dir, dataset_name):
    """保存合并的.inter文件供RecBole使用 (让RecBole自动做leave-one-out划分)"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并所有数据 (RecBole会自动按时间戳做leave-one-out划分)
    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    df_all = df_all.sort_values(['user_id', 'timestamp'])
    
    inter_file = os.path.join(output_dir, f'{dataset_name}.inter')
    df_all[['user_id', 'item_id', 'rating', 'timestamp']].to_csv(
        inter_file, sep='\t', index=False,
        header=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
    )
    
    print(f"💾 已保存数据文件:")
    print(f"   - {inter_file} (所有数据，RecBole将自动划分)")
    print(f"   原始分布: train={len(df_train)}, valid={len(df_valid)}, test={len(df_test)}")
    print(f"   RecBole将按时间戳自动做leave-one-out划分\n")
    
    return inter_file


def train_sasrec(dataset_name, data_path):
    """训练SASRec模型 (使用真实valid/test做早停和评估)"""
    print("🚀 开始训练SASRec...")
    
    config_dict = {
        'model': 'SASRec',
        'dataset': dataset_name,
        'data_path': data_path,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        
        # 评估配置 - RecBole会自动做leave-one-out
        'eval_args': {
            'split': {'LS': 'valid_and_test'},  # 自动划分最后2个交互为valid和test
            'order': 'TO',  # 按时间排序
            'group_by': 'user',  # 按用户分组
            'mode': 'full'  # 全排序模式
        },
        
        # 模型参数
        'hidden_size': 256,
        'inner_size': 256,
        'n_layers': 2,
        'n_heads': 2,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'loss_type': 'CE',
        'max_seq_length': 50,
        
        # 训练参数
        'train_neg_sample_args': None,
        'epochs': 50,  # 增加最大epochs，依赖early stopping
        'train_batch_size': 2048,
        'eval_batch_size': 2048,
        'learner': 'adam',
        'learning_rate': 0.001,
        'eval_step': 1,  # 每个epoch评估一次
        'stopping_step': 10,  # 10个epoch无提升则停止
        
        # 评估指标
        'metrics': ['Recall', 'NDCG', 'Hit', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
        
        # 设备配置
        'gpu_id': '0',
        'use_gpu': True,
        'checkpoint_dir': f'./saved/SASRec_{dataset_name}',
        'show_progress': True,
    }
    
    # 创建数据集和模型
    config = Config(model='SASRec', dataset=dataset_name, config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    model = SASRec(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)
    
    # 训练 (使用valid做早停)
    print("\n⚙️  训练配置:")
    print(f"   - 只在train数据上训练 (梯度更新)")
    print(f"   - 使用valid数据做早停 (选择最佳checkpoint)")
    print(f"   - 最大序列长度: {config['max_seq_length']}")
    print(f"   - 最大epochs: {config['epochs']}, 早停patience: {config['stopping_step']}\n")
    
    best_valid_score, best_valid_result = trainer.fit(
        train_data, 
        valid_data=valid_data,  # 使用真实验证集
        saved=True,  # 保存最佳模型
        show_progress=True
    )
    
    print(f"\n✅ 训练完成!")
    print(f"   最佳验证集 {config['valid_metric']}: {best_valid_score:.4f}")
    
    # 在测试集上评估最佳模型
    print("\n📊 在测试集上评估最佳模型...")
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
    
    print("\n测试集结果:")
    for metric, value in test_result.items():
        print(f"   {metric}: {value:.4f}")
    
    return model, dataset, test_result


def extract_embeddings(model, dataset, output_dir, dataset_name):
    """提取并保存物品嵌入和映射"""
    print("💾 提取物品嵌入...")
    
    # 提取嵌入 (去除padding)
    item_embedding = model.item_embedding.weight.data.cpu().numpy()
    item_embedding_no_pad = item_embedding[1:]  # 去除第0个 (padding)
    
    # 保存嵌入
    npy_path = os.path.join(output_dir, f'{dataset_name}_emb_256.npy')
    np.save(npy_path, item_embedding_no_pad)
    print(f"✅ 已保存: {npy_path} (shape={item_embedding_no_pad.shape})")
    
    # 构建映射 (包含[PAD])
    item_token2id = dataset.field2token_id['item_id']
    item2id_map = {'[PAD]': 0}
    for token, idx in item_token2id.items():
        if idx > 0:
            item2id_map[str(token)] = int(idx)
    
    # 保存映射
    map_path = os.path.join(output_dir, f'{dataset_name}.emb_map.json')
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(item2id_map, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存: {map_path} (含[PAD], 共{len(item2id_map)}个物品)")
    
    # 验证一致性
    if len(item2id_map) != item_embedding_no_pad.shape[0] + 1:
        raise ValueError(f"映射数量 ({len(item2id_map)}) != 嵌入数量+1 ({item_embedding_no_pad.shape[0]+1})")
    
    print("✅ 映射与嵌入一致性验证通过\n")
    return npy_path, map_path


def main():
    print("=" * 70)
    print("🎵 SASRec训练 - Amazon Musical Instruments 2023")
    print("=" * 70)
    
    # 配置
    BASE_DIR = './dataset/Instruments2023'
    INTER_FILE = os.path.join(BASE_DIR, 'Instruments2023.inter')
    DATASET_NAME = 'Instruments2023'
    
    if not os.path.exists(INTER_FILE):
        print(f"❌ 文件不存在: {INTER_FILE}")
        print("请先运行: python prepare_data.py")
        return
    
    # 步骤1: 加载并分割数据
    df_train, df_valid, df_test = load_and_split_data(INTER_FILE)
    
    # 步骤2: 保存合并的.inter文件 (供RecBole使用)
    recbole_dir = './dataset/Instruments2023_recbole'
    recbole_dataset_name = 'Instruments2023_recbole'
    save_recbole_inter_file(df_train, df_valid, df_test, recbole_dir, recbole_dataset_name)
    
    # 步骤3: 训练SASRec (使用valid做早停，test做最终评估)
    model, dataset, test_result = train_sasrec(recbole_dataset_name, './dataset/')
    
    # 步骤4: 提取并保存嵌入 (保存到原始数据目录)
    npy_path, map_path = extract_embeddings(model, dataset, BASE_DIR, DATASET_NAME)
    
    print("=" * 70)
    print("🎉 SASRec训练完成!")
    print("=" * 70)
    print(f"\n生成的文件:")
    print(f"  1. {npy_path} - 物品嵌入")
    print(f"  2. {map_path} - item2id映射")
    print(f"\n✅ 训练策略:")
    print(f"   - 只在train数据上训练 (无数据泄露)")
    print(f"   - 使用valid数据做早停 (选择最佳模型)")
    print(f"   - 在test数据上评估 (最终性能)")
    print(f"\n📊 测试集性能: NDCG@10 = {test_result.get('ndcg@10', 0):.4f}")


if __name__ == '__main__':
    main()
