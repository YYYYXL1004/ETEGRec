#!/usr/bin/env python3
"""
train_sasrec_unified.py

更稳健的 SASRec 训练脚本（使用统一 split 标签），修复列名解析、评估设置问题，
并在训练前做额外的检查以避免评估阶段返回 None 的情况。
"""
import os
import json
import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer

def read_inter_file_normalized(inter_file):
    """
    读取 .inter 文件并规范化列名：
    将 'user_id:token' -> 'user_id'，'split:token' -> 'split' 等。
    返回 pandas.DataFrame
    """
    print(f"📖 读取交互文件: {inter_file}")
    # 使用 header=0 读取，保留原始列名
    df = pd.read_csv(inter_file, sep='\t', header=0, dtype=str, keep_default_na=False)
    # 规范化列名（取 ':' 前面的部分）
    new_cols = []
    for c in df.columns.tolist():
        if isinstance(c, str) and ':' in c:
            new_cols.append(c.split(':')[0])
        else:
            new_cols.append(c)
    df.columns = new_cols
    # 把空字符串转为 NaN 以利于后续类型转换
    df = df.replace({'': pd.NA})
    return df

def quick_checks_df(df):
    # 必需列检查
    for col in ['user_id', 'item_id', 'rating', 'timestamp', 'split']:
        if col not in df.columns:
            raise KeyError(f"缺少必需列: {col}。请确认 Instruments2023.inter 包含该列（可能名为 'split:token'）")
    # 检查 split 取值
    uniques = set(df['split'].dropna().unique())
    if not {'train','valid','test'}.issubset({u.lower() for u in uniques}):
        raise ValueError(f"split 列值应包含 'train','valid','test' 三类，目前发现: {sorted(list(uniques))}")
    # 检查 timestamp/rating 类型可转为数值
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    if df['timestamp'].isna().any():
        raise ValueError("timestamp 列包含无法解析为数值的值，请检查 .inter 文件中的时间戳")
    if df['rating'].isna().any():
        # 允许缺失 rating（可默认为1.0），但提醒用户
        print("⚠️ warning: rating 列包含无法解析为数值的值，会用 1.0 填充")
        df['rating'] = df['rating'].fillna(1.0)
    return df

def save_split_files(df, output_dir):
    """
    保存三个分割文件（RecBole 可读），并返回每个文件路径
    """
    train_df = df[df['split'] == 'train'][['user_id','item_id','rating','timestamp']]
    valid_df = df[df['split'] == 'valid'][['user_id','item_id','rating','timestamp']]
    test_df  = df[df['split'] == 'test'][['user_id','item_id','rating','timestamp']]

    # 写文件，header 需要带 recbole 的列类型标注
    train_file = os.path.join(output_dir, 'Instruments2023_train.inter')
    valid_file = os.path.join(output_dir, 'Instruments2023_valid.inter')
    test_file  = os.path.join(output_dir, 'Instruments2023_test.inter')

    train_df.to_csv(train_file, sep='\t', index=False,
                    header=['user_id:token','item_id:token','rating:float','timestamp:float'])
    valid_df.to_csv(valid_file, sep='\t', index=False,
                    header=['user_id:token','item_id:token','rating:float','timestamp:float'])
    test_df.to_csv(test_file, sep='\t', index=False,
                    header=['user_id:token','item_id:token','rating:float','timestamp:float'])

    print(f"✅ 已保存分割文件: {train_file}, {valid_file}, {test_file}")
    return train_file, valid_file, test_file

def train_and_extract_embeddings():
    print("=" * 70)
    print("🎵 SASRec 训练 - 使用统一数据划分（稳健版）")
    print("=" * 70)

    config_dict = {
        'model': 'SASRec',
        'dataset': 'Instruments2023',
        'data_path': './dataset/',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp']
        },
        # 使用 Leave-one-out split（LS），与我们按用户最后两个交互划分一致。
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'group_by': 'user',
            'mode': 'full'   # 使用 full 模式（不依赖外部负采样表）
        },
        'hidden_size': 256,
        'inner_size': 256,
        'n_layers': 2,
        'n_heads': 2,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'loss_type': 'CE',
        'max_seq_length': 50,
        'train_neg_sample_args': None,
        'epochs': 200,
        'train_batch_size': 2048,
        'eval_batch_size': 2048,
        'learner': 'adam',
        'learning_rate': 0.001,
        'eval_step': 1,
        'stopping_step': 10,
        'metrics': ['Recall', 'NDCG', 'Hit', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
        'gpu_id': '0',
        'use_gpu': True,
        'checkpoint_dir': './saved/SASRec_unified',
        'show_progress': True,
    }

    try:
        inter_file = './dataset/Instruments2023/Instruments2023.inter'
        if not os.path.exists(inter_file):
            raise FileNotFoundError(f"{inter_file} 不存在，请先运行 prepare_amazon_data_unified.py")

        # 读取并规范化列名
        df = read_inter_file_normalized(inter_file)

        # quick checks: ensure columns and types are OK
        df = quick_checks_df(df)

        # 把 split 列值标准化小写并去除空白
        df['split'] = df['split'].astype(str).str.strip().str.lower()

        # 打印分布（便于调试）
        print(f"split 值分布:\n{df['split'].value_counts()}")

        # 保存为 RecBole 可读的分割文件（RecBole 会基于 dataset name 去读取）
        output_dir = './dataset/Instruments2023'
        os.makedirs(output_dir, exist_ok=True)
        train_file, valid_file, test_file = save_split_files(df, output_dir)

        # 使用 RecBole 的配置加载数据
        # RecBole 会在 dataset/Instruments2023/ 下查找数据文件
        config = Config(model='SASRec', dataset='Instruments2023', config_dict=config_dict)
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # quick runtime checks: ensure dataloaders non-empty
        if len(train_data) == 0:
            raise RuntimeError("训练 DataLoader 为空！检查 Instruments2023_train.inter 是否正确")
        if len(valid_data) == 0:
            raise RuntimeError("验证 DataLoader 为空！检查 Instruments2023_valid.inter 是否正确")
        if len(test_data) == 0:
            raise RuntimeError("测试 DataLoader 为空！检查 Instruments2023_test.inter 是否正确")

        # 创建并训练模型
        model = SASRec(config, train_data.dataset).to(config['device'])
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=True, show_progress=config['show_progress'])

        print(f"\n✅ 训练完成! 最佳验证 NDCG@10: {best_valid_score:.4f}")

        # 评估
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
        print("\n📊 测试结果:")
        for metric, value in test_result.items():
            print(f"   {metric}: {value:.4f}")

        # 提取并保存嵌入
        print("\n💾 正在提取物品嵌入...")
        item_embedding = model.item_embedding.weight.data.cpu().numpy()
        item_embedding_no_pad = item_embedding[1:]
        npy_path = os.path.join(output_dir, 'Instruments2023_emb_256.npy')
        np.save(npy_path, item_embedding_no_pad)
        print(f"✅ 嵌入文件已保存: {npy_path} (shape={item_embedding_no_pad.shape})")

        # 保存映射（包含 [PAD]）
        item_token2id = dataset.field2token_id['item_id']
        item2id_etegrec = {'[PAD]': 0}
        for token, idx in item_token2id.items():
            if idx > 0:
                item2id_etegrec[str(token)] = int(idx)
        map_path = os.path.join(output_dir, 'Instruments2023.emb_map.json')
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(item2id_etegrec, f, indent=2, ensure_ascii=False)
        print(f"✅ Item2ID 映射已保存: {map_path} (含 [PAD])")

        # Validate mapping size vs embeddings
        loaded_emb = np.load(npy_path)
        with open(map_path, 'r', encoding='utf-8') as f:
            loaded_map = json.load(f)
        expected_map_size = loaded_emb.shape[0] + 1
        if len(loaded_map) != expected_map_size:
            raise AssertionError(f"映射数量 ({len(loaded_map)}) != 嵌入数量+1 ({expected_map_size})")

        print("✅ 映射数量与嵌入数量一致 (含 [PAD])")
        return model, dataset, item_embedding_no_pad, test_result

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback; traceback.print_exc()
        return None, None, None, None

if __name__ == '__main__':
    train_and_extract_embeddings()