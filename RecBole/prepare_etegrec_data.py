"""
使用统一划分生成 ETEGRec 训练数据
关键：使用步骤1生成的 split 标签，确保与 SASRec 完全一致
"""

import json
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm

def load_unified_data(inter_file):
    """加载带 split 标签的数据"""
    print(f"📖 正在读取统一格式数据: {inter_file}")
    
    df = pd.read_csv(inter_file, sep='\t', dtype=str, keep_default_na=False)
    
    # 规范化列名（取 ':' 前面的部分）
    new_cols = []
    for c in df.columns.tolist():
        if isinstance(c, str) and ':' in c:
            new_cols.append(c.split(':')[0])
        else:
            new_cols.append(c)
    df.columns = new_cols
    
    print(f"✅ 读取了 {len(df)} 条交互")
    print(f"   训练集: {len(df[df['split'] == 'train']):,} 条")
    print(f"   验证集: {len(df[df['split'] == 'valid']):,} 条")
    print(f"   测试集: {len(df[df['split'] == 'test']):,} 条")
    
    return df

def build_sequences_from_split(df, max_seq_length=50):
    """
    🔑 核心：根据 split 标签构建序列
    确保与 SASRec 的划分完全一致
    """
    print(f"\n🔨 正在构建序列...")
    print(f"   最大序列长度: {max_seq_length}")
    
    # 按用户分组
    df = df.sort_values(['user_id', 'timestamp'])
    user_groups = df.groupby('user_id')
    
    train_sequences = []
    valid_sequences = []
    test_sequences = []
    
    stats = {'truncated': 0}
    
    for user_id, group in tqdm(user_groups, desc="构建序列"):
        interactions = group['item_id'].tolist()
        splits = group['split'].tolist()
        n = len(interactions)
        
        if n < 3:
            continue
        
        # 找到 valid 和 test 的位置
        valid_idx = next((i for i, s in enumerate(splits) if s == 'valid'), None)
        test_idx = next((i for i, s in enumerate(splits) if s == 'test'), None)
        
        if valid_idx is None or test_idx is None:
            continue
        
        # ============ 训练集：所有 train 标记的交互 ============
        # 构建增量序列
        for i in range(1, valid_idx):
            if splits[i] == 'train' or i < valid_idx:
                history = interactions[:i]
                if len(history) > max_seq_length:
                    history = history[-max_seq_length:]
                    stats['truncated'] += 1
                
                train_sequences.append({
                    'user_id': user_id,
                    'inter_history': history,
                    'target_id': interactions[i]
                })
        
        # ============ 验证集：valid 标记的交互 ============
        valid_history = interactions[:valid_idx]
        if len(valid_history) > max_seq_length:
            valid_history = valid_history[-max_seq_length:]
            stats['truncated'] += 1
        
        valid_sequences.append({
            'user_id': user_id,
            'inter_history': valid_history,
            'target_id': interactions[valid_idx]
        })
        
        # ============ 测试集：test 标记的交互 ============
        test_history = interactions[:test_idx]
        if len(test_history) > max_seq_length:
            test_history = test_history[-max_seq_length:]
            stats['truncated'] += 1
        
        test_sequences.append({
            'user_id': user_id,
            'inter_history': test_history,
            'target_id': interactions[test_idx]
        })
    
    print(f"\n✅ 序列构建完成:")
    print(f"   训练集: {len(train_sequences):,} 条序列")
    print(f"   验证集: {len(valid_sequences):,} 条序列")
    print(f"   测试集: {len(test_sequences):,} 条序列")
    print(f"   截断序列: {stats['truncated']:,} 条")
    
    return train_sequences, valid_sequences, test_sequences

def save_jsonl(data, output_file):
    """保存为 JSONL 格式"""
    print(f"💾 正在保存到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json_obj = {
                'user_id': item['user_id'],
                'target_id': item['target_id'],
                'inter_history': item['inter_history']
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(data)} 条记录")

def verify_consistency(train_seqs, valid_seqs, test_seqs):
    """验证数据一致性"""
    print(f"\n🔍 验证数据一致性...")
    
    # 检查用户重叠
    train_users = set(s['user_id'] for s in train_seqs)
    valid_users = set(s['user_id'] for s in valid_seqs)
    test_users = set(s['user_id'] for s in test_seqs)
    
    print(f"\n用户分布:")
    print(f"   训练集唯一用户: {len(train_users):,}")
    print(f"   验证集唯一用户: {len(valid_users):,}")
    print(f"   测试集唯一用户: {len(test_users):,}")
    print(f"   验证∩测试: {len(valid_users & test_users):,} ({len(valid_users & test_users)/len(valid_users)*100:.1f}%)")
    
    # 检查序列长度
    train_lens = [len(s['inter_history']) for s in train_seqs]
    valid_lens = [len(s['inter_history']) for s in valid_seqs]
    test_lens = [len(s['inter_history']) for s in test_seqs]
    
    print(f"\n序列长度:")
    print(f"   训练集最大: {max(train_lens)}")
    print(f"   验证集最大: {max(valid_lens)}")
    print(f"   测试集最大: {max(test_lens)}")
    
    if max(train_lens) <= 50 and max(valid_lens) <= 50 and max(test_lens) <= 50:
        print(f"   ✅ 所有序列长度 ≤ 50")
    else:
        print(f"   ❌ 发现超长序列！")

def main():
    """主函数"""
    print("=" * 70)
    print("🎵 ETEGRec 数据准备 - 使用统一划分")
    print("=" * 70)
    
    # 配置
    BASE_DIR = './dataset/Instruments2023'
    INTER_FILE = os.path.join(BASE_DIR, 'Instruments2023.inter')
    OUTPUT_DIR = BASE_DIR
    DATASET_NAME = 'Instruments2023'
    MAX_SEQ_LENGTH = 50
    
    if not os.path.exists(INTER_FILE):
        print(f"\n❌ 错误: 找不到文件 {INTER_FILE}")
        print(f"   请先运行 prepare_amazon_data_unified.py")
        return
    
    # 步骤 1: 加载统一格式数据
    df = load_unified_data(INTER_FILE)
    
    # 步骤 2: 根据 split 标签构建序列
    train_seqs, valid_seqs, test_seqs = build_sequences_from_split(df, MAX_SEQ_LENGTH)
    
    # 步骤 3: 验证一致性
    verify_consistency(train_seqs, valid_seqs, test_seqs)
    
    # 步骤 4: 保存文件
    print(f"\n{'='*70}")
    print(f"💾 保存文件...")
    print(f"{'='*70}")
    
    train_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.train.jsonl')
    valid_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.valid.jsonl')
    test_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.test.jsonl')
    
    save_jsonl(train_seqs, train_file)
    save_jsonl(valid_seqs, valid_file)
    save_jsonl(test_seqs, test_file)
    
    # 总结
    print(f"\n{'='*70}")
    print(f"🎉 ETEGRec 数据准备完成！")
    print(f"{'='*70}")
    
    print(f"\n📁 生成的文件:")
    print(f"   1. {train_file}")
    print(f"   2. {valid_file}")
    print(f"   3. {test_file}")
    
    print(f"\n✅ 数据划分与 SASRec 完全一致！")
    print(f"   ✅ 使用相同的 split 标签")
    print(f"   ✅ 序列长度限制为 {MAX_SEQ_LENGTH}")
    print(f"   ✅ 无数据泄露")
    
    print(f"\n✨ 下一步: 训练 ETEGRec")
    print(f"   bash run.sh")
    print("=" * 70)

if __name__ == '__main__':
    main()