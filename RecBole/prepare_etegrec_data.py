import json
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def load_recbole_interactions(inter_file):
    """
    加载 RecBole 的 .inter 文件
    """
    print(f"📖 正在读取 RecBole 交互文件: {inter_file}")
    
    data = []
    with open(inter_file, 'r', encoding='utf-8') as f:
        # 跳过表头
        header = f.readline().strip().split('\t')
        print(f"   表头: {header}")
        
        # 读取数据
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                data.append({
                    'user_id': parts[0],
                    'item_id': parts[1],
                    'rating': float(parts[2]) if len(parts) > 2 else 1.0,
                    'timestamp': float(parts[3]) if len(parts) > 3 else 0.0
                })
    
    df = pd.DataFrame(data)
    print(f"✅ 读取了 {len(df)} 条交互")
    return df

def split_sequences_by_user(df, max_seq_length=50):
    """
    按用户划分数据，使用 leave-one-out 策略
    🔧 限制序列最大长度（与作者一致）
    
    Args:
        df: 交互数据
        max_seq_length: 最大序列长度（默认50）
    """
    print(f"\n🔪 正在划分数据集...")
    print(f"   策略: Leave-one-out")
    print(f"   最大序列长度: {max_seq_length}")
    
    # 按用户和时间排序
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    train_sequences = []
    valid_sequences = []
    test_sequences = []
    
    # 按用户分组
    user_groups = df.groupby('user_id')
    print(f"   用户数: {len(user_groups)}")
    
    stats = {
        'total_users': 0,
        'train_users': 0,
        'valid_users': 0,
        'test_users': 0,
        'skipped_users': 0,
        'truncated_sequences': 0
    }
    
    for user_id, group in tqdm(user_groups, desc="处理用户"):
        stats['total_users'] += 1
        interactions = group['item_id'].tolist()
        n = len(interactions)
        
        if n < 3:
            stats['skipped_users'] += 1
            continue
        
        # ============ 训练集：增量序列 ============
        for i in range(1, n - 2):
            # 🔧 限制历史长度
            history = interactions[:i]
            if len(history) > max_seq_length:
                history = history[-max_seq_length:]
                stats['truncated_sequences'] += 1
            
            train_sequences.append({
                'user_id': user_id,
                'inter_history': history,
                'target_id': interactions[i]
            })
        
        if n > 3:
            stats['train_users'] += 1
        
        # ============ 验证集 ============
        valid_history = interactions[:-2]
        if len(valid_history) > max_seq_length:
            valid_history = valid_history[-max_seq_length:]
            stats['truncated_sequences'] += 1
        
        valid_sequences.append({
            'user_id': user_id,
            'inter_history': valid_history,
            'target_id': interactions[-2]
        })
        stats['valid_users'] += 1
        
        # ============ 测试集 ============
        test_history = interactions[:-1]
        if len(test_history) > max_seq_length:
            test_history = test_history[-max_seq_length:]
            stats['truncated_sequences'] += 1
        
        test_sequences.append({
            'user_id': user_id,
            'inter_history': test_history,
            'target_id': interactions[-1]
        })
        stats['test_users'] += 1
    
    print(f"\n✅ 数据划分完成:")
    print(f"   总用户数: {stats['total_users']:,}")
    print(f"   训练集序列: {len(train_sequences):,} (来自 {stats['train_users']:,} 个用户)")
    print(f"   验证集序列: {len(valid_sequences):,} (来自 {stats['valid_users']:,} 个用户)")
    print(f"   测试集序列: {len(test_sequences):,} (来自 {stats['test_users']:,} 个用户)")
    print(f"   跳过用户: {stats['skipped_users']:,} (交互少于3次)")
    print(f"   截断序列: {stats['truncated_sequences']:,} (超过 {max_seq_length} 长度)")
    
    return train_sequences, valid_sequences, test_sequences

def save_jsonl(data, output_file):
    """
    保存为 JSONL 格式
    """
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

def verify_data(train_seqs, valid_seqs, test_seqs, max_seq_length):
    """
    验证数据质量
    """
    print(f"\n{'='*70}")
    print(f"🔍 数据质量验证")
    print(f"{'='*70}")
    
    all_seqs = train_seqs + valid_seqs + test_seqs
    
    # 检查1: 历史长度
    hist_lens = [len(seq['inter_history']) for seq in all_seqs]
    max_len = max(hist_lens)
    
    print(f"\n✅ 历史长度检查:")
    print(f"   最大长度: {max_len}")
    print(f"   限制长度: {max_seq_length}")
    if max_len <= max_seq_length:
        print(f"   ✅ 所有序列长度都在限制范围内")
    else:
        print(f"   ❌ 发现超长序列！")
    
    # 检查2: 空历史
    empty_count = sum(1 for seq in all_seqs if len(seq['inter_history']) == 0)
    print(f"\n✅ 空历史检查:")
    print(f"   空历史序列数: {empty_count}")
    if empty_count == 0:
        print(f"   ✅ 没有空历史序列")
    else:
        print(f"   ❌ 发现 {empty_count} 个空历史序列")
    
    # 检查3: 统计分布
    print(f"\n✅ 统计分布:")
    for name, seqs in [('训练集', train_seqs), ('验证集', valid_seqs), ('测试集', test_seqs)]:
        lens = [len(s['inter_history']) for s in seqs]
        print(f"   {name}:")
        print(f"      平均长度: {np.mean(lens):.2f}")
        print(f"      中位数: {np.median(lens):.2f}")
        print(f"      最大长度: {np.max(lens)}")

def main():
    """
    主函数
    """
    print("=" * 70)
    print("🎵 ETEGRec 数据准备工具 - Musical Instruments 2023 (优化版)")
    print("=" * 70)
    print(f"当前时间: 2025-11-14 09:12:40 UTC")
    print(f"用户: YYYYXL1004")
    print("=" * 70)
    
    # ============ 配置 ============
    BASE_DIR = './dataset/Instruments2023'
    INTER_FILE = os.path.join(BASE_DIR, 'Instruments2023.inter')
    MAP_FILE = os.path.join(BASE_DIR, 'Instruments2023.emb_map.json')
    OUTPUT_DIR = BASE_DIR
    DATASET_NAME = 'Instruments2023'
    
    # 🔧 关键参数（与作者对齐）
    MAX_SEQ_LENGTH = 50  # 限制序列最大长度为50
    
    print(f"\n⚙️  配置参数:")
    print(f"   最大序列长度: {MAX_SEQ_LENGTH} (与作者一致)")
    
    # 检查文件
    if not os.path.exists(INTER_FILE):
        print(f"\n❌ 错误: 找不到交互文件 {INTER_FILE}")
        return
    
    if not os.path.exists(MAP_FILE):
        print(f"\n❌ 错误: 找不到映射文件 {MAP_FILE}")
        print(f"   请先运行 train_sasrec_instruments.py 生成映射文件")
        return
    
    # ============ 步骤 1: 加载数据 ============
    df = load_recbole_interactions(INTER_FILE)
    
    # 检查映射
    print(f"\n📖 正在读取 item2id 映射: {MAP_FILE}")
    with open(MAP_FILE, 'r', encoding='utf-8') as f:
        item2id = json.load(f)
    print(f"✅ 映射条目数: {len(item2id)}")
    if '[PAD]' in item2id:
        print(f"   包含 [PAD] token: ✅")
    else:
        print(f"   ⚠️  警告: 映射不包含 [PAD] token")
    
    # ============ 步骤 2: 划分数据集 ============
    train_sequences, valid_sequences, test_sequences = split_sequences_by_user(
        df, 
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    # ============ 步骤 3: 验证数据 ============
    verify_data(train_sequences, valid_sequences, test_sequences, MAX_SEQ_LENGTH)
    
    # ============ 步骤 4: 保存文件 ============
    print(f"\n{'='*70}")
    print(f"💾 保存文件...")
    print(f"{'='*70}")
    
    train_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.train.jsonl')
    valid_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.valid.jsonl')
    test_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.test.jsonl')
    
    save_jsonl(train_sequences, train_file)
    save_jsonl(valid_sequences, valid_file)
    save_jsonl(test_sequences, test_file)
    
    # ============ 步骤 5: 显示样例 ============
    print(f"\n{'='*70}")
    print(f"📊 数据样例")
    print(f"{'='*70}")
    
    print(f"\n训练集前3条:")
    for i, seq in enumerate(train_sequences[:3], 1):
        hist_str = str(seq['inter_history'][:3])
        if len(seq['inter_history']) > 3:
            hist_str = hist_str[:-1] + ', ...]'
        print(f"   {i}. user={seq['user_id'][:20]}..., target={seq['target_id']}, history_len={len(seq['inter_history'])}, history={hist_str}")
    
    print(f"\n验证集前3条:")
    for i, seq in enumerate(valid_sequences[:3], 1):
        hist_str = str(seq['inter_history'][:3])
        if len(seq['inter_history']) > 3:
            hist_str = hist_str[:-1] + ', ...]'
        print(f"   {i}. user={seq['user_id'][:20]}..., target={seq['target_id']}, history_len={len(seq['inter_history'])}, history={hist_str}")
    
    # ============ 总结 ============
    print(f"\n{'='*70}")
    print(f"🎉 数据准备完成!")
    print(f"{'='*70}")
    
    print(f"\n📁 生成的文件:")
    print(f"   1. {train_file}")
    print(f"      - 序列数: {len(train_sequences):,}")
    print(f"   2. {valid_file}")
    print(f"      - 序列数: {len(valid_sequences):,}")
    print(f"   3. {test_file}")
    print(f"      - 序列数: {len(test_sequences):,}")
    
    print(f"\n📁 已有的文件:")
    print(f"   4. {MAP_FILE}")
    print(f"   5. {os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_emb_256.npy')}")
    
    print(f"\n✨ 与作者数据集对齐:")
    print(f"   ✅ 最大序列长度限制为 {MAX_SEQ_LENGTH}")
    print(f"   ✅ 数据格式: {{user_id, target_id, inter_history}}")
    print(f"   ✅ 映射包含 [PAD] token")
    
    print(f"\n✨ 下一步: 训练 ETEGRec!")
    print(f"   1. 创建配置文件 config/instruments.yaml")
    print(f"   2. 修改 run.sh 中的 DATASET=Instruments2023")
    print(f"   3. 运行: bash run.sh")
    
    print(f"\n{'='*70}")

if __name__ == '__main__':
    main()