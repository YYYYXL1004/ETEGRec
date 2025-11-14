import json
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm

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

def split_sequences_by_user(df):
    """
    按用户划分数据，使用 leave-one-out 策略
    每个用户的交互序列：
    - 训练集：每个时间点的增量序列（历史 -> 下一个物品）
    - 验证集：前 n-2 个 -> 倒数第2个物品
    - 测试集：前 n-1 个 -> 最后一个物品
    """
    print(f"\n🔪 正在划分数据集...")
    print(f"   策略: Leave-one-out (每个用户最后2个交互作为验证和测试)")
    
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
        'skipped_users': 0
    }
    
    for user_id, group in tqdm(user_groups, desc="处理用户"):
        stats['total_users'] += 1
        interactions = group['item_id'].tolist()
        n = len(interactions)
        
        if n < 3:
            # 交互太少（少于3个），跳过该用户
            stats['skipped_users'] += 1
            continue
        
        # ============ 训练集：增量序列 ============
        # 从第2个交互开始到倒数第3个交互，每个位置都生成一个训练样本
        # 例如：[A, B, C, D, E] -> 
        #   {history: [A], target: B}
        #   {history: [A, B], target: C}
        #   {history: [A, B, C], target: D} (不包括最后两个)
        for i in range(1, n - 2):  # 从索引1到n-3
            train_sequences.append({
                'user_id': user_id,
                'inter_history': interactions[:i],
                'target_id': interactions[i]
            })
        
        if n > 3:  # 至少有4个交互才有训练数据
            stats['train_users'] += 1
        
        # ============ 验证集 ============
        # 使用前 n-2 个作为历史，倒数第2个作为目标
        # 例如：[A, B, C, D, E] -> {history: [A, B, C], target: D}
        valid_sequences.append({
            'user_id': user_id,
            'inter_history': interactions[:-2],
            'target_id': interactions[-2]
        })
        stats['valid_users'] += 1
        
        # ============ 测试集 ============
        # 使用前 n-1 个作为历史，最后一个作为目标
        # 例如：[A, B, C, D, E] -> {history: [A, B, C, D], target: E}
        test_sequences.append({
            'user_id': user_id,
            'inter_history': interactions[:-1],
            'target_id': interactions[-1]
        })
        stats['test_users'] += 1
    
    print(f"\n✅ 数据划分完成:")
    print(f"   总用户数: {stats['total_users']}")
    print(f"   训练集序列: {len(train_sequences)} (来自 {stats['train_users']} 个用户)")
    print(f"   验证集序列: {len(valid_sequences)} (来自 {stats['valid_users']} 个用户)")
    print(f"   测试集序列: {len(test_sequences)} (来自 {stats['test_users']} 个用户)")
    print(f"   跳过用户: {stats['skipped_users']} (交互少于3次)")
    
    return train_sequences, valid_sequences, test_sequences

def save_jsonl(data, output_file):
    """
    保存为 JSONL 格式
    格式：{"user_id": "xxx", "target_id": "xxx", "inter_history": [...]}
    """
    print(f"💾 正在保存到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 按照作者的格式：user_id, target_id, inter_history 的顺序
            json_obj = {
                'user_id': item['user_id'],
                'target_id': item['target_id'],
                'inter_history': item['inter_history']
            }
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存 {len(data)} 条记录")

def load_item2id_mapping(map_file):
    """
    加载 item2id 映射
    """
    print(f"\n📖 正在读取 item2id 映射: {map_file}")
    
    with open(map_file, 'r', encoding='utf-8') as f:
        item2id = json.load(f)
    
    print(f"✅ 读取了 {len(item2id)} 个物品的映射")
    return item2id

def verify_consistency(train_seqs, valid_seqs, test_seqs, item2id):
    """
    验证数据一致性
    """
    print(f"\n🔍 验证数据一致性...")
    
    # 收集所有出现的物品
    all_items = set()
    for seq in train_seqs + valid_seqs + test_seqs:
        all_items.update(seq['inter_history'])
        all_items.add(seq['target_id'])
    
    print(f"   数据中的物品数: {len(all_items)}")
    print(f"   映射中的物品数: {len(item2id)}")
    
    # 检查是否所有物品都在映射中
    missing_items = all_items - set(item2id.keys())
    if missing_items:
        print(f"⚠️  警告: 有 {len(missing_items)} 个物品不在映射中")
        print(f"   示例: {list(missing_items)[:5]}")
    else:
        print(f"✅ 所有物品都在映射中")
    
    return len(missing_items) == 0

def print_statistics(sequences, dataset_name):
    """
    打印数据集统计信息
    """
    if len(sequences) == 0:
        print(f"\n{dataset_name}:")
        print(f"  序列数量: 0")
        return
    
    hist_lens = [len(seq['inter_history']) for seq in sequences]
    unique_users = len(set(seq['user_id'] for seq in sequences))
    
    print(f"\n{dataset_name}:")
    print(f"  序列数量: {len(sequences)}")
    print(f"  唯一用户数: {unique_users}")
    print(f"  平均历史长度: {sum(hist_lens)/len(hist_lens):.2f}")
    print(f"  最小历史长度: {min(hist_lens)}")
    print(f"  最大历史长度: {max(hist_lens)}")

def main():
    """
    主函数
    """
    print("=" * 70)
    print("🎵 ETEGRec 数据准备工具 - Musical Instruments 2023")
    print("=" * 70)
    print(f"当前时间: 2025-11-14 08:28:01 UTC")
    print(f"用户: YYYYXL1004")
    print("=" * 70)
    
    # ============ 配置 ============
    BASE_DIR = './dataset/Instruments2023'
    INTER_FILE = os.path.join(BASE_DIR, 'Instruments2023.inter')
    MAP_FILE = os.path.join(BASE_DIR, 'Instruments2023.emb_map.json')
    OUTPUT_DIR = BASE_DIR
    DATASET_NAME = 'Instruments2023'
    
    # 检查文件
    if not os.path.exists(INTER_FILE):
        print(f"❌ 错误: 找不到交互文件 {INTER_FILE}")
        return
    
    if not os.path.exists(MAP_FILE):
        print(f"❌ 错误: 找不到映射文件 {MAP_FILE}")
        print(f"   请先运行 train_sasrec_instruments.py 生成映射文件")
        return
    
    # ============ 步骤 1: 加载数据 ============
    df = load_recbole_interactions(INTER_FILE)
    item2id = load_item2id_mapping(MAP_FILE)
    
    # ============ 步骤 2: 划分数据集并构建序列 ============
    train_sequences, valid_sequences, test_sequences = split_sequences_by_user(df)
    
    # ============ 步骤 3: 验证一致性 ============
    verify_consistency(train_sequences, valid_sequences, test_sequences, item2id)
    
    # ============ 步骤 4: 保存文件 ============
    print("\n" + "=" * 70)
    print("保存 JSONL 文件...")
    
    train_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.train.jsonl')
    valid_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.valid.jsonl')
    test_file = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}.test.jsonl')
    
    save_jsonl(train_sequences, train_file)
    save_jsonl(valid_sequences, valid_file)
    save_jsonl(test_sequences, test_file)
    
    # ============ 步骤 5: 显示样例 ============
    print("\n" + "=" * 70)
    print("📊 数据样例:")
    print("=" * 70)
    
    if len(train_sequences) > 0:
        print("\n训练集样例:")
        for i, seq in enumerate(train_sequences[:3]):
            print(f"  样例 {i+1}:")
            print(f"    User ID: {seq['user_id']}")
            print(f"    历史长度: {len(seq['inter_history'])}")
            hist_display = seq['inter_history'][:5]
            if len(seq['inter_history']) > 5:
                print(f"    历史: {hist_display}...")
            else:
                print(f"    历史: {seq['inter_history']}")
            print(f"    目标: {seq['target_id']}")
            # 显示完整的 JSON 格式
            json_str = json.dumps({
                'user_id': seq['user_id'],
                'target_id': seq['target_id'],
                'inter_history': seq['inter_history'][:3] + (['...'] if len(seq['inter_history']) > 3 else [])
            }, ensure_ascii=False)
            print(f"    JSON: {json_str}")
    
    if len(valid_sequences) > 0:
        print("\n验证集样例:")
        for i, seq in enumerate(valid_sequences[:3]):
            print(f"  样例 {i+1}:")
            print(f"    User ID: {seq['user_id']}")
            print(f"    历史长度: {len(seq['inter_history'])}")
            hist_display = seq['inter_history'][:5]
            if len(seq['inter_history']) > 5:
                print(f"    历史: {hist_display}...")
            else:
                print(f"    历史: {seq['inter_history']}")
            print(f"    目标: {seq['target_id']}")
            # 显示完整的 JSON 格式
            json_str = json.dumps({
                'user_id': seq['user_id'],
                'target_id': seq['target_id'],
                'inter_history': seq['inter_history'][:3] + (['...'] if len(seq['inter_history']) > 3 else [])
            }, ensure_ascii=False)
            print(f"    JSON: {json_str}")
    
    if len(test_sequences) > 0:
        print("\n测试集样例:")
        for i, seq in enumerate(test_sequences[:3]):
            print(f"  样例 {i+1}:")
            print(f"    User ID: {seq['user_id']}")
            print(f"    历史长度: {len(seq['inter_history'])}")
            hist_display = seq['inter_history'][:5]
            if len(seq['inter_history']) > 5:
                print(f"    历史: {hist_display}...")
            else:
                print(f"    历史: {seq['inter_history']}")
            print(f"    目标: {seq['target_id']}")
            # 显示完整的 JSON 格式
            json_str = json.dumps({
                'user_id': seq['user_id'],
                'target_id': seq['target_id'],
                'inter_history': seq['inter_history'][:3] + (['...'] if len(seq['inter_history']) > 3 else [])
            }, ensure_ascii=False)
            print(f"    JSON: {json_str}")
    
    # ============ 步骤 6: 统计信息 ============
    print("\n" + "=" * 70)
    print("📈 数据统计:")
    print("=" * 70)
    
    print_statistics(train_sequences, "训练集")
    print_statistics(valid_sequences, "验证集")
    print_statistics(test_sequences, "测试集")
    
    # ============ 步骤 7: 验证文件格式 ============
    print("\n" + "=" * 70)
    print("🔍 验证生成的文件格式...")
    print("=" * 70)
    
    # 读取第一行验证
    for name, file_path in [('训练集', train_file), ('验证集', valid_file), ('测试集', test_file)]:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if first_line:
                    obj = json.loads(first_line)
                    print(f"\n{name}第一行:")
                    print(f"  键: {list(obj.keys())}")
                    print(f"  完整内容: {json.dumps(obj, ensure_ascii=False)[:200]}...")
    
    # ============ 总结 ============
    print("\n" + "=" * 70)
    print("🎉 数据准备完成!")
    print("=" * 70)
    
    print(f"\n📁 生成的文件:")
    print(f"   1. {train_file}")
    print(f"      - 序列数: {len(train_sequences)}")
    print(f"   2. {valid_file}")
    print(f"      - 序列数: {len(valid_sequences)}")
    print(f"   3. {test_file}")
    print(f"      - 序列数: {len(test_sequences)}")
    
    print(f"\n📁 已有的文件:")
    print(f"   4. {MAP_FILE}")
    print(f"   5. {os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_emb_256.npy')}")
    
    print(f"\n✨ 下一步: 训练 ETEGRec!")
    print(f"\n1. 创建配置文件 config/instruments.yaml")
    print(f"2. 修改 run.sh 中的 DATASET=Instruments2023")
    print(f"3. 运行: bash run.sh")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()