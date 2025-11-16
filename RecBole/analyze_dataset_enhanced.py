#!/usr/bin/env python3
"""
增强版数据集分析工具
- 可视化分布
- 深入对比分析
- 分析数据差异原因
"""

import json
import numpy as np
import os
from collections import defaultdict, Counter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class EnhancedDatasetAnalyzer:
    """增强版数据集分析工具"""
    
    def __init__(self, dataset_dir, dataset_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.train_file = os.path.join(dataset_dir, f'{dataset_name}.train.jsonl')
        self.valid_file = os.path.join(dataset_dir, f'{dataset_name}.valid.jsonl')
        self.test_file = os.path.join(dataset_dir, f'{dataset_name}.test.jsonl')
        self.map_file = os.path.join(dataset_dir, f'{dataset_name}.emb_map.json')
        
        self.train_data = []
        self.valid_data = []
        self.test_data = []
        self.item2id = {}
        self.stats = {}
    
    def load_data(self):
        """加载所有数据文件"""
        print(f"\n{'='*70}")
        print(f"📂 加载数据集: {self.dataset_name}")
        print(f"{'='*70}")
        
        if os.path.exists(self.train_file):
            print(f"📖 加载训练集...")
            with open(self.train_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="  读取"):
                    self.train_data.append(json.loads(line.strip()))
            print(f"   ✅ 训练集: {len(self.train_data):,} 条序列")
        
        if os.path.exists(self.valid_file):
            print(f"📖 加载验证集...")
            with open(self.valid_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="  读取"):
                    self.valid_data.append(json.loads(line.strip()))
            print(f"   ✅ 验证集: {len(self.valid_data):,} 条序列")
        
        if os.path.exists(self.test_file):
            print(f"📖 加载测试集...")
            with open(self.test_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="  读取"):
                    self.test_data.append(json.loads(line.strip()))
            print(f"   ✅ 测试集: {len(self.test_data):,} 条序列")
        
        if os.path.exists(self.map_file):
            with open(self.map_file, 'r', encoding='utf-8') as f:
                self.item2id = json.load(f)
            print(f"   ✅ 映射: {len(self.item2id):,} 个物品")
    
    def compute_statistics(self):
        """计算统计信息"""
        print(f"\n📊 计算统计信息...")
        
        all_data = self.train_data + self.valid_data + self.test_data
        
        all_users = set(seq['user_id'] for seq in all_data)
        user_interactions = defaultdict(int)
        for seq in all_data:
            user_interactions[seq['user_id']] += len(seq['inter_history']) + 1
        
        all_items = set()
        item_counts = defaultdict(int)
        for seq in all_data:
            for item in seq['inter_history']:
                all_items.add(item)
                item_counts[item] += 1
            all_items.add(seq['target_id'])
            item_counts[seq['target_id']] += 1
        
        hist_lengths = [len(seq['inter_history']) for seq in all_data]
        
        self.stats = {
            'num_users': len(all_users),
            'num_items': len(all_items),
            'num_sequences': len(all_data),
            'num_train': len(self.train_data),
            'num_valid': len(self.valid_data),
            'num_test': len(self.test_data),
            'user_interactions': user_interactions,
            'item_counts': item_counts,
            'hist_lengths': hist_lengths,
            'all_users': all_users,
            'all_items': all_items
        }
        
        print(f"   ✅ 统计完成")
    
    def plot_distributions(self, output_dir=None):
        """绘制分布图"""
        if output_dir is None:
            output_dir = self.dataset_dir
        
        print(f"\n📈 生成可视化图表...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Dataset: {self.dataset_name}', fontsize=16, fontweight='bold')
        
        # 1. 序列长度分布
        ax = axes[0, 0]
        hist_lengths = self.stats['hist_lengths']
        ax.hist(hist_lengths, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Sequence Length Distribution\nMean: {np.mean(hist_lengths):.2f}')
        ax.axvline(np.mean(hist_lengths), color='r', linestyle='--', label='Mean')
        ax.legend()
        
        # 2. 用户交互数分布
        ax = axes[0, 1]
        user_inter_counts = list(self.stats['user_interactions'].values())
        ax.hist(user_inter_counts, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Interactions per User')
        ax.set_ylabel('Number of Users (log)')
        ax.set_title(f'User Interaction Distribution')
        ax.set_yscale('log')
        
        # 3. 物品流行度分布
        ax = axes[0, 2]
        item_counts_sorted = sorted(self.stats['item_counts'].values(), reverse=True)
        ax.plot(range(len(item_counts_sorted)), item_counts_sorted)
        ax.set_xlabel('Item Rank (log)')
        ax.set_ylabel('Interaction Count (log)')
        ax.set_title('Item Popularity (Long-tail)')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # 4. 用户交互数分箱
        ax = axes[1, 0]
        bins = [0, 5, 10, 20, 50, 100, float('inf')]
        labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '100+']
        bin_counts = []
        for i in range(len(bins)-1):
            count = sum(1 for c in user_inter_counts if bins[i] < c <= bins[i+1])
            bin_counts.append(count)
        ax.bar(labels, bin_counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Interaction Range')
        ax.set_ylabel('Number of Users')
        ax.set_title('User Interaction (Binned)')
        
        # 5. 物品流行度分箱
        ax = axes[1, 1]
        item_counts_list = list(self.stats['item_counts'].values())
        bins = [0, 5, 10, 20, 50, 100, 500, float('inf')]
        labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '101-500', '500+']
        bin_counts = []
        for i in range(len(bins)-1):
            count = sum(1 for c in item_counts_list if bins[i] < c <= bins[i+1])
            bin_counts.append(count)
        ax.bar(labels, bin_counts, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Popularity Range')
        ax.set_ylabel('Number of Items')
        ax.set_title('Item Popularity (Binned)')
        ax.tick_params(axis='x', rotation=45)
        
        # 6. 数据集划分
        ax = axes[1, 2]
        split_data = [self.stats['num_train'], self.stats['num_valid'], self.stats['num_test']]
        split_labels = ['Train', 'Valid', 'Test']
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax.pie(split_data, labels=split_labels, autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Dataset Split')
        
        plt.tight_layout()
        
        plot_file = os.path.join(output_dir, f'{self.dataset_name}_distributions.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ 图表已保存: {plot_file}")
        plt.close()
    
    def print_summary(self):
        """打印摘要统计"""
        print(f"\n{'='*70}")
        print(f"📊 数据集摘要: {self.dataset_name}")
        print(f"{'='*70}")
        
        print(f"\n基本统计:")
        print(f"  总用户数: {self.stats['num_users']:,}")
        print(f"  总物品数: {self.stats['num_items']:,}")
        print(f"  总序列数: {self.stats['num_sequences']:,}")
        print(f"    - 训练集: {self.stats['num_train']:,}")
        print(f"    - 验证集: {self.stats['num_valid']:,}")
        print(f"    - 测试集: {self.stats['num_test']:,}")


def compare_datasets_detailed(analyzer1, analyzer2, output_dir):
    """详细对比两个数据集"""
    print(f"\n{'='*70}")
    print(f"🔍 详细对比分析")
    print(f"{'='*70}")
    
    users1 = analyzer1.stats['all_users']
    users2 = analyzer2.stats['all_users']
    items1 = analyzer1.stats['all_items']
    items2 = analyzer2.stats['all_items']
    
    common_users = users1 & users2
    common_items = items1 & items2
    only_in_2_users = users2 - users1
    only_in_2_items = items2 - items1
    
    print(f"\n用户分析:")
    print(f"  {analyzer1.dataset_name}: {len(users1):,} 用户")
    print(f"  {analyzer2.dataset_name}: {len(users2):,} 用户")
    print(f"  共同: {len(common_users):,}")
    print(f"  仅在 {analyzer2.dataset_name}: {len(only_in_2_users):,}")
    
    print(f"\n物品分析:")
    print(f"  {analyzer1.dataset_name}: {len(items1):,} 物品")
    print(f"  {analyzer2.dataset_name}: {len(items2):,} 物品")
    print(f"  共同: {len(common_items):,}")
    print(f"  仅在 {analyzer2.dataset_name}: {len(only_in_2_items):,}")
    
    if len(only_in_2_users) > 0:
        print(f"\n多出用户的特征:")
        extra_user_inters = [analyzer2.stats['user_interactions'][u] for u in only_in_2_users]
        print(f"  平均交互数: {np.mean(extra_user_inters):.2f}")
        print(f"  中位数: {np.median(extra_user_inters):.0f}")
    
    if len(only_in_2_items) > 0:
        print(f"\n多出物品的特征:")
        extra_item_counts = [analyzer2.stats['item_counts'][i] for i in only_in_2_items]
        print(f"  平均流行度: {np.mean(extra_item_counts):.2f}")
        print(f"  中位数: {np.median(extra_item_counts):.0f}")
    
    print(f"\n💡 可能原因:")
    print(f"  1. 数据过滤阈值不同")
    print(f"  2. 数据源或时间范围不同")
    print(f"  3. 预处理方式不同")


def main():
    """主函数"""
    print(f"{'='*70}")
    print(f"🎵 增强版数据集分析工具")
    print(f"{'='*70}")
    
    MY_DIR = './dataset/Instruments2023'
    MY_NAME = 'Instruments2023'
    AUTHOR_DIR = '../dataset/instrument'
    AUTHOR_NAME = 'instrument'
    
    # 分析自己的数据集
    my_analyzer = EnhancedDatasetAnalyzer(MY_DIR, MY_NAME)
    my_analyzer.load_data()
    my_analyzer.compute_statistics()
    my_analyzer.print_summary()
    my_analyzer.plot_distributions()
    
    # 分析作者的数据集
    if os.path.exists(AUTHOR_DIR):
        author_analyzer = EnhancedDatasetAnalyzer(AUTHOR_DIR, AUTHOR_NAME)
        author_analyzer.load_data()
        author_analyzer.compute_statistics()
        author_analyzer.print_summary()
        author_analyzer.plot_distributions()
        
        # 对比
        compare_datasets_detailed(author_analyzer, my_analyzer, MY_DIR)
    
    print(f"\n✅ 分析完成！")


if __name__ == '__main__':
    main()
