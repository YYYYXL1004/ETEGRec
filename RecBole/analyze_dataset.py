import json
import numpy as np
import os
from collections import defaultdict, Counter
from tqdm import tqdm

class DatasetAnalyzer:
    """
    数据集分析工具
    """
    
    def __init__(self, dataset_dir, dataset_name):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.train_file = os.path.join(dataset_dir, f'{dataset_name}.train.jsonl')
        self.valid_file = os.path.join(dataset_dir, f'{dataset_name}.valid.jsonl')
        self.test_file = os.path.join(dataset_dir, f'{dataset_name}.test.jsonl')
        self.map_file = os.path.join(dataset_dir, f'{dataset_name}.emb_map.json')
        self.emb_file = os.path.join(dataset_dir, f'{dataset_name}_emb_256.npy')
        
        self.train_data = []
        self.valid_data = []
        self.test_data = []
        self.item2id = {}
        self.embeddings = None
    
    def load_data(self):
        """
        加载所有数据文件
        """
        print(f"\n{'='*70}")
        print(f"📂 加载数据集: {self.dataset_name}")
        print(f"{'='*70}")
        
        # 加载训练集
        if os.path.exists(self.train_file):
            print(f"📖 加载训练集: {self.train_file}")
            with open(self.train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.train_data.append(json.loads(line.strip()))
            print(f"   ✅ 训练集: {len(self.train_data)} 条序列")
        else:
            print(f"   ❌ 训练集文件不存在")
        
        # 加载验证集
        if os.path.exists(self.valid_file):
            print(f"📖 加载验证集: {self.valid_file}")
            with open(self.valid_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.valid_data.append(json.loads(line.strip()))
            print(f"   ✅ 验证集: {len(self.valid_data)} 条序列")
        else:
            print(f"   ❌ 验证集文件不存在")
        
        # 加载测试集
        if os.path.exists(self.test_file):
            print(f"📖 加载测试集: {self.test_file}")
            with open(self.test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.test_data.append(json.loads(line.strip()))
            print(f"   ✅ 测试集: {len(self.test_data)} 条序列")
        else:
            print(f"   ❌ 测试集文件不存在")
        
        # 加载映射
        if os.path.exists(self.map_file):
            print(f"📖 加载映射文件: {self.map_file}")
            with open(self.map_file, 'r', encoding='utf-8') as f:
                self.item2id = json.load(f)
            print(f"   ✅ 映射: {len(self.item2id)} 个物品")
        else:
            print(f"   ❌ 映射文件不存在")
        
        # 加载嵌入
        if os.path.exists(self.emb_file):
            print(f"📖 加载嵌入文件: {self.emb_file}")
            self.embeddings = np.load(self.emb_file)
            print(f"   ✅ 嵌入形状: {self.embeddings.shape}")
            print(f"   ✅ 嵌入大小: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")
        else:
            print(f"   ❌ 嵌入文件不存在")
    
    def analyze_basic_stats(self):
        """
        基本统计信息
        """
        print(f"\n{'='*70}")
        print(f"📊 基本统计信息")
        print(f"{'='*70}")
        
        all_data = {
            '训练集': self.train_data,
            '验证集': self.valid_data,
            '测试集': self.test_data
        }
        
        for name, data in all_data.items():
            if len(data) == 0:
                print(f"\n{name}: 无数据")
                continue
            
            # 统计历史长度
            hist_lens = [len(seq['inter_history']) for seq in data]
            
            # 统计唯一用户
            unique_users = len(set(seq['user_id'] for seq in data))
            
            # 统计唯一物品
            all_items = set()
            for seq in data:
                all_items.update(seq['inter_history'])
                all_items.add(seq['target_id'])
            
            print(f"\n{name}:")
            print(f"  序列数量: {len(data):,}")
            print(f"  唯一用户数: {unique_users:,}")
            print(f"  唯一物品数: {len(all_items):,}")
            print(f"  历史长度:")
            print(f"    平均: {np.mean(hist_lens):.2f}")
            print(f"    中位数: {np.median(hist_lens):.2f}")
            print(f"    最小: {np.min(hist_lens)}")
            print(f"    最大: {np.max(hist_lens)}")
            print(f"    标准差: {np.std(hist_lens):.2f}")
    
    def analyze_user_distribution(self):
        """
        用户分布分析
        """
        print(f"\n{'='*70}")
        print(f"👥 用户分布分析")
        print(f"{'='*70}")
        
        all_data = self.train_data + self.valid_data + self.test_data
        
        if len(all_data) == 0:
            print("无数据可分析")
            return
        
        # 统计每个用户的交互次数
        user_interactions = defaultdict(int)
        for seq in all_data:
            user_interactions[seq['user_id']] += len(seq['inter_history']) + 1
        
        interaction_counts = list(user_interactions.values())
        
        print(f"\n总用户数: {len(user_interactions):,}")
        print(f"总交互数: {sum(interaction_counts):,}")
        print(f"每用户平均交互数: {np.mean(interaction_counts):.2f}")
        print(f"每用户中位数交互数: {np.median(interaction_counts):.2f}")
        print(f"每用户最少交互数: {np.min(interaction_counts)}")
        print(f"每用户最多交互数: {np.max(interaction_counts)}")
        
        # 分布统计
        print(f"\n用户交互数分布:")
        bins = [0, 5, 10, 20, 50, 100, float('inf')]
        labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '100+']
        
        for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
            count = sum(1 for c in interaction_counts if low < c <= high)
            pct = count / len(user_interactions) * 100
            print(f"  {label:>10}次: {count:>6} 用户 ({pct:>5.2f}%)")
    
    def analyze_item_distribution(self):
        """
        物品分布分析
        """
        print(f"\n{'='*70}")
        print(f"🎸 物品分布分析")
        print(f"{'='*70}")
        
        all_data = self.train_data + self.valid_data + self.test_data
        
        if len(all_data) == 0:
            print("无数据可分析")
            return
        
        # 统计每个物品出现的次数
        item_counts = defaultdict(int)
        for seq in all_data:
            for item in seq['inter_history']:
                item_counts[item] += 1
            item_counts[seq['target_id']] += 1
        
        counts = list(item_counts.values())
        
        print(f"\n总物品数: {len(item_counts):,}")
        print(f"总出现次数: {sum(counts):,}")
        print(f"每物品平均出现次数: {np.mean(counts):.2f}")
        print(f"每物品中位数出现次数: {np.median(counts):.2f}")
        print(f"每物品最少出现次数: {np.min(counts)}")
        print(f"每物品最多出现次数: {np.max(counts)}")
        
        # 分布统计
        print(f"\n物品流行度分布:")
        bins = [0, 5, 10, 20, 50, 100, 500, float('inf')]
        labels = ['1-5', '6-10', '11-20', '21-50', '51-100', '101-500', '500+']
        
        for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
            count = sum(1 for c in counts if low < c <= high)
            pct = count / len(item_counts) * 100
            print(f"  {label:>10}次: {count:>6} 物品 ({pct:>5.2f}%)")
        
        # Top 热门物品
        print(f"\nTop 10 热门物品:")
        top_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (item, count) in enumerate(top_items, 1):
            print(f"  {i:2d}. {item}: {count:,} 次")
    
    def analyze_data_format(self):
        """
        数据格式分析
        """
        print(f"\n{'='*70}")
        print(f"📝 数据格式分析")
        print(f"{'='*70}")
        
        all_data = {
            '训练集': self.train_data,
            '验证集': self.valid_data,
            '测试集': self.test_data
        }
        
        for name, data in all_data.items():
            if len(data) == 0:
                continue
            
            print(f"\n{name}:")
            
            # 检查字段
            sample = data[0]
            print(f"  字段: {list(sample.keys())}")
            
            # 检查是否所有记录都有相同字段
            all_keys = set()
            for seq in data:
                all_keys.update(seq.keys())
            print(f"  所有字段: {all_keys}")
            
            # 检查字段类型
            print(f"  字段类型:")
            print(f"    user_id: {type(sample.get('user_id')).__name__}")
            print(f"    target_id: {type(sample.get('target_id')).__name__}")
            print(f"    inter_history: {type(sample.get('inter_history')).__name__} (长度: {len(sample.get('inter_history', []))})")
            
            # 显示样例
            print(f"  样例 (前3条):")
            for i, seq in enumerate(data[:3], 1):
                hist_preview = seq['inter_history'][:3]
                if len(seq['inter_history']) > 3:
                    hist_preview_str = str(hist_preview)[:-1] + ', ...]'
                else:
                    hist_preview_str = str(seq['inter_history'])
                print(f"    {i}. user_id={seq['user_id']}, target={seq['target_id']}, history={hist_preview_str}")
    
    def check_data_consistency(self):
        """
        检查数据一致性
        """
        print(f"\n{'='*70}")
        print(f"🔍 数据一致性检查")
        print(f"{'='*70}")
        
        issues = []
        
        # 1. 检查物品是否都在映射中
        all_items = set()
        for data in [self.train_data, self.valid_data, self.test_data]:
            for seq in data:
                all_items.update(seq['inter_history'])
                all_items.add(seq['target_id'])
        
        missing_in_map = all_items - set(self.item2id.keys())
        if missing_in_map:
            issues.append(f"❌ 有 {len(missing_in_map)} 个物品不在映射中")
            print(f"  示例: {list(missing_in_map)[:5]}")
        else:
            print(f"✅ 所有物品都在映射中")
        
        # 2. 检查映射和嵌入是否匹配
        if self.embeddings is not None:
            if len(self.item2id) == self.embeddings.shape[0]:
                print(f"✅ 映射数量 ({len(self.item2id)}) 与嵌入数量 ({self.embeddings.shape[0]}) 一致")
            else:
                issues.append(f"❌ 映射数量 ({len(self.item2id)}) 与嵌入数量 ({self.embeddings.shape[0]}) 不一致")
        
        # 3. 检查用户在不同集合中的分布
        train_users = set(seq['user_id'] for seq in self.train_data)
        valid_users = set(seq['user_id'] for seq in self.valid_data)
        test_users = set(seq['user_id'] for seq in self.test_data)
        
        print(f"\n用户分布:")
        print(f"  训练集用户: {len(train_users):,}")
        print(f"  验证集用户: {len(valid_users):,}")
        print(f"  测试集用户: {len(test_users):,}")
        print(f"  验证∩测试: {len(valid_users & test_users):,}")
        print(f"  训练∩验证: {len(train_users & valid_users):,}")
        print(f"  训练∩测试: {len(train_users & test_users):,}")
        
        # 4. 检查是否有空历史
        empty_history = 0
        for data in [self.train_data, self.valid_data, self.test_data]:
            for seq in data:
                if len(seq['inter_history']) == 0:
                    empty_history += 1
        
        if empty_history > 0:
            issues.append(f"❌ 有 {empty_history} 个序列的历史为空")
        else:
            print(f"✅ 没有空历史序列")
        
        # 总结
        print(f"\n{'='*40}")
        if len(issues) == 0:
            print(f"✅ 所有检查通过！数据一致性良好")
        else:
            print(f"⚠️  发现 {len(issues)} 个问题:")
            for issue in issues:
                print(f"  {issue}")
        print(f"{'='*40}")
    
    def run_full_analysis(self):
        """
        运行完整分析（不保存到文件，只打印）
        """
        self.analyze_basic_stats()
        self.analyze_user_distribution()
        self.analyze_item_distribution()
        self.analyze_data_format()
        self.check_data_consistency()
    
    def generate_report(self, output_file=None):
        """
        生成完整报告
        """
        if output_file is None:
            output_file = os.path.join(self.dataset_dir, f'{self.dataset_name}_analysis_report.txt')
        
        print(f"\n{'='*70}")
        print(f"📄 生成分析报告")
        print(f"{'='*70}")
        
        import sys
        from io import StringIO
        
        # 重定向输出到字符串
        old_stdout = sys.stdout
        sys.stdout = report_buffer = StringIO()
        
        # 运行所有分析
        self.run_full_analysis()
        
        # 恢复输出
        sys.stdout = old_stdout
        report_content = report_buffer.getvalue()
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"数据集分析报告\n")
            f.write(f"数据集: {self.dataset_name}\n")
            f.write(f"时间: 2025-11-14 08:57:57 UTC\n")
            f.write(f"用户: YYYYXL1004\n")
            f.write(f"{'='*70}\n\n")
            f.write(report_content)
        
        print(f"✅ 报告已保存到: {output_file}")
        
        # 同时打印到控制台
        print(report_content)


def compare_datasets(dataset1_dir, dataset1_name, dataset2_dir, dataset2_name):
    """
    对比两个数据集
    """
    print(f"\n{'='*70}")
    print(f"🔄 对比数据集")
    print(f"{'='*70}")
    print(f"数据集1: {dataset1_name} ({dataset1_dir})")
    print(f"数据集2: {dataset2_name} ({dataset2_dir})")
    
    # 加载两个数据集
    analyzer1 = DatasetAnalyzer(dataset1_dir, dataset1_name)
    analyzer1.load_data()
    
    analyzer2 = DatasetAnalyzer(dataset2_dir, dataset2_name)
    analyzer2.load_data()
    
    # 对比统计
    print(f"\n{'='*70}")
    print(f"📊 数据集对比")
    print(f"{'='*70}")
    
    stats = {
        '训练集序列数': (len(analyzer1.train_data), len(analyzer2.train_data)),
        '验证集序列数': (len(analyzer1.valid_data), len(analyzer2.valid_data)),
        '测试集序列数': (len(analyzer1.test_data), len(analyzer2.test_data)),
        '物品映射数': (len(analyzer1.item2id), len(analyzer2.item2id)),
    }
    
    if analyzer1.embeddings is not None and analyzer2.embeddings is not None:
        stats['嵌入形状'] = (analyzer1.embeddings.shape, analyzer2.embeddings.shape)
    
    print(f"\n{'指标':<20} {'数据集1':>15} {'数据集2':>15} {'差异':>15}")
    print(f"{'-'*70}")
    
    for metric, (val1, val2) in stats.items():
        if isinstance(val1, tuple):
            diff = "N/A"
            print(f"{metric:<20} {str(val1):>15} {str(val2):>15} {diff:>15}")
        else:
            diff = val2 - val1
            diff_pct = (diff / val1 * 100) if val1 > 0 else 0
            print(f"{metric:<20} {val1:>15,} {val2:>15,} {diff:>+15,} ({diff_pct:+.1f}%)")
    
    # 对比用户和物品
    print(f"\n用户和物品分析:")
    
    all_users1 = set()
    all_items1 = set()
    for data in [analyzer1.train_data, analyzer1.valid_data, analyzer1.test_data]:
        for seq in data:
            all_users1.add(seq['user_id'])
            all_items1.update(seq['inter_history'])
            all_items1.add(seq['target_id'])
    
    all_users2 = set()
    all_items2 = set()
    for data in [analyzer2.train_data, analyzer2.valid_data, analyzer2.test_data]:
        for seq in data:
            all_users2.add(seq['user_id'])
            all_items2.update(seq['inter_history'])
            all_items2.add(seq['target_id'])
    
    print(f"  用户数: {len(all_users1):,} vs {len(all_users2):,}")
    print(f"  物品数: {len(all_items1):,} vs {len(all_items2):,}")
    print(f"  共同用户: {len(all_users1 & all_users2):,}")
    print(f"  共同物品: {len(all_items1 & all_items2):,}")
    
    # 对比历史长度分布
    print(f"\n历史长度分布对比:")
    
    dataset_pairs = [
        ('训练集', analyzer1.train_data, analyzer2.train_data),
        ('验证集', analyzer1.valid_data, analyzer2.valid_data),
        ('测试集', analyzer1.test_data, analyzer2.test_data)
    ]
    
    for dataset_name, data1, data2 in dataset_pairs:
        if len(data1) > 0 and len(data2) > 0:
            lens1 = [len(seq['inter_history']) for seq in data1]
            lens2 = [len(seq['inter_history']) for seq in data2]
            
            print(f"\n  {dataset_name}:")
            print(f"    平均长度: {np.mean(lens1):.2f} vs {np.mean(lens2):.2f}")
            print(f"    中位数: {np.median(lens1):.2f} vs {np.median(lens2):.2f}")
            print(f"    最大长度: {np.max(lens1)} vs {np.max(lens2)}")


def main():
    """
    主函数
    """
    print(f"{'='*70}")
    print(f"🎵 数据集分析工具")
    print(f"{'='*70}")
    print(f"当前时间: 2025-11-14 08:57:57 UTC")
    print(f"当前用户: YYYYXL1004")
    print(f"{'='*70}")
    
    # ============ 配置 ============
    # 你自己生成的数据集
    MY_DATASET_DIR = './dataset/Instruments2023'
    MY_DATASET_NAME = 'Instruments2023'
    
    # 作者提供的数据集（如果有的话）
    AUTHOR_DATASET_DIR = '../dataset/instrument'  # 修改为作者数据集路径
    AUTHOR_DATASET_NAME = 'instrument'
    
    # ============ 第一部分：分析自己的数据集 ============
    print(f"\n{'#'*70}")
    print(f"# 第一部分: 分析自己生成的数据集")
    print(f"{'#'*70}")
    
    my_analyzer = DatasetAnalyzer(MY_DATASET_DIR, MY_DATASET_NAME)
    my_analyzer.load_data()
    
    # 打印分析结果到控制台
    my_analyzer.run_full_analysis()
    
    # 生成报告文件
    my_report_file = os.path.join(MY_DATASET_DIR, f'{MY_DATASET_NAME}_analysis_report.txt')
    my_analyzer.generate_report(my_report_file)
    
    # ============ 第二部分：分析作者的数据集 ============
    if os.path.exists(AUTHOR_DATASET_DIR):
        print(f"\n{'#'*70}")
        print(f"# 第二部分: 分析作者提供的数据集")
        print(f"{'#'*70}")
        
        author_analyzer = DatasetAnalyzer(AUTHOR_DATASET_DIR, AUTHOR_DATASET_NAME)
        author_analyzer.load_data()
        
        # 打印分析结果到控制台
        author_analyzer.run_full_analysis()
        
        # 生成报告文件
        author_report_file = os.path.join(AUTHOR_DATASET_DIR, f'{AUTHOR_DATASET_NAME}_analysis_report.txt')
        author_analyzer.generate_report(author_report_file)
        
        # ============ 第三部分：对比两个数据集 ============
        print(f"\n{'#'*70}")
        print(f"# 第三部分: 对比两个数据集")
        print(f"{'#'*70}")
        
        compare_datasets(
            MY_DATASET_DIR, MY_DATASET_NAME,
            AUTHOR_DATASET_DIR, AUTHOR_DATASET_NAME
        )
        
        # 生成对比报告
        print(f"\n{'='*70}")
        print(f"📄 生成对比报告")
        print(f"{'='*70}")
        
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = comparison_buffer = StringIO()
        
        compare_datasets(
            MY_DATASET_DIR, MY_DATASET_NAME,
            AUTHOR_DATASET_DIR, AUTHOR_DATASET_NAME
        )
        
        sys.stdout = old_stdout
        comparison_content = comparison_buffer.getvalue()
        
        comparison_file = os.path.join(MY_DATASET_DIR, 'comparison_report.txt')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(f"数据集对比报告\n")
            f.write(f"时间: 2025-11-14 08:57:57 UTC\n")
            f.write(f"用户: YYYYXL1004\n")
            f.write(f"{'='*70}\n\n")
            f.write(comparison_content)
        
        print(f"✅ 对比报告已保存到: {comparison_file}")
        
    else:
        print(f"\n⚠️  未找到作者数据集目录: {AUTHOR_DATASET_DIR}")
        print(f"   如果你有作者的数据集，请修改脚本中的 AUTHOR_DATASET_DIR 变量")
    
    # ============ 总结 ============
    print(f"\n{'='*70}")
    print(f"✅ 分析完成！")
    print(f"{'='*70}")
    
    print(f"\n📁 生成的报告文件:")
    if os.path.exists(my_report_file):
        print(f"   1. {my_report_file}")
    if os.path.exists(AUTHOR_DATASET_DIR):
        author_report_file = os.path.join(AUTHOR_DATASET_DIR, f'{AUTHOR_DATASET_NAME}_analysis_report.txt')
        if os.path.exists(author_report_file):
            print(f"   2. {author_report_file}")
        comparison_file = os.path.join(MY_DATASET_DIR, 'comparison_report.txt')
        if os.path.exists(comparison_file):
            print(f"   3. {comparison_file}")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()