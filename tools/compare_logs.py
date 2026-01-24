import re
import matplotlib.pyplot as plt
import ast
import sys
import os

# 设置 matplotlib 支持中文显示 (如果系统有中文字体)
# 这里为了通用性，尽量用英文标签，注释用中文

def parse_log(file_path):
    """
    解析日志文件，区分预训练(Pre-train)和微调(STF)阶段
    """
    config = {}
    
    # 数据结构: {'pre': {'train': [], 'val': []}, 'stf': {'train': [], 'val': []}}
    data = {
        'pre': {'train': [], 'val': []},
        'stf': {'train': [], 'val': []}
    }
    
    current_phase = 'pre'
    last_epoch = -1
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 1. 提取配置 (只取第一个出现的配置)
            if "Config:" in line and not config:
                try:
                    config_str = line.split("Config:", 1)[1].strip()
                    # Sanitize the config string for ast.literal_eval
                    # 1. Handle device(...)
                    config_str = re.sub(r"device\(type=['\"](.*?)['\"]\)", r"'\1'", config_str)
                    # 2. Handle objects like <accelerate...>
                    config_str = re.sub(r"<.*?>", "'<object>'", config_str)
                    
                    config = ast.literal_eval(config_str)
                except Exception as e:
                    # print(f"Config parse error: {e}") # Debug if needed
                    pass

            # 2. 检测阶段切换
            # 如果出现 "Pre Best Validation Score"，说明预训练结束，即将开始 STF
            if "Pre Best Validation Score" in line:
                current_phase = 'stf'
                last_epoch = -1
                continue
                
            # 或者通过 Epoch 重置来判断 (如果错过了上面的标志)
            # 例如从 Epoch 20 变为 Epoch 0
            epoch_match = re.search(r"\[Epoch (\d+)\]", line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                if current_phase == 'pre' and epoch < last_epoch and last_epoch > 5:
                    current_phase = 'stf'
                last_epoch = epoch

            # 3. 提取训练 Loss
            # 格式: ... [Epoch X] [time: ..., train loss[('loss', 0.0044), ...]]
            if "train loss[" in line:
                try:
                    epoch_match = re.search(r"\[Epoch (\d+)\]", line)
                    loss_match = re.search(r"train loss\[(.*?)\]\]", line)
                    if epoch_match and loss_match:
                        epoch = int(epoch_match.group(1))
                        losses_str = loss_match.group(1)
                        # 将 ('loss', 0.123), ... 转为字典
                        losses = dict(ast.literal_eval(f"[{losses_str}]"))
                        losses['epoch'] = epoch
                        data[current_phase]['train'].append(losses)
                except Exception:
                    pass

            # 4. 提取验证结果
            # 格式: ... [Epoch X] Val Results: {'recall@1': ...}
            if "Val Results:" in line:
                try:
                    epoch_match = re.search(r"\[Epoch (\d+)\]", line)
                    val_match = re.search(r"Val Results: (\{.*?\})", line)
                    if epoch_match and val_match:
                        epoch = int(epoch_match.group(1))
                        metrics = ast.literal_eval(val_match.group(1))
                        metrics['epoch'] = epoch
                        data[current_phase]['val'].append(metrics)
                except Exception:
                    pass
                    
    return config, data

def compare_configs(config1, config2, name1, name2):
    """
    对比两个配置文件的差异
    """
    print(f"\n{'='*20} 配置对比: {name1} vs {name2} {'='*20}")
    
    # 忽略的字段 (运行时差异)
    ignore_keys = ['run_local_time', 'save_path', 'accelerator', 'device', 'rqvae_path', 'log_dir']
    
    keys1 = set(config1.keys())
    keys2 = set(config2.keys())
    all_keys = keys1.union(keys2)
    
    diffs = []
    for key in all_keys:
        if key in ignore_keys:
            continue
            
        val1 = config1.get(key, "Not Found")
        val2 = config2.get(key, "Not Found")
        
        if str(val1) != str(val2):
            diffs.append((key, val1, val2))
            
    if not diffs:
        print("✅ 核心训练参数一致")
    else:
        print("⚠️ 发现以下参数差异:")
        for key, v1, v2 in diffs:
            print(f"  - {key}: \n    {name1}: {v1}\n    {name2}: {v2}")

def save_plot(fig, filename):
    """保存图片并关闭"""
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"📊 图表已保存至: {filename}")

def plot_separate_train_loss(data1, data2, name1, name2, output_dir):
    """
    单独绘制 Train Loss，包含 Pre-train 和 STF。
    由于 cycle=2 导致 Loss 震荡（REC task vs ID task），这里将它们拆分开绘制。
    """
    # 准备画布：上下两张图，分别画 REC Loss (大) 和 ID Loss (小)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # 辅助函数：拆分奇偶 Epoch
    def split_even_odd(phase_data):
        if not phase_data:
            return [], [], [], []
        
        # 转换为 (epoch, loss) 列表
        points = [(item['epoch'], item.get('loss', 0)) for item in phase_data]
        
        # 偶数 (ID Task, Low Loss)
        even_x = [p[0] for p in points if p[0] % 2 == 0]
        even_y = [p[1] for p in points if p[0] % 2 == 0]
        
        # 奇数 (REC Task, High Loss)
        odd_x = [p[0] for p in points if p[0] % 2 != 0]
        odd_y = [p[1] for p in points if p[0] % 2 != 0]
        
        return even_x, even_y, odd_x, odd_y

    # 绘制函数
    def plot_phase(ax_odd, ax_even, data, name, color_pre, color_stf, lw=1.5, alpha=0.7):
        # Pre-train
        ex, ey, ox, oy = split_even_odd(data['pre']['train'])
        if ox: ax_odd.plot(ox, oy, label=f'{name} (Pre)', color=color_pre, linestyle='-', linewidth=lw, alpha=alpha)
        if ex: ax_even.plot(ex, ey, label=f'{name} (Pre)', color=color_pre, linestyle='-', linewidth=lw, alpha=alpha)
        
        # STF
        ex, ey, ox, oy = split_even_odd(data['stf']['train'])
        if ox: ax_odd.plot(ox, oy, label=f'{name} (STF)', color=color_stf, linestyle='--', linewidth=lw, alpha=alpha)
        if ex: ax_even.plot(ex, ey, label=f'{name} (STF)', color=color_stf, linestyle='--', linewidth=lw, alpha=alpha)

    # 绘制 Log 1 (底层，稍粗，可见度高)
    plot_phase(ax1, ax2, data1, name1, 'blue', 'cyan', lw=2.5, alpha=0.9)
    # 绘制 Log 2 (上层，稍细，半透明)
    plot_phase(ax1, ax2, data2, name2, 'red', 'orange', lw=1.2, alpha=0.6)
    
    # 设置标题和标签
    ax1.set_title(f'REC Loss (Odd Epochs) - High Loss Task')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2.set_title(f'ID/VQ Loss (Even Epochs) - Low Loss Task')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    save_plot(fig, os.path.join(output_dir, "train_loss_comparison_split.png"))

def plot_validation_metrics(data1, data2, name1, name2, output_dir):
    """
    绘制验证集指标 (NDCG, Recall)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics = [
        ('ndcg@10', 'Val NDCG@10'),
        ('recall@10', 'Val Recall@10')
    ]
    
    for i, (key, title) in enumerate(metrics):
        ax = axes[i]
        # Log 1
        plot_single_metric_curve_ax(ax, data1, key, 'val', name1, 'blue', 'cyan', lw=2.5, alpha=0.9)
        # Log 2
        plot_single_metric_curve_ax(ax, data2, key, 'val', name2, 'red', 'orange', lw=1.2, alpha=0.6)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        
    save_plot(fig, os.path.join(output_dir, "validation_metrics_comparison.png"))

def plot_single_metric_curve(data, metric_key, source, name, color_pre, color_stf):
    """plt 直接绘制"""
    # Pre-train
    pre_data = data['pre'][source]
    if pre_data:
        x = [item['epoch'] for item in pre_data]
        y = [item.get(metric_key, 0) for item in pre_data]
        plt.plot(x, y, label=f'{name} (Pre)', color=color_pre, linestyle='-')
        
    # STF
    stf_data = data['stf'][source]
    if stf_data:
        x = [item['epoch'] for item in stf_data]
        y = [item.get(metric_key, 0) for item in stf_data]
        plt.plot(x, y, label=f'{name} (STF)', color=color_stf, linestyle='--')

def plot_single_metric_curve_ax(ax, data, metric_key, source, name, color_pre, color_stf, lw=1.5, alpha=1.0):
    """ax 对象绘制"""
    # Pre-train
    pre_data = data['pre'][source]
    if pre_data:
        x = [item['epoch'] for item in pre_data]
        y = [item.get(metric_key, 0) for item in pre_data]
        ax.plot(x, y, label=f'{name} (Pre)', color=color_pre, linestyle='-', linewidth=lw, alpha=alpha)
        
    # STF
    stf_data = data['stf'][source]
    if stf_data:
        x = [item['epoch'] for item in stf_data]
        y = [item.get(metric_key, 0) for item in stf_data]
        ax.plot(x, y, label=f'{name} (STF)', color=color_stf, linestyle='--', linewidth=lw, alpha=alpha)

def main():
    # 使用绝对路径，避免路径问题
    base_dir = "/data/yaoxianglin/ETEGRec"
    log1_path = os.path.join(base_dir, "logs/Instrument2018/Dec-04-2025_00-55-54f96c.log")
    log2_path = os.path.join(base_dir, "logs/Instrument2018/Dec-17-2025_22-34-aedc1b.log")
    
    # 结果输出目录
    output_dir = os.path.join(base_dir, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    name1 = "Dec-04"
    name2 = "Dec-17"
    
    print(f"正在解析日志...")
    conf1, data1 = parse_log(log1_path)
    conf2, data2 = parse_log(log2_path)
    
    if not conf1:
        print(f"❌ 无法解析配置: {log1_path}")
    if not conf2:
        print(f"❌ 无法解析配置: {log2_path}")
        
    # 1. 检查参数
    compare_configs(conf1, conf2, name1, name2)
    
    # 2. 单独绘制 Train Loss
    print(f"正在生成图表到目录: {output_dir}")
    plot_separate_train_loss(data1, data2, name1, name2, output_dir)
    
    # 3. 绘制验证集指标
    plot_validation_metrics(data1, data2, name1, name2, output_dir)
    
    # 简单统计输出
    print(f"\n{'='*20} 统计摘要 {'='*20}")
    for name, data in [(name1, data1), (name2, data2)]:
        print(f"[{name}]")
        print(f"  Pre-train Epochs: {len(data['pre']['train'])}")
        if data['pre']['val']:
            print(f"  Best Pre Val NDCG@10: {max([x.get('ndcg@10', 0) for x in data['pre']['val']]):.4f}")
        
        print(f"  STF Epochs: {len(data['stf']['train'])}")
        if data['stf']['val']:
            print(f"  Best STF Val NDCG@10: {max([x.get('ndcg@10', 0) for x in data['stf']['val']]):.4f}")

if __name__ == "__main__":
    main()
