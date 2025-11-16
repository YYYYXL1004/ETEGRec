# ETEGRec 数据准备流程

## 概述

重构后的数据准备流程分为两个独立的脚本:

1. **`prepare_data.py`** - 数据预处理和格式转换
2. **`train_sasrec.py`** - SASRec训练和嵌入提取

## 使用方法

### 步骤1: 准备数据

从Amazon原始数据生成RecBole格式和ETEGRec格式的数据:

```bash
python prepare_data.py
```

**输入:**
- `./dataset/Instruments2023/Musical_Instruments.jsonl` (Amazon原始数据)

**输出:**
- `Instruments2023.inter` - RecBole格式 (带split标签)
- `Instruments2023.train.jsonl` - ETEGRec训练集
- `Instruments2023.valid.jsonl` - ETEGRec验证集
- `Instruments2023.test.jsonl` - ETEGRec测试集
- `dataset_stats.json` - 数据统计

### 步骤2: 训练SASRec

训练SASRec模型并提取物品嵌入:

```bash
python train_sasrec.py
```

**输入:**
- `Instruments2023.inter` (步骤1生成)

**输出:**
- `Instruments2023_emb_256.npy` - 物品嵌入向量
- `Instruments2023.emb_map.json` - item2id映射 (含[PAD])

## 数据划分策略

### Leave-One-Out 划分

对每个用户:
- **最后一个交互** → test
- **倒数第二个交互** → valid
- **其余交互** → train

### 训练集构建

**ETEGRec**: 为每个train交互构建增量序列
```
用户交互: [A, B, C, D] (train) + [E] (valid) + [F] (test)
训练样本:
  {history: [A], target: B}
  {history: [A,B], target: C}
  {history: [A,B,C], target: D}
```

**SASRec**: 同样只使用train交互进行训练
- 使用valid数据做早停 (选择最佳checkpoint)
- 使用test数据做最终评估
- **无数据泄露**: 只有train参与梯度更新，valid/test仅用于模型选择

## 关键特性

### ✅ 数据一致性
- 两个脚本使用相同的数据源和划分策略
- SASRec和ETEGRec都只在train数据上训练
- 无数据泄露

### ✅ 序列长度一致
- 所有序列截断到max_seq_length=50
- 确保SASRec和ETEGRec使用相同的上下文长度

### ✅ 简洁明了
- 每个脚本功能单一，职责清晰
- 代码简洁，易于理解和维护

## 注意事项

### 早停策略

SASRec训练使用valid数据做早停:
- **训练**: 只在train数据上进行梯度更新
- **验证**: 每个epoch在valid上评估，选择最佳checkpoint
- **测试**: 训练结束后在test上评估最佳模型
- **无数据泄露**: valid/test不参与训练，只用于模型选择和评估

### RecBole配置说明

使用`benchmark_filename`参数读取预先划分的文件:
```python
'benchmark_filename': ['train', 'valid', 'test']
```
这会让RecBole读取:
- `{dataset_name}.train.inter` - 训练集
- `{dataset_name}.valid.inter` - 验证集  
- `{dataset_name}.test.inter` - 测试集

而不是使用`eval_args.split`进行随机划分。

### 文件依赖

```
Musical_Instruments.jsonl (原始数据)
    ↓
prepare_data.py
    ↓
Instruments2023.inter (带split标签)
    ├─→ train_sasrec.py → 生成嵌入
    └─→ 直接使用 .train/.valid/.test.jsonl
```

## 数据统计示例

```json
{
  "num_users": 12345,
  "num_items": 6789,
  "num_interactions": 123456,
  "train_interactions": 110000,
  "valid_interactions": 12345,
  "test_interactions": 12345,
  "train_sequences": 95000,
  "valid_sequences": 12345,
  "test_sequences": 12345
}
```

## 旧文件说明

以下文件为旧版本，已被重构:
- `prepare_amazon_instruments.py` → 功能整合到 `prepare_data.py`
- `prepare_etegrec_data.py` → 功能整合到 `prepare_data.py`
- `train_sasrec_instruments.py` → 重构为 `train_sasrec.py`

可以保留作为参考，或删除以保持目录整洁。
