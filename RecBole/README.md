# ETEGRec 数据准备流程

## 总览

从 Amazon 原始数据到 ETEGRec 可用的数据集，分为 4 步：

```
Amazon 原始数据 (review JSON + meta JSON)
    |
    v
[Step 1] prepare_data_*.py -- 数据清洗、过滤、划分
    |   输出: .inter, .train.jsonl, .valid.jsonl, .test.jsonl
    |
    v
[Step 2] get_collab_emb.py -- 训练 SASRec，提取协同嵌入
    |   输出: *_colab_emb_256.npy, *.emb_map.json
    |
    v
[Step 3] get_text_emb.py -- 用预训练语言模型提取文本嵌入
    |   输出: *_sentence-transformer_text_768.npy
    |
    v
[Step 4] get_image_emb.py -- CLIP 提取图像嵌入 (多模态)
        输出: *_clip_image_768.npy
```

## 数据集说明

| 数据集 | 说明 | 准备脚本 |
|--------|------|----------|
| Instrument2018_5090 | Text-only 基线 (全部 item) | prepare_data_2018.py |
| Instrument2018_MM | 多模态版本 (仅保留有图片的 item) | prepare_data_mm.py |
| Instrument2014 | Amazon 2014 版本 | prepare_data_2014.py |
| Instruments2023 | Amazon 2023 版本 | prepare_data.py |

## 使用方法

以多模态数据集 Instrument2018_MM 为例：

### Step 1: 数据预处理

```bash
python prepare_data_mm.py
```

处理流程：
1. 读取 meta JSON，筛选有图片 URL 的 item（仅检查 URL 存在性，不下载）
2. 用 URL 存在性初筛交互数据 + 5-core 迭代过滤
3. 只下载 5-core 后最终 item 的图片（多线程并发），验证 JPG 完整性
4. 踢掉下载失败的 item，必要时补一轮 5-core
5. Leave-one-out 划分，输出到 dataset/Instrument2018_MM/

### Step 2: 协同嵌入 (SASRec)

```bash
python get_collab_emb.py --dataset Instrument2018_MM
```

在新数据集上训练 SASRec，提取 item embedding (256维) 和 emb_map.json。
文本嵌入
### Step 3: 

```bash
python get_text_emb.py --dataset Instrument2018_MM
```

用 SentenceTransformer 从 meta 数据构建文本描述并编码 (768维)。

### Step 4: 图像嵌入

```bash
python get_image_emb.py --dataset Instrument2018_MM
```

从 images/ 目录读取已下载的图片，用 CLIP ViT-L/14 编码 (768维)。
图片已在 Step 1 由 prepare_data_mm.py 下载，此步骤仅做提取。

注意：Step 2/3/4 相互独立，可以并行执行。它们都依赖 Step 1 生成的数据和 Step 2 的 emb_map.json 来确定 item 顺序（所以 Step 3/4 需要等 Step 2 完成）。

## 数据划分策略

Leave-One-Out：对每个用户按时间排序后，最后一个交互为 test，倒数第二个为 valid，其余为 train。

```
用户交互序列: [A, B, C, D, E, F]
               ----------  -  -
                 train      V  T
```

ETEGRec 训练集构建增量序列：
```
{history: [A], target: B}
{history: [A,B], target: C}
{history: [A,B,C], target: D}
```

## 最终输出文件

以 Instrument2018_MM 为例，ETEGRec 训练需要的文件：

```
dataset/Instrument2018_MM/
  Instrument2018_MM.inter              # RecBole 格式交互数据
  Instrument2018_MM.train.jsonl        # ETEGRec 训练序列
  Instrument2018_MM.valid.jsonl        # ETEGRec 验证序列
  Instrument2018_MM.test.jsonl         # ETEGRec 测试序列
  Instrument2018_MM.emb_map.json       # item2id 映射 ([PAD]=0)
  Instrument2018_MM_emb_256.npy        # 协同嵌入 (N_items, 256)
  Instrument2018_MM_sentence-transformer_text_768.npy  # 文本嵌入 (N_items, 768)
  Instrument2018_MM_clip_image_768.npy # 图像嵌入 (N_items, 768)
  meta_Musical_Instruments.json        # 元数据
  dataset_stats.json                   # 数据统计
  images/                              # 下载的图片 (prepare_data_mm.py 生成)
```

## 脚本一览

| 脚本 | 功能 |
|------|------|
| prepare_data_2018.py | Amazon 2018 数据预处理 (text-only) |
| prepare_data_mm.py | Amazon 2018 多模态数据预处理 (下载验证图片 + 过滤) |
| prepare_data_2014.py | Amazon 2014 数据预处理 |
| prepare_data.py | Amazon 2023 数据预处理 |
| get_collab_emb.py | 训练 SASRec + 提取协同嵌入 |
| get_text_emb.py | 提取文本语义嵌入 (SentenceTransformer / Qwen) |
| get_image_emb.py | 提取图像嵌入 (CLIP ViT-L/14) |
| analyze_dataset.py | 数据集统计分析 |
