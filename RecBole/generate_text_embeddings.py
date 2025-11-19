"""
生成 Instrument2018 数据集的文本嵌入 - 离线版本

如果网络不稳定，可以使用本地已下载的模型或配置镜像
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

# 设置 HF 镜像（如果需要）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 配置参数
DATASET_NAME = "Instrument2018"
DATASET_DIR = f"./dataset/{DATASET_NAME}"
META_FILE = f"{DATASET_DIR}/meta_Musical_Instruments.json"
ITEM2ID_FILE = f"{DATASET_DIR}/{DATASET_NAME}.emb_map.json"
MODEL_NAME = "sentence-transformers/sentence-t5-base"  
EMBEDDING_DIM = 768

# 模型配置 - 使用本地路径或镜像
# 方式 1: 如果已经下载过模型，指定本地路径
# MODEL_NAME = "/path/to/local/model"

# 方式 2: 使用 HF 镜像（中国大陆用户）
# 在运行前设置环境变量: export HF_ENDPOINT=https://hf-mirror.com
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 方式 3: 使用轻量级模型（推荐）
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384维
# EMBEDDING_DIM = 384

OUTPUT_FILE = f"{DATASET_DIR}/{DATASET_NAME}_text_{EMBEDDING_DIM}.npy"

print(f"=" * 80)
print(f"生成 {DATASET_NAME} 数据集的文本语义嵌入")
print(f"模型: {MODEL_NAME}")
print(f"嵌入维度: {EMBEDDING_DIM}")
print(f"=" * 80)

# 1. 加载 item2id 映射
print(f"\n[1/5] 加载 item2id 映射: {ITEM2ID_FILE}")
with open(ITEM2ID_FILE, 'r') as f:
    item2id = json.load(f)

n_items = len(item2id)
print(f"  - 物品总数: {n_items} (包括 [PAD])")

# 创建 id2item 反向映射
id2item = {v: k for k, v in item2id.items()}

# 2. 加载元数据
print(f"\n[2/5] 加载元数据: {META_FILE}")
meta_data = {}
with open(META_FILE, 'r') as f:
    for line in tqdm(f, desc="  读取元数据"):
        item = json.loads(line.strip())
        asin = item.get('asin', '')
        if asin:
            meta_data[asin] = item

print(f"  - 元数据条目数: {len(meta_data)}")

# 3. 构建文本描述
print(f"\n[3/5] 为每个物品构建文本描述")
item_texts = {}

for item_id in range(n_items):
    asin = id2item.get(item_id, '')
    
    if asin == '[PAD]':
        item_texts[item_id] = ""
        continue
    
    meta = meta_data.get(asin, {})
    
    title = meta.get('title', '')
    description = meta.get('description', [])
    if isinstance(description, list):
        description = ' '.join(description)
    
    categories = meta.get('category', [])
    if isinstance(categories, list):
        categories = ' '.join(categories)
    
    brand = meta.get('brand', '')
    feature = meta.get('feature', [])
    if isinstance(feature, list):
        feature = ' '.join(feature)
    
    text_parts = []
    if title:
        text_parts.append(f"Title: {title}")
    if brand:
        text_parts.append(f"Brand: {brand}")
    if categories:
        text_parts.append(f"Category: {categories}")
    if feature:
        text_parts.append(f"Features: {feature}")
    if description:
        text_parts.append(f"Description: {description}")
    
    if not text_parts:
        text_parts.append(f"Product ID: {asin}")
    
    item_texts[item_id] = ' '.join(text_parts)

print(f"  - 成功构建 {len(item_texts)} 个物品的文本描述")

# 4. 加载模型并生成嵌入
print(f"\n[4/5] 加载模型: {MODEL_NAME}")
try:
    model = SentenceTransformer(MODEL_NAME)
    print(f"  - 模型加载成功")
except Exception as e:
    print(f"  ✗ 模型加载失败: {e}")
    raise

print(f"\n  生成文本嵌入...")
embeddings = np.zeros((n_items, EMBEDDING_DIM), dtype=np.float32)

batch_size = 32
item_ids = list(range(n_items))
texts = [item_texts[i] for i in item_ids]

for i in tqdm(range(0, len(texts), batch_size), desc="  编码文本"):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
    embeddings[i:i+batch_size] = batch_embeddings

embeddings[0] = 0.0

print(f"  - 嵌入生成完成")
print(f"  - Shape: {embeddings.shape}")

# 5. 保存
print(f"\n[5/5] 保存到: {OUTPUT_FILE}")
np.save(OUTPUT_FILE, embeddings)
print(f"  - 保存成功!")

# 验证
loaded = np.load(OUTPUT_FILE)
print(f"\n验证:")
print(f"  - Shape: {loaded.shape}")
print(f"  - [PAD] 是零向量: {np.allclose(loaded[0], 0.0)}")

print(f"\n" + "=" * 80)
print(f"✓ 完成!")
print(f"  输出: {OUTPUT_FILE}")
print(f"  Shape: {embeddings.shape}")
print(f"=" * 80)
