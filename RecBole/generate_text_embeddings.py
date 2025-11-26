"""
生成 Instrument2018 数据集的文本嵌入 - 离线版本

如果网络不稳定，可以使用本地已下载的模型或配置镜像
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import torch
from transformers import AutoTokenizer, AutoModel

# 设置 HF 镜像（如果需要）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ========== 配置参数 ==========
DATASET_NAME = "Instrument2018"
DATASET_DIR = f"./dataset/{DATASET_NAME}"
META_FILE = f"{DATASET_DIR}/meta_Musical_Instruments.json"
ITEM2ID_FILE = f"{DATASET_DIR}/{DATASET_NAME}.emb_map.json"

# 模型类型选择: "sentence-transformer" 或 "qwen"
MODEL_TYPE = "qwen"  # 默认使用 sentence-transformer
# MODEL_TYPE = "qwen"  # 使用 Qwen-7B

# 模型配置
if MODEL_TYPE == "sentence-transformer":
    MODEL_NAME = "sentence-transformers/sentence-t5-base" 
    # MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 768
    USE_DIMENSION_REDUCTION = False  # sentence-transformer 不需要降维
    QWEN_HIDDEN_DIM = None
    
elif MODEL_TYPE == "qwen":
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # 或使用本地路径
    # MODEL_NAME = "/path/to/local/qwen-7b"
    
    # 注意：隐藏层维度会在模型加载后自动获取，不同版本 Qwen 可能不同
    # Qwen-7B: 4096, Qwen2.5-7B-Instruct: 3584
    QWEN_HIDDEN_DIM = None  # 稍后从模型配置中获取
    
    # 降维配置
    # 说明：Qwen 原始输出通常是 3584-4096 维
    # 建议先生成全维向量，最后使用 PCA 降维，而不是使用随机投影层
    USE_DIMENSION_REDUCTION = False  # 是否对 Qwen 输出降维 (使用 PCA)
    EMBEDDING_DIM = 4096  # 降维后的目标维度
    # EMBEDDING_DIM = 768  # 或使用 768，拼接后总维度 1024
else:
    raise ValueError(f"不支持的模型类型: {MODEL_TYPE}")

# 模型配置 - 使用本地路径或镜像
# 方式 1: 如果已经下载过模型，指定本地路径
# MODEL_NAME = "/path/to/local/model"

# 方式 2: 使用 HF 镜像（中国大陆用户）
# 在运行前设置环境变量: export HF_ENDPOINT=https://hf-mirror.com
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 方式 3: 使用轻量级模型（推荐）
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384维
# EMBEDDING_DIM = 384

OUTPUT_FILE = f"{DATASET_DIR}/{DATASET_NAME}_{MODEL_TYPE}_text_{EMBEDDING_DIM}.npy"

print(f"=" * 80)
print(f"生成 {DATASET_NAME} 数据集的文本语义嵌入")
print(f"模型类型: {MODEL_TYPE}")
print(f"模型: {MODEL_NAME}")
if MODEL_TYPE == "qwen" and USE_DIMENSION_REDUCTION:
    print(f"目标嵌入维度: {EMBEDDING_DIM} (将自动降维)")
else:
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

if MODEL_TYPE == "sentence-transformer":
    # 使用 SentenceTransformer
    try:
        model = SentenceTransformer(MODEL_NAME)
        print(f"  - 模型加载成功")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        raise
    
    def encode_texts(texts, batch_size=32):
        """使用 SentenceTransformer 编码文本"""
        embeddings_list = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  编码文本"):
            batch_texts = texts[i:i+batch_size]
            batch_embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings_list.append(batch_embs)
        return np.vstack(embeddings_list)

elif MODEL_TYPE == "qwen":
    # 使用 Qwen 模型
    try:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f"  - 使用设备: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # 确保 pad_token 存在
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"  - 设置 pad_token 为 eos_token: {tokenizer.pad_token}")
        
        # 使用半精度 (float16) 加载，节省显存
        # 7B 模型：float32 需要 ~28GB，float16 只需要 ~14GB
        print(f"  - 使用 float16 精度加载模型以节省显存...")
        model = AutoModel.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 半精度（注：新版本推荐用 dtype= 参数）
            low_cpu_mem_usage=True,     # 减少 CPU 内存占用
            device_map=device           # 直接加载到指定设备
        )
        model.eval()
        
        # 动态获取模型的实际隐藏层维度
        actual_hidden_dim = model.config.hidden_size
        print(f"  - 模型加载成功，隐藏层维度: {actual_hidden_dim}")
        
        # 移除随机投影层，改为后续 PCA 降维
        if USE_DIMENSION_REDUCTION:
            print(f"  - 将在生成完成后使用 PCA 降维: {actual_hidden_dim} → {EMBEDDING_DIM}")
            
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        raise
    
    def encode_texts(texts, batch_size=8):
        """使用 Qwen 模型编码文本 (取 Last Token Pooling)"""
        embeddings_list = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="  编码文本"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,
                    return_tensors="pt"
                ).to(device)
                
                # 获取模型输出
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
                
                # Last Token Pooling (对于 Decoder 模型更准确)
                # 找到每个序列最后一个非 padding token 的位置
                attention_mask = inputs['attention_mask']
                # seq_lengths = attention_mask.sum(dim=1) - 1  # 索引从0开始
                # 注意：如果 padding 在左边（left padding），这行代码需要调整。
                # 这里假设 HuggingFace 默认对 feature extraction 做 right padding，或者 tokenizer 自动处理。
                # 为了安全，显式计算最后一个 1 的位置。
                last_token_indices = attention_mask.sum(dim=1) - 1
                
                # Gather last token embeddings
                batch_indices = torch.arange(last_token_indices.shape[0], device=device)
                batch_embs = hidden_states[batch_indices, last_token_indices]  # [batch, hidden_dim]
                
                # 转为 float32 避免 NaN/Inf，再转 numpy
                batch_embs = batch_embs.float().cpu().numpy()  # float16 → float32 → numpy
                embeddings_list.append(batch_embs)
        
        return np.vstack(embeddings_list)

print(f"\n  生成文本嵌入...")
embeddings = np.zeros((n_items, EMBEDDING_DIM), dtype=np.float32)

item_ids = list(range(n_items))
texts = [item_texts[i] for i in item_ids]

# 使用统一的编码接口
batch_size = 32 if MODEL_TYPE == "sentence-transformer" else 8
embeddings = encode_texts(texts, batch_size=batch_size)

# PCA 降维 (如果需要)
if MODEL_TYPE == "qwen" and USE_DIMENSION_REDUCTION:
    try:
        from sklearn.decomposition import PCA
        print(f"\n  [PCA] 正在训练 PCA 将维度从 {embeddings.shape[1]} 降至 {EMBEDDING_DIM} ...")
        # 确保 float32
        embeddings = embeddings.astype(np.float32)
        pca = PCA(n_components=EMBEDDING_DIM)
        embeddings = pca.fit_transform(embeddings)
        print(f"  - PCA 降维完成: {embeddings.shape}")
    except ImportError:
        print("  ! 警告: 未找到 sklearn，无法进行 PCA 降维。保存原始维度。")

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
