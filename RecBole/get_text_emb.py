#!/usr/bin/env python3
"""
文本语义Embedding提取脚本 (get_text_emb.py)

从元数据中构建物品文本描述，使用预训练语言模型编码为向量，保存为 .npy 文件。

支持模型:
    - sentence-transformer (默认, 768维)
    - qwen (3584/4096维, 可选PCA降维)

用法:
    python get_text_emb.py --dataset Instrument2018_MM
    python get_text_emb.py --dataset Instrument2018_MM --model_type qwen --model_name Qwen/Qwen2.5-7B-Instruct
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

# ================= 5090 精度补丁 =================
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def parse_args():
    parser = argparse.ArgumentParser(description="提取文本语义Embedding")
    parser.add_argument("--dataset", type=str, default="Instrument2018_5090")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--meta_file", type=str, default="meta_Musical_Instruments.json")
    parser.add_argument("--model_type", type=str, default="sentence-transformer",
                        choices=["sentence-transformer", "qwen"])
    parser.add_argument("--model_name", type=str, default=None,
                        help="模型名称或本地路径 (默认根据model_type自动选择)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="编码batch size (默认: st=32, qwen=8)")
    parser.add_argument("--pca_dim", type=int, default=None,
                        help="PCA降维目标维度 (仅qwen, 不指定则不降维)")
    parser.add_argument("--device", type=str, default="cuda:1")
    return parser.parse_args()


def load_item_mapping(emb_map_path):
    """加载 item2id 映射，返回 id2asin 字典 (跳过 [PAD])"""
    with open(emb_map_path, 'r') as f:
        item2id = json.load(f)
    id2asin = {v: k for k, v in item2id.items()}
    n_items = len(item2id)
    print(f"物品总数: {n_items} (含 [PAD])")
    return id2asin, n_items


def load_meta(meta_path):
    """加载元数据，返回 asin -> meta_dict 映射"""
    meta_data = {}
    with open(meta_path, 'r') as f:
        for line in tqdm(f, desc="读取元数据"):
            item = json.loads(line.strip())
            asin = item.get('asin', '')
            if asin:
                meta_data[asin] = item
    print(f"元数据条目数: {len(meta_data)}")
    return meta_data


def build_item_texts(id2asin, meta_data, n_items):
    """为每个item构建文本描述，返回按id排序的文本列表"""
    texts = []
    for item_id in range(n_items):
        asin = id2asin.get(item_id, '')
        if asin == '[PAD]':
            texts.append("")
            continue

        meta = meta_data.get(asin, {})
        title = meta.get('title', '')
        brand = meta.get('brand', '')
        categories = meta.get('category', [])
        if isinstance(categories, list):
            categories = ' '.join(categories)
        feature = meta.get('feature', [])
        if isinstance(feature, list):
            feature = ' '.join(feature)
        description = meta.get('description', [])
        if isinstance(description, list):
            description = ' '.join(description)

        parts = []
        if title:       parts.append(f"Title: {title}")
        if brand:       parts.append(f"Brand: {brand}")
        if categories:  parts.append(f"Category: {categories}")
        if feature:     parts.append(f"Features: {feature}")
        if description: parts.append(f"Description: {description}")
        if not parts:   parts.append(f"Product ID: {asin}")

        texts.append(' '.join(parts))

    print(f"构建文本描述: {len(texts)} 个item")
    return texts


def encode_with_sentence_transformer(texts, model_name, batch_size=32):
    """使用 SentenceTransformer 编码"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"模型加载成功: {model_name}")

    embeddings_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings_list.append(embs)
    return np.vstack(embeddings_list)


def encode_with_qwen(texts, model_name, device, batch_size=8):
    """使用 Qwen 模型编码 (Last Token Pooling)"""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device
    )
    model.eval()
    print(f"模型加载成功: {model_name}, hidden_size={model.config.hidden_size}")

    embeddings_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device)
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            # Last Token Pooling
            last_idx = inputs['attention_mask'].sum(dim=1) - 1
            batch_idx = torch.arange(last_idx.shape[0], device=hidden_states.device)
            embs = hidden_states[batch_idx, last_idx]
            embeddings_list.append(embs.float().cpu().numpy())

    return np.vstack(embeddings_list)


def main():
    args = parse_args()

    # 路径
    dataset_dir = os.path.join(args.data_root, args.dataset)
    emb_map_path = os.path.join(dataset_dir, f"{args.dataset}.emb_map.json")
    meta_path = os.path.join(dataset_dir, args.meta_file)

    # 默认模型名
    if args.model_name is None:
        args.model_name = {
            "sentence-transformer": "sentence-transformers/sentence-t5-base",
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
        }[args.model_type]

    # 默认 batch size
    if args.batch_size is None:
        args.batch_size = 32 if args.model_type == "sentence-transformer" else 8

    print("=" * 70)
    print(f"文本Embedding提取: {args.dataset}")
    print(f"模型: {args.model_type} ({args.model_name})")
    print("=" * 70)

    # 1. 加载映射和元数据
    id2asin, n_items = load_item_mapping(emb_map_path)
    meta_data = load_meta(meta_path)

    # 2. 构建文本
    texts = build_item_texts(id2asin, meta_data, n_items)

    # 3. 编码
    if args.model_type == "sentence-transformer":
        embeddings = encode_with_sentence_transformer(texts, args.model_name, args.batch_size)
    else:
        embeddings = encode_with_qwen(texts, args.model_name, args.device, args.batch_size)

    # 4. PCA 降维 (可选, 仅 qwen)
    if args.pca_dim is not None and args.model_type == "qwen":
        from sklearn.decomposition import PCA
        print(f"PCA 降维: {embeddings.shape[1]} → {args.pca_dim}")
        embeddings = PCA(n_components=args.pca_dim).fit_transform(embeddings.astype(np.float32))

    # 5. PAD 位置置零
    embeddings[0] = 0.0

    # 6. 保存
    emb_dim = embeddings.shape[1]
    output_file = os.path.join(dataset_dir,
                               f"{args.dataset}_{args.model_type}_text_{emb_dim}.npy")
    np.save(output_file, embeddings)

    print(f"\n✅ 已保存: {output_file}")
    print(f"   Shape: {embeddings.shape}")
    print(f"   [PAD] 零向量: {np.allclose(embeddings[0], 0.0)}")


if __name__ == "__main__":
    main()
