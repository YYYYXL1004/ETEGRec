#!/usr/bin/env python3
"""
图像Embedding提取脚本 - 用CLIP提取图像embedding

前置条件: 图片应已由 prepare_data_mm.py 下载到 dataset/{dataset}/images/ 目录

流程:
1. 读取 emb_map.json 获取 item 顺序 (index 0=[PAD], 1~N=真实item)
2. 检查图片是否已下载 (缺失时尝试从 meta 补充下载)
3. 用 CLIP ViT-L/14 提取图像embedding
4. 保存为 .npy 文件, shape=(N_items, 768)

用法:
    python get_image_emb.py --dataset Instrument2018_MM

依赖:
    pip install clip-by-openai pillow tqdm
    或者: pip install git+https://github.com/openai/CLIP.git
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO


def download_image(url, save_path, timeout=10):
    """下载图片到本地，返回是否成功"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        return False


def main(args):
    dataset_dir = os.path.join(args.data_root, args.dataset)
    image_dir = os.path.join(dataset_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    # 1. 加载 item 顺序
    emb_map_path = os.path.join(dataset_dir, f"{args.dataset}.emb_map.json")
    with open(emb_map_path, 'r') as f:
        emb_map = json.load(f)

    # id2asin: index -> asin (跳过 [PAD])
    id2asin = {}
    for asin, idx in emb_map.items():
        if asin != "[PAD]":
            id2asin[idx] = asin
    n_items = len(id2asin)
    print(f"共 {n_items} 个 item (不含PAD)")

    # 2. 检查图片 (图片应已由 prepare_data_mm.py 下载到 images/ 目录)
    print(f"\n📂 检查图片目录: {image_dir}")
    found = 0
    missing = 0
    for idx in range(1, n_items + 1):
        asin = id2asin[idx]
        if os.path.exists(os.path.join(image_dir, f"{asin}.jpg")):
            found += 1
        else:
            missing += 1
    print(f"  已有图片: {found}, 缺失: {missing}")
    if missing > 0:
        print(f"  ⚠️  如果使用 prepare_data_mm.py 生成的数据集，不应有缺失。")
        print(f"     尝试从 meta 补充下载...")
        # fallback: 从 meta 补下载
        meta_path = os.path.join(dataset_dir, args.meta_file)
        asin2url = {}
        with open(meta_path, 'r') as f:
            for line in f:
                d = json.loads(line.strip())
                asin = d.get('asin', '')
                urls = d.get('imageURLHighRes', [])
                if urls:
                    asin2url[asin] = urls[0]
        for idx in tqdm(range(1, n_items + 1), desc="补充下载"):
            asin = id2asin[idx]
            save_path = os.path.join(image_dir, f"{asin}.jpg")
            if not os.path.exists(save_path) and asin in asin2url:
                download_image(asin2url[asin], save_path)

    # 3. 用 CLIP 提取 embedding
    print(f"\n🔧 加载 CLIP 模型: {args.clip_model}")
    try:
        from clip import clip
    except ImportError:
        print("请安装 CLIP: pip install git+https://github.com/openai/CLIP.git")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.clip_model, device=device,
                                   download_root=args.clip_cache_dir)
    model.eval()

    print(f"\n🖼️  提取图像embedding...")
    embeddings = []
    missing_items = []

    with torch.no_grad():
        for idx in tqdm(range(1, n_items + 1), desc="提取embedding"):
            asin = id2asin[idx]
            image_path = os.path.join(image_dir, f"{asin}.jpg")

            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    feat = model.encode_image(image_input)
                    embeddings.append(feat[0].cpu().float())
                    continue
                except Exception as e:
                    pass

            # 没有图片或加载失败 → 报错 (新数据集应保证所有item都有图片)
            missing_items.append((idx, asin))
            embeddings.append(torch.zeros(768))

    embeddings = torch.stack(embeddings, dim=0).numpy()

    if missing_items:
        print(f"⚠️  警告: {len(missing_items)} 个item缺少图片!")
        print(f"   如果使用 prepare_data_mm.py 生成的数据集，不应出现此情况。")
        print(f"   缺失item: {missing_items[:10]}...")  # 只打印前10个

    print(f"\n📊 Embedding shape: {embeddings.shape}")

    # 4. 保存
    save_name = f"{args.dataset}_clip_image_{embeddings.shape[1]}.npy"
    save_path = os.path.join(dataset_dir, save_name)
    np.save(save_path, embeddings)
    print(f"✅ 已保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取图像Embedding (CLIP)")
    parser.add_argument("--dataset", type=str, default="Instrument2018_5090")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--meta_file", type=str, default="meta_Musical_Instruments.json",
                        help="meta JSON 文件名")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14",
                        help="CLIP模型名称")
    parser.add_argument("--clip_cache_dir", type=str, default=None,
                        help="CLIP模型缓存目录")
    args = parser.parse_args()
    main(args)
