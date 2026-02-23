"""
伪标签生成脚本 (Cross-modal Quantization 预处理)

对 text embedding 和 image embedding 分别做 KMeans 聚类，
生成 item2item_list 映射文件，用于 CrossRQVAE 的跨模态对比学习。

输出格式 (JSON):
{
    "0": [0, 3, 7, 12, ...],   # item 0 所在 cluster 的所有 item indices
    "1": [1, 5, 9, ...],
    ...
}

用法:
python gen_pseudo_labels.py \
    --text_path /path/to/text_768.npy \
    --image_path /path/to/image_768.npy \
    --n_clusters 512 \
    --output_dir /path/to/output/
"""

import argparse
import json
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


def generate_pseudo_labels(emb_path, n_clusters, random_state=42):
    """
    对 embedding 做 KMeans 聚类，返回 item2item_list。
    
    Args:
        emb_path: embedding npy 文件路径
        n_clusters: KMeans 聚类数
        random_state: 随机种子
    
    Returns:
        item2item_list: dict, {item_idx: [同 cluster 的所有 item indices]}
        avg_cluster_size: 平均每个 cluster 的 item 数量
    """
    embeddings = np.load(emb_path)
    print(f"Loaded embeddings: {embeddings.shape} from {emb_path}")

    # KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, verbose=0)
    labels = kmeans.fit_predict(embeddings)

    # 构建 cluster -> item list 映射
    cluster2items = defaultdict(list)
    for item_idx, cluster_id in enumerate(labels):
        cluster2items[cluster_id].append(item_idx)

    # 构建 item -> 同 cluster item list 映射
    item2item_list = {}
    for item_idx, cluster_id in enumerate(labels):
        item2item_list[item_idx] = cluster2items[cluster_id]

    avg_cluster_size = np.mean([len(v) for v in cluster2items.values()])
    print(f"  n_clusters: {n_clusters}, avg_cluster_size: {avg_cluster_size:.1f}")

    return item2item_list, avg_cluster_size


def save_pseudo_labels(item2item_list, output_path):
    """保存为 JSON，key 转为 str（JSON 要求）"""
    str_key_dict = {str(k): v for k, v in item2item_list.items()}
    with open(output_path, 'w') as f:
        json.dump(str_key_dict, f)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels for cross-modal contrastive learning")
    parser.add_argument("--text_path", type=str, required=True, help="Path to text embedding npy file")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image embedding npy file")
    parser.add_argument("--n_clusters", type=int, default=512, help="Number of KMeans clusters")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for pseudo-label JSON files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Text 伪标签
    print("=== Generating text pseudo-labels ===")
    text_item2items, _ = generate_pseudo_labels(args.text_path, args.n_clusters, args.seed)
    text_output = os.path.join(args.output_dir, "text_class_info.json")
    save_pseudo_labels(text_item2items, text_output)

    # Image 伪标签
    print("=== Generating image pseudo-labels ===")
    image_item2items, _ = generate_pseudo_labels(args.image_path, args.n_clusters, args.seed)
    image_output = os.path.join(args.output_dir, "image_class_info.json")
    save_pseudo_labels(image_item2items, image_output)

    print("Done.")


if __name__ == "__main__":
    main()
