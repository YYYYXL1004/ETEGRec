"""
将高维嵌入通过 PCA 投影到低维空间。
用于三模态融合前降低 text/image 嵌入维度，避免 RQVAE collision 过高。

用法:
    python project_emb.py \
        --input dataset/Instrument2018_MM/Instrument2018_MM_sentence-transformer_text_768.npy \
        --output dataset/Instrument2018_MM/Instrument2018_MM_text_proj128.npy \
        --target_dim 128

    python project_emb.py \
        --input dataset/Instrument2018_MM/Instrument2018_MM_clip_image_768.npy \
        --output dataset/Instrument2018_MM/Instrument2018_MM_image_proj128.npy \
        --target_dim 128
"""
import argparse
import numpy as np
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser(description="PCA projection for embeddings")
    parser.add_argument("--input", type=str, required=True, help="Input .npy file")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file")
    parser.add_argument("--target_dim", type=int, default=128, help="Target dimension")
    args = parser.parse_args()

    emb = np.load(args.input)
    print(f"Input: {args.input}")
    print(f"  shape: {emb.shape}, dtype: {emb.dtype}")

    # 检查是否有 PAD token (第一行全零或长度比预期多1)
    has_pad = False
    if np.allclose(emb[0], 0):
        print("  Detected PAD token at index 0, excluding from PCA fit")
        has_pad = True
        pad_row = emb[0:1]
        emb_data = emb[1:]
    else:
        emb_data = emb

    print(f"  Fitting PCA: {emb_data.shape[-1]} -> {args.target_dim}")
    pca = PCA(n_components=args.target_dim)
    projected = pca.fit_transform(emb_data)

    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance ratio: {explained:.4f}")

    if has_pad:
        # PAD token 投影为零向量
        pad_proj = np.zeros((1, args.target_dim), dtype=np.float32)
        projected = np.vstack([pad_proj, projected])

    projected = projected.astype(np.float32)
    np.save(args.output, projected)
    print(f"Output: {args.output}, shape: {projected.shape}")


if __name__ == "__main__":
    main()
