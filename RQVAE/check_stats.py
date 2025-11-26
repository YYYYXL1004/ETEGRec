import numpy as np

collab_path = "/sda/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018/Instrument2018_emb_256.npy"
semantic_path = "/sda/data/yaoxianglin/ETEGRec/RecBole/dataset/Instrument2018/Instrument2018_text_768.npy"

try:
    print(f"Loading collab: {collab_path}")
    collab = np.load(collab_path)
    print(f"Collab shape: {collab.shape}")
    print(f"Collab range: [{collab.min():.4f}, {collab.max():.4f}]")
    print(f"Collab mean norm: {np.linalg.norm(collab, axis=1).mean():.4f}")

    print(f"\nLoading semantic: {semantic_path}")
    semantic = np.load(semantic_path)
    print(f"Semantic shape: {semantic.shape}")
    print(f"Semantic range: [{semantic.min():.4f}, {semantic.max():.4f}]")
    print(f"Semantic mean norm: {np.linalg.norm(semantic, axis=1).mean():.4f}")

    collab = collab / (np.linalg.norm(collab, axis=1, keepdims=True) + 1e-9)
    semantic = semantic / (np.linalg.norm(semantic, axis=1, keepdims=True) + 1e-9)
    
    print(f"\nNormalized Collab mean norm: {np.linalg.norm(collab, axis=1).mean():.4f}")
    print(f"Normalized Semantic mean norm: {np.linalg.norm(semantic, axis=1).mean():.4f}")
except Exception as e:
    print(f"Error: {e}")
