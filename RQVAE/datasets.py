import numpy as np
import torch
import torch.utils.data as data


class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class DualEmbDataset(data.Dataset):

    def __init__(self, collab_path, semantic_path, normalize=False):
        self.collab_embeddings = np.load(collab_path)
        self.semantic_embeddings = np.load(semantic_path)
            
        assert len(self.collab_embeddings) == len(self.semantic_embeddings), \
            f"Length mismatch: Collab {len(self.collab_embeddings)} vs Semantic {len(self.semantic_embeddings)}"
            
        # Normalize embeddings to unit sphere to balance contribution to MSE Loss
        if normalize:
            print("Normalizing Collab and Semantic embeddings to unit sphere...")
            self.collab_embeddings = self.collab_embeddings / (np.linalg.norm(self.collab_embeddings, axis=1, keepdims=True) + 1e-9)
            self.semantic_embeddings = self.semantic_embeddings / (np.linalg.norm(self.semantic_embeddings, axis=1, keepdims=True) + 1e-9)
        else:
            print("Skipping Normalization (using raw embeddings)...")
        
        self.dim = self.collab_embeddings.shape[-1] + self.semantic_embeddings.shape[-1]

    def __getitem__(self, index):
        collab_emb = self.collab_embeddings[index]
        semantic_emb = self.semantic_embeddings[index]
        combined_emb = np.concatenate((collab_emb, semantic_emb), axis=0)
        return torch.FloatTensor(combined_emb)

    def __len__(self):
        return len(self.collab_embeddings)


class TripleEmbDataset(data.Dataset):
    """三模态 Dataset: collab + text + image concat"""

    def __init__(self, collab_path, semantic_path, image_path, normalize=False):
        self.collab_embeddings = np.load(collab_path)
        self.semantic_embeddings = np.load(semantic_path)
        self.image_embeddings = np.load(image_path)
            
        assert len(self.collab_embeddings) == len(self.semantic_embeddings) == len(self.image_embeddings), \
            f"Length mismatch: Collab {len(self.collab_embeddings)} vs Semantic {len(self.semantic_embeddings)} vs Image {len(self.image_embeddings)}"
            
        if normalize:
            print("Normalizing Collab, Semantic, and Image embeddings to unit sphere...")
            self.collab_embeddings = self.collab_embeddings / (np.linalg.norm(self.collab_embeddings, axis=1, keepdims=True) + 1e-9)
            self.semantic_embeddings = self.semantic_embeddings / (np.linalg.norm(self.semantic_embeddings, axis=1, keepdims=True) + 1e-9)
            self.image_embeddings = self.image_embeddings / (np.linalg.norm(self.image_embeddings, axis=1, keepdims=True) + 1e-9)
        else:
            print("Skipping Normalization (using raw embeddings)...")
        
        self.dim = self.collab_embeddings.shape[-1] + self.semantic_embeddings.shape[-1] + self.image_embeddings.shape[-1]
        print(f"TripleEmbDataset: collab({self.collab_embeddings.shape[-1]}) + text({self.semantic_embeddings.shape[-1]}) + image({self.image_embeddings.shape[-1]}) = {self.dim}")

    def __getitem__(self, index):
        combined = np.concatenate((
            self.collab_embeddings[index],
            self.semantic_embeddings[index],
            self.image_embeddings[index]
        ), axis=0)
        return torch.FloatTensor(combined)

    def __len__(self):
        return len(self.collab_embeddings)


class CrossModalEmbDataset(data.Dataset):
    """
    双路 Dataset: 分别返回 (collab+text, collab+image, item_index)
    用于 CrossRQVAE 预训练。
    """

    def __init__(self, collab_path, text_path, image_path, normalize=False):
        self.collab_emb = np.load(collab_path)
        self.text_emb = np.load(text_path)
        self.image_emb = np.load(image_path)

        assert len(self.collab_emb) == len(self.text_emb) == len(self.image_emb), \
            f"Length mismatch: collab {len(self.collab_emb)}, text {len(self.text_emb)}, image {len(self.image_emb)}"

        if normalize:
            print("Normalizing embeddings to unit sphere...")
            for attr in ['collab_emb', 'text_emb', 'image_emb']:
                emb = getattr(self, attr)
                setattr(self, attr, emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9))

        # 两路输入维度 (collab + text 或 collab + image)
        self.text_route_dim = self.collab_emb.shape[-1] + self.text_emb.shape[-1]
        self.image_route_dim = self.collab_emb.shape[-1] + self.image_emb.shape[-1]
        assert self.text_route_dim == self.image_route_dim, \
            f"Dimension mismatch: text_route {self.text_route_dim} vs image_route {self.image_route_dim}"
        self.dim = self.text_route_dim

        print(f"CrossModalEmbDataset: collab({self.collab_emb.shape[-1]}) + "
              f"text({self.text_emb.shape[-1]}) / image({self.image_emb.shape[-1]}) "
              f"= {self.dim} per route, {len(self)} items")

    def __getitem__(self, index):
        collab = self.collab_emb[index]
        text_route = np.concatenate((collab, self.text_emb[index]), axis=0)
        image_route = np.concatenate((collab, self.image_emb[index]), axis=0)
        return torch.FloatTensor(text_route), torch.FloatTensor(image_route), index

    def __len__(self):
        return len(self.collab_emb)
