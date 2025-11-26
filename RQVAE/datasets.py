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
        
        # Handle PAD token mismatch
        # User hint: "第一个是[PAD]" and Semantic has 1 more item than Collab
        if len(self.semantic_embeddings) == len(self.collab_embeddings) + 1:
            print(f"Detected PAD token in Semantic embeddings. Slicing [1:] to align with Collab.")
            self.semantic_embeddings = self.semantic_embeddings[1:]
            
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
