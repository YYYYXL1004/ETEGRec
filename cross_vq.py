"""
CrossRQVAE (交替训练用)

与根目录 vq.py 的角色相同：纯 encoder/RQ/decoder 结构，
loss 由外部 trainer.py 计算。不含 compute_loss。

与 RQVAE/cross_vq.py 的区别：
- quant_loss_weight 读 config['alpha'] (与根目录 vq.py 一致)
- 不含 compute_loss, compute_gini 等方法
- 不含 cross-modal contrastive loss (交替训练阶段不需要，
  因为 codebook 已经在预训练阶段学好了跨模态信息)
- 提供与原 RQVAE 兼容的接口供 trainer.py 调用

trainer.py 调用接口:
  1. model_id.text_forward(x) → (out, rq_loss, indices, code_one_hot, logit)
  2. model_id.image_forward(x) → (out, rq_loss, indices, code_one_hot, logit)
  3. model_id.text_rq(latent) → (x_q, loss, indices, one_hot, logit)
  4. model_id.image_rq(latent) → (x_q, loss, indices, one_hot, logit)
  5. model_id.get_text_indices(x) → indices
  6. model_id.get_image_indices(x) → indices
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class CrossRQVAE(nn.Module):
    """
    双路 RQ-VAE，用于 ETEGRec 交替训练阶段。
    text 和 image 各自有独立的 encoder/RQ/decoder。

    注意: 这里不包含 cross-modal contrastive loss，
    因为交替训练阶段的 codebook 已经在预训练阶段学好了跨模态信息。
    """

    def __init__(self, config, in_dim=1024):
        super(CrossRQVAE, self).__init__()

        self.in_dim = in_dim
        self.e_dim = config['e_dim']
        self.layers = config['layers']
        self.dropout_prob = config['dropout_prob']
        self.bn = config['bn']
        self.quant_loss_weight = config['alpha']  # 注意: 交替训练用 'alpha'
        self.beta = config['beta']
        self.vq_type = config['vq_type']

        if self.vq_type in ["vq"]:
            encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
            decode_layer_dims = encode_layer_dims[::-1]
        else:
            raise NotImplementedError

        # ---- Text 路 ----
        self.text_encoder = MLPLayers(layers=encode_layer_dims,
                                      dropout=self.dropout_prob, bn=self.bn)
        self.text_rq = ResidualVectorQuantizer(config=config)
        self.text_decoder = MLPLayers(layers=decode_layer_dims,
                                      dropout=self.dropout_prob, bn=self.bn)

        # ---- Image 路 ----
        self.image_encoder = MLPLayers(layers=encode_layer_dims,
                                       dropout=self.dropout_prob, bn=self.bn)
        self.image_rq = ResidualVectorQuantizer(config=config)
        self.image_decoder = MLPLayers(layers=decode_layer_dims,
                                       dropout=self.dropout_prob, bn=self.bn)

    def text_forward(self, x):
        """
        Text 路 forward，接口与原 RQVAE.forward 完全一致。
        Returns: (out, rq_loss, indices, code_one_hot, logit)
        """
        latent = self.text_encoder(x)
        x_q, rq_loss, indices, code_one_hot, logit = self.text_rq(latent)
        out = self.text_decoder(x_q)
        return out, rq_loss, indices, code_one_hot, logit

    def image_forward(self, x):
        """
        Image 路 forward，接口与原 RQVAE.forward 完全一致。
        Returns: (out, rq_loss, indices, code_one_hot, logit)
        """
        latent = self.image_encoder(x)
        x_q, rq_loss, indices, code_one_hot, logit = self.image_rq(latent)
        out = self.image_decoder(x_q)
        return out, rq_loss, indices, code_one_hot, logit

    @torch.no_grad()
    def get_text_indices(self, xs):
        """获取 text 路的 code indices"""
        x_e = self.text_encoder(xs)
        return self.text_rq.get_indices(x_e)

    @torch.no_grad()
    def get_image_indices(self, xs):
        """获取 image 路的 code indices"""
        x_e = self.image_encoder(xs)
        return self.image_rq.get_indices(x_e)

    def get_text_codebook(self):
        return self.text_rq.get_codebook()

    def get_image_codebook(self):
        return self.image_rq.get_codebook()


class ResidualVectorQuantizer(nn.Module):
    """
    残差向量量化器 (交替训练用)
    与根目录 vq.py 中的 RVQ 结构一致，使用 config['dist'] 作为距离度量。
    """

    def __init__(self, config):
        super().__init__()
        self.n_e_list = config['num_emb_list']
        self.num_quantizers = len(self.n_e_list)
        self.vq_type = config['vq_type']
        self.dist = config['dist']
        if self.vq_type == "vq":
            self.vq_layers = nn.ModuleList([
                VectorQuantizer(config=config, n_e=n_e, dist=self.dist)
                for n_e in self.n_e_list
            ])
        else:
            raise NotImplementedError

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook.detach().cpu())
        return torch.stack(all_codebook)

    @torch.no_grad()
    def get_indices(self, x):
        all_indices = []
        residual = x
        for vq in self.vq_layers:
            x_res, _, indices, _, _ = vq(residual)
            residual = residual - x_res
            all_indices.append(indices)
        return torch.stack(all_indices, dim=-1)

    def forward(self, x):
        all_losses = []
        all_indices = []
        all_one_hots = []
        all_logits = []
        x_q = 0
        residual = x

        for quantizer in self.vq_layers:
            x_res, loss, indices, one_hot, logit = quantizer(residual)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)
            all_one_hots.append(one_hot)
            all_logits.append(logit)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_one_hots = torch.stack(all_one_hots, dim=1)  # (batch, code_len, code_num)
        all_logits = torch.stack(all_logits, dim=1)

        return x_q, mean_losses, all_indices, all_one_hots, all_logits


class VectorQuantizer(nn.Module):
    """
    向量量化器 (交替训练用)
    与根目录 vq.py 中的 VQ 结构一致。
    """

    def __init__(self, config, n_e, dist):
        super().__init__()
        self.n_e = n_e
        self.dist = dist
        self.e_dim = config['e_dim']
        self.beta = config['beta']
        self.kmeans_init = config['kmeans_init']
        self.kmeans_iters = config['kmeans_iters']
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.initted = False if self.kmeans_init else True
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, detach=True):
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.dist.lower() == 'l2':
            d = (torch.sum(latent ** 2, dim=1, keepdim=True)
                 + torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t()
                 - 2 * torch.matmul(latent, self.embedding.weight.t()))
        elif self.dist.lower() == 'dot':
            d = -torch.matmul(latent, self.embedding.weight.t())
        elif self.dist.lower() == 'cos':
            d = -torch.matmul(
                F.normalize(latent, dim=-1),
                F.normalize(self.embedding.weight, dim=-1).t()
            )
        else:
            raise NotImplementedError

        indices = torch.argmin(d, dim=-1)
        code_one_hot = F.one_hot(indices, self.n_e).float()
        x_q = self.embedding(indices).view(x.shape)

        # VQ loss
        if self.dist.lower() == 'l2':
            codebook_loss = F.mse_loss(x_q, x.detach())
            commitment_loss = F.mse_loss(x_q.detach(), x)
            loss = codebook_loss + self.beta * commitment_loss
        elif self.dist.lower() in ['dot', 'cos']:
            loss = self.beta * F.cross_entropy(-d, indices.detach())
        else:
            raise NotImplementedError

        # Straight-through estimator
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        logit = d

        return x_q, loss, indices, code_one_hot, logit
