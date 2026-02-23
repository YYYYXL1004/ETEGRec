"""
CrossRQVAE: 双路 RQ-VAE + 跨模态对比学习 (预训练用)

参考 MACRec 的 Cross-modal Quantization 机制:
- text 路: concat(collab, text) → encoder → RQ → decoder → recon
- image 路: concat(collab, image) → encoder → RQ → decoder → recon
- 从 begin_cross_layer 开始，在残差层加入跨模态对比学习
- 额外的 reconstruction alignment loss 对齐两路量化表示

与 RQVAE/vq.py 的区别:
- 双路 encoder/RQ/decoder
- 包含 compute_loss (自包含训练，不依赖外部 trainer 算 loss)
- 包含 cross-modal contrastive loss 和 alignment loss
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MLPLayers, kmeans, compute_gini


class CrossRQVAE(nn.Module):
    """
    双路 RQ-VAE，text 和 image 各自有独立的 encoder/RQ/decoder。
    从 begin_cross_layer 开始，在 RQ 残差层引入跨模态对比学习。
    
    Args:
        config: 配置字典，需包含以下 key:
            - num_emb_list: list[int], 每层 codebook 大小, e.g. [256, 256, 256]
            - e_dim: int, codebook embedding 维度
            - layers: list[int], encoder/decoder 中间层维度
            - dropout_prob, bn, loss_type, beta, vq_type, dist, tau: RQ-VAE 基础参数
            - begin_cross_layer: int, 从第几层开始做 cross-modal contrastive
            - text_contrast_weight: float, text 路对比学习权重
            - image_contrast_weight: float, image 路对比学习权重
            - recon_contrast_weight: float, reconstruction alignment 权重
        in_dim: int, 每路输入维度 (collab+text 或 collab+image)
        text_class_info: dict, {item_idx: [同 image cluster 的 item indices]}
        image_class_info: dict, {item_idx: [同 text cluster 的 item indices]}
    """

    def __init__(self, config, in_dim=1024,
                 text_class_info=None, image_class_info=None):
        super(CrossRQVAE, self).__init__()

        self.in_dim = in_dim
        self.e_dim = config['e_dim']
        self.num_emb_list = config['num_emb_list']
        self.num_rq_layers = len(self.num_emb_list)
        self.layers = config['layers']
        self.dropout_prob = config['dropout_prob']
        self.bn = config['bn']
        self.loss_type = config['loss_type']
        self.quant_loss_weight = config['quant_loss_weight']
        self.beta = config['beta']
        self.vq_type = config['vq_type']
        self.dist = config['dist']
        self.tau = config['tau']

        # Cross-modal 参数
        self.begin_cross_layer = config.get('begin_cross_layer', 1)
        self.text_contrast_weight = config.get('text_contrast_weight', 0.1)
        self.image_contrast_weight = config.get('image_contrast_weight', 0.1)
        self.recon_contrast_weight = config.get('recon_contrast_weight', 0.001)

        # 伪标签 (item_idx -> 同 cluster 的 item list)
        # text_class_info: 用 image 聚类的伪标签，指导 text 残差
        # image_class_info: 用 text 聚类的伪标签，指导 image 残差
        self.text_class_info = text_class_info  # image pseudo-labels -> guide text
        self.image_class_info = image_class_info  # text pseudo-labels -> guide image

        # Encoder/Decoder 层维度
        encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        decode_layer_dims = encode_layer_dims[::-1]

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

    def cross_modal_contrastive_rq(
        self, text_vq, image_vq, residual_text, residual_image,
        item_index=None, temperature=0.1
    ):
        """
        在单层 RQ 上执行跨模态对比学习。
        
        1. 正常做 VQ 量化 (text_vq, image_vq)
        2. 用对方模态的伪标签构建正样本对
        3. 计算 InfoNCE loss，加到 vq_loss 上
        
        Args:
            text_vq: text 路的 VectorQuantizer (单层)
            image_vq: image 路的 VectorQuantizer (单层)
            residual_text: (B, e_dim), text 路当前层的残差
            residual_image: (B, e_dim), image 路当前层的残差
            item_index: (B,), batch 中每个样本的全局 item index
            temperature: 对比学习温度
            
        Returns:
            text_x_res, text_loss, text_indices: text 路量化结果
            image_x_res, image_loss, image_indices: image 路量化结果
        """
        # 正常量化
        text_x_res, text_loss, text_indices, _, _ = text_vq(residual_text)
        image_x_res, image_loss, image_indices, _, _ = image_vq(residual_image)

        if item_index is not None and self.text_class_info is not None:
            batch_size = residual_text.size(0)

            # L2 归一化残差
            text_feat = F.normalize(residual_text, p=2, dim=1)
            image_feat = F.normalize(residual_image, p=2, dim=1)

            # 全局 index -> batch 内 index 的映射
            global2batch = {int(idx): i for i, idx in enumerate(item_index)}

            # ---- Text 残差的对比学习 (用 image 伪标签指导) ----
            # 对于每个 anchor text 残差，正样本是共享同一 image cluster 的其他 item 的 text 残差
            pos_idx_text = []
            for i in range(batch_size):
                anchor_global = int(item_index[i])
                # text_class_info 存的是 image 聚类的伪标签
                pos_globals = set(self.text_class_info.get(anchor_global, []))
                batch_pos = [global2batch[g] for g in pos_globals
                             if g in global2batch and global2batch[g] != i]
                pos_idx_text.append(random.choice(batch_pos) if batch_pos else i)
            pos_idx_text = torch.tensor(pos_idx_text, device=residual_text.device)

            sim_text = torch.matmul(text_feat, text_feat.T) / temperature
            pos_sim_text = sim_text[torch.arange(batch_size), pos_idx_text]
            loss_text = -torch.log(
                torch.exp(pos_sim_text) / torch.exp(sim_text).sum(dim=1)
            ).mean()

            # ---- Image 残差的对比学习 (用 text 伪标签指导) ----
            pos_idx_image = []
            for i in range(batch_size):
                anchor_global = int(item_index[i])
                pos_globals = set(self.image_class_info.get(anchor_global, []))
                batch_pos = [global2batch[g] for g in pos_globals
                             if g in global2batch and global2batch[g] != i]
                pos_idx_image.append(random.choice(batch_pos) if batch_pos else i)
            pos_idx_image = torch.tensor(pos_idx_image, device=residual_image.device)

            sim_image = torch.matmul(image_feat, image_feat.T) / temperature
            pos_sim_image = sim_image[torch.arange(batch_size), pos_idx_image]
            loss_image = -torch.log(
                torch.exp(pos_sim_image) / torch.exp(sim_image).sum(dim=1)
            ).mean()

            # 将对比 loss 加到 vq_loss 上
            text_loss = text_loss + self.text_contrast_weight * loss_text
            image_loss = image_loss + self.image_contrast_weight * loss_image

        return text_x_res, text_loss, text_indices, image_x_res, image_loss, image_indices

    def recon_alignment_loss(self, text_xq, image_xq, temperature=0.1):
        """
        Reconstruction Alignment: 对齐同一 item 的 text 和 image 量化表示。
        双向 InfoNCE loss。
        
        Args:
            text_xq: (B, e_dim), text 路量化后的表示 (sum of codebook vectors)
            image_xq: (B, e_dim), image 路量化后的表示
            temperature: 温度参数
        """
        text_norm = F.normalize(text_xq, p=2, dim=1)
        image_norm = F.normalize(image_xq, p=2, dim=1)
        sim = torch.matmul(text_norm, image_norm.T) / temperature
        batch_size = text_xq.size(0)
        labels = torch.arange(batch_size, device=text_xq.device)
        loss = F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        return loss

    def forward(self, text_x, image_x, item_index=None):
        """
        双路 forward。
        
        Args:
            text_x: (B, in_dim), collab+text 拼接的 embedding
            image_x: (B, in_dim), collab+image 拼接的 embedding
            item_index: (B,), 全局 item index (用于跨模态对比学习)
            
        Returns:
            text_out: (B, in_dim), text 路重建
            image_out: (B, in_dim), image 路重建
            text_rq_loss: scalar, text 路 VQ loss (含 cross-modal contrastive)
            image_rq_loss: scalar, image 路 VQ loss (含 cross-modal contrastive)
            text_indices: (B, num_rq_layers), text 路 code indices
            image_indices: (B, num_rq_layers), image 路 code indices
            text_xq: (B, e_dim), text 路量化表示
            image_xq: (B, e_dim), image 路量化表示
        """
        # Encode
        text_latent = self.text_encoder(text_x)
        image_latent = self.image_encoder(image_x)

        # 逐层 RQ，从 begin_cross_layer 开始加 cross-modal contrastive
        text_rq_losses = []
        image_rq_losses = []
        text_indices_list = []
        image_indices_list = []
        text_xq = 0
        image_xq = 0
        residual_text = text_latent
        residual_image = image_latent

        for i in range(self.num_rq_layers):
            text_vq = self.text_rq.vq_layers[i]
            image_vq = self.image_rq.vq_layers[i]

            if i >= self.begin_cross_layer:
                # 跨模态对比学习层
                (text_x_res, text_loss, text_idx,
                 image_x_res, image_loss, image_idx) = \
                    self.cross_modal_contrastive_rq(
                        text_vq, image_vq,
                        residual_text, residual_image,
                        item_index=item_index
                    )
            else:
                # 普通独立量化层
                text_x_res, text_loss, text_idx, _, _ = text_vq(residual_text)
                image_x_res, image_loss, image_idx, _, _ = image_vq(residual_image)

            # 更新残差
            residual_text = residual_text - text_x_res
            residual_image = residual_image - image_x_res
            text_xq = text_xq + text_x_res
            image_xq = image_xq + image_x_res

            text_rq_losses.append(text_loss)
            image_rq_losses.append(image_loss)
            text_indices_list.append(text_idx)
            image_indices_list.append(image_idx)

        text_rq_loss = torch.stack(text_rq_losses).mean()
        image_rq_loss = torch.stack(image_rq_losses).mean()
        text_indices = torch.stack(text_indices_list, dim=-1)   # (B, num_rq_layers)
        image_indices = torch.stack(image_indices_list, dim=-1)

        # Decode
        text_out = self.text_decoder(text_xq)
        image_out = self.image_decoder(image_xq)

        return (text_out, image_out,
                text_rq_loss, image_rq_loss,
                text_indices, image_indices,
                text_xq, image_xq)

    def compute_loss(self, text_out, image_out, text_rq_loss, image_rq_loss,
                     text_xq, image_xq, text_x, image_x):
        """
        计算总 loss = recon_loss + vq_loss + alignment_loss
        
        Args:
            text_out, image_out: 重建输出
            text_rq_loss, image_rq_loss: VQ loss (已含 cross-modal contrastive)
            text_xq, image_xq: 量化表示 (用于 alignment)
            text_x, image_x: 原始输入 (用于 recon loss)
        """
        # Reconstruction loss
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(text_out, text_x) + F.mse_loss(image_out, image_x)
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(text_out, text_x) + F.l1_loss(image_out, image_x)
        else:
            raise ValueError(f'Unsupported loss_type: {self.loss_type}')

        # VQ loss (已包含 cross-modal contrastive loss)
        loss_vq = text_rq_loss + image_rq_loss

        # Reconstruction alignment loss
        loss_align = self.recon_alignment_loss(text_xq, image_xq)

        loss_total = loss_recon + self.quant_loss_weight * loss_vq + \
                     self.recon_contrast_weight * loss_align

        return loss_total, loss_recon, loss_align

    @torch.no_grad()
    def get_text_indices(self, text_x):
        """获取 text 路的 code indices"""
        latent = self.text_encoder(text_x)
        return self.text_rq.get_indices(latent)

    @torch.no_grad()
    def get_image_indices(self, image_x):
        """获取 image 路的 code indices"""
        latent = self.image_encoder(image_x)
        return self.image_rq.get_indices(latent)

    def get_text_codebook(self):
        return self.text_rq.get_codebook()

    def get_image_codebook(self):
        return self.image_rq.get_codebook()


# ============================================================
# 以下是底层组件，复用自 RQVAE/vq.py，保持一致
# ============================================================

class ResidualVectorQuantizer(nn.Module):
    """残差向量量化器，与 RQVAE/vq.py 中的实现一致"""

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
        all_one_hots = torch.stack(all_one_hots, dim=1)
        all_logits = torch.stack(all_logits, dim=1)
        return x_q, mean_losses, all_indices, all_one_hots, all_logits


class VectorQuantizer(nn.Module):
    """向量量化器，与 RQVAE/vq.py 中的实现一致"""

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
