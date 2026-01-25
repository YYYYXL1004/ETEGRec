"""
LLaMA 版本的训练器
基于 T5_to_LLaMA2_Migration_Plan v3.2

核心修改：
1. 使用 LlamaRecModel 替代 T5 Model
2. 使用 LlamaRecDataset 的新数据格式
3. 保持 SIA/PSA 对齐逻辑
"""

import os
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from time import time
from torch import optim
from tqdm import tqdm
import json
import math
from colorama import init
from utils import ensure_dir, set_color, log, safe_load
from accelerate import PartialState
from transformers.optimization import get_scheduler
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from metrics import ndcg_at_k
from collections import defaultdict
from logging import getLogger
import zlib

init(autoreset=True)


class LlamaTrainer:
    """
    LLaMA 推荐模型训练器
    
    与原 Trainer 的主要区别：
    1. 数据格式: 直接使用 codes 序列 + 位置索引
    2. Forward: 传入 seq_end_positions 和 target_positions
    3. 评估: 使用 generate() 方法
    """
    
    def __init__(self, config, model_rec, model_id, accelerator,
                 train_data=None, valid_data=None, test_data=None):
        self.config = config
        self.model_rec = model_rec
        self.model_id = model_id
        self.logger = getLogger()
        self.accelerator = accelerator
        
        # === 配置参数 ===
        self.code_num = config["code_num"]
        self.code_length = config["code_length"]
        self.learner = config["learner"]
        self.lr_rec = config['lr_rec']
        self.lr_id = config['lr_id']
        self.lr_scheduler_type = config["lr_scheduler_type"]
        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.early_stop = config["early_stop"]
        self.eval_step = min(config["eval_step"], self.epochs)
        self.gradient_accumulation_steps = config["gradient_accumulation_steps"]
        self.save_path = config["save_path"]
        ensure_dir(self.save_path)
        
        self.alpha = config['alpha']
        self.loss_type = config['loss_type']
        self.tau = config['tau']
        self.warm_epoch = config['warm_epoch']
        self.cycle = config['cycle']
        self.sim = config['sim']
        
        # === 设备 ===
        self.state = PartialState()
        self.world_size = self.state.num_processes
        self.device = self.state.device
        self.all_item_code = None
        
        # === 评估指标 ===
        self.all_metrics = config["metrics"].split(",")
        self.valid_metric = config["valid_metric"]
        self.max_topk = 0
        self.all_metric_name = []
        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            self.max_topk = max(self.max_topk, int(top_k))
            if m_name.lower() not in self.all_metric_name:
                self.all_metric_name.append(m_name.lower())
        
        # === 数据 ===
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        
        # === 优化器 ===
        self.max_steps = self._get_train_steps()
        self.warmup_steps = config["warmup_steps"]
        self.rec_optimizer = self._build_optimizer(model_rec, self.lr_rec, self.weight_decay)
        self.id_optimizer = self._build_optimizer(model_id, self.lr_id, self.weight_decay)
        
        # === 学习率调度器 ===
        self._setup_lr_schedulers()
        
        self.best_score = 0
        self.best_ckpt = None
        
        # === Accelerate 准备 ===
        (self.model_rec, self.rec_optimizer, self.rec_lr_scheduler,
         self.model_id, self.id_optimizer, self.id_lr_scheduler,
         self.train_data, self.valid_data, self.test_data) = \
            self.accelerator.prepare(
                self.model_rec, self.rec_optimizer, self.rec_lr_scheduler,
                self.model_id, self.id_optimizer, self.id_lr_scheduler,
                self.train_data, self.valid_data, self.test_data
            )
    
    def _build_optimizer(self, model, lr, weight_decay):
        """构建优化器"""
        params = model.parameters()
        
        if self.learner.lower() == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif self.learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            self.log(f"未识别的优化器 {self.learner}，使用 AdamW", level='warning')
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        return optimizer
    
    def _setup_lr_schedulers(self):
        """设置学习率调度器"""
        if self.lr_scheduler_type == "cosine":
            self.rec_lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.rec_optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
            self.id_lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.id_optimizer,
                num_warmup_steps=self.warmup_steps // self.cycle,
                num_training_steps=self.max_steps // self.cycle,
            )
        elif self.lr_scheduler_type == "linear":
            self.rec_lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.rec_optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps
            )
            self.id_lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.id_optimizer,
                num_warmup_steps=self.warmup_steps // self.cycle,
                num_training_steps=self.max_steps // self.cycle
            )
        else:
            self.rec_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.rec_optimizer,
                num_warmup_steps=self.warmup_steps
            )
            self.id_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.id_optimizer,
                num_warmup_steps=self.warmup_steps // self.cycle
            )
    
    def _get_train_steps(self, epochs=None):
        """计算总训练步数"""
        len_dataloader = len(self.train_data)
        num_update_steps_per_epoch = len_dataloader // self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if epochs is None:
            epochs = self.epochs
        return math.ceil(epochs * num_update_steps_per_epoch)
    
    # === Loss 计算函数 ===
    
    @staticmethod
    def compute_discrete_contrastive_loss_kl(x_logits, y_logits):
        """KL 散度 Loss (SIA)"""
        code_num = x_logits.size(-1)
        x_logits = F.log_softmax(x_logits.view(-1, code_num), dim=-1)
        y_logits = F.log_softmax(y_logits.view(-1, code_num), dim=-1)
        loss = F.kl_div(x_logits, y_logits, reduction='batchmean', log_target=True)
        return loss
    
    @staticmethod
    def compute_contrastive_loss(query_embeds, semantic_embeds, temperature=0.07, sim="cos"):
        """InfoNCE Loss (PSA)"""
        if sim == "cos":
            query_embeds = F.normalize(query_embeds, dim=-1)
            semantic_embeds = F.normalize(semantic_embeds, dim=-1)
        
        effective_bsz = query_embeds.size(0)
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(query_embeds, semantic_embeds.transpose(0, 1)) / temperature
        
        return F.cross_entropy(similarities, labels)
    
    # === 训练循环 ===
    
    def _train_epoch_rec(self, epoch_idx, loss_w, verbose=True):
        """训练推荐器 (冻结 Tokenizer)"""
        self.model_rec.train()
        self.model_id.eval()
        
        total_num = 0
        total_loss = defaultdict(float)
        
        iter_data = tqdm(
            self.train_data,
            total=len(self.train_data),
            ncols=100,
            desc=set_color(f"Train Rec {epoch_idx}", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )
        
        for batch_idx, batch in enumerate(iter_data):
            with self.accelerator.accumulate(self.model_rec):
                total_num += 1
                self.rec_optimizer.zero_grad()
                
                # === 数据准备 ===
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_end_positions = batch['seq_end_positions'].to(self.device)
                target_positions = batch['target_positions'].to(self.device)
                labels = batch['labels'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # === 获取目标 item 的语义 embedding ===
                if dist.is_initialized():
                    target_semantic_embs = self.model_rec.module.semantic_embedding(targets)
                else:
                    target_semantic_embs = self.model_rec.semantic_embedding(targets)
                
                target_recon_embs, _, _, _, target_code_logits = self.model_id(target_semantic_embs)
                
                # 去重优化
                _, unq_index = np.unique(targets.cpu().numpy(), return_index=True)
                unq_index = torch.tensor(unq_index).to(self.device)
                
                # === Forward ===
                outputs = self.model_rec(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    seq_end_positions=seq_end_positions,
                    target_positions=target_positions,
                )
                
                # === Loss 计算 ===
                
                # 1. Code Loss (生成任务) - 梯度会流向 Codebook!
                code_loss = F.cross_entropy(
                    outputs.logits.view(-1, self.code_num),
                    labels.view(-1)
                )
                
                # 2. SIA Loss (KL 散度)
                if dist.is_initialized():
                    _, _, _, _, seq_code_logits = self.model_id.module.rq(outputs.seq_project_latents)
                else:
                    _, _, _, _, seq_code_logits = self.model_id.rq(outputs.seq_project_latents)
                
                kl_loss = (
                    self.compute_discrete_contrastive_loss_kl(seq_code_logits[unq_index], target_code_logits[unq_index]) +
                    self.compute_discrete_contrastive_loss_kl(target_code_logits[unq_index], seq_code_logits[unq_index])
                )
                
                # 3. PSA Loss (InfoNCE)
                dec_cl_loss = (
                    self.compute_contrastive_loss(target_recon_embs[unq_index], outputs.dec_latents[unq_index], sim=self.sim) +
                    self.compute_contrastive_loss(outputs.dec_latents[unq_index], target_recon_embs[unq_index], sim=self.sim)
                )
                
                # === 总 Loss ===
                loss = (
                    loss_w['code_loss'] * code_loss +
                    loss_w['kl_loss'] * kl_loss +
                    loss_w['dec_cl_loss'] * dec_cl_loss
                )
                
                # === 反向传播 ===
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1.0)
                self.rec_optimizer.step()
                self.rec_lr_scheduler.step()
                
                # === 记录 Loss ===
                loss_dict = {
                    'loss': self.accelerator.gather(loss).mean().item(),
                    'code_loss': self.accelerator.gather(code_loss).mean().item(),
                    'kl_loss': self.accelerator.gather(kl_loss).mean().item(),
                    'dec_cl_loss': self.accelerator.gather(dec_cl_loss).mean().item(),
                }
                
                for k, v in loss_dict.items():
                    total_loss[k] += v
                iter_data.set_postfix(loss=loss_dict['loss'])
        
        for k in total_loss:
            total_loss[k] = round(total_loss[k] / total_num, 4)
        
        self.accelerator.wait_for_everyone()
        return total_loss
    
    def _train_epoch_id(self, epoch_idx, loss_w, verbose=True):
        """训练 Tokenizer (冻结 Recommender)"""
        self.model_id.train()
        self.model_rec.eval()
        
        total_num = 0
        total_loss = defaultdict(float)
        
        iter_data = tqdm(
            self.train_data,
            total=len(self.train_data),
            ncols=100,
            desc=set_color(f"Train ID {epoch_idx}", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )
        
        for batch_idx, batch in enumerate(iter_data):
            with self.accelerator.accumulate(self.model_id):
                total_num += 1
                self.id_optimizer.zero_grad()
                
                # === 数据准备 ===
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                seq_end_positions = batch['seq_end_positions'].to(self.device)
                target_positions = batch['target_positions'].to(self.device)
                labels = batch['labels'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # === 获取目标 item 的语义 embedding ===
                if dist.is_initialized():
                    target_semantic_embs = self.model_rec.module.semantic_embedding(targets)
                else:
                    target_semantic_embs = self.model_rec.semantic_embedding(targets)
                
                target_recon_embs, _, _, _, target_code_logits = self.model_id(target_semantic_embs)
                
                # 去重
                target_flatten = targets.flatten()
                unq_input, unq_index = np.unique(target_flatten.cpu().numpy(), return_index=True)
                unq_input = torch.tensor(unq_input).to(self.device)
                unq_index = torch.tensor(unq_index).to(self.device)
                
                if dist.is_initialized():
                    unq_semantic_embs = self.model_rec.module.semantic_embedding(unq_input)
                else:
                    unq_semantic_embs = self.model_rec.semantic_embedding(unq_input)
                
                unq_recon_embs, commit_loss, _, _, _ = self.model_id(unq_semantic_embs)
                
                # === Forward (不计算梯度) ===
                with torch.no_grad():
                    outputs = self.model_rec(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        seq_end_positions=seq_end_positions,
                        target_positions=target_positions,
                    )
                
                # === Loss 计算 ===
                
                # 1. VQ Loss (重建 + commitment)
                if self.loss_type == 'mse':
                    recon_loss = F.mse_loss(unq_recon_embs, unq_semantic_embs, reduction='mean')
                elif self.loss_type == 'l1':
                    recon_loss = F.l1_loss(unq_recon_embs, unq_semantic_embs, reduction='mean')
                else:
                    recon_loss = self.compute_contrastive_loss(
                        unq_recon_embs, unq_semantic_embs, temperature=self.tau
                    )
                
                vq_loss = recon_loss + self.alpha * commit_loss
                
                # 2. SIA Loss
                if dist.is_initialized():
                    _, _, _, _, seq_code_logits = self.model_id.module.rq(outputs.seq_project_latents)
                else:
                    _, _, _, _, seq_code_logits = self.model_id.rq(outputs.seq_project_latents)
                
                kl_loss = (
                    self.compute_discrete_contrastive_loss_kl(seq_code_logits[unq_index], target_code_logits[unq_index]) +
                    self.compute_discrete_contrastive_loss_kl(target_code_logits[unq_index], seq_code_logits[unq_index])
                )
                
                # 3. PSA Loss
                dec_cl_loss = (
                    self.compute_contrastive_loss(target_recon_embs[unq_index], outputs.dec_latents[unq_index], sim=self.sim) +
                    self.compute_contrastive_loss(outputs.dec_latents[unq_index], target_recon_embs[unq_index], sim=self.sim)
                )
                
                # === 总 Loss ===
                loss = (
                    loss_w['vq_loss'] * vq_loss +
                    loss_w['kl_loss'] * kl_loss +
                    loss_w['dec_cl_loss'] * dec_cl_loss
                )
                
                # === 反向传播 ===
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model_id.parameters(), 1.0)
                self.id_optimizer.step()
                self.id_lr_scheduler.step()
                
                # === 记录 Loss ===
                loss_dict = {
                    'loss': self.accelerator.gather(loss).mean().item(),
                    'vq_loss': self.accelerator.gather(vq_loss).mean().item(),
                    'kl_loss': self.accelerator.gather(kl_loss).mean().item(),
                    'dec_cl_loss': self.accelerator.gather(dec_cl_loss).mean().item(),
                }
                
                for k, v in loss_dict.items():
                    total_loss[k] += v
                iter_data.set_postfix(loss=loss_dict['loss'])
        
        for k in total_loss:
            total_loss[k] = round(total_loss[k] / total_num, 4)
        
        self.accelerator.wait_for_everyone()
        return total_loss
    
    # === 评估 ===
    
    def evaluate(self, outputs, labels):
        """计算评估指标"""
        batch_size, k, _ = outputs.shape
        recall_at_1, recall_at_5, recall_at_10 = [], [], []
        ndcg_at_1, ndcg_at_5, ndcg_at_10 = [], [], []
        
        for i in range(batch_size):
            label = labels[i].unsqueeze(0)
            out = outputs[i]
            
            matches = torch.all(torch.eq(out.unsqueeze(1), label.unsqueeze(0)), dim=2)
            matches = matches.any(dim=1).cpu().numpy()
            
            recall_at_1.append(matches[:1].sum() / 1.0)
            recall_at_5.append(matches[:5].sum() / 1.0)
            recall_at_10.append(matches.sum() / 1.0)
            
            ndcg_at_1.append(ndcg_at_k(matches, 1))
            ndcg_at_5.append(ndcg_at_k(matches, 5))
            ndcg_at_10.append(ndcg_at_k(matches, 10))
        
        return {
            "recall@1": np.sum(recall_at_1),
            "recall@5": np.sum(recall_at_5),
            "recall@10": np.sum(recall_at_10),
            "ndcg@1": np.sum(ndcg_at_1),
            "ndcg@5": np.sum(ndcg_at_5),
            "ndcg@10": np.sum(ndcg_at_10),
        }
    
    @torch.no_grad()
    def _test_epoch(self, test_data=None, verbose=True):
        """评估一个 epoch"""
        if test_data is None:
            test_data = self.test_data
        
        self.model_rec.eval()
        self.model_id.eval()
        
        iter_data = tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color("Evaluate", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )
        
        total = 0
        metrics = {m: 0 for m in self.all_metrics}
        
        for batch in iter_data:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 生成预测
            if dist.is_initialized():
                preds = self.model_rec.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_return_sequences=10
                )
                all_preds, all_labels = self.accelerator.gather_for_metrics((preds, labels))
                _metrics = self.evaluate(all_preds, all_labels)
                total += len(all_labels)
            else:
                preds = self.model_rec.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_return_sequences=10
                )
                _metrics = self.evaluate(preds, labels)
                total += len(labels)
            
            for m in metrics:
                metrics[m] += _metrics[m]
        
        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)
        
        return metrics
    
    # === 主训练循环 ===
    
    def train(self, verbose=True):
        """主训练循环"""
        stop = False
        cur_eval_step = 0
        loss_w = defaultdict(int)
        
        # 初始化 item code 表
        all_item_code = self.get_code(epoch_idx=-1, verbose=verbose)
        self.all_item_code = torch.tensor(all_item_code).to(self.device)
        
        for epoch_idx in range(self.epochs):
            # 设置 Loss 权重和冻结策略
            if epoch_idx % self.cycle == 0:
                # 训练 Tokenizer
                loss_w['vq_loss'] = self.config['id_vq_loss']
                loss_w['code_loss'] = self.config['id_code_loss'] if epoch_idx >= self.warm_epoch else 0
                loss_w['kl_loss'] = self.config['id_kl_loss'] if epoch_idx >= self.warm_epoch else 0
                loss_w['dec_cl_loss'] = self.config['id_dec_cl_loss'] if epoch_idx >= self.warm_epoch else 0
                
                self._freeze_model(self.model_rec)
                self._unfreeze_model(self.model_id)
            else:
                # 训练 Recommender
                loss_w['vq_loss'] = self.config['rec_vq_loss']
                loss_w['code_loss'] = self.config['rec_code_loss']
                loss_w['kl_loss'] = self.config['rec_kl_loss'] if epoch_idx >= self.warm_epoch else 0
                loss_w['dec_cl_loss'] = self.config['rec_dec_cl_loss'] if epoch_idx >= self.warm_epoch else 0
                
                self._unfreeze_model(self.model_rec, exclude_semantic=True)
                self._freeze_model(self.model_id)
            
            self.accelerator.wait_for_everyone()
            
            # 训练
            training_start_time = time()
            if epoch_idx % self.cycle == 0:
                train_loss = self._train_epoch_id(epoch_idx, loss_w=loss_w, verbose=verbose)
                all_item_code = self.get_code(epoch_idx=epoch_idx, verbose=verbose)
                self.all_item_code = torch.tensor(all_item_code).to(self.device)
            else:
                train_loss = self._train_epoch_rec(epoch_idx, loss_w=loss_w, verbose=verbose)
            training_end_time = time()
            
            # 日志
            self.log(f"[Epoch {epoch_idx}] time: {training_end_time - training_start_time:.2f}s, loss: {train_loss}")
            self.log(f"[Epoch {epoch_idx}] REC lr: {self.rec_lr_scheduler.get_last_lr()} ID lr: {self.id_lr_scheduler.get_last_lr()}")
            
            # 评估
            if (epoch_idx + 1) % self.eval_step == 0:
                metrics = self._test_epoch(test_data=self.valid_data, verbose=verbose)
                
                if metrics[self.valid_metric] > self.best_score:
                    self.best_score = metrics[self.valid_metric]
                    self.best_result = metrics
                    cur_eval_step = 0
                    self.best_ckpt = self._save_checkpoint(epoch_idx)
                else:
                    cur_eval_step += 1
                
                if cur_eval_step >= self.early_stop:
                    stop = True
                
                self.log(f"[Epoch {epoch_idx}] Val Results: {metrics}")
            
            self.accelerator.wait_for_everyone()
            
            if stop:
                break
        
        return self.best_score
    
    # === 辅助方法 ===
    
    def _freeze_model(self, model):
        """冻结模型参数"""
        for param in model.parameters():
            param.requires_grad = False
    
    def _unfreeze_model(self, model, exclude_semantic=False):
        """解冻模型参数"""
        for name, param in model.named_parameters():
            if exclude_semantic and 'semantic_embedding' in name:
                continue
            param.requires_grad = True
    
    def _save_checkpoint(self, epoch):
        """保存检查点"""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrap_model_rec = self.accelerator.unwrap_model(self.model_rec)
            unwrap_model_id = self.accelerator.unwrap_model(self.model_id)
            
            # 保存模型
            self.accelerator.save(unwrap_model_rec.state_dict(), f'{self.save_path}/{epoch}.pt')
            self.accelerator.save(unwrap_model_id.state_dict(), f'{self.save_path}/{epoch}.pt.rqvae')
            
            # 保存 code 表
            json.dump(self.all_item_code.cpu().tolist(), open(f'{self.save_path}/{epoch}.code.json', 'w'))
            
            self.log(f"[Epoch {epoch}] Save model to {self.save_path}/{epoch}.pt")
        
        return f'{self.save_path}/{epoch}.pt'
    
    @torch.no_grad()
    def get_code(self, epoch_idx, verbose=True):
        """生成 item code 表 (与原版逻辑一致)"""
        self.model_rec.eval()
        self.model_id.eval()
        
        # 获取语义 embedding
        if dist.is_initialized():
            all_item_embs = self.model_rec.module.semantic_embedding.weight.data[1:]
            all_item_prefix = self.model_id.module.get_indices(all_item_embs).detach().cpu().numpy()
        else:
            all_item_embs = self.model_rec.semantic_embedding.weight.data[1:]
            all_item_prefix = self.model_id.get_indices(all_item_embs).detach().cpu().numpy()
        
        all_item_prefix = all_item_prefix.tolist()
        
        # FORGE 策略
        if verbose:
            self.log(f'[Epoch {epoch_idx}] Applying FORGE load balancing...')
        
        prefix_groups = defaultdict(list)
        for idx, code in enumerate(all_item_prefix):
            if len(code) >= 2:
                prefix_key = tuple(code[:-1])
            else:
                prefix_key = "global"
            prefix_groups[prefix_key].append(idx)
        
        for key, item_indices in prefix_groups.items():
            if key == "global":
                start_offset = 0
            else:
                key_str = str(key).encode('utf-8')
                start_offset = zlib.adler32(key_str) % self.code_num
            
            for i, item_idx in enumerate(item_indices):
                new_c3 = (start_offset + i) % self.code_num
                all_item_prefix[item_idx][-1] = new_c3
        
        # 构建最终 code 表
        tokens2item = defaultdict(list)
        all_item_tokens = [[-1] * self.code_length]
        max_conflict = 0
        
        for i in range(len(all_item_prefix)):
            str_id = ' '.join(map(str, all_item_prefix[i]))
            tokens2item[str_id].append(i + 1)
            
            count = len(tokens2item[str_id]) - 1
            suffix = count
            
            all_item_tokens.append(all_item_prefix[i] + [suffix])
            max_conflict = max(max_conflict, suffix + 1)
        
        self.log(f'[Epoch {epoch_idx}] Max conflict: {max_conflict}')
        
        return all_item_tokens
    
    def log(self, message, level='info'):
        """日志输出"""
        return log(message, self.accelerator, self.logger, level=level)

