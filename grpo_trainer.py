"""
GRPO (Group Relative Policy Optimization) Trainer for ETEGRec
基于 MiniOneRec 的 GRPO 实现，适配 ETEGRec 的 T5 + RQ-VAE 架构
"""
import os
import copy
import json
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from time import time
from torch import optim
from tqdm import tqdm
from collections import defaultdict
from logging import getLogger

from trainer import Trainer
from utils import set_color, log, safe_load
from transformers.optimization import get_scheduler


class GRPOTrainer(Trainer):
    """
    GRPO Trainer: 在 SFT 训练好的模型基础上，使用 GRPO 进行强化学习微调
    
    核心思想：
    1. 对每个用户历史，采样 G 个候选推荐
    2. 用混合奖励函数打分 (Rule + Semantic Similarity)
    3. 组内归一化计算 Advantage
    4. 用 Policy Gradient + KL Penalty 更新模型
    """
    
    def __init__(self, config, model_rec, model_id, ref_model, accelerator,
                 train_data=None, valid_data=None, test_data=None, 
                 all_item_code=None, eos_token_id=None):
        """
        Args:
            ref_model: 冻结的 SFT 模型副本，用于计算 KL divergence
            all_item_code: [n_items+1, code_length] 预计算好的 item code
        """
        # 调用父类初始化（会设置 optimizer, scheduler 等）
        # 但我们需要重新配置一些参数
        self.config = config
        self.model_rec = model_rec
        self.model_id = model_id
        self.ref_model = ref_model
        self.logger = getLogger()
        
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
        self.code_num = config["code_num"]
        self.code_length = config["code_length"]
        
        # GRPO 特有参数
        self.G = config.get('grpo_group_size', 4)
        self.beta = config.get('grpo_kl_coeff', 0.05)
        self.temperature = config.get('grpo_temperature', 1.0)
        self.top_k = config.get('grpo_top_k', 50)
        self.reward_alpha = config.get('grpo_reward_alpha', 0.5)  # soft reward 权重
        self.clip_range = config.get('grpo_clip_range', 0.2)
        
        # 训练参数
        self.learner = config.get("grpo_learner", "adamw")
        self.lr = config.get('grpo_lr', 1e-5)
        self.weight_decay = config.get("weight_decay", 0.01)
        self.epochs = config.get('grpo_epochs', 20)
        self.early_stop = config.get("grpo_early_stop", 5)
        self.eval_step = config.get("grpo_eval_step", 1)
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.save_path = config["save_path"]
        
        self.accelerator = accelerator
        self.device = accelerator.device
        self.world_size = accelerator.num_processes
        
        # 数据
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        
        # Item codes
        if all_item_code is not None:
            self.all_item_code = torch.tensor(all_item_code).to(self.device)
        else:
            self.all_item_code = None
            
        # 构建反向索引
        self.code_to_item = None
        
        # Metrics
        self.all_metrics = config["metrics"].split(",")
        self.valid_metric = config["valid_metric"]
        self.max_topk = 0
        self.all_metric_name = []
        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            self.max_topk = max(self.max_topk, int(top_k))
            if m_name.lower() not in self.all_metric_name:
                self.all_metric_name.append(m_name.lower())
        
        self.best_score = 0
        self.best_ckpt = None
        
        # 冻结 ref_model 和 model_id
        self._freeze_models()
        
        # 构建 optimizer 和 scheduler
        self._build_grpo_optimizer()
        
        # Accelerator prepare
        self.model_rec, self.optimizer, self.lr_scheduler, \
        self.train_data, self.valid_data, self.test_data = \
        self.accelerator.prepare(
            self.model_rec, self.optimizer, self.lr_scheduler,
            self.train_data, self.valid_data, self.test_data
        )
        
    def _freeze_models(self):
        """冻结 ref_model, model_id, 和 semantic_embedding
        注意：此方法在 accelerator.prepare() 之前调用，模型尚未被 DDP 包装
        """
        # 冻结 ref_model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        # 冻结 model_id (RQ-VAE)
        for param in self.model_id.parameters():
            param.requires_grad = False
        self.model_id.eval()
        
        # 冻结 semantic_embedding (此时模型未被 wrap，直接访问)
        self.model_rec.semantic_embedding.requires_grad_(False)
            
        self.log("[GRPO] Frozen: ref_model, model_id, semantic_embedding")
        
    def _build_grpo_optimizer(self):
        """构建 GRPO 专用的 optimizer 和 scheduler
        注意：此方法在 accelerator.prepare() 之前调用，模型尚未被 DDP 包装
        """
        # 只训练 model_rec 中非 semantic_embedding 的参数
        trainable_params = []
        # 此时模型未被 wrap，直接访问
        model = self.model_rec
            
        for name, param in model.named_parameters():
            if not name.startswith('semantic_embedding') and param.requires_grad:
                trainable_params.append(param)
        
        if self.learner.lower() == 'adamw':
            self.optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
            
        # Scheduler
        train_steps = len(self.train_data) * self.epochs // self.gradient_accumulation_steps
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=train_steps,
        )
        
        self.log(f"[GRPO] Trainable params: {sum(p.numel() for p in trainable_params)}")
        
    def set_item_codes(self, all_item_code):
        """设置 item codes 并构建反向索引"""
        if isinstance(all_item_code, list):
            self.all_item_code = torch.tensor(all_item_code).to(self.device)
        else:
            self.all_item_code = all_item_code.to(self.device)
        self.code_to_item = self.build_code_to_item_index(self.all_item_code)
        self.log(f"[GRPO] Built code_to_item index with {len(self.code_to_item)} entries")
        
    def codes_to_items(self, codes):
        """
        将生成的 codes 转换为 item IDs
        
        Args:
            codes: [B, G, code_length] 或 [B, code_length]
            
        Returns:
            item_ids: [B, G] 或 [B], -1 表示 OOV
        """
        original_shape = codes.shape[:-1]
        codes_flat = codes.view(-1, self.code_length)  # [B*G, code_length]
        
        item_ids = []
        for i in range(codes_flat.shape[0]):
            code_tuple = tuple(codes_flat[i].cpu().tolist())
            item_id = self.code_to_item.get(code_tuple, -1)  # -1 for OOV
            item_ids.append(item_id)
            
        item_ids = torch.tensor(item_ids, device=self.device)
        return item_ids.view(original_shape)
    
    def compute_reward(self, gen_codes, targets):
        """
        计算混合奖励：Rule Reward + Soft Reward (Semantic Similarity)
        
        Args:
            gen_codes: [B, G, code_length] 生成的 code 序列
            targets: [B] 或 [B, 1] ground truth item IDs
            
        Returns:
            rewards: [B, G] 奖励分数
        """
        B, G, _ = gen_codes.shape
        targets = targets.view(B)
        
        # 1. 将 codes 转为 item IDs
        gen_item_ids = self.codes_to_items(gen_codes)  # [B, G]
        
        # 2. Rule Reward: hit = 1, miss = 0, OOV = -0.1
        rule_reward = torch.zeros(B, G, device=self.device)
        for b in range(B):
            target_id = targets[b].item()
            for g in range(G):
                gen_id = gen_item_ids[b, g].item()
                if gen_id == target_id:
                    rule_reward[b, g] = 1.0
                elif gen_id == -1:  # OOV
                    rule_reward[b, g] = -0.1
                else:
                    rule_reward[b, g] = 0.0
        
        # 3. Soft Reward: Semantic Similarity
        soft_reward = torch.zeros(B, G, device=self.device)
        
        if dist.is_initialized():
            semantic_emb = self.model_rec.module.semantic_embedding
        else:
            semantic_emb = self.model_rec.semantic_embedding
            
        # 获取 target embeddings
        target_embs = semantic_emb(targets)  # [B, emb_dim]
        target_embs = F.normalize(target_embs, dim=-1)
        
        for b in range(B):
            for g in range(G):
                gen_id = gen_item_ids[b, g].item()
                if gen_id > 0:  # 有效 item
                    gen_emb = semantic_emb(torch.tensor([gen_id], device=self.device))
                    gen_emb = F.normalize(gen_emb, dim=-1)
                    sim = F.cosine_similarity(gen_emb, target_embs[b:b+1], dim=-1)
                    soft_reward[b, g] = sim.item()
                else:  # OOV
                    soft_reward[b, g] = 0.0
        
        # 4. 合并奖励
        rewards = rule_reward + self.reward_alpha * soft_reward
        
        return rewards
    
    def compute_advantage(self, rewards):
        """
        GRPO 核心：组内归一化计算 Advantage
        
        Args:
            rewards: [B, G]
            
        Returns:
            advantages: [B, G] 归一化后的优势值
        """
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards - mean) / std
        return advantages
    
    def compute_grpo_loss(self, current_log_probs, ref_log_probs, advantages):
        """
        计算 GRPO Loss
        
        Args:
            current_log_probs: [B*G] 当前策略的 log probability
            ref_log_probs: [B*G] 参考策略的 log probability
            advantages: [B*G] 归一化后的优势值
            
        Returns:
            loss: scalar
        """
        # Policy ratio
        ratio = torch.exp(current_log_probs - ref_log_probs)
        
        # Clipped ratio
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        
        # Policy loss (取 min 是为了保守更新)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # KL penalty
        kl_penalty = (current_log_probs - ref_log_probs).mean()
        
        loss = policy_loss + self.beta * kl_penalty
        
        return loss, policy_loss.item(), kl_penalty.item()
    
    def _train_epoch_grpo(self, epoch_idx, verbose=True):
        """GRPO 训练一个 epoch"""
        self.model_rec.train()
        self.ref_model.eval()
        self.model_id.eval()
        
        total_num = 0
        total_loss = defaultdict(float)
        
        iter_data = tqdm(
            self.train_data,
            total=len(self.train_data),
            ncols=100,
            desc=set_color(f"GRPO Train {epoch_idx}", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )
        
        for batch_idx, batch in enumerate(iter_data):
            with self.accelerator.accumulate(self.model_rec):
                total_num += 1
                self.optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                
                B = input_ids.size(0)
                
                # 转换为 codes
                input_codes = self.all_item_code[input_ids].contiguous().clone().view(B, -1)
                attention_mask = (input_codes != -1).bool()
                
                # ============ Step 1: 采样生成 G 个候选 ============
                if dist.is_initialized():
                    model = self.model_rec.module
                else:
                    model = self.model_rec
                    
                with torch.no_grad():
                    gen_codes = model.sample_generate(
                        input_ids=input_codes,
                        attention_mask=attention_mask,
                        num_samples=self.G,
                        temperature=self.temperature,
                        top_k=self.top_k,
                        return_log_probs=False
                    )  # [B, G, code_length]
                
                # ============ Step 2: 计算奖励 ============
                rewards = self.compute_reward(gen_codes, targets)  # [B, G]
                
                # ============ Step 3: 计算 Advantage ============
                advantages = self.compute_advantage(rewards)  # [B, G]
                advantages_flat = advantages.view(-1)  # [B*G]
                
                # ============ Step 4: 计算 Log Probs ============
                # 展平 gen_codes: [B, G, code_len] -> [B*G, code_len]
                gen_codes_flat = gen_codes.view(B * self.G, -1)
                
                # 展开 input_codes 和 attention_mask
                input_codes_expanded = input_codes.repeat_interleave(self.G, dim=0)  # [B*G, seq_len]
                attention_mask_expanded = attention_mask.repeat_interleave(self.G, dim=0)
                
                # 当前模型的 log probs
                current_log_probs = model.compute_log_probs(
                    input_ids=input_codes_expanded,
                    attention_mask=attention_mask_expanded,
                    target_codes=gen_codes_flat
                )  # [B*G]
                
                # 参考模型的 log probs
                with torch.no_grad():
                    ref_log_probs = self.ref_model.compute_log_probs(
                        input_ids=input_codes_expanded,
                        attention_mask=attention_mask_expanded,
                        target_codes=gen_codes_flat
                    )  # [B*G]
                
                # ============ Step 5: 计算 GRPO Loss ============
                loss, policy_loss, kl_penalty = self.compute_grpo_loss(
                    current_log_probs, ref_log_probs, advantages_flat
                )
                
                # Backward
                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                
                # Logging
                loss_mean = self.accelerator.gather(loss).mean().item()
                reward_mean = rewards.mean().item()
                hit_rate = (rewards >= 1.0).float().mean().item()
                
                total_loss['loss'] += loss_mean
                total_loss['policy_loss'] += policy_loss
                total_loss['kl_penalty'] += kl_penalty
                total_loss['reward'] += reward_mean
                total_loss['hit_rate'] += hit_rate
                
                iter_data.set_postfix(loss=loss_mean, reward=reward_mean)
        
        for k in total_loss.keys():
            total_loss[k] = round(total_loss[k] / total_num, 4)
            
        self.accelerator.wait_for_everyone()
        return total_loss
    
    def train_grpo(self, verbose=True):
        """GRPO 主训练循环"""
        self.log(f"[GRPO] Starting GRPO training for {self.epochs} epochs")
        self.log(f"[GRPO] Config: G={self.G}, beta={self.beta}, temp={self.temperature}, "
                 f"reward_alpha={self.reward_alpha}, lr={self.lr}")
        
        if self.code_to_item is None:
            raise ValueError("Must call set_item_codes() before training")
        
        stop = False
        cur_eval_step = 0
        
        for epoch_idx in range(self.epochs):
            self.accelerator.wait_for_everyone()
            
            # Train
            training_start_time = time()
            train_loss = self._train_epoch_grpo(epoch_idx, verbose=verbose)
            training_end_time = time()
            
            # Log
            self.log(f"[GRPO Epoch {epoch_idx}] time: {training_end_time - training_start_time:.2f}s, "
                     f"loss: {train_loss['loss']:.4f}, policy: {train_loss['policy_loss']:.4f}, "
                     f"kl: {train_loss['kl_penalty']:.4f}, reward: {train_loss['reward']:.4f}, "
                     f"hit_rate: {train_loss['hit_rate']:.4f}")
            self.log(f"[GRPO Epoch {epoch_idx}] lr: {self.lr_scheduler.get_last_lr()}")
            
            # Eval
            if (epoch_idx + 1) % self.eval_step == 0:
                metrics = self._test_epoch(test_data=self.valid_data, code=self.all_item_code, verbose=verbose)
                
                if metrics[self.valid_metric] > self.best_score:
                    self.best_score = metrics[self.valid_metric]
                    self.best_result = metrics
                    cur_eval_step = 0
                    self.best_ckpt = self.safe_save_grpo(epoch_idx)
                else:
                    cur_eval_step += 1
                    
                if cur_eval_step >= self.early_stop:
                    stop = True
                    
                self.log(f"[GRPO Epoch {epoch_idx}] Val Results: {metrics}")
                
            self.accelerator.wait_for_everyone()
            
            if stop:
                break
                
        self.log(f"[GRPO] Training finished. Best {self.valid_metric}: {self.best_score:.4f}")
        return self.best_score
    
    def safe_save_grpo(self, epoch):
        """保存 GRPO checkpoint"""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            os.makedirs(self.save_path, exist_ok=True)  # 确保目录存在
            unwrap_model_rec = self.accelerator.unwrap_model(self.model_rec)
            save_path = f'{self.save_path}/grpo_{epoch}.pt'
            self.accelerator.save(unwrap_model_rec.state_dict(), save_path)
            self.log(f'[GRPO Epoch {epoch}] Saved model to {save_path}')
        return f'{self.save_path}/grpo_{epoch}.pt'
    
    def log(self, message, level='info'):
        """日志输出"""
        return log(message, self.accelerator, self.logger, level=level)
    
    @torch.no_grad()
    def test(self, verbose=True, model_file=None):
        """
        测试 GRPO 模型
        注意：GRPO checkpoint 只保存了 model_rec，rqvae 使用原始的
        """
        if self.test_data is None:
            return None
            
        # 加载最佳 checkpoint
        ckpt_file = model_file or self.best_ckpt
        if ckpt_file:
            if dist.is_initialized():
                safe_load(self.model_rec.module, ckpt_file, verbose=verbose)
            else:
                safe_load(self.model_rec, ckpt_file, verbose=verbose)
            self.log(f"[GRPO Test] Loading model from {ckpt_file}")
        
        # 使用父类的 _test_epoch，但不加载 checkpoint（已经加载过了）
        metrics = self._test_epoch(
            test_data=self.test_data, 
            code=self.all_item_code, 
            load_best_model=False,  # 不再加载，上面已经加载了
            verbose=verbose
        )
        return metrics
