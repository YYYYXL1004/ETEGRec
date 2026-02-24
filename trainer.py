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
from utils import ensure_dir, set_color, get_local_time
from accelerate import PartialState
from model import Model
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import get_scheduler
from metrics import *
from utils import *
from collections import defaultdict
from logging import getLogger
from torch.utils.tensorboard import SummaryWriter
import zlib
init(autoreset=True)


class Trainer(object):
    def __init__(self, config, model_rec: Model, model_id, accelerator, train_data=None,
                 valid_data=None, test_data=None, eos_token_id=None):
        self.config = config
        self.model_rec = model_rec
        self.model_id = model_id
        self.logger = getLogger()

        self.eos_token_id = eos_token_id
        self.pad_token_id = 0
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
        self.accelerator = accelerator

        # 是否启用 cross-modal 双路模式
        self.cross_modal = config.get('cross_modal', False)

        assert self.cycle % self.eval_step == 0, 'cycle should be divisible by eval_step'

        self.state = PartialState()
        self.world_size = self.state.num_processes
        self.device = self.state.device
        self.all_item_code = None
        # cross-modal 双路 code
        self.all_item_text_code = None
        self.all_item_image_code = None
        self.model_rec.device = self.device

        self.all_metrics = config["metrics"].split(",")
        self.valid_metric = config["valid_metric"]
        self.max_topk = 0
        self.all_metric_name = []
        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            self.max_topk = max(self.max_topk, int(top_k))
            if m_name.lower() not in self.all_metric_name:
                self.all_metric_name.append(m_name.lower())

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.max_steps = self.get_train_steps()
        self.warmup_steps = config["warmup_steps"]
        self.rec_optimizer = self._build_optimizer(model_rec, self.lr_rec, self.weight_decay)
        self.id_optimizer = self._build_optimizer(model_id, self.lr_id, self.weight_decay)

        if self.lr_scheduler_type == "linear":
            self.rec_lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.rec_optimizer, num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps)
            self.id_lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.id_optimizer, num_warmup_steps=self.warmup_steps // self.cycle,
                num_training_steps=self.max_steps // self.cycle)
        elif self.lr_scheduler_type == "constant":
            self.rec_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.rec_optimizer, num_warmup_steps=self.warmup_steps)
            self.id_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=self.id_optimizer, num_warmup_steps=self.warmup_steps // self.cycle)
        elif self.lr_scheduler_type == "cosine":
            self.rec_lr_scheduler = get_scheduler(
                name="cosine", optimizer=self.rec_optimizer,
                num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps)
            self.id_lr_scheduler = get_scheduler(
                name="cosine", optimizer=self.id_optimizer,
                num_warmup_steps=self.warmup_steps // self.cycle,
                num_training_steps=self.max_steps // self.cycle)

        self.best_score = 0
        self.best_ckpt = None

        self.model_rec, self.rec_optimizer, self.rec_lr_scheduler, \
        self.model_id, self.id_optimizer, self.id_lr_scheduler, \
        self.train_data, self.valid_data, self.test_data = \
        self.accelerator.prepare(self.model_rec, self.rec_optimizer, self.rec_lr_scheduler,
                                 self.model_id, self.id_optimizer, self.id_lr_scheduler,
                                 self.train_data, self.valid_data, self.test_data)

        # TensorBoard (main process only)
        self.writer = None
        if self.accelerator.is_main_process:
            tb_dir = os.path.join(self.save_path, 'tb_logs')
            ensure_dir(tb_dir)
            self.writer = SummaryWriter(log_dir=tb_dir)
        self.global_step_rec = 0
        self.global_step_id = 0

    def _build_optimizer(self, model, lr, weight_decay):
        params = model.parameters()
        learner = self.learner

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            self.logger.warning("Received unrecognized optimizer, set default Adam optimizer")
            optimizer = optim.Adam(params, lr=lr)
        return optimizer

    @staticmethod
    def _gather_tensor(t, local_rank):
        all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(all_tensors, t)
        all_tensors[local_rank] = t
        return all_tensors

    @staticmethod
    def gather_tensors(t, local_rank=None):
        if local_rank is None:
            local_rank = dist.get_rank()
        return torch.cat(Trainer._gather_tensor(t, local_rank))

    @staticmethod
    def compute_discrete_contrastive_loss_kl(x_logits, y_logits):
        code_num = x_logits.size(-1)
        x_logits = F.log_softmax(x_logits.view(-1, code_num), dim=-1)
        y_logits = F.log_softmax(y_logits.view(-1, code_num), dim=-1)
        loss = F.kl_div(x_logits, y_logits, reduction='batchmean', log_target=True)
        return loss

    @staticmethod
    def compute_contrastive_loss(query_embeds, semantic_embeds, temperature=0.07, sim="cos", gathered=True):
        if gathered:
            gathered_query_embeds = Trainer.gather_tensors(query_embeds)
            gathered_semantic_embeds = Trainer.gather_tensors(semantic_embeds)
        else:
            gathered_query_embeds = query_embeds
            gathered_semantic_embeds = semantic_embeds

        if sim == "cos":
            gathered_query_embeds = F.normalize(gathered_query_embeds, dim=-1)
            gathered_semantic_embeds = F.normalize(gathered_semantic_embeds, dim=-1)

        effective_bsz = gathered_query_embeds.size(0)
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(gathered_query_embeds, gathered_semantic_embeds.transpose(0, 1)) / temperature
        co_loss = F.cross_entropy(similarities, labels)
        return co_loss

    @staticmethod
    def get_unique_index(inputs):
        unique_value = torch.unique(inputs).to(inputs.device)
        unique_index = torch.zeros_like(unique_value, device=inputs.device)
        for i, value in enumerate(unique_value):
            unique_index[i] = torch.argwhere(inputs == value).flatten()[0]
        unique_index = unique_index.to(inputs.device)
        return unique_index

    def get_train_steps(self, epochs=None):
        len_dataloader = len(self.train_data)
        num_update_steps_per_epoch = len_dataloader // self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if epochs is None:
            epochs = self.epochs
        max_steps = math.ceil(epochs * num_update_steps_per_epoch)
        return max_steps

    # =====================================================================
    # _train_epoch_rec: 训练 T5 (model_rec)，冻结 RQVAE (model_id)
    # cross_modal 模式下: 分别对 text/image 两路做 forward，loss 求和
    # =====================================================================
    def _train_epoch_rec(self, epoch_idx, loss_w, verbose=True):
        self.model_rec.train()
        self.model_id.eval()

        total_num = 0
        total_loss = defaultdict(int)
        iter_data = tqdm(
            self.train_data, total=len(self.train_data), ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )

        for batch_idx, batch in enumerate(iter_data):
            with self.accelerator.accumulate(self.model_rec):
                total_num += 1
                self.rec_optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                B = input_ids.size(0)
                target_flatten = targets.flatten()

                if self.cross_modal:
                    loss, loss_dict = self._rec_step_cross(
                        input_ids, attention_mask, targets, target_flatten, B, loss_w)
                else:
                    loss, loss_dict = self._rec_step_single(
                        input_ids, attention_mask, targets, target_flatten, B, loss_w)

                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1)
                self.rec_optimizer.step()
                self.rec_lr_scheduler.step()

                # Gather and log
                loss_mean = self.accelerator.gather(loss).mean().item()
                gathered_dict = {k: self.accelerator.gather(v).mean().item() for k, v in loss_dict.items()}
                gathered_dict['loss'] = loss_mean

                for k, v in gathered_dict.items():
                    total_loss[k] += v
                iter_data.set_postfix(loss=loss_mean)

                if self.writer is not None:
                    self.global_step_rec += 1
                    for k, v in gathered_dict.items():
                        self.writer.add_scalar(f'rec_step/{k}', v, self.global_step_rec)
                    self.writer.add_scalar('rec_step/lr', self.rec_lr_scheduler.get_last_lr()[0], self.global_step_rec)

        for k in total_loss.keys():
            total_loss[k] = round(total_loss[k] / total_num, 4)

        self.accelerator.wait_for_everyone()
        return total_loss

    def _rec_step_single(self, input_ids, attention_mask, targets, target_flatten, B, loss_w):
        """原始单路 rec step"""
        input_ids = self.all_item_code[input_ids].contiguous().clone().view(B, -1)
        labels = self.all_item_code[targets].contiguous().clone().view(B, -1)
        attention_mask = (input_ids != -1).bool()

        if dist.is_initialized():
            target_semantic_embs = self.model_rec.module.semantic_embedding(target_flatten)
        else:
            target_semantic_embs = self.model_rec.semantic_embedding(target_flatten)
        target_recon_embs, _, _, _, target_code_logits = self.model_id(target_semantic_embs)

        _, unq_index = np.unique(target_flatten.cpu().numpy(), return_index=True)
        unq_index = torch.tensor(unq_index).to(self.device)

        outputs = self.model_rec(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        seq_project_latents = outputs.seq_project_latents
        dec_latents = outputs.dec_latents

        if dist.is_initialized():
            _, _, _, _, seq_code_logits = self.model_id.module.rq(seq_project_latents)
        else:
            _, _, _, _, seq_code_logits = self.model_id.rq(seq_project_latents)

        code_loss = F.cross_entropy(logits.view(-1, self.code_num), labels.detach().reshape(-1))
        kl_loss = self.compute_discrete_contrastive_loss_kl(
            seq_code_logits[unq_index], target_code_logits[unq_index]) + \
            self.compute_discrete_contrastive_loss_kl(
            target_code_logits[unq_index], seq_code_logits[unq_index])
        dec_cl_loss = self.compute_contrastive_loss(
            target_recon_embs[unq_index], dec_latents[unq_index], sim=self.sim, gathered=False) + \
            self.compute_contrastive_loss(
            dec_latents[unq_index], target_recon_embs[unq_index], sim=self.sim, gathered=False)

        losses = dict(code_loss=code_loss, kl_loss=kl_loss, dec_cl_loss=dec_cl_loss)
        loss = sum([v * loss_w[k] for k, v in losses.items()])
        return loss, losses

    def _rec_step_cross(self, input_ids, attention_mask, targets, target_flatten, B, loss_w):
        """
        双路 rec step: text 路 + image 路分别 forward，loss 求和。
        共享 T5 backbone，但 token_embeddings / adapter 各自独立。
        """
        # ---- Text 路 ----
        text_input_ids = self.all_item_text_code[input_ids].contiguous().clone().view(B, -1)
        text_labels = self.all_item_text_code[targets].contiguous().clone().view(B, -1)
        text_attn_mask = (text_input_ids != -1).bool()

        if dist.is_initialized():
            text_semantic_embs = self.model_rec.module.text_semantic_embedding(target_flatten)
        else:
            text_semantic_embs = self.model_rec.text_semantic_embedding(target_flatten)
        text_recon_embs, _, _, _, text_code_logits = self.model_id.text_forward(text_semantic_embs) \
            if not dist.is_initialized() else self.model_id.module.text_forward(text_semantic_embs)

        _, unq_index = np.unique(target_flatten.cpu().numpy(), return_index=True)
        unq_index = torch.tensor(unq_index).to(self.device)

        text_outputs = self.model_rec(input_ids=text_input_ids, attention_mask=text_attn_mask,
                                      labels=text_labels, route='text')
        if dist.is_initialized():
            _, _, _, _, text_seq_logits = self.model_id.module.text_rq(text_outputs.seq_project_latents)
        else:
            _, _, _, _, text_seq_logits = self.model_id.text_rq(text_outputs.seq_project_latents)

        text_code_loss = F.cross_entropy(text_outputs.logits.view(-1, self.code_num),
                                         text_labels.detach().reshape(-1))
        text_kl_loss = self.compute_discrete_contrastive_loss_kl(
            text_seq_logits[unq_index], text_code_logits[unq_index]) + \
            self.compute_discrete_contrastive_loss_kl(
            text_code_logits[unq_index], text_seq_logits[unq_index])
        text_dec_cl_loss = self.compute_contrastive_loss(
            text_recon_embs[unq_index], text_outputs.dec_latents[unq_index], sim=self.sim, gathered=False) + \
            self.compute_contrastive_loss(
            text_outputs.dec_latents[unq_index], text_recon_embs[unq_index], sim=self.sim, gathered=False)

        # ---- Image 路 ----
        img_input_ids = self.all_item_image_code[input_ids].contiguous().clone().view(B, -1)
        img_labels = self.all_item_image_code[targets].contiguous().clone().view(B, -1)
        img_attn_mask = (img_input_ids != -1).bool()

        if dist.is_initialized():
            img_semantic_embs = self.model_rec.module.image_semantic_embedding(target_flatten)
        else:
            img_semantic_embs = self.model_rec.image_semantic_embedding(target_flatten)
        img_recon_embs, _, _, _, img_code_logits = self.model_id.image_forward(img_semantic_embs) \
            if not dist.is_initialized() else self.model_id.module.image_forward(img_semantic_embs)

        img_outputs = self.model_rec(input_ids=img_input_ids, attention_mask=img_attn_mask,
                                     labels=img_labels, route='image')
        if dist.is_initialized():
            _, _, _, _, img_seq_logits = self.model_id.module.image_rq(img_outputs.seq_project_latents)
        else:
            _, _, _, _, img_seq_logits = self.model_id.image_rq(img_outputs.seq_project_latents)

        img_code_loss = F.cross_entropy(img_outputs.logits.view(-1, self.code_num),
                                        img_labels.detach().reshape(-1))
        img_kl_loss = self.compute_discrete_contrastive_loss_kl(
            img_seq_logits[unq_index], img_code_logits[unq_index]) + \
            self.compute_discrete_contrastive_loss_kl(
            img_code_logits[unq_index], img_seq_logits[unq_index])
        img_dec_cl_loss = self.compute_contrastive_loss(
            img_recon_embs[unq_index], img_outputs.dec_latents[unq_index], sim=self.sim, gathered=False) + \
            self.compute_contrastive_loss(
            img_outputs.dec_latents[unq_index], img_recon_embs[unq_index], sim=self.sim, gathered=False)

        # ---- 合并 loss ----
        code_loss = text_code_loss + img_code_loss
        kl_loss = text_kl_loss + img_kl_loss
        dec_cl_loss = text_dec_cl_loss + img_dec_cl_loss

        losses = dict(code_loss=code_loss, kl_loss=kl_loss, dec_cl_loss=dec_cl_loss)
        loss = sum([v * loss_w[k] for k, v in losses.items()])

        # 额外记录分路 loss 用于 TensorBoard
        losses['text_code_loss'] = text_code_loss
        losses['img_code_loss'] = img_code_loss
        losses['text_kl_loss'] = text_kl_loss
        losses['img_kl_loss'] = img_kl_loss

        return loss, losses

    # =====================================================================
    # _train_epoch_id: 训练 RQVAE (model_id)，冻结 T5 (model_rec)
    # cross_modal 模式下: 分别对 text/image 两路做 forward，loss 求和
    # =====================================================================
    def _train_epoch_id(self, epoch_idx, loss_w, verbose=True):
        self.model_id.train()

        total_num = 0
        total_loss = defaultdict(int)
        iter_data = tqdm(
            self.train_data, total=len(self.train_data), ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )

        for batch_idx, batch in enumerate(iter_data):
            with self.accelerator.accumulate(self.model_id):
                total_num += 1
                self.id_optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                targets = batch["targets"].to(self.device)
                B = input_ids.size(0)
                target_flatten = targets.flatten()

                if self.cross_modal:
                    loss, loss_dict = self._id_step_cross(
                        input_ids, attention_mask, targets, target_flatten, B, loss_w)
                else:
                    loss, loss_dict = self._id_step_single(
                        input_ids, attention_mask, targets, target_flatten, B, loss_w)

                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(self.model_id.parameters(), 1)
                self.id_optimizer.step()
                self.id_lr_scheduler.step()

                loss_mean = self.accelerator.gather(loss).mean().item()
                gathered_dict = {k: self.accelerator.gather(v).mean().item() for k, v in loss_dict.items()}
                gathered_dict['loss'] = loss_mean

                for k, v in gathered_dict.items():
                    total_loss[k] += v
                iter_data.set_postfix(loss=loss_mean)

                if self.writer is not None:
                    self.global_step_id += 1
                    for k, v in gathered_dict.items():
                        self.writer.add_scalar(f'id_step/{k}', v, self.global_step_id)
                    self.writer.add_scalar('id_step/lr', self.id_lr_scheduler.get_last_lr()[0], self.global_step_id)

        for k in total_loss.keys():
            total_loss[k] = round(total_loss[k] / total_num, 4)

        self.accelerator.wait_for_everyone()
        return total_loss

    def _id_step_single(self, input_ids, attention_mask, targets, target_flatten, B, loss_w):
        """原始单路 id step"""
        input_ids = self.all_item_code[input_ids].contiguous().clone().view(B, -1)
        labels = self.all_item_code[targets].contiguous().clone().view(B, -1)
        attention_mask = (input_ids != -1).bool()

        if dist.is_initialized():
            target_semantic_embs = self.model_rec.module.semantic_embedding(target_flatten)
        else:
            target_semantic_embs = self.model_rec.semantic_embedding(target_flatten)
        target_recon_embs, _, _, _, target_code_logits = self.model_id(target_semantic_embs)

        unq_input, unq_index = np.unique(target_flatten.cpu().numpy(), return_index=True)
        unq_input = torch.tensor(unq_input).to(self.device)
        unq_index = torch.tensor(unq_index).to(self.device)
        if dist.is_initialized():
            unq_semantic_embs = self.model_rec.module.semantic_embedding(unq_input)
        else:
            unq_semantic_embs = self.model_rec.semantic_embedding(unq_input)
        unq_recon_embs, commit_loss, _, _, _ = self.model_id(unq_semantic_embs)

        outputs = self.model_rec(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        seq_project_latents = outputs.seq_project_latents
        dec_latents = outputs.dec_latents

        if dist.is_initialized():
            _, _, _, _, seq_code_logits = self.model_id.module.rq(seq_project_latents)
        else:
            _, _, _, _, seq_code_logits = self.model_id.rq(seq_project_latents)

        code_loss = F.cross_entropy(logits.view(-1, self.code_num), labels.detach().reshape(-1))

        if self.loss_type == 'mse':
            recon_loss = F.mse_loss(unq_recon_embs, unq_semantic_embs, reduction='mean')
        elif self.loss_type == 'l1':
            recon_loss = F.l1_loss(unq_recon_embs, unq_semantic_embs, reduction='mean')
        elif self.loss_type == 'infonce':
            recon_loss = self.compute_contrastive_loss(
                unq_recon_embs, unq_semantic_embs, temperature=self.tau, gathered=False)
        else:
            raise ValueError('incompatible loss type')

        vq_loss = recon_loss + self.alpha * commit_loss

        kl_loss = self.compute_discrete_contrastive_loss_kl(
            seq_code_logits[unq_index], target_code_logits[unq_index]) + \
            self.compute_discrete_contrastive_loss_kl(
            target_code_logits[unq_index], seq_code_logits[unq_index])
        dec_cl_loss = self.compute_contrastive_loss(
            target_recon_embs[unq_index], dec_latents[unq_index], sim=self.sim, gathered=False) + \
            self.compute_contrastive_loss(
            dec_latents[unq_index], target_recon_embs[unq_index], sim=self.sim, gathered=False)

        losses = dict(vq_loss=vq_loss, code_loss=code_loss, kl_loss=kl_loss, dec_cl_loss=dec_cl_loss)
        loss = sum([v * loss_w[k] for k, v in losses.items()])
        return loss, losses

    def _id_step_cross(self, input_ids, attention_mask, targets, target_flatten, B, loss_w):
        """
        双路 id step: text 路 + image 路分别 forward CrossRQVAE，loss 求和。
        """
        # ---- 公共: 获取 unique items ----
        unq_input, unq_index = np.unique(target_flatten.cpu().numpy(), return_index=True)
        unq_input = torch.tensor(unq_input).to(self.device)
        unq_index = torch.tensor(unq_index).to(self.device)

        # ---- Text 路 ----
        text_input_ids = self.all_item_text_code[input_ids].contiguous().clone().view(B, -1)
        text_labels = self.all_item_text_code[targets].contiguous().clone().view(B, -1)
        text_attn_mask = (text_input_ids != -1).bool()

        if dist.is_initialized():
            text_semantic_embs = self.model_rec.module.text_semantic_embedding(target_flatten)
            text_unq_semantic = self.model_rec.module.text_semantic_embedding(unq_input)
        else:
            text_semantic_embs = self.model_rec.text_semantic_embedding(target_flatten)
            text_unq_semantic = self.model_rec.text_semantic_embedding(unq_input)

        text_recon_embs, _, _, _, text_code_logits = self.model_id.text_forward(text_semantic_embs) \
            if not dist.is_initialized() else self.model_id.module.text_forward(text_semantic_embs)
        text_unq_recon, text_commit_loss, _, _, _ = self.model_id.text_forward(text_unq_semantic) \
            if not dist.is_initialized() else self.model_id.module.text_forward(text_unq_semantic)

        text_outputs = self.model_rec(input_ids=text_input_ids, attention_mask=text_attn_mask,
                                      labels=text_labels, route='text')
        if dist.is_initialized():
            _, _, _, _, text_seq_logits = self.model_id.module.text_rq(text_outputs.seq_project_latents)
        else:
            _, _, _, _, text_seq_logits = self.model_id.text_rq(text_outputs.seq_project_latents)

        text_code_loss = F.cross_entropy(text_outputs.logits.view(-1, self.code_num),
                                         text_labels.detach().reshape(-1))
        if self.loss_type == 'mse':
            text_recon_loss = F.mse_loss(text_unq_recon, text_unq_semantic, reduction='mean')
        elif self.loss_type == 'l1':
            text_recon_loss = F.l1_loss(text_unq_recon, text_unq_semantic, reduction='mean')
        elif self.loss_type == 'infonce':
            text_recon_loss = self.compute_contrastive_loss(
                text_unq_recon, text_unq_semantic, temperature=self.tau, gathered=False)
        else:
            raise ValueError('incompatible loss type')
        text_vq_loss = text_recon_loss + self.alpha * text_commit_loss

        text_kl_loss = self.compute_discrete_contrastive_loss_kl(
            text_seq_logits[unq_index], text_code_logits[unq_index]) + \
            self.compute_discrete_contrastive_loss_kl(
            text_code_logits[unq_index], text_seq_logits[unq_index])
        text_dec_cl_loss = self.compute_contrastive_loss(
            text_recon_embs[unq_index], text_outputs.dec_latents[unq_index], sim=self.sim, gathered=False) + \
            self.compute_contrastive_loss(
            text_outputs.dec_latents[unq_index], text_recon_embs[unq_index], sim=self.sim, gathered=False)

        # ---- Image 路 ----
        img_input_ids = self.all_item_image_code[input_ids].contiguous().clone().view(B, -1)
        img_labels = self.all_item_image_code[targets].contiguous().clone().view(B, -1)
        img_attn_mask = (img_input_ids != -1).bool()

        if dist.is_initialized():
            img_semantic_embs = self.model_rec.module.image_semantic_embedding(target_flatten)
            img_unq_semantic = self.model_rec.module.image_semantic_embedding(unq_input)
        else:
            img_semantic_embs = self.model_rec.image_semantic_embedding(target_flatten)
            img_unq_semantic = self.model_rec.image_semantic_embedding(unq_input)

        img_recon_embs, _, _, _, img_code_logits = self.model_id.image_forward(img_semantic_embs) \
            if not dist.is_initialized() else self.model_id.module.image_forward(img_semantic_embs)
        img_unq_recon, img_commit_loss, _, _, _ = self.model_id.image_forward(img_unq_semantic) \
            if not dist.is_initialized() else self.model_id.module.image_forward(img_unq_semantic)

        img_outputs = self.model_rec(input_ids=img_input_ids, attention_mask=img_attn_mask,
                                     labels=img_labels, route='image')
        if dist.is_initialized():
            _, _, _, _, img_seq_logits = self.model_id.module.image_rq(img_outputs.seq_project_latents)
        else:
            _, _, _, _, img_seq_logits = self.model_id.image_rq(img_outputs.seq_project_latents)

        img_code_loss = F.cross_entropy(img_outputs.logits.view(-1, self.code_num),
                                        img_labels.detach().reshape(-1))
        if self.loss_type == 'mse':
            img_recon_loss = F.mse_loss(img_unq_recon, img_unq_semantic, reduction='mean')
        elif self.loss_type == 'l1':
            img_recon_loss = F.l1_loss(img_unq_recon, img_unq_semantic, reduction='mean')
        elif self.loss_type == 'infonce':
            img_recon_loss = self.compute_contrastive_loss(
                img_unq_recon, img_unq_semantic, temperature=self.tau, gathered=False)
        else:
            raise ValueError('incompatible loss type')
        img_vq_loss = img_recon_loss + self.alpha * img_commit_loss

        img_kl_loss = self.compute_discrete_contrastive_loss_kl(
            img_seq_logits[unq_index], img_code_logits[unq_index]) + \
            self.compute_discrete_contrastive_loss_kl(
            img_code_logits[unq_index], img_seq_logits[unq_index])
        img_dec_cl_loss = self.compute_contrastive_loss(
            img_recon_embs[unq_index], img_outputs.dec_latents[unq_index], sim=self.sim, gathered=False) + \
            self.compute_contrastive_loss(
            img_outputs.dec_latents[unq_index], img_recon_embs[unq_index], sim=self.sim, gathered=False)

        # ---- 合并 loss ----
        vq_loss = text_vq_loss + img_vq_loss
        code_loss = text_code_loss + img_code_loss
        kl_loss = text_kl_loss + img_kl_loss
        dec_cl_loss = text_dec_cl_loss + img_dec_cl_loss

        losses = dict(vq_loss=vq_loss, code_loss=code_loss, kl_loss=kl_loss, dec_cl_loss=dec_cl_loss)
        loss = sum([v * loss_w[k] for k, v in losses.items()])

        # 额外记录分路 loss
        losses['text_vq_loss'] = text_vq_loss
        losses['img_vq_loss'] = img_vq_loss
        losses['text_code_loss'] = text_code_loss
        losses['img_code_loss'] = img_code_loss

        return loss, losses

    def safe_save(self, epoch, code, text_code=None, image_code=None):
        """保存模型和 code。cross_modal 模式下额外保存双路 code。"""
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrap_model_rec = self.accelerator.unwrap_model(self.model_rec)
            unwrap_model_id = self.accelerator.unwrap_model(self.model_id)
            self.accelerator.save(unwrap_model_rec.state_dict(), f'{self.save_path}/{epoch}.pt')
            self.accelerator.save(unwrap_model_id.state_dict(), f'{self.save_path}/{epoch}.pt.rqvae')

            if self.cross_modal and text_code is not None and image_code is not None:
                json.dump(text_code.cpu().tolist(), open(f'{self.save_path}/{epoch}.text_code.json', 'w'))
                json.dump(image_code.cpu().tolist(), open(f'{self.save_path}/{epoch}.image_code.json', 'w'))
                # 兼容: 也保存一份 text_code 作为默认 code (用于单路 test)
                json.dump(text_code.cpu().tolist(), open(f'{self.save_path}/{epoch}.code.json', 'w'))
            else:
                json.dump(code.cpu().tolist(), open(f'{self.save_path}/{epoch}.code.json', 'w'))

            self.log(f'[Epoch {epoch}] Save model {self.save_path}/{epoch}.pt')

        last_checkpoint = f'{self.save_path}/{epoch}.pt'
        return last_checkpoint

    def evaluate(self, outputs, labels):
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

        metrics = {
            "recall@1": np.sum(recall_at_1),
            "recall@5": np.sum(recall_at_5),
            "recall@10": np.sum(recall_at_10),
            "ndcg@1": np.sum(ndcg_at_1),
            "ndcg@5": np.sum(ndcg_at_5),
            "ndcg@10": np.sum(ndcg_at_10),
        }
        return metrics

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss_dict):
        train_loss_output = "[Epoch %d] [time: %.2fs, " % (epoch_idx, e_time - s_time)
        if isinstance(loss_dict, dict):
            train_loss_output += "train loss" + str(list(loss_dict.items()))
        else:
            train_loss_output += "train loss" + ": %.4f" % loss_dict
        return train_loss_output + "]"

    def train(self, verbose=True):
        stop = False
        cur_eval_step = 0
        loss_w = defaultdict(int)

        if self.cross_modal:
            text_code, image_code = self.get_code(epoch_idx=-1, verbose=verbose)
            self.all_item_text_code = torch.tensor(text_code).to(self.device)
            self.all_item_image_code = torch.tensor(image_code).to(self.device)
        else:
            all_item_code = self.get_code(epoch_idx=-1, verbose=verbose)
            self.all_item_code = torch.tensor(all_item_code).to(self.device)

        for epoch_idx in range(self.epochs):

            if epoch_idx % self.cycle == 0:
                loss_w['vq_loss'] = self.config['id_vq_loss']
                loss_w['code_loss'] = self.config['id_code_loss'] if epoch_idx >= self.warm_epoch else 0
                loss_w['kl_loss'] = self.config['id_kl_loss'] if epoch_idx >= self.warm_epoch else 0
                loss_w['dec_cl_loss'] = self.config['id_dec_cl_loss'] if epoch_idx >= self.warm_epoch else 0

                for name, param in self.model_rec.named_parameters():
                    param.requires_grad = False
                for param in self.model_id.parameters():
                    param.requires_grad = True
            else:
                loss_w['vq_loss'] = self.config['rec_vq_loss']
                loss_w['code_loss'] = self.config['rec_code_loss']
                loss_w['kl_loss'] = self.config['rec_kl_loss'] if epoch_idx >= self.warm_epoch else 0
                loss_w['dec_cl_loss'] = self.config['rec_dec_cl_loss'] if epoch_idx >= self.warm_epoch else 0

                for name, param in self.model_rec.named_parameters():
                    # 冻结 semantic_embedding (cross_modal 下冻结两个)
                    sem_prefix = 'module.text_semantic_embedding' if dist.is_initialized() else 'text_semantic_embedding'
                    img_prefix = 'module.image_semantic_embedding' if dist.is_initialized() else 'image_semantic_embedding'
                    single_prefix = 'module.semantic_embedding' if dist.is_initialized() else 'semantic_embedding'
                    if not (name.startswith(sem_prefix) or name.startswith(img_prefix) or name.startswith(single_prefix)):
                        param.requires_grad = True
                for param in self.model_id.parameters():
                    param.requires_grad = False

            self.accelerator.wait_for_everyone()
            training_start_time = time()

            if epoch_idx % self.cycle == 0:
                train_loss = self._train_epoch_id(epoch_idx, loss_w=loss_w, verbose=verbose)
                if self.cross_modal:
                    text_code, image_code = self.get_code(epoch_idx=epoch_idx, verbose=verbose)
                    self.all_item_text_code = torch.tensor(text_code).to(self.device)
                    self.all_item_image_code = torch.tensor(image_code).to(self.device)
                else:
                    all_item_code = self.get_code(epoch_idx=epoch_idx, verbose=verbose)
                    self.all_item_code = torch.tensor(all_item_code).to(self.device)
            else:
                train_loss = self._train_epoch_rec(epoch_idx, loss_w=loss_w, verbose=verbose)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)
            self.log(train_loss_output)
            self.log(f'[Epoch {epoch_idx}] REC lr: {self.rec_lr_scheduler.get_lr()} ID lr: {self.id_lr_scheduler.get_lr()}')

            if self.writer is not None:
                phase = 'id' if epoch_idx % self.cycle == 0 else 'rec'
                for k, v in train_loss.items():
                    self.writer.add_scalar(f'epoch/{phase}_{k}', v, epoch_idx)
                self.writer.add_scalar('epoch/rec_lr', self.rec_lr_scheduler.get_last_lr()[0], epoch_idx)
                self.writer.add_scalar('epoch/id_lr', self.id_lr_scheduler.get_last_lr()[0], epoch_idx)

            if (epoch_idx + 1) % self.eval_step == 0:
                if self.cross_modal:
                    # 双路分别评估 + ensemble 评估
                    text_metrics = self._test_epoch(test_data=self.valid_data,
                                                    code=self.all_item_text_code, verbose=verbose, route='text')
                    image_metrics = self._test_epoch(test_data=self.valid_data,
                                                     code=self.all_item_image_code, verbose=verbose, route='image')
                    ensemble_metrics = self._test_epoch_ensemble(
                        test_data=self.valid_data, verbose=verbose)
                    total_metrics = ensemble_metrics
                    self.log(f'[Epoch {epoch_idx}] Text Val: {text_metrics}')
                    self.log(f'[Epoch {epoch_idx}] Image Val: {image_metrics}')
                    self.log(f'[Epoch {epoch_idx}] Ensemble Val: {ensemble_metrics}')
                else:
                    total_metrics = self._test_epoch(test_data=self.valid_data,
                                                     code=self.all_item_code, verbose=verbose)

                if total_metrics[self.valid_metric] > self.best_score:
                    self.best_score = total_metrics[self.valid_metric]
                    self.best_result = total_metrics
                    cur_eval_step = 0
                    if self.cross_modal:
                        self.best_ckpt = self.safe_save(epoch_idx, None,
                                                        self.all_item_text_code, self.all_item_image_code)
                    else:
                        self.best_ckpt = self.safe_save(epoch_idx, self.all_item_code)
                else:
                    cur_eval_step += 1

                if cur_eval_step >= self.early_stop:
                    stop = True

                self.log(f'[Epoch {epoch_idx}] Val Results: {total_metrics}')

                if self.writer is not None:
                    for metric_name, metric_val in total_metrics.items():
                        self.writer.add_scalar(f'val/{metric_name}', metric_val, epoch_idx)
                    self.writer.add_scalar('val/best_score', self.best_score, epoch_idx)

            self.accelerator.wait_for_everyone()
            if stop:
                break

        if self.writer is not None:
            self.writer.close()
        return self.best_score

    def finetune(self, verbose=True):
        stop = False
        cur_eval_step = 0
        self.best_score = 0
        self.early_stop = 10
        self.eval_step = 1
        self.epochs = 100
        loss_w = defaultdict(int)

        model_rec = self.accelerator.unwrap_model(self.model_rec)
        self.rec_optimizer = self._build_optimizer(model_rec, 5e-4, self.weight_decay)
        train_steps = self.get_train_steps(epochs=100) * self.world_size
        self.rec_lr_scheduler = get_scheduler(name='cosine', optimizer=self.rec_optimizer,
                                              num_warmup_steps=0, num_training_steps=train_steps)
        self.rec_optimizer, self.rec_lr_scheduler = self.accelerator.prepare(
            self.rec_optimizer, self.rec_lr_scheduler)

        loss_w['code_loss'], loss_w['vq_loss'], loss_w['kl_loss'], loss_w['dec_cl_loss'] = 1, 0, 0, 0

        if self.best_ckpt:
            if dist.is_initialized():
                safe_load(self.model_rec.module, self.best_ckpt, verbose=verbose)
                safe_load(self.model_id.module, f'{self.best_ckpt}.rqvae', verbose=verbose)
            else:
                safe_load(self.model_rec, self.best_ckpt, verbose=verbose)
                safe_load(self.model_id, f'{self.best_ckpt}.rqvae', verbose=verbose)
        else:
            self.log('No best checkpoint found; skip loading and use current model', level='warning')

        if self.cross_modal:
            text_code, image_code = self.get_code(epoch_idx=0, verbose=False)
            self.all_item_text_code = torch.tensor(text_code).to(self.device)
            self.all_item_image_code = torch.tensor(image_code).to(self.device)
        else:
            all_item_code = self.get_code(epoch_idx=0, verbose=False)
            self.all_item_code = torch.tensor(all_item_code).to(self.device)

        for name, param in self.model_rec.named_parameters():
            sem_prefix = 'module.text_semantic_embedding' if dist.is_initialized() else 'text_semantic_embedding'
            img_prefix = 'module.image_semantic_embedding' if dist.is_initialized() else 'image_semantic_embedding'
            single_prefix = 'module.semantic_embedding' if dist.is_initialized() else 'semantic_embedding'
            if not (name.startswith(sem_prefix) or name.startswith(img_prefix) or name.startswith(single_prefix)):
                param.requires_grad = True

        for param in self.model_id.parameters():
            param.requires_grad = False

        for epoch_idx in range(self.epochs):
            self.accelerator.wait_for_everyone()
            training_start_time = time()
            train_loss = self._train_epoch_rec(epoch_idx, loss_w=loss_w, verbose=verbose)
            training_end_time = time()

            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)
            self.log(train_loss_output)
            self.log(f'[Epoch {epoch_idx}] Current REC lr: {self.rec_lr_scheduler.get_lr()}')

            if self.writer is not None:
                for k, v in train_loss.items():
                    self.writer.add_scalar(f'finetune_epoch/{k}', v, epoch_idx)
                self.writer.add_scalar('finetune_epoch/lr', self.rec_lr_scheduler.get_last_lr()[0], epoch_idx)

            if (epoch_idx + 1) % self.eval_step == 0:
                if self.cross_modal:
                    text_metrics = self._test_epoch(test_data=self.valid_data,
                                                    code=self.all_item_text_code, verbose=verbose, route='text')
                    image_metrics = self._test_epoch(test_data=self.valid_data,
                                                     code=self.all_item_image_code, verbose=verbose, route='image')
                    ensemble_metrics = self._test_epoch_ensemble(
                        test_data=self.valid_data, verbose=verbose)
                    total_metrics = ensemble_metrics
                    self.log(f'[FT Epoch {epoch_idx}] Text Val: {text_metrics}')
                    self.log(f'[FT Epoch {epoch_idx}] Image Val: {image_metrics}')
                    self.log(f'[FT Epoch {epoch_idx}] Ensemble Val: {ensemble_metrics}')
                else:
                    total_metrics = self._test_epoch(test_data=self.valid_data,
                                                     code=self.all_item_code, verbose=verbose)

                if total_metrics[self.valid_metric] > self.best_score:
                    self.best_score = total_metrics[self.valid_metric]
                    self.best_result = total_metrics
                    cur_eval_step = 0
                    if self.cross_modal:
                        self.best_ckpt = self.safe_save(epoch_idx, None,
                                                        self.all_item_text_code, self.all_item_image_code)
                    else:
                        self.best_ckpt = self.safe_save(epoch_idx, self.all_item_code)
                else:
                    cur_eval_step += 1

                if cur_eval_step >= self.early_stop:
                    stop = True

                self.log(f'[Epoch {epoch_idx}] Val Results: {total_metrics}')

                if self.writer is not None:
                    for metric_name, metric_val in total_metrics.items():
                        self.writer.add_scalar(f'finetune_val/{metric_name}', metric_val, epoch_idx)
                    self.writer.add_scalar('finetune_val/best_score', self.best_score, epoch_idx)

            self.accelerator.wait_for_everyone()
            if stop:
                break

        if self.writer is not None:
            self.writer.close()
        return self.best_score

    @torch.no_grad()
    def test(self, verbose=True, model_file=None, prefix_allowed_tokens_fn=None):
        test_results = None
        if self.test_data is not None:
            if self.cross_modal:
                # 双路: 分别测试 + ensemble
                text_metrics = self._test_epoch(load_best_model=True, model_file=model_file,
                                                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                                verbose=verbose, route='text')
                image_metrics = self._test_epoch(load_best_model=True, model_file=model_file,
                                                 prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                                 verbose=verbose, route='image')
                ensemble_metrics = self._test_epoch_ensemble(
                    load_best_model=True, model_file=model_file, verbose=verbose)
                self.log(f'Test Text: {text_metrics}')
                self.log(f'Test Image: {image_metrics}')
                self.log(f'Test Ensemble: {ensemble_metrics}')
                test_results = ensemble_metrics
            else:
                metrics = self._test_epoch(load_best_model=True, model_file=model_file,
                                           prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, verbose=verbose)
                test_results = metrics
        return test_results

    @torch.no_grad()
    def _test_epoch(self, code=None, test_data=None, load_best_model=False, model_file=None,
                    prefix_allowed_tokens_fn=None, verbose=True, route=None):
        """
        单路评估。cross_modal 模式下通过 route 指定使用哪路 code。
        """
        if test_data is None:
            test_data = self.test_data

        if load_best_model:
            ckpt_file = model_file or self.best_ckpt
            if ckpt_file:
                if dist.is_initialized():
                    safe_load(self.model_rec.module, ckpt_file, verbose=verbose)
                    safe_load(self.model_id.module, ckpt_file + '.rqvae', verbose=verbose)
                else:
                    safe_load(self.model_rec, ckpt_file, verbose=verbose)
                    safe_load(self.model_id, ckpt_file + '.rqvae', verbose=verbose)

                if self.cross_modal and route == 'text':
                    code_file = ckpt_file[:-3] + '.text_code.json'
                    if os.path.exists(code_file):
                        code = json.load(open(code_file))
                    else:
                        code = json.load(open(ckpt_file[:-3] + '.code.json'))
                elif self.cross_modal and route == 'image':
                    code = json.load(open(ckpt_file[:-3] + '.image_code.json'))
                else:
                    code = json.load(open(ckpt_file[:-3] + '.code.json'))

                self.log(f"Loading model from {ckpt_file}")
            else:
                if code is None:
                    if self.cross_modal:
                        if route == 'text':
                            code = self.all_item_text_code
                        elif route == 'image':
                            code = self.all_item_image_code
                        else:
                            code = self.all_item_text_code  # fallback
                    else:
                        code = self.all_item_code if self.all_item_code is not None else self.get_code(epoch_idx=-1, verbose=False)
                self.log("No checkpoint available; evaluating current in-memory model", level='warning')

        self.model_rec.eval()
        self.model_id.eval()

        iter_data = tqdm(
            test_data, total=len(test_data), ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )

        if isinstance(code, torch.Tensor):
            code = code.cpu().tolist()

        total = 0
        metrics = {m: 0 for m in self.all_metrics}

        code2item = defaultdict(list)
        for i, c in enumerate(code[1:]):
            code2item[str(c)].append(i + 1)

        item_code = torch.tensor(code).to(self.device)

        for batch_idx, data in enumerate(iter_data):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            labels = data["targets"].to(self.device)

            B = input_ids.size(0)
            input_ids = item_code[input_ids].contiguous().clone().view(B, -1)
            labels = item_code[labels].contiguous().clone().view(B, -1)
            attention_mask = (input_ids != -1).bool()

            if dist.is_initialized():
                preds = self.model_rec.module.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    n_return_sequences=10, route=route)
                all_preds, all_labels = self.accelerator.gather_for_metrics((preds, labels))
                _metrics = self.evaluate(all_preds, all_labels)
                total += len(all_labels)
            else:
                preds = self.model_rec.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    n_return_sequences=10, route=route)
                _metrics = self.evaluate(preds, labels)
                total += len(labels)

            for m in metrics.keys():
                metrics[m] += _metrics[m]

        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)

        return metrics

    @torch.no_grad()
    def _test_epoch_ensemble(self, test_data=None, load_best_model=False, model_file=None, verbose=True):
        """
        Ensemble 评估: 对 text 路和 image 路分别做 beam search，
        然后通过 score fusion 合并结果。参考 MACRec/ensemble.py 的逻辑。
        """
        if test_data is None:
            test_data = self.test_data

        if load_best_model:
            ckpt_file = model_file or self.best_ckpt
            if ckpt_file:
                if dist.is_initialized():
                    safe_load(self.model_rec.module, ckpt_file, verbose=verbose)
                    safe_load(self.model_id.module, ckpt_file + '.rqvae', verbose=verbose)
                else:
                    safe_load(self.model_rec, ckpt_file, verbose=verbose)
                    safe_load(self.model_id, ckpt_file + '.rqvae', verbose=verbose)

                text_code_file = ckpt_file[:-3] + '.text_code.json'
                image_code_file = ckpt_file[:-3] + '.image_code.json'
                if os.path.exists(text_code_file):
                    text_code_list = json.load(open(text_code_file))
                    image_code_list = json.load(open(image_code_file))
                    self.all_item_text_code = torch.tensor(text_code_list).to(self.device)
                    self.all_item_image_code = torch.tensor(image_code_list).to(self.device)

        self.model_rec.eval()
        self.model_id.eval()

        text_code = self.all_item_text_code
        image_code = self.all_item_image_code

        if isinstance(text_code, torch.Tensor):
            text_code_list = text_code.cpu().tolist()
        else:
            text_code_list = text_code
        if isinstance(image_code, torch.Tensor):
            image_code_list = image_code.cpu().tolist()
        else:
            image_code_list = image_code

        # 构建 code → item_id 反向索引
        text_code2items = defaultdict(list)
        for i, c in enumerate(text_code_list[1:]):
            text_code2items[tuple(c)].append(i + 1)
        image_code2items = defaultdict(list)
        for i, c in enumerate(image_code_list[1:]):
            image_code2items[tuple(c)].append(i + 1)

        text_item_code = torch.tensor(text_code_list).to(self.device)
        image_item_code = torch.tensor(image_code_list).to(self.device)

        iter_data = tqdm(
            test_data, total=len(test_data), ncols=100,
            desc=set_color(f"Ensemble   ", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )

        # DDP 同步: 确保所有 rank 在 ensemble 开始前对齐
        # (前面的 _test_epoch text/image 评估可能导致 rank 间进度不一致)
        if dist.is_initialized():
            dist.barrier()

        num_beams = self.config['num_beams']  # 20
        total = 0
        metrics = {m: 0 for m in self.all_metrics}

        for batch_idx, data in enumerate(iter_data):
            input_ids_raw = data["input_ids"].to(self.device)
            labels_raw = data["targets"].to(self.device)
            B = input_ids_raw.size(0)

            # ---- Text 路 beam search (返回 score) ----
            text_input = text_item_code[input_ids_raw].contiguous().clone().view(B, -1)
            text_attn = (text_input != -1).bool()
            text_labels = text_item_code[labels_raw].contiguous().clone().view(B, -1)

            model_rec = self.model_rec.module if dist.is_initialized() else self.model_rec
            text_seqs, text_scores = model_rec.my_beam_search(
                input_ids=text_input, attention_mask=text_attn,
                max_length=self.code_length + 1, num_beams=num_beams,
                num_return_sequences=num_beams, return_score=True, route='text')
            text_preds = text_seqs[:, 1:].view(B, num_beams, self.code_length)
            text_scores = text_scores.view(B, num_beams)

            # ---- Image 路 beam search (返回 score) ----
            img_input = image_item_code[input_ids_raw].contiguous().clone().view(B, -1)
            img_attn = (img_input != -1).bool()

            img_seqs, img_scores = model_rec.my_beam_search(
                input_ids=img_input, attention_mask=img_attn,
                max_length=self.code_length + 1, num_beams=num_beams,
                num_return_sequences=num_beams, return_score=True, route='image')
            img_preds = img_seqs[:, 1:].view(B, num_beams, self.code_length)
            img_scores = img_scores.view(B, num_beams)

            # ---- Score Fusion (参考 MACRec) ----
            batch_results = self._ensemble_fusion(
                text_preds, text_scores, img_preds, img_scores,
                text_code2items, image_code2items,
                text_labels, B, num_beams)

            # 转为 tensor 以支持 DDP gather (固定长度避免 rank 间 shape 不一致)
            fixed_len = self.max_topk
            results_tensor = torch.zeros(len(batch_results), fixed_len, device=self.device)
            for i, r in enumerate(batch_results):
                rlen = min(len(r), fixed_len)
                results_tensor[i, :rlen] = torch.tensor(r[:rlen], dtype=torch.float, device=self.device)

            if dist.is_initialized():
                results_tensor = self.accelerator.gather_for_metrics(results_tensor)

            for i in range(results_tensor.size(0)):
                total += 1
                result = results_tensor[i].cpu().numpy()
                for m in self.all_metrics:
                    m_name, top_k = m.split("@")
                    top_k = int(top_k)
                    if m_name.lower() == 'recall':
                        metrics[m] += 1.0 if any(result[:top_k] > 0.5) else 0.0
                    elif m_name.lower() == 'ndcg':
                        metrics[m] += ndcg_at_k(result, top_k)

        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)

        return metrics

    def _ensemble_fusion(self, text_preds, text_scores, img_preds, img_scores,
                         text_code2items, image_code2items, text_labels, B, num_beams):
        """
        Score fusion: 合并 text/image 两路的 beam 候选。
        参考 MACRec/ensemble.py 的 get_topk_results_ensemble 逻辑:
        - 如果同一个 item 在两路都出现，分数取 (s1+s2)/2 + 1 (bonus)
        - 否则保留原始分数
        返回: list of list, 每个样本的 hit/miss 列表 (按分数排序)
        """
        results = []
        for b in range(B):
            item_id2score = {}
            target_code = text_labels[b].cpu().tolist()

            # Text 路候选
            for k in range(num_beams):
                code_tuple = tuple(text_preds[b, k].cpu().tolist())
                score = text_scores[b, k].item()
                items = text_code2items.get(code_tuple, [-1])
                item_id = items[0]
                if item_id == -1:
                    score = -1000
                if item_id in item_id2score and item_id != -1:
                    item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
                else:
                    item_id2score[item_id] = score

            # Image 路候选
            for k in range(num_beams):
                code_tuple = tuple(img_preds[b, k].cpu().tolist())
                score = img_scores[b, k].item()
                items = image_code2items.get(code_tuple, [-1])
                item_id = items[0]
                if item_id == -1:
                    score = -1000
                if item_id in item_id2score and item_id != -1:
                    item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
                else:
                    item_id2score[item_id] = score

            # 按分数排序
            sorted_items = sorted(item_id2score.items(), key=lambda x: x[1], reverse=True)

            # 判断 hit/miss: 需要知道 target item_id
            # target_code 是 text 路的 label code，需要反查 item_id
            target_items = text_code2items.get(tuple(target_code), [-1])

            one_result = []
            for item_id, _ in sorted_items:
                if item_id in target_items:
                    one_result.append(1)
                else:
                    one_result.append(0)

            # 确保至少有 max_topk 个结果
            while len(one_result) < self.max_topk:
                one_result.append(0)

            results.append(one_result)

        return results

    @torch.no_grad()
    def get_code(self, epoch_idx, verbose=True, use_forge=True):
        """
        生成 item code。
        cross_modal 模式: 返回 (text_code, image_code) 两个列表
        单路模式: 返回 all_item_tokens 列表
        """
        self.model_rec.eval()
        self.model_id.eval()

        if self.cross_modal:
            return self._get_code_cross(epoch_idx, verbose, use_forge)
        else:
            return self._get_code_single(epoch_idx, verbose, use_forge)

    def _get_code_single(self, epoch_idx, verbose, use_forge):
        """原始单路 code 生成 (与原版逻辑完全一致)"""
        if dist.is_initialized():
            all_item_embs = self.model_rec.module.semantic_embedding.weight.data[1:]
            all_item_prefix = self.model_id.module.get_indices(all_item_embs).detach().cpu().numpy()
        else:
            all_item_embs = self.model_rec.semantic_embedding.weight.data[1:]
            all_item_prefix = self.model_id.get_indices(all_item_embs).detach().cpu().numpy()

        if verbose:
            self.log(f'[Epoch {epoch_idx}] Original Code conflict (Before FORGE opt): {conflict(all_item_prefix.tolist())}')
            if use_forge:
                self.log(f'[Epoch {epoch_idx}] [FORGE] Applied Random/Sequential Policy to New Suffix Code Layer.')

        all_item_prefix = all_item_prefix.tolist()
        if use_forge:
            all_item_prefix = self._apply_forge(all_item_prefix, epoch_idx)

        return self._build_item_tokens(all_item_prefix, epoch_idx)

    def _get_code_cross(self, epoch_idx, verbose, use_forge):
        """双路 code 生成: 分别对 text/image 路生成 code"""
        model_id = self.model_id.module if dist.is_initialized() else self.model_id
        model_rec = self.model_rec.module if dist.is_initialized() else self.model_rec

        # ---- Text 路 ----
        text_embs = model_rec.text_semantic_embedding.weight.data[1:]
        text_prefix = model_id.get_text_indices(text_embs).detach().cpu().numpy()

        if verbose:
            self.log(f'[Epoch {epoch_idx}] Text Code conflict (Before FORGE): {conflict(text_prefix.tolist())}')

        text_prefix = text_prefix.tolist()
        if use_forge:
            self.log(f'[Epoch {epoch_idx}] [FORGE] Applying FORGE on Text route...')
            text_prefix = self._apply_forge(text_prefix, epoch_idx)
        text_tokens = self._build_item_tokens(text_prefix, epoch_idx, route_name='Text')

        # ---- Image 路 ----
        image_embs = model_rec.image_semantic_embedding.weight.data[1:]
        image_prefix = model_id.get_image_indices(image_embs).detach().cpu().numpy()

        if verbose:
            self.log(f'[Epoch {epoch_idx}] Image Code conflict (Before FORGE): {conflict(image_prefix.tolist())}')

        image_prefix = image_prefix.tolist()
        if use_forge:
            self.log(f'[Epoch {epoch_idx}] [FORGE] Applying FORGE on Image route...')
            image_prefix = self._apply_forge(image_prefix, epoch_idx)
        image_tokens = self._build_item_tokens(image_prefix, epoch_idx, route_name='Image')

        return text_tokens, image_tokens

    def _apply_forge(self, all_item_prefix, epoch_idx):
        """FORGE 策略: 在 RQ-VAE 最后一层进行 load balancing 分流"""
        self.log(f'[Epoch {epoch_idx}] [FORGE] Applying Load Balancing on the last RQ-VAE layer...')

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
                new_last = (start_offset + i) % self.code_num
                all_item_prefix[item_idx][-1] = new_last

        return all_item_prefix

    def _build_item_tokens(self, all_item_prefix, epoch_idx, route_name=''):
        """将 RQ prefix + FORGE suffix 构建为最终的 item tokens"""
        tokens2item = defaultdict(list)
        all_item_tokens = [[-1] * self.code_length]  # index 0 = padding
        max_conflict = 0

        for i in range(len(all_item_prefix)):
            str_id = ' '.join(map(str, all_item_prefix[i]))
            tokens2item[str_id].append(i + 1)
            count = len(tokens2item[str_id]) - 1
            suffix = count
            all_item_tokens.append(all_item_prefix[i] + [suffix])
            max_conflict = max(max_conflict, suffix + 1)

        prefix = f'[{route_name}] ' if route_name else ''
        self.log(f'[Epoch {epoch_idx}] {prefix}[TOKENIZER] Final Code Maximum Conflict (Suffix Size): {max_conflict}')

        if max_conflict > self.code_num:
            self.log(f'WARNING: Max conflict {max_conflict} > code_num {self.code_num}. '
                     f'Consider increasing code_num or layers.', level='warning')

        return all_item_tokens

    def build_code_to_item_index(self, all_item_code):
        """构建 Code 序列 -> Item ID 的反向索引"""
        code_to_item = {}
        for item_id in range(1, len(all_item_code)):
            code = tuple(all_item_code[item_id].tolist())
            if code not in code_to_item:
                code_to_item[code] = item_id
        return code_to_item

    def log(self, message, level='info'):
        return log(message, self.accelerator, self.logger, level=level)
