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
import copy
from data import DPODataset, DPOCollator
from model import Model
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from transformers.optimization import get_scheduler
from metrics import *
from utils import *
from collections import defaultdict
from logging import getLogger
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
        
        assert self.cycle % self.eval_step == 0, 'cycle should be divisible by eval_step'

        self.state = PartialState()
        self.world_size = self.state.num_processes
        self.device = self.state.device
        self.all_item_code = None
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
            self.rec_lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.rec_optimizer,
                                                                    num_warmup_steps=self.warmup_steps,
                                                                    num_training_steps=self.max_steps)
            self.id_lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.id_optimizer,
                                                                   num_warmup_steps=self.warmup_steps // self.cycle,
                                                                   num_training_steps=self.max_steps // self.cycle)
        elif self.lr_scheduler_type == "constant":
            self.rec_lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.rec_optimizer,
                                                                      num_warmup_steps=self.warmup_steps)
            self.id_lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.id_optimizer,
                                                                     num_warmup_steps=self.warmup_steps // self.cycle)
        elif self.lr_scheduler_type == "cosine":
            self.rec_lr_scheduler = get_scheduler(
                            name="cosine",
                            optimizer=self.rec_optimizer,
                            num_warmup_steps=self.warmup_steps,
                            num_training_steps=self.max_steps,
                        )
            self.id_lr_scheduler = get_scheduler(
                            name="cosine",
                            optimizer=self.id_optimizer,
                            num_warmup_steps=self.warmup_steps  // self.cycle,
                            num_training_steps=self.max_steps // self.cycle,
                        )

        self.best_score = 0
        self.best_ckpt = None
        
        self.model_rec, self.rec_optimizer, self.rec_lr_scheduler, \
        self.model_id, self.id_optimizer, self.id_lr_scheduler, \
        self.train_data, self.valid_data, self.test_data = \
        self.accelerator.prepare(self.model_rec, self.rec_optimizer, self.rec_lr_scheduler,
                                 self.model_id, self.id_optimizer, self.id_lr_scheduler,
                                 self.train_data, self.valid_data, self.test_data)

    def _build_optimizer(self, model, lr, weight_decay):
        params = model.parameters()
        learner =  self.learner

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=lr, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=lr, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=lr, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
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
        # kl loss
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

        if sim=="cos":
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

    def _train_epoch_rec(self, epoch_idx, loss_w, verbose=True):

        self.model_rec.train()
        self.model_id.eval()

        total_num = 0
        total_loss = defaultdict(int)
        iter_data = tqdm(
                    self.train_data,
                    total=len(self.train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
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
                input_ids = self.all_item_code[input_ids].contiguous().clone().view(B, -1)
                labels = self.all_item_code[targets].contiguous().clone().view(B, -1)
                attention_mask = (input_ids != -1).bool() 
                
                target_flatten = targets.flatten()
                if dist.is_initialized():
                    target_semantic_embs = self.model_rec.module.semantic_embedding(target_flatten)
                else:
                    target_semantic_embs = self.model_rec.semantic_embedding(target_flatten)
                target_recon_embs, _, _, _, target_code_logits = \
                self.model_id(target_semantic_embs)

                _, unq_index = np.unique(target_flatten.cpu().numpy(), return_index=True)
                unq_index = torch.tensor(unq_index).to(self.device)

                outputs = self.model_rec(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
          
                logits = outputs.logits  # (batch, code_len, code_num)

                seq_project_latents = outputs.seq_project_latents
                dec_latents = outputs.dec_latents
                
                if dist.is_initialized():
                    _, _, _, _, seq_code_logits = self.model_id.module.rq(seq_project_latents)
                else:
                    _, _, _, _, seq_code_logits = self.model_id.rq(seq_project_latents)
                
                code_loss = F.cross_entropy(logits.view(-1, self.code_num), labels.detach().reshape(-1)) 
                

                # kl divergence
                kl_loss = self.compute_discrete_contrastive_loss_kl(seq_code_logits[unq_index], 
                                                                    target_code_logits[unq_index]) + \
                          self.compute_discrete_contrastive_loss_kl(target_code_logits[unq_index],
                                                                    seq_code_logits[unq_index])
            
                
                dec_cl_loss = self.compute_contrastive_loss(target_recon_embs[unq_index], dec_latents[unq_index], sim=self.sim, gathered=False) + \
                          self.compute_contrastive_loss(dec_latents[unq_index], target_recon_embs[unq_index], sim=self.sim, gathered=False)

                losses = dict(
                    code_loss=code_loss,
                    kl_loss=kl_loss,
                    dec_cl_loss=dec_cl_loss,
                )
                
                loss = sum([v * loss_w[k] for k, v in losses.items()])

                self.accelerator.backward(loss)

                self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1)
                self.rec_optimizer.step()
                self.rec_lr_scheduler.step()

                
                kl_loss_mean = self.accelerator.gather(kl_loss).mean().item()
                code_loss_mean = self.accelerator.gather(code_loss).mean().item()
                dec_cl_loss_mean = self.accelerator.gather(dec_cl_loss).mean().item()
                
                loss_mean = self.accelerator.gather(loss).mean().item()
                loss = dict(
                    loss=loss_mean,
                    kl_loss=kl_loss_mean,
                    code_loss=code_loss_mean,
                    dec_cl_loss=dec_cl_loss_mean,
                )

                for k,v in loss.items():
                    total_loss[k] += v
                iter_data.set_postfix(loss=loss_mean)

        for k in total_loss.keys():
            total_loss[k] = round(total_loss[k]/total_num, 4)
                
        self.accelerator.wait_for_everyone()
        
        return total_loss
    

    def _train_epoch_id(self, epoch_idx, loss_w, verbose=True):
        self.model_id.train()


        total_num = 0
        total_loss = defaultdict(int)
        iter_data = tqdm(
                    self.train_data,
                    total=len(self.train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
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
                input_ids = self.all_item_code[input_ids].contiguous().clone().view(B, -1)
                labels = self.all_item_code[targets].contiguous().clone().view(B, -1)
                attention_mask = (input_ids != -1).bool() 
                
                
                target_flatten = targets.flatten()
                if dist.is_initialized():
                    target_semantic_embs = self.model_rec.module.semantic_embedding(target_flatten)
                else:
                    target_semantic_embs = self.model_rec.semantic_embedding(target_flatten)
                target_recon_embs, _, _, _, target_code_logits = \
                self.model_id(target_semantic_embs)
                
                unq_input, unq_index = np.unique(target_flatten.cpu().numpy(), return_index=True)
                unq_input = torch.tensor(unq_input).to(self.device)
                unq_index = torch.tensor(unq_index).to(self.device)
                if dist.is_initialized():
                    unq_semantic_embs = self.model_rec.module.semantic_embedding(unq_input)
                else:
                    unq_semantic_embs = self.model_rec.semantic_embedding(unq_input)
                unq_recon_embs, commit_loss, _, _, _ = self.model_id(unq_semantic_embs)


                outputs = self.model_rec(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         labels=labels)
          
                logits = outputs.logits  # (batch, code_len, code_num)

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
                    recon_loss = self.compute_contrastive_loss(unq_recon_embs, unq_semantic_embs,
                                                               temperature=self.tau, gathered=False)
                else:
                    raise ValueError('incompatible loss type')
                
                vq_loss = recon_loss + self.alpha * commit_loss

                # kl divergence
                kl_loss = self.compute_discrete_contrastive_loss_kl(seq_code_logits[unq_index], 
                                                                    target_code_logits[unq_index]) + \
                          self.compute_discrete_contrastive_loss_kl(target_code_logits[unq_index],
                                                                    seq_code_logits[unq_index])
                
                
                dec_cl_loss = self.compute_contrastive_loss(target_recon_embs[unq_index], dec_latents[unq_index],
                                                            sim=self.sim, gathered=False) + \
                              self.compute_contrastive_loss(dec_latents[unq_index], target_recon_embs[unq_index],
                                                            sim=self.sim, gathered=False)
                
                losses = dict(
                    vq_loss=vq_loss,
                    code_loss=code_loss,
                    kl_loss=kl_loss,
                    dec_cl_loss=dec_cl_loss,
                )
                
                loss = sum([v * loss_w[k] for k, v in losses.items()])

                self.accelerator.backward(loss)

                self.accelerator.clip_grad_norm_(self.model_id.parameters(), 1)
                self.id_optimizer.step()
                self.id_lr_scheduler.step()
                
                
                vq_loss_mean = self.accelerator.gather(vq_loss).mean().item()
                code_loss_mean = self.accelerator.gather(code_loss).mean().item()
                kl_loss_mean = self.accelerator.gather(kl_loss).mean().item()
                dec_cl_loss_mean = self.accelerator.gather(dec_cl_loss).mean().item()
                
                loss_mean = self.accelerator.gather(loss).mean().item()
                loss = dict(
                    loss=loss_mean,
                    vq_loss=vq_loss_mean,
                    code_loss=code_loss_mean,
                    kl_loss=kl_loss_mean,
                    dec_cl_loss=dec_cl_loss_mean,
                )

                for k,v in loss.items():
                    total_loss[k] += v
                iter_data.set_postfix(loss=loss_mean)

        for k in total_loss.keys():
            total_loss[k] = round(total_loss[k]/total_num, 4)
                
        self.accelerator.wait_for_everyone()
        
        return total_loss

    def safe_save(self, epoch, code):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrap_model_rec = self.accelerator.unwrap_model(self.model_rec)
            unwrap_model_id = self.accelerator.unwrap_model(self.model_id)
            self.accelerator.save(unwrap_model_rec.state_dict(), f'{self.save_path}/{epoch}.pt')
            self.accelerator.save(unwrap_model_id.state_dict(), f'{self.save_path}/{epoch}.pt.rqvae')
            json.dump(code.cpu().tolist(), open(f'{self.save_path}/{epoch}.code.json', 'w'))
            self.log(f'[Epoch {epoch}] Save model {self.save_path}/{epoch}.pt')

        last_checkpoint = f'{self.save_path}/{epoch}.pt'
        return last_checkpoint

    def evaluate(self, outputs, labels):
        batch_size, k, _ = outputs.shape  # Assuming outputs is [batch_size, 10, seq_len]
        recall_at_1, recall_at_5, recall_at_10 = [], [], []
        ndcg_at_1, ndcg_at_5, ndcg_at_10 = [], [], []

        for i in range(batch_size):
            label = labels[i].unsqueeze(0)  # [1, seq_len]
            out = outputs[i]
                
            matches = torch.all(torch.eq(out.unsqueeze(1), label.unsqueeze(0)), dim=2)  # [10, 1, seq_len] -> [10, 1]
            matches = matches.any(dim=1).cpu().numpy()  # [10]

            # Recall
            recall_at_1.append(matches[:1].sum() / 1.0)
            recall_at_5.append(matches[:5].sum() / 1.0)  # Assuming each label has only 1 correct match.
            recall_at_10.append(matches.sum() / 1.0)

            # NDCG (binary relevance)
            ndcg_at_1.append(ndcg_at_k(matches, 1))
            ndcg_at_5.append(ndcg_at_k(matches, 5))
            ndcg_at_10.append(ndcg_at_k(matches, 10))

        # Calculate mean metrics
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
        train_loss_output = (
            "[Epoch %d] [time: %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        if isinstance(loss_dict, dict):
            train_loss_output += "train loss" + str(list(loss_dict.items()))
        else:
            train_loss_output += "train loss" + ": %.4f" % loss_dict
        return train_loss_output + "]"

    def train(self, verbose=True):
        stop = False
        cur_eval_step = 0
        loss_w = defaultdict(int)
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
                    semantic_emb_name = 'module.semantic_embedding' if dist.is_initialized() else 'semantic_embedding'
                    if not name.startswith(semantic_emb_name):
                        param.requires_grad = True
                for param in self.model_id.parameters():
                    param.requires_grad = False

            
            self.accelerator.wait_for_everyone()
            # train
            training_start_time = time()
            if epoch_idx % self.cycle == 0:
                train_loss = self._train_epoch_id(epoch_idx, loss_w=loss_w, verbose=verbose)
                all_item_code = self.get_code(epoch_idx=epoch_idx, verbose=verbose)
                self.all_item_code = torch.tensor(all_item_code).to(self.device)
            else:
                train_loss = self._train_epoch_rec(epoch_idx, loss_w=loss_w, verbose=verbose)
            training_end_time = time()
   
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            
            self.log(train_loss_output)
            self.log(f'[Epoch {epoch_idx}] REC lr: {self.rec_lr_scheduler.get_lr()} ID lr: {self.id_lr_scheduler.get_lr()}')
            

            if (epoch_idx + 1) % self.eval_step == 0:
                metrics = self._test_epoch(test_data=self.valid_data, code=self.all_item_code, verbose=verbose)
                total_metrics = metrics

                if total_metrics[self.valid_metric] > self.best_score:
                    self.best_score = total_metrics[self.valid_metric]
                    self.best_result = total_metrics
                    cur_eval_step = 0
                    self.best_ckpt = self.safe_save(epoch_idx, self.all_item_code)
                else:
                    cur_eval_step += 1

                if cur_eval_step >= self.early_stop:
                    stop = True

                self.log(f'[Epoch {epoch_idx}] Val Results: {total_metrics}')
            
            self.accelerator.wait_for_everyone()

            if stop:
                break

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
        self.rec_lr_scheduler = get_scheduler(name='cosine',
                                              optimizer=self.rec_optimizer,
                                              num_warmup_steps=0,
                                              num_training_steps=train_steps)
        self.rec_optimizer, self.rec_lr_scheduler = self.accelerator.prepare(self.rec_optimizer, self.rec_lr_scheduler)
        
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
        all_item_code = self.get_code(epoch_idx=0, verbose=False)
        self.all_item_code = torch.tensor(all_item_code).to(self.device)

        for name, param in self.model_rec.named_parameters():
            if not name.startswith('module.semantic_embedding'):
                param.requires_grad = True

        for param in self.model_id.parameters():
            param.requires_grad = False

        for epoch_idx in range(self.epochs):
            
            self.accelerator.wait_for_everyone()
            # train
            training_start_time = time()
            train_loss = self._train_epoch_rec(epoch_idx, loss_w=loss_w, verbose=verbose)
            training_end_time = time()
   
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss
            )
            
            self.log(train_loss_output)
            self.log(f'[Epoch {epoch_idx}] Current REC lr: {self.rec_lr_scheduler.get_lr()}')
            

            if (epoch_idx + 1) % self.eval_step == 0:
                metrics = self._test_epoch(test_data=self.valid_data, code=self.all_item_code, verbose=verbose)
                total_metrics = metrics

                if total_metrics[self.valid_metric] > self.best_score:
                    self.best_score = total_metrics[self.valid_metric]
                    self.best_result = total_metrics
                    cur_eval_step = 0
                    self.best_ckpt = self.safe_save(epoch_idx, self.all_item_code)
                else:
                    cur_eval_step += 1

                if cur_eval_step >= self.early_stop:
                    stop = True

                self.log(f'[Epoch {epoch_idx}] Val Results: {total_metrics}')
            
            self.accelerator.wait_for_everyone()

            if stop:
                break
        

        return self.best_score
    
    @torch.no_grad()
    def test(self, verbose=True, model_file=None, prefix_allowed_tokens_fn=None):
        test_results=None
        if self.test_data is not None:
            metrics = self._test_epoch(load_best_model=True, model_file=model_file,
                                       prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, verbose=verbose)

            test_results = metrics
        return test_results


    @torch.no_grad()
    def generate_dpo_data_from_model(self, verbose=True):
        """
        生成 DPO 数据。
        使用当前 SFT 模型对训练集进行预测 (Beam Search k=5)，
        构造 (Input, Chosen, Rejected) 三元组。
        """
        self.model_rec.eval()
        self.model_rec.to(self.device) # 确保模型在 device 上
        
        # 确保 item_code 映射表已加载
        if self.all_item_code is None:
            self.all_item_code = torch.tensor(self.get_code(epoch_idx=-1, verbose=False)).to(self.device)
            
        item_code = self.all_item_code
        dpo_data = []
        
        # 使用 k=5 进行生成
        k = 5
        
        # OOM 优化：生成时使用较小的 Batch Size
        gen_batch_size = self.config['dpo_batch_size'] 
        
        # 创建一个新的 DataLoader 用于生成
        gen_dataloader = torch.utils.data.DataLoader(
            self.train_data.dataset, # 复用 dataset
            batch_size=gen_batch_size,
            shuffle=False, # 生成数据不需要 shuffle
            collate_fn=self.train_data.collate_fn,
            num_workers=self.config['num_workers']
        )
        
        # 如果是多卡，使用 prepare 加速
        if self.accelerator.num_processes > 1:
            gen_dataloader = self.accelerator.prepare_data_loader(gen_dataloader)

        iter_data = tqdm(
            gen_dataloader,
            total=len(gen_dataloader),
            ncols=100,
            desc=set_color(f"Gen DPO Data", "cyan"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )
        
        for batch in iter_data:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            targets = batch["targets"].to(self.device) # (B, 1) or (B,)
            
            B = input_ids.size(0)
            # 将 Item ID 映射为离散 Code 序列
            # input_ids: (B, seq_len) -> (B, seq_len, code_len) -> (B, seq_len * code_len)
            input_ids_mapped = item_code[input_ids].contiguous().clone().view(B, -1)
            attention_mask_mapped = (input_ids_mapped != -1).bool()
            
            # 生成推荐结果 (Beam Search, Return Top-K)
            # 注意：我们需要保证 num_beams >= k
            num_beams = max(self.config['num_beams'], k)
            
            if dist.is_initialized():
                preds = self.model_rec.module.generate(
                    input_ids=input_ids_mapped, 
                    attention_mask=attention_mask_mapped, 
                    n_return_sequences=k
                )
            else:
                preds = self.model_rec.generate(
                    input_ids=input_ids_mapped, 
                    attention_mask=attention_mask_mapped, 
                    n_return_sequences=k
                )
            
            # preds shape: (B * k, code_len) -> reshape to (B, k, code_len)
            preds = preds.view(B, k, -1)
            
            # 收集数据
            input_ids_cpu = input_ids.cpu().tolist()
            targets_cpu = item_code[targets].view(B, -1).cpu().tolist() # Ground Truth (Chosen)
            preds_cpu = preds.cpu().tolist() # Model Generation Candidates
            
            hit_count = 0
            for i in range(B):
                chosen = targets_cpu[i]
                rejected = None
                
                # 统计 Hit Rate (Chosen 是否在 Candidates 中)
                if chosen in preds_cpu[i]:
                    hit_count += 1

                # 从 Top-K 中寻找一个 Rejected
                # 策略：找到第一个不等于 Chosen 的生成结果
                for candidate in preds_cpu[i]:
                    if candidate != chosen:
                        rejected = candidate
                        break
                
                # 如果找到了有效的 Rejected 样本
                if rejected is not None:
                    # 移除 input_ids 中的 padding (0)
                    clean_input_ids = [x for x in input_ids_cpu[i] if x != 0]
                    dpo_data.append({
                        'input_ids': clean_input_ids,
                        'chosen': chosen,
                        'rejected': rejected
                    })
        
        # 打印采样统计信息
        if verbose and len(dpo_data) > 0:
            sample_hit_rate = hit_count / B # 只统计最后一个 Batch 的，虽然不精确但够看了
            self.log(f"DPO Data Gen (Last Batch): Hit Rate @ Top-{k} = {sample_hit_rate:.4f}")
            self.log(f"Sample DPO Triplet:\n  Input: {dpo_data[0]['input_ids']}\n  Chosen (GT): {dpo_data[0]['chosen']}\n  Rejected (Model): {dpo_data[0]['rejected']}")

        return dpo_data

    def train_dpo(self, saved_sft_ckpt=None, verbose=True):
        """
        DPO (Direct Preference Optimization) 训练循环。
        """
        self.log("Starting DPO Training...")

        # 0. 加载 SFT Checkpoint (如果指定)
        if saved_sft_ckpt:
            self.log(f"Loading SFT Checkpoint for DPO: {saved_sft_ckpt}")
            if dist.is_initialized():
                safe_load(self.model_rec.module, saved_sft_ckpt, verbose=verbose)
                # 尝试加载对应的 RQVAE，如果存在
                if os.path.exists(f'{saved_sft_ckpt}.rqvae'):
                    safe_load(self.model_id.module, f'{saved_sft_ckpt}.rqvae', verbose=verbose)
            else:
                safe_load(self.model_rec, saved_sft_ckpt, verbose=verbose)
                if os.path.exists(f'{saved_sft_ckpt}.rqvae'):
                    safe_load(self.model_id, f'{saved_sft_ckpt}.rqvae', verbose=verbose)
        
        # 1. 生成 Self-Play 数据
        # 注意：这里使用加载后的 SFT 模型生成数据
        self.log("Generating Self-Play Data... (This may take a while)")
        dpo_data = self.generate_dpo_data_from_model(verbose=verbose)
        
        if len(dpo_data) == 0:
            self.log("No DPO data generated (all predictions matched ground truth?). Skipping DPO.")
            return

        self.log(f"Process {self.accelerator.process_index}: Generated {len(dpo_data)} DPO triplets.")

        # === OOM 关键优化 ===
        # 生成完数据后，先将模型移回 CPU 并清空显存，为加载 Reference Model 腾出空间
        self.model_rec.to('cpu')
        self.model_id.to('cpu')
        torch.cuda.empty_cache()
        self.log("Memory Optimized: Model moved to CPU, Cache cleared.")
        # ===================

        # 2. 准备 DataLoader
        dpo_dataset = DPODataset(dpo_data)
        dpo_collator = DPOCollator(pad_token_id=0, max_length=self.config['max_length'])
        
        dpo_dataloader = torch.utils.data.DataLoader(
            dpo_dataset, 
            batch_size=self.config['dpo_batch_size'], 
            shuffle=True, 
            collate_fn=dpo_collator,
            num_workers=self.config['num_workers']
        )
        
        # 使用 accelerator 准备 dataloader (处理分布式采样)
        dpo_dataloader = self.accelerator.prepare(dpo_dataloader)
        
        # 3. 准备 Reference Model (参考模型)
        self.log("Initializing Reference Model...")
        
        # 此时 model_rec 在 CPU，复制也在 CPU 进行，节省显存
        if dist.is_initialized():
             # 如果是 DDP，model_rec 是 DistributedDataParallel，需要取 module
             # 但我们已经 move 到 cpu 了，DDP wrapper 可能还在？
             # 安全起见，unwrap 一下
             unwrapped_model = self.accelerator.unwrap_model(self.model_rec)
             self.ref_model = copy.deepcopy(unwrapped_model)
        else:
             self.ref_model = copy.deepcopy(self.model_rec)

        # 将两个模型都移回 GPU
        self.model_rec.to(self.device)
        self.model_id.to(self.device)
        self.ref_model.to(self.device)
        
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        # 4. 优化器配置
        # 关键修复：DPO 阶段必须冻结 Semantic Embedding，防止 Code 漂移
        self.log("Freezing Semantic Embeddings for DPO...")
        for name, param in self.model_rec.named_parameters():
            if 'semantic_embedding' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        # 仅优化 requires_grad=True 的参数
        trainable_params = [p for p in self.model_rec.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=self.config['dpo_lr'])
        
        # DPO 通常使用较小的恒定学习率或简单的 Warmup
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=10)
        
        optimizer, lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)
        
        # 5. 训练循环
        self.model_rec.train()
        
        for epoch in range(self.config['dpo_epochs']):
            total_loss = 0
            steps = 0
            
            iter_data = tqdm(
                dpo_dataloader,
                total=len(dpo_dataloader),
                ncols=100,
                desc=set_color(f"DPO Epoch {epoch}", "green"),
                disable=(not verbose) or (not self.accelerator.is_main_process),
            )
            
            for batch in iter_data:
                with self.accelerator.accumulate(self.model_rec):
                    optimizer.zero_grad()
                    
                    # 准备数据
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    chosen_ids = batch['chosen_ids'].to(self.device)
                    rejected_ids = batch['rejected_ids'].to(self.device)
                    
                    B = input_ids.size(0)
                    # 将 Item ID 序列映射为 Code 序列输入
                    input_ids_mapped = self.all_item_code[input_ids].contiguous().clone().view(B, -1)
                    attention_mask_mapped = (input_ids_mapped != -1).bool()
                    
                    # 计算 Policy Model (当前训练模型) 的 Log Probs
                    if dist.is_initialized():
                        policy_chosen = self.model_rec.module.compute_log_probs(input_ids_mapped, attention_mask_mapped, chosen_ids)
                        policy_rejected = self.model_rec.module.compute_log_probs(input_ids_mapped, attention_mask_mapped, rejected_ids)
                    else:
                        policy_chosen = self.model_rec.compute_log_probs(input_ids_mapped, attention_mask_mapped, chosen_ids)
                        policy_rejected = self.model_rec.compute_log_probs(input_ids_mapped, attention_mask_mapped, rejected_ids)
                        
                    # 计算 Reference Model (冻结参考模型) 的 Log Probs
                    with torch.no_grad():
                        ref_chosen = self.ref_model.compute_log_probs(input_ids_mapped, attention_mask_mapped, chosen_ids)
                        ref_rejected = self.ref_model.compute_log_probs(input_ids_mapped, attention_mask_mapped, rejected_ids)
                    
                    # 计算 DPO Loss
                    # Log Ratio: log(pi(y|x) / ref(y|x)) = log(pi) - log(ref)
                    pi_logratios = policy_chosen - policy_rejected
                    ref_logratios = ref_chosen - ref_rejected
                    
                    logits = pi_logratios - ref_logratios
                    
                    # DPO Loss = -log(sigmoid(beta * logits))
                    loss = -F.logsigmoid(self.config['dpo_beta'] * logits).mean()
                    
                    self.accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    
                    loss_val = self.accelerator.gather(loss).mean().item()
                    total_loss += loss_val
                    steps += 1
                    iter_data.set_postfix(dpo_loss=loss_val)
            
            avg_loss = total_loss / steps if steps > 0 else 0
            self.log(f"[DPO Epoch {epoch}] Avg Loss: {avg_loss:.4f}")
            
            # 保存 Checkpoint
            if (epoch + 1) % self.config['dpo_epochs'] == 0: # 仅在最后保存，避免占用过多空间
                 self.safe_save(f"dpo_final", self.all_item_code)
        
        # 清理 Reference Model 释放显存
        del self.ref_model
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _test_epoch(self, code=None, test_data=None, load_best_model=False, model_file=None,
                    prefix_allowed_tokens_fn=None, verbose=True):
        
        if test_data is None:
            test_data = self.test_data

        if load_best_model:
            ckpt_file = model_file or self.best_ckpt
            if ckpt_file:
                if dist.is_initialized():
                    safe_load(self.model_rec.module, ckpt_file, verbose=verbose)
                    safe_load(self.model_id.module, ckpt_file+'.rqvae', verbose=verbose)
                else:
                    safe_load(self.model_rec, ckpt_file, verbose=verbose)
                    safe_load(self.model_id, ckpt_file+'.rqvae', verbose=verbose)

                code = json.load(open(ckpt_file[:-3]+'.code.json'))

                message_output = "Loading model parameters from {}".format(
                    ckpt_file
                )
                self.log(message_output)
            else:
                if code is None:
                    code = self.all_item_code if self.all_item_code is not None else self.get_code(epoch_idx=-1, verbose=False)
                self.log("No checkpoint available; evaluating current in-memory model", level='warning')

        self.model_rec.eval()
        self.model_id.eval()

        iter_data = tqdm(
            test_data,
            total=len(test_data),
            ncols=100,
            desc=set_color(f"Evaluate   ", "pink"),
            disable=(not verbose) or (not self.accelerator.is_main_process),
        )

        if isinstance(code, torch.Tensor):
            code = code.cpu().tolist()

        total = 0
        metrics = {m: 0 for m in self.all_metrics}

        code2item = defaultdict(list)
        for i, c in enumerate(code[1:]):
            code2item[str(c)].append(i+1)

        item_code = torch.tensor(code).to(self.device)

        for batch_idx, data in enumerate(iter_data):
            input_ids, attention_mask, labels \
                = data["input_ids"].to(self.device), data["attention_mask"].to(self.device), data["targets"].to(self.device)

            B = input_ids.size(0)
            input_ids = item_code[input_ids].contiguous().clone().view(B, -1)
            labels = item_code[labels].contiguous().clone().view(B, -1)
            attention_mask = (input_ids != -1).bool() 

            if dist.is_initialized():
                preds = self.model_rec.module.generate(input_ids=input_ids, attention_mask=attention_mask, n_return_sequences=10)
                all_preds, all_labels = self.accelerator.gather_for_metrics((preds, labels))
                _metrics = self.evaluate(all_preds, all_labels)
                total += len(all_labels)
            else:
                preds = self.model_rec.generate(input_ids=input_ids, attention_mask=attention_mask, n_return_sequences=10)
                _metrics = self.evaluate(preds, labels)
                total += len(labels)

            for m in metrics.keys():
                metrics[m] += _metrics[m]

        for m in metrics:
            metrics[m] = round(metrics[m] / total, 6)

        return metrics
    
    @torch.no_grad()
    def get_code(self, epoch_idx, verbose=True):
        self.model_rec.eval()
        self.model_id.eval()
        if dist.is_initialized():
            all_item_embs = self.model_rec.module.semantic_embedding.weight.data[1:]
            all_item_prefix = self.model_id.module.get_indices(all_item_embs).detach().cpu().numpy()
        else:
            all_item_embs = self.model_rec.semantic_embedding.weight.data[1:]
            all_item_prefix = self.model_id.get_indices(all_item_embs).detach().cpu().numpy()
        

        if verbose:
            for i in range(self.code_length-1):
                self.log(f'[Epoch {epoch_idx}] Evaluation {self.save_path}/{epoch_idx}.pt Code balance {balance(all_item_prefix[:, i].tolist(), ncentroids=self.code_num)} Used code num of level {i+1}: {len(set(all_item_prefix[:, i].tolist()))}')

            self.log(f'[Epoch {epoch_idx}] Evaluation {self.save_path}/{epoch_idx}.pt Code confilct {conflict(all_item_prefix.tolist())}')
        
        all_item_prefix = all_item_prefix.tolist()

        tokens2item = defaultdict(list)
        all_item_tokens = [[-1,-1,-1,-1]]
        max_conflict = 0
        for i in range(len(all_item_prefix)):
            str_id = ' '.join(map(str, all_item_prefix[i]))
            tokens2item[str_id].append(i+1)
            all_item_tokens.append(all_item_prefix[i]+[len(tokens2item[str_id])-1])
            max_conflict = max(max_conflict, len(tokens2item[str_id]))
        self.log(f'[Epoch {epoch_idx}] [TOKENIZER] RQ-VAE semantic IDs, maximum conflict: {max_conflict}')
        if max_conflict > self.code_num:
            raise ValueError(
                f'[TOKENIZER] RQ-VAE semantic IDs conflict with codebook size: '
                f'{max_conflict} > {self.code_num}. Please increase the codebook size.'
            )

        return all_item_tokens

    def log(self, message, level='info'):
        return log(message, self.accelerator, self.logger, level=level)

