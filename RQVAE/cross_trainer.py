"""
CrossTrainer: CrossRQVAE 预训练的 Trainer

训练双路 RQ-VAE，评估两路各自的 collision rate 和 gini coefficient。
"""

import os
import torch
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import ensure_dir, set_color, get_local_time, delete_file
from vq import compute_gini


class CrossTrainer(object):
    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.device = args.device
        self.model.to(self.device)

        self.lr = args.lr
        self.epochs = args.epochs
        self.eval_step = args.eval_step
        self.patience = getattr(args, 'patience', 20)
        self.save_limit = getattr(args, 'save_limit', 3)
        self.ckpt_dir = args.ckpt_dir

        # TensorBoard
        local_time = get_local_time()
        self.run_dir = os.path.join(self.ckpt_dir, local_time)
        ensure_dir(self.run_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.run_dir, 'tb_logs'))
        self.global_step = 0

        # Optimizer & Scheduler
        self.optimizer = self._build_optimizer()
        total_steps = data_num * self.epochs
        warmup_epochs = getattr(args, 'warmup_epochs', 50)
        warmup_steps = data_num * warmup_epochs
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(warmup_steps, 1))
            if step < warmup_steps
            else max(0.01, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))
        )

        # Checkpoint tracking
        self.saved_ckpts = []  # (score, path) 按 score 升序

    def _build_optimizer(self):
        args = self.args
        if args.learner.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                    weight_decay=args.weight_decay)
        elif args.learner.lower() == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                     weight_decay=args.weight_decay)
        else:
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr,
                                     weight_decay=args.weight_decay)

    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_align = 0
        total_vq = 0

        iter_data = tqdm(train_data, total=len(train_data), ncols=120,
                         desc=set_color(f"Train {epoch_idx}", "pink"))

        for batch_idx, (text_data, image_data, index) in enumerate(iter_data):
            text_data = text_data.to(self.device)
            image_data = image_data.to(self.device)

            self.optimizer.zero_grad()

            (text_out, image_out,
             text_rq_loss, image_rq_loss,
             text_indices, image_indices,
             text_xq, image_xq) = self.model(text_data, image_data, item_index=index)

            loss, loss_recon, loss_align = self.model.compute_loss(
                text_out, image_out,
                text_rq_loss, image_rq_loss,
                text_xq, image_xq,
                text_data, image_data
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_align += loss_align.item()
            total_vq += (text_rq_loss + image_rq_loss).item()

            # TensorBoard per-step
            self.global_step += 1
            self.writer.add_scalar('step/total_loss', loss.item(), self.global_step)
            self.writer.add_scalar('step/recon_loss', loss_recon.item(), self.global_step)
            self.writer.add_scalar('step/vq_loss', (text_rq_loss + image_rq_loss).item(), self.global_step)
            self.writer.add_scalar('step/align_loss', loss_align.item(), self.global_step)
            self.writer.add_scalar('step/lr', self.scheduler.get_last_lr()[0], self.global_step)

        n = len(train_data)
        return total_loss / n, total_recon / n, total_align / n, total_vq / n

    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        """评估两路各自的 collision rate 和 gini coefficient"""
        self.model.eval()
        all_text_indices = []
        all_image_indices = []

        for text_data, image_data, index in valid_data:
            text_data = text_data.to(self.device)
            image_data = image_data.to(self.device)

            text_indices = self.model.get_text_indices(text_data)
            image_indices = self.model.get_image_indices(image_data)
            all_text_indices.append(text_indices.cpu())
            all_image_indices.append(image_indices.cpu())

        all_text_indices = torch.cat(all_text_indices, dim=0)   # (N, num_rq_layers)
        all_image_indices = torch.cat(all_image_indices, dim=0)

        # Collision rate: 有多少 item 的 code 完全相同
        def calc_collision_and_gini(indices, code_num):
            N = indices.shape[0]
            # 将多层 code 拼成 tuple 作为唯一标识
            code_tuples = [tuple(row.tolist()) for row in indices]
            unique_codes = set(code_tuples)
            collision_rate = 1.0 - len(unique_codes) / N

            # 每层的 gini
            gini_list = []
            for layer in range(indices.shape[1]):
                g = compute_gini(indices[:, layer], code_num)
                gini_list.append(g)
            avg_gini = np.mean(gini_list)

            return collision_rate, avg_gini

        code_num = self.args.num_emb_list[0]
        text_collision, text_gini = calc_collision_and_gini(all_text_indices, code_num)
        image_collision, image_gini = calc_collision_and_gini(all_image_indices, code_num)

        return text_collision, text_gini, image_collision, image_gini

    def _save_checkpoint(self, epoch, text_collision, image_collision, text_gini, image_gini):
        """保存 checkpoint，维护 save_limit 个最优"""
        ckpt_name = (f"best_tc{text_collision:.4f}_ic{image_collision:.4f}"
                     f"_tg{text_gini:.4f}_ig{image_gini:.4f}.pth")
        ckpt_path = os.path.join(self.run_dir, ckpt_name)
        torch.save(self.model.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # 用 (text_collision + image_collision) 作为排序依据，越小越好
        score = text_collision + image_collision
        self.saved_ckpts.append((score, ckpt_path))
        self.saved_ckpts.sort(key=lambda x: x[0])

        # 删除多余的 checkpoint
        while len(self.saved_ckpts) > self.save_limit:
            _, old_path = self.saved_ckpts.pop()
            delete_file(old_path)

        return ckpt_path

    def fit(self, data_loader):
        """主训练循环"""
        best_text_collision = 1.0
        best_image_collision = 1.0
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time()
            avg_loss, avg_recon, avg_align, avg_vq = self._train_epoch(data_loader, epoch)
            t1 = time()

            # TensorBoard per-epoch
            self.writer.add_scalar('epoch/total_loss', avg_loss, epoch)
            self.writer.add_scalar('epoch/recon_loss', avg_recon, epoch)
            self.writer.add_scalar('epoch/align_loss', avg_align, epoch)
            self.writer.add_scalar('epoch/vq_loss', avg_vq, epoch)

            print(f"Epoch {epoch} [{t1 - t0:.1f}s] "
                  f"loss={avg_loss:.4f} recon={avg_recon:.4f} "
                  f"vq={avg_vq:.4f} align={avg_align:.6f}")

            # 定期评估
            if epoch % self.eval_step == 0:
                text_col, text_gini, image_col, image_gini = self._valid_epoch(data_loader)

                self.writer.add_scalar('eval/text_collision', text_col, epoch)
                self.writer.add_scalar('eval/text_gini', text_gini, epoch)
                self.writer.add_scalar('eval/image_collision', image_col, epoch)
                self.writer.add_scalar('eval/image_gini', image_gini, epoch)

                print(f"  [Eval] text_collision={text_col:.4f} text_gini={text_gini:.4f} "
                      f"image_collision={image_col:.4f} image_gini={image_gini:.4f}")

                # 以两路 collision 之和作为优化目标
                total_collision = text_col + image_col
                if total_collision < (best_text_collision + best_image_collision):
                    best_text_collision = text_col
                    best_image_collision = image_col
                    best_loss = avg_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, text_col, image_col, text_gini, image_gini)
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self.writer.close()
        return best_loss, best_text_collision, best_image_collision
