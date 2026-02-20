## 🚀 ETEGRec 工程修改计划：引入 WandB 可视化监控

**目标**：将原本基于终端日志 (Logger) 的监控升级为 WandB 可视化看板，以便实时分析 Loss 曲线、Collision Rate 变化、FORGE 策略效果 (Max Conflict) 以及最终的 NDCG 指标。

**涉及阶段**：

1.  **RQ-VAE 预训练阶段** (`RQVAE/`)：监控语义重构质量与 Codebook 分布。
2.  **联合训练阶段** (`./`)：监控推荐性能、对齐损失及 FORGE 策略的有效性。

-----

### 第一部分：环境准备

在项目根目录的 `requirements.txt` 中添加：

```text
wandb
```

-----

### 第二部分：RQ-VAE 预训练阶段 (`RQVAE/`)

此阶段代码较为独立，未使用 Accelerate，建议直接使用 `wandb` 原生库进行打点。

#### 1\. 修改 `RQVAE/main.py` (初始化)

**位置**：在文件头部导入库，并在 `main` 函数开始处初始化。

```python
# [Add] 导入 wandb
import wandb 

# ... (在 if __name__ == '__main__': 下方)

    args = parse_args()
    
    # [Add] 初始化 WandB
    wandb.init(
        project="ETEGRec-RQVAE-Pretrain", 
        name=f"{args.dataset}_layers{len(args.layers)}_edim{args.e_dim}",
        config=vars(args)
    )

    print("=================================================")
    # ...
```

#### 2\. 修改 `RQVAE/trainer.py` (埋点)

**位置**：`Trainer.fit` 方法中的训练循环和验证循环后。

```python
# [Add] 头部导入
import wandb

    # ... 在 Trainer.fit 方法内部 ...

    def fit(self, data):
        cur_eval_step = 0
        for epoch_idx in range(self.epochs):
            # 1. Training Loop
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            
            # [Add] 记录训练指标
            wandb.log({
                "train/loss_total": train_loss,
                "train/loss_recon": train_recon_loss,
                "epoch": epoch_idx
            })

            # ... 日志打印代码 ...

            # 2. Validation Loop
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate, avg_gini = self._valid_epoch(data)

                # [Add] 记录验证指标 (重点关注 Collision 和 Gini)
                wandb.log({
                    "val/collision_rate": collision_rate,
                    "val/gini_coefficient": avg_gini,
                    "epoch": epoch_idx
                })

                # ... 后续的模型保存逻辑 ...
```

-----

### 第三部分：ETEGRec 联合训练阶段 (`./`)

此阶段使用了 HuggingFace `Accelerate`，**强烈建议使用 `accelerator.log`** 接口，以确保多卡训练 (DDP) 时的安全性（只在主进程记录）。

#### 1\. 修改 `main.py` (配置 Accelerator)

**位置**：`train` 函数开头，初始化 `Accelerator` 时。

```python
def train(config, verbose=True, rank=0):
    # ... 之前的代码 ...
    
    # [Modify] 修改 Accelerator 初始化，指定 log_with="wandb"
    # 原代码: accelerator = config['accelerator'] 
    # 修改为:
    accelerator = Accelerator(log_with="wandb")  # 开启 wandb 支持
    
    # [Add] 初始化 Trackers (建议在 config 打印之后)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ETEGRec-Joint", 
            config=config,
            init_kwargs={"wandb": {"name": config.get("ckpt_name", "experiment")}}
        )

    # ... 中间的数据加载和模型初始化代码保持不变 ...
    
    # ... 在 train 函数的最末尾添加 ...
    accelerator.end_training()
```

#### 2\. 修改 `trainer.py` (埋点)

需要修改三个地方：训练循环、FORGE 策略监测、测试循环。

**A. 训练循环 (`_train_epoch_rec` 和 `_train_epoch_id`)**

在计算出 `total_loss` 字典后记录。

```python
    # 以 _train_epoch_rec 为例 (id 的同理)
    def _train_epoch_rec(self, epoch_idx, loss_w, verbose=True):
        # ... 训练循环 ...
        
        # 在循环结束后，return total_loss 之前：
        
        # [Add] 使用 accelerator 记录训练指标
        self.accelerator.log({
            "train/rec_loss_total": total_loss['loss'],
            "train/rec_code_loss": total_loss['code_loss'],
            "train/rec_kl_loss": total_loss['kl_loss'],
            "train/rec_dec_cl_loss": total_loss['dec_cl_loss'],
            "lr/rec_lr": self.rec_lr_scheduler.get_last_lr()[0],
            "epoch": epoch_idx
        })
        
        return total_loss
```

**B. FORGE 策略监测 (`get_code`)**

这是监控你修改后的代码是否生效的关键。

```python
    def get_code(self, epoch_idx, verbose=True, use_forge=True):
        # ... 这里的 FORGE 逻辑代码保持你最新的修改不变 ...
        
        # 在计算出 max_conflict 之后，return 之前：
        
        # [Add] 记录 Max Conflict (监控 FORGE 效果)
        self.accelerator.log({
            "monitor/max_conflict": max_conflict,
            "epoch": epoch_idx
        })
        
        return all_item_tokens
```

**C. 验证/测试循环 (`_test_epoch`)**

```python
    def _test_epoch(self, ...):
        # ... 计算 metrics ...
        
        # 在 return metrics 之前：
        
        # [Add] 记录评估指标
        if self.accelerator.is_main_process:
            self.accelerator.log({
                f"eval/{k}": v for k, v in metrics.items()
            })
            # 同时记录当前的 step/epoch
            self.accelerator.log({"epoch": self.epochs}) # 注意这里可能需要传入当前的 epoch_idx
            
        return metrics
```

-----

### 💡 核心监控指标说明 (发给程序员参考)

| 阶段 | 指标 Key (WandB) | 含义 | 理想趋势 | 作用 |
| :--- | :--- | :--- | :--- | :--- |
| **预训练** | `val/collision_rate` | 碰撞率 | 接近 0 | 判断 RQ-VAE 是否区分开了物品 |
| **预训练** | `train/loss_recon` | 重构损失 | 下降 | 判断语义是否丢失 |
| **预训练** | `val/gini_coefficient` | 基尼系数 | \< 0.6 | 监控 Codebook 是否坍塌 |
| **联合训练** | `monitor/max_conflict` | 最大后缀冲突 | **\< 256** | **验证 FORGE 策略是否生效的核心指标** |
| **联合训练** | `train/rec_code_loss` | T5生成损失 | 下降 | T5 是否学会了预测 ID |
| **联合训练** | `eval/ndcg@10` | 推荐准确率 | 上升 | 最终模型效果 |
