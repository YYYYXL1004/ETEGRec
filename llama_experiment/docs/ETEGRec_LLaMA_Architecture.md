# ETEGRec-LLaMA: 基于 LLaMA2-7B 的生成式推荐系统

> 将 ETEGRec 从 T5 迁移到 LLaMA2-7B，实现端到端的生成式推荐

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      ETEGRec-LLaMA 架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [用户历史序列: item codes]                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────┐                                │
│  │   SoftEmbedding (查表)       │  codes → codebook → 128d      │
│  └─────────────┬───────────────┘                                │
│                ▼                                                 │
│        ┌───────────────┐                                        │
│        │ scid_projector │  128d → 4096d                         │
│        └───────┬───────┘                                        │
│                ▼                                                 │
│  ┌─────────────────────────────┐                                │
│  │     LLaMA2-7B (LoRA)        │  Gradient Checkpointing        │
│  │     attn: SDPA              │  bf16 精度                      │
│  └─────────────┬───────────────┘                                │
│                │                                                 │
│       ┌────────┼────────┐                                       │
│       ▼        ▼        ▼                                       │
│   [SIA]    [PSA]   [Generation]                                 │
│     │        │          │                                       │
│     ▼        ▼          ▼                                       │
│  enc_adapter dec_adapter output_projector                       │
│  [128d]    [1024d]      [128d]                                  │
│     │        │          │                                       │
│     ▼        ▼          ▼                                       │
│  KL Loss  InfoNCE   MatMul(Codebook.T)                          │
│                         │                                       │
│                         ▼                                       │
│                    Code Logits → CE Loss                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 核心设计

### 2.1 T5 vs LLaMA 对比

| 特性 | T5 (原版) | LLaMA2 (本实现) |
|------|----------|----------------|
| 架构 | Encoder-Decoder | Decoder-only |
| 序列表示 | Encoder Mean Pool | Last Token |
| 生成方式 | Encoder缓存 + Decoder自回归 | 纯自回归 |
| 参数量 | ~220M | ~7B (LoRA微调) |

### 2.2 关键技术点

| 设计点 | 方案 |
|-------|------|
| **Input Embedding** | SoftEmbedding: codes → Codebook查表 → Projector(128→4096) |
| **Output Logits** | Weight Tying: Projector(4096→128) → MatMul(Codebook.T) |
| **序列表示** | Last Token (历史序列最后一个位置) |
| **Codebook** | 独立副本 + Trainer手动同步 (DDP兼容) |
| **微调策略** | LoRA (q/k/v/o_proj, r=64, α=128) |
| **显存优化** | Gradient Checkpointing + 8-bit AdamW + 分批Generate |

## 3. 数据格式

### Prompt 结构
```
<c0_151> <c1_19> <c2_62> <c3_0> | <c0_74> <c1_44> ... | <c0_?> <c1_?> <c2_?> <c3_?>
|<--------- Item 1 --------->|   |<---- Item 2 ---->|   |<-- 目标 Item (预测) -->|
                                                    ↑
                                          seq_end_position
```

- 每个 item 用 4 个 codes 表示 (RQ-VAE 3层 + suffix 1层)
- `seq_end_position`: 历史序列最后一个 token 的位置
- 左 Padding，padding value = -1

## 4. 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│                    交替训练策略                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Epoch 0, 2, 4...  ──►  Train Tokenizer (RQ-VAE)            │
│       │                    - 冻结 LLaMA                      │
│       │                    - 更新 Codebook                   │
│       │                    - 重新计算 item codes             │
│       ▼                                                      │
│  Epoch 1, 3, 5...  ──►  Train Recommender (LLaMA)           │
│                            - 冻结 RQ-VAE                     │
│                            - 更新 LoRA + Projectors          │
│                            - Codebook 梯度流通               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Loss 组成

| Loss | 作用 | 权重 |
|------|------|------|
| **Code Loss** | 生成正确的 item codes | 1.0 |
| **SIA Loss** | 序列-目标 code分布对齐 (KL散度) | 0.0001 |
| **PSA Loss** | 序列-目标 语义对齐 (InfoNCE) | 0.0003 |

## 5. 推理流程

```python
# Beam Search 生成
for code_idx in range(4):  # 4个codes
    # 1. LLaMA forward (分批避免OOM)
    hidden = llama(current_embeds)[:, -1, :]
    
    # 2. 投影 + 点积 Codebook
    query = output_projector(hidden)  # [B, 128]
    logits = query @ codebook[code_idx].T  # [B, 256]
    
    # 3. Beam Search 更新
    next_codes = topk(log_softmax(logits) + beam_scores)
    
    # 4. 拼接新 embedding 继续生成
    current_embeds = cat([current_embeds, embed(next_codes)])
```

## 6. 配置示例

```yaml
# 模型配置
llama_path: models/Llama-2-7b-hf
lora_r: 64
lora_alpha: 128
code_num: 256
code_length: 4
e_dim: 128

# 训练配置
batch_size: 16
eval_batch_size: 4
lr_rec: 0.0001
epochs: 50
cycle: 2  # 每2个epoch切换训练目标

# 显存优化
gradient_checkpointing: true
num_beams: 15
generate_chunk_size: 4
```

## 7. 快速开始

```bash
# 单卡训练
python main_llama.py --config ./config/llama_instrument2018.yaml

# 多卡训练
accelerate launch --config_file accelerate_config_llama.yaml \
    main_llama.py --config ./config/llama_instrument2018.yaml

# Debug 模式 (小数据集验证)
python main_llama.py --config ./config/llama_instrument2018.yaml \
    --debug --debug_samples 1000
```

## 8. 文件结构

```
├── model_llama.py      # LlamaRecModel 定义
├── data_llama.py       # Dataset 和 Collator
├── trainer_llama.py    # 训练器
├── main_llama.py       # 入口文件
└── config/
    └── llama_instrument2018.yaml  # 配置文件
```

## 9. 关键实现细节

### 9.1 Causal LM 位置修复
```python
# 预测第 i 个 code 时，使用第 i-1 个位置的 hidden state
if i == 0:
    pos = seq_end_positions  # 历史最后一个
else:
    pos = target_positions[:, i - 1]  # 前一个目标位置
```

### 9.2 Code Table 动态同步
```python
# RQ-VAE 更新后，同步 codes 到数据集
def _sync_code_table_to_datasets(self):
    for dataloader in [train, valid, test]:
        dataloader.dataset.all_item_code = self.all_item_code.cpu()
```

### 9.3 DDP 兼容
```python
# 独立 Codebook 副本，避免共享参数问题
self.codebook_embeddings = nn.ModuleList([
    nn.Embedding(256, 128) for _ in range(3)
])
# Trainer 中手动同步: model_rec ↔ model_id
```

## 10. 性能指标

在 Instrument2018 数据集上的表现：

| 指标 | 值 |
|------|-----|
| Recall@10 | - |
| NDCG@10 | - |
| 训练显存 | ~28GB (5090) |
| 推理速度 | ~X samples/s |

---

**参考论文**: ETEGRec, Align3GR, MiniOneRec, OpenOneRec
