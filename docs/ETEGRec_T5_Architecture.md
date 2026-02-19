# ETEGRec: 基于 T5 的端到端生成式推荐系统

> End-to-End Generative Recommender with Learnable Item Tokenization

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                       ETEGRec 架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [用户历史序列: item codes]                                       │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────┐                                │
│  │   SoftEmbedding (查表)       │  codes → token_embeddings     │
│  └─────────────┬───────────────┘                                │
│                ▼                                                 │
│  ┌─────────────────────────────┐                                │
│  │       T5 Encoder            │  6层 Transformer               │
│  │       (d_model=128)         │  Mean Pooling → 序列表示       │
│  └─────────────┬───────────────┘                                │
│                │                                                 │
│       ┌────────┼────────┐                                       │
│       ▼        │        ▼                                       │
│   [SIA]        │    [PSA]                                       │
│  enc_adapter   │   dec_adapter                                  │
│  [128d]        │   [1024d]                                      │
│       │        │        │                                       │
│       ▼        ▼        ▼                                       │
│  KL Loss   T5 Decoder  InfoNCE                                  │
│            (自回归)                                              │
│                │                                                 │
│                ▼                                                 │
│         MatMul(token_emb.T)                                     │
│                │                                                 │
│                ▼                                                 │
│           Code Logits → CE Loss                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 核心组件

### 2.1 模型架构

| 组件 | 配置 |
|------|------|
| **T5 Encoder** | 6层, d_model=128, d_ff=512, 4 heads |
| **T5 Decoder** | 6层, 自回归生成 codes |
| **token_embeddings** | 4个独立 Embedding(256, 128)，每层一个 |
| **enc_adapter** | MLP [128 → 128]，用于 SIA |
| **dec_adapter** | MLP [128 → 1024]，用于 PSA |
| **semantic_embedding** | Embedding(n_items, 1024)，冻结 |

### 2.2 Item Tokenizer (RQ-VAE)

```
┌─────────────────────────────────────────────────────────────────┐
│                      RQ-VAE 结构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [语义 Embedding: 1024d]                                         │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────┐                                │
│  │       Encoder               │  MLP: 1024→512→256→128         │
│  └─────────────┬───────────────┘                                │
│                ▼                                                 │
│  ┌─────────────────────────────┐                                │
│  │   Residual Quantization     │  3层 VQ，每层 256 codes        │
│  │   Layer 1: x → c1, r1       │                                │
│  │   Layer 2: r1 → c2, r2      │                                │
│  │   Layer 3: r2 → c3, r3      │                                │
│  └─────────────┬───────────────┘                                │
│                ▼                                                 │
│  ┌─────────────────────────────┐                                │
│  │       Decoder               │  MLP: 128→256→512→1024         │
│  └─────────────┬───────────────┘                                │
│                ▼                                                 │
│  [重建 Embedding: 1024d]                                         │
│                                                                  │
│  输出: [c1, c2, c3, suffix] (4个 codes)                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 数据格式

### 输入格式
```python
# DataLoader 输出
{
    'input_ids': [B, seq_len],      # 历史 item IDs
    'attention_mask': [B, seq_len], # padding mask
    'targets': [B, 1]               # 目标 item ID
}

# Trainer 内部转换为 codes
input_ids = all_item_code[input_ids].view(B, -1)  # [B, seq_len * 4]
labels = all_item_code[targets].view(B, -1)       # [B, 4]
```

### Code 结构
```
每个 item 用 4 个 codes 表示:
[c0, c1, c2, c3] = [RQ层1, RQ层2, RQ层3, suffix]

suffix: 用于处理 code 冲突 (FORGE 策略)
```

## 4. 训练流程

### 4.1 交替训练策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    交替训练 (cycle=2)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Epoch 0, 2, 4...  ──►  Train Tokenizer (RQ-VAE)                │
│       │                    - 冻结 T5                             │
│       │                    - 更新 RQ-VAE codebook               │
│       │                    - 重新计算 all_item_code              │
│       ▼                                                          │
│  Epoch 1, 3, 5...  ──►  Train Recommender (T5)                  │
│                            - 冻结 RQ-VAE                         │
│                            - 更新 T5 + token_embeddings          │
│                            - 更新 adapters                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Loss 组成

| Loss | 公式 | 作用 | 权重 |
|------|------|------|------|
| **Code Loss** | CE(logits, labels) | 生成正确的 codes | 1.0 |
| **VQ Loss** | MSE(recon, input) + β·commit | RQ-VAE 重建 | 1.0 (train_id) |
| **SIA Loss** | KL(seq_logits, target_logits) | 序列-目标 code 分布对齐 | 0.0001 |
| **PSA Loss** | InfoNCE(dec_latents, target_emb) | 序列-目标 语义对齐 | 0.0003 |

### 4.3 关键实现

```python
# Forward 流程
def forward(self, input_ids, attention_mask, labels):
    # 1. SoftEmbedding: 按层查表
    inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
    
    # 2. T5 Encoder-Decoder
    outputs = self.model(
        inputs_embeds=inputs_embeds,
        decoder_inputs_embeds=decoder_embeds,
        output_hidden_states=True
    )
    
    # 3. 序列表示: Encoder Mean Pooling
    seq_latents = encoder_hidden.mean(dim=1)  # [B, 128]
    seq_project_latents = self.enc_adapter(seq_latents)  # [B, 128] for SIA
    
    # 4. 解码表示: Decoder 第一个位置
    dec_latents = decoder_hidden[:, 0, :]  # [B, 128]
    dec_latents = self.dec_adapter(dec_latents)  # [B, 1024] for PSA
    
    # 5. 生成 Logits: 点积 token_embeddings
    code_logits = []
    for i in range(code_length):
        centroid = self.token_embeddings[i].weight.t()  # [128, 256]
        logits = decoder_hidden[:, i] @ centroid  # [B, 256]
        code_logits.append(logits)
    
    return code_logits, seq_project_latents, dec_latents
```

## 5. 推理流程

```python
# Beam Search 生成
def generate(self, input_ids, attention_mask, num_beams=20):
    # 1. Encoder 编码历史序列
    encoder_outputs = self.get_encoder()(inputs_embeds)
    
    # 2. Decoder 自回归生成
    for step in range(code_length):
        outputs = self.forward(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids
        )
        # Beam Search 选择 top-k
        next_tokens = beam_search_step(outputs.logits)
        decoder_input_ids = cat([decoder_input_ids, next_tokens])
    
    return decoder_input_ids[:, 1:]  # 去掉 start token
```

## 6. 配置示例

```yaml
# T5 模型配置
encoder_layers: 6
decoder_layers: 6
d_model: 128
d_ff: 512
num_heads: 4
d_kv: 64

# Item Tokenizer 配置
code_num: 256
code_length: 4
num_emb_list: [256, 256, 256]  # RQ-VAE 3层
e_dim: 128
layers: [1024, 512, 256]

# 训练配置
epochs: 400
batch_size: 512
lr_rec: 0.005      # T5 学习率
lr_id: 0.0001      # RQ-VAE 学习率
cycle: 2           # 交替周期
warm_epoch: 10     # 预热期 (只训练 VQ Loss)

# Loss 权重
rec_code_loss: 1
rec_kl_loss: 0.0001
rec_dec_cl_loss: 0.0003
```

## 7. 快速开始

```bash
# 单卡训练
python main.py --config ./config/Instrument2018_5090.yaml

# 多卡训练
accelerate launch --config_file accelerate_config_ddp.yaml \
    main.py --config ./config/Instrument2018_5090.yaml

# GRPO 强化学习微调
bash run_grpo.sh
```

## 8. 文件结构

```
├── model.py          # T5 推荐模型 (Model 类)
├── vq.py             # RQ-VAE 实现
├── data.py           # Dataset 和 Collator
├── trainer.py        # 训练器
├── main.py           # 入口文件
├── train_grpo.py     # GRPO 训练
└── config/
    └── Instrument2018_5090.yaml
```

## 9. 关键设计细节

### 9.1 SoftEmbedding (按层查表)

```python
def get_input_embeddings(self, input_ids, attention_mask):
    # input_ids: [B, seq_len * code_length]
    # 每 code_length 个位置对应一个 item
    for i in range(self.code_length):
        # 第 i 层的 codes 在位置 i, i+4, i+8, ...
        inputs_embeds[:, i::self.code_length] = \
            self.token_embeddings[i](input_ids[:, i::self.code_length])
```

### 9.2 FORGE 冲突处理

```python
# 使用 suffix (第4层) 处理 code 冲突
# 相同 prefix [c0, c1, c2] 的 items 分配不同 suffix
prefix_groups = defaultdict(list)
for idx, code in enumerate(all_item_prefix):
    prefix_key = tuple(code[:-1])
    prefix_groups[prefix_key].append(idx)

for key, items in prefix_groups.items():
    start_offset = hash(key) % code_num
    for i, item_idx in enumerate(items):
        suffix = (start_offset + i) % code_num
        all_item_code[item_idx][-1] = suffix
```

### 9.3 Dual SCID (双重语义)

```python
# 拼接协同过滤 embedding 和文本 embedding
collab_emb = np.load("collab_emb_256.npy")   # [n_items, 256]
text_emb = np.load("text_emb_768.npy")       # [n_items, 768]
semantic_emb = np.concatenate([collab_emb, text_emb], axis=-1)  # [n_items, 1024]
```

## 10. 性能指标

在 Instrument2018 数据集上的表现：

| 阶段 | Recall@10 | NDCG@10 |
|------|-----------|---------|
| Pre-train | ~0.08 | ~0.04 |
| Finetune | ~0.10 | ~0.05 |
| +GRPO | ~0.11 | ~0.06 |

---

**参考论文**: [Generative Recommender with End-to-End Learnable Item Tokenization](https://doi.org/10.1145/3726302.3729989)
