# ETEGRec: T5 → LLaMA2-7B-HF 迁移方案 (Final v3.2)

> **目标设备**: RTX 5090 32GB × N 卡  
> **参考**: Align3GR, MiniOneRec, OpenOneRec

---

## 0. 核心架构差异说明

| 特性 | T5 (原) | LLaMA2 (目标) |
|------|---------|---------------|
| 架构 | Encoder-Decoder | Decoder-only |
| SIA提取 | `encoder_last_hidden_state` Mean Pool | **Last Token** (seq_end_pos) |
| PSA提取 | `decoder_hidden_states[-1][:, 0]` | **Last Token** (seq_end_pos) |
| 生成Logits | decoder输出 × codebook.T | **Output Proj → MatMul(Codebook.T)** |
| 生成方式 | encoder缓存 + decoder自回归 | 纯自回归续写 |

### ⭐ v3.1 关键修正 (vs v3.0)

1. **移除 `code_heads`**：改回 Weight Tying (点积 Codebook)，保证 End-to-End 梯度流
2. **SIA 改用 Last Token**：符合 Causal LM 特性，去掉 Mean Pooling
3. **显式管理 Projector 梯度**：确保自定义层参与训练

### ⭐ v3.2 工程优化 (vs v3.1)

1. **Projector 初始化**：使用小方差初始化 (std=0.02)，防止梯度爆炸/消失
2. **澄清 SEQ_END 实现**：不需要显式特殊token，仅通过位置索引区分历史/目标

---

## 1. 核心设计决策

| 设计点 | 决策 |
|-------|------|
| **Input Embedding** | SoftEmbedding + scid_projector (128 → 4096)，按间隔查表 |
| **Output Logits** | output_projector (4096 → 128) + MatMul(Codebook.T) ⭐ Weight Tying |
| **SIA/PSA 位置** | Last Token (seq_end_pos)，不用 Mean Pooling |
| **LoRA** | q/k/v/o_proj，显式训练 Projectors |
| **显存优化** | Gradient Checkpointing + bf16 |

---

## 2. Prompt 格式设计

```
<c0_151> <c1_19> <c2_62> <c3_0> | <c0_74> <c1_44> ... | <c0_?> <c1_?> <c2_?> <c3_?>
|<--------- Item 1 --------->|   |<---- Item 2 ---->|   |<-- 目标 Item (预测) -->|
                                                    ↑
                                          seq_end_position (位置索引)
```

### ⭐ 关键澄清：不需要显式 `[SEQ_END]` Token！

- **实现方式**：纯 codes 序列，通过**位置索引**区分历史/目标
- **seq_end_position**：历史序列最后一个 token 的位置索引
- **无需 resize 词表**：不引入新 token，零显存开销
- **SIA/PSA 提取**：直接用 `hidden_states[batch_idx, seq_end_position]`

```python
# DataLoader 中的位置计算
seq_end_position = len(history_codes) - 1  # 最后一个历史 code 的位置
target_positions = range(len(history_codes), len(history_codes) + code_length)
```

---

## 3. 关键代码实现

### 3.1 模型定义

```python
class LlamaRecModel(nn.Module):
    def __init__(self, config, llama_model, rqvae):
        super().__init__()
        
        # === LLaMA 基座 ===
        self.llama = llama_model
        self.llama.gradient_checkpointing_enable()
        
        # === RQ-VAE ===
        self.rqvae = rqvae
        self.code_length = config['code_length']  # 4 (3 from RQVAE + 1 collision)
        self.code_num = config['code_num']  # 256
        
        # === 维度配置 ===
        self.codebook_dim = config['e_dim']  # 128
        self.hidden_size = llama_model.config.hidden_size  # 4096
        self.semantic_dim = config['semantic_hidden_size']  # 256
        
        # === Input Projector: Codebook → LLaMA ===
        self.scid_projector = nn.Linear(self.codebook_dim, self.hidden_size, bias=False)
        
        # === ⭐ Output Projector: LLaMA → Codebook (用于点积) ===
        # 不用 code_heads！保持 Weight Tying 让梯度流向 Codebook
        self.output_projector = nn.Linear(self.hidden_size, self.codebook_dim, bias=False)
        
        # === 对齐层 ===
        self.enc_adapter = MLPLayers([self.hidden_size, self.codebook_dim])  # SIA
        self.dec_adapter = MLPLayers([self.hidden_size, self.semantic_dim])  # PSA
        
        # === 语义 Embedding (冻结) ===
        self.semantic_embedding = nn.Embedding(config['n_items'], self.semantic_dim)
        self.semantic_embedding.requires_grad_(False)
        
        # === LoRA ===
        lora_config = LoraConfig(
            r=64, lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=[],  # Projectors 不在 llama 内部，无需加入
        )
        self.llama = get_peft_model(self.llama, lora_config)
        
        # === ⭐ 显式确保自定义层参与训练 ===
        self.scid_projector.requires_grad_(True)
        self.output_projector.requires_grad_(True)
        for param in self.enc_adapter.parameters():
            param.requires_grad_(True)
        for param in self.dec_adapter.parameters():
            param.requires_grad_(True)
        
        # === ⭐ v3.2: 初始化自定义层 (防止梯度爆炸/消失) ===
        self._init_custom_weights()
    
    def _init_custom_weights(self):
        """
        使用小方差初始化 Projector 层
        防止训练初期 Logits 过大导致 Softmax 变 one-hot (梯度消失)
        或 Logits 过小导致学习缓慢
        """
        # Projectors
        torch.nn.init.normal_(self.scid_projector.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.output_projector.weight, mean=0.0, std=0.02)
        
        # Adapters (如果是 nn.Linear)
        for module in [self.enc_adapter, self.dec_adapter]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)

    def get_codebooks(self):
        """获取 RQ-VAE 的码本 Embedding 层列表"""
        return self.rqvae.rq.vq_layers

    def get_input_embeddings(self, input_ids, attention_mask):
        """
        SoftEmbedding: 按间隔查表，与原 T5 版本逻辑一致
        
        input_ids 布局: [c0, c1, c2, c3, c0, c1, c2, c3, ...]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        embeddings = torch.zeros(
            batch_size, seq_len, self.hidden_size,
            dtype=torch.bfloat16, device=device
        )
        
        # 处理 padding (-1 → 0)
        input_ids_safe = input_ids.clone()
        input_ids_safe[input_ids == -1] = 0
        
        # 按间隔查表
        codebooks = self.get_codebooks()
        for level in range(self.code_length):
            # 取每隔 code_length 的位置
            codes_at_level = input_ids_safe[:, level::self.code_length]  # [B, seq_len/K]
            
            # 从对应层的码本查 embedding
            raw_embeds = codebooks[level].embedding(codes_at_level)  # [B, seq_len/K, 128]
            
            # 投影到 LLaMA 维度
            proj_embeds = self.scid_projector(raw_embeds)  # [B, seq_len/K, 4096]
            
            # 放回对应位置
            embeddings[:, level::self.code_length] = proj_embeds
        
        # Padding 位置置零
        padding_mask = ~attention_mask.bool()
        embeddings[padding_mask] = 0
        
        return embeddings

    def forward(self, input_ids, attention_mask, seq_end_positions, 
                target_positions, labels=None, targets=None):
        """
        Args:
            input_ids: [B, total_len] - 历史 + 目标的 codes
            attention_mask: [B, total_len]
            seq_end_positions: [B] - 历史序列结束位置 (用于 SIA/PSA)
            target_positions: [B, code_length] - 目标 code 各位置的索引
            labels: [B, code_length] - 目标 item 的真实 code
            targets: [B] - 目标 item ID (用于 SIA/PSA)
        """
        # === 1. 获取输入嵌入 ===
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
        
        if self.training:
            inputs_embeds = inputs_embeds.requires_grad_(True)
        
        # === 2. LLaMA Forward ===
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[-1]  # [B, L, 4096]
        batch_size = hidden_states.size(0)
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        
        # === 3. ⭐ SIA/PSA: Last Token (不用 Mean Pooling) ===
        # Causal LM 中，最后一个 token 已经看过所有历史，信息最丰富
        last_hidden = hidden_states[batch_indices, seq_end_positions]  # [B, 4096]
        
        seq_project_latents = self.enc_adapter(last_hidden)  # [B, 128] for SIA
        dec_latents = self.dec_adapter(last_hidden)  # [B, 256] for PSA
        
        # === 4. ⭐ 生成 Logits: Weight Tying (点积 Codebook) ===
        code_logits = []
        codebooks = self.get_codebooks()
        
        for i in range(self.code_length):
            pos_i = target_positions[:, i]  # [B]
            hidden_at_pos = hidden_states[batch_indices, pos_i]  # [B, 4096]
            
            # Step 1: 投影回 Codebook 维度
            query_emb = self.output_projector(hidden_at_pos)  # [B, 128]
            
            # Step 2: 与第 i 层 Codebook 做点积 (Weight Tying!)
            # codebook.embedding.weight: [256, 128] → 转置 → [128, 256]
            codebook_weight = codebooks[i].embedding.weight.t()  # [128, 256]
        
            # Step 3: 计算相似度 logits
            logits = torch.matmul(query_emb, codebook_weight)  # [B, 256]
            code_logits.append(logits)
        
        code_logits = torch.stack(code_logits, dim=1)  # [B, code_length, code_num]
        
        return QuantizeOutput(
            logits=code_logits,
            seq_latents=last_hidden,  # 原始 hidden，供调试用
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents
        )

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, seq_end_positions,
                 num_beams=20, num_return_sequences=10):
        """
        自回归生成目标 item 的 codes
        使用 Beam Search，每步用 output_projector + Codebook 点积
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        codebooks = self.get_codebooks()
        
        # Beam Search 初始化
        input_ids_expanded = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask_expanded = attention_mask.repeat_interleave(num_beams, dim=0)
        
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        
        generated_codes = []
        current_embeds = self.get_input_embeddings(input_ids_expanded, attention_mask_expanded)
        
        beam_idx_offset = torch.arange(batch_size, device=device).repeat_interleave(num_beams) * num_beams
        
        for code_idx in range(self.code_length):
            # Forward
            outputs = self.llama(
                inputs_embeds=current_embeds,
                attention_mask=attention_mask_expanded,
                output_hidden_states=True
            )
            
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B*beams, 4096]
            
            # 投影 + 点积 Codebook
            query_emb = self.output_projector(last_hidden)  # [B*beams, 128]
            codebook_weight = codebooks[code_idx].embedding.weight.t()  # [128, 256]
            logits = torch.matmul(query_emb, codebook_weight)  # [B*beams, 256]
            
            # Beam Search 更新
            log_probs = F.log_softmax(logits, dim=-1)
            next_scores = log_probs + beam_scores.unsqueeze(-1)
            
            vocab_size = log_probs.size(-1)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=-1)
            
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_codes = next_tokens % vocab_size
            
            beam_scores = next_scores.view(-1)
            
            # 记录生成的 code
            generated_codes.append(next_codes)
            
            # 准备下一步的 embedding
            beam_idx = (next_indices + beam_idx_offset.view(batch_size, num_beams)).view(-1)
            current_embeds = current_embeds[beam_idx]
            attention_mask_expanded = attention_mask_expanded[beam_idx]
            
            # 添加新生成的 code embedding
            next_codes_flat = next_codes.view(-1)
            next_embeds = codebooks[code_idx].embedding(next_codes_flat)  # [B*beams, 128]
            next_embeds = self.scid_projector(next_embeds).unsqueeze(1)  # [B*beams, 1, 4096]
            
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
            attention_mask_expanded = torch.cat([
                attention_mask_expanded,
                torch.ones(attention_mask_expanded.size(0), 1, device=device, dtype=attention_mask_expanded.dtype)
            ], dim=1)
        
        # 整理输出
        generated_codes = torch.stack(generated_codes, dim=-1)  # [B, beams, code_length]
        return generated_codes[:, :num_return_sequences, :]
```

### 3.2 DataLoader 设计

```python
class LlamaRecDataset(Dataset):
    def __init__(self, data, all_item_code, code_length=4, max_seq_len=50):
        self.data = data
        self.all_item_code = all_item_code  # [n_items+1, code_length]
        self.code_length = code_length
        self.max_seq_len = max_seq_len
    
    def __getitem__(self, idx):
        user_seq = self.data[idx]['history']  # [L] item IDs
        target_item = self.data[idx]['target']  # item ID
        
        # 1. 构造历史序列的 codes
        history_codes = []
        for item_id in user_seq[-self.max_seq_len:]:
            item_codes = self.all_item_code[item_id]
            history_codes.extend(item_codes.tolist())
        
        # 2. 构造目标序列的 codes
        target_codes = self.all_item_code[target_item].tolist()
        
        # 3. 拼接
        input_ids = history_codes + target_codes
        
        # 4. 计算关键位置
        seq_end_position = len(history_codes) - 1  # 最后一个历史 token 的位置
        target_positions = list(range(len(history_codes), len(history_codes) + self.code_length))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
            'seq_end_position': seq_end_position,
            'target_positions': torch.tensor(target_positions, dtype=torch.long),
            'labels': torch.tensor(target_codes, dtype=torch.long),
            'target_item': target_item,
        }


def collate_fn(batch):
    """动态 Padding"""
    max_len = max(len(b['input_ids']) for b in batch)
    
    input_ids = []
    attention_mask = []
    seq_end_positions = []
    target_positions = []
    labels = []
    targets = []
    
    for b in batch:
        pad_len = max_len - len(b['input_ids'])
        # 左 Padding (LLaMA 习惯)
        input_ids.append(F.pad(b['input_ids'], (pad_len, 0), value=-1))
        attention_mask.append(F.pad(b['attention_mask'], (pad_len, 0), value=0))
        
        # 位置索引需要加上 padding 偏移
        seq_end_positions.append(b['seq_end_position'] + pad_len)
        target_positions.append(b['target_positions'] + pad_len)
        
        labels.append(b['labels'])
        targets.append(b['target_item'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'seq_end_positions': torch.tensor(seq_end_positions),
        'target_positions': torch.stack(target_positions),
        'labels': torch.stack(labels),
        'targets': torch.tensor(targets),
    }
```

### 3.3 Trainer 核心修改

```python
def _train_epoch_rec(self, epoch_idx, loss_w, verbose=True):
    """训练推荐器 (冻结 Tokenizer)"""
    self.model_rec.train()
    self.model_id.eval()
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        seq_end_positions = batch['seq_end_positions'].to(self.device)
        target_positions = batch['target_positions'].to(self.device)
        labels = batch['labels'].to(self.device)
        targets = batch['targets'].to(self.device)
        
        # 目标 item 的语义 embedding
        target_semantic_embs = self.model_rec.semantic_embedding(targets)
        target_recon_embs, _, _, _, target_code_logits = self.model_id(target_semantic_embs)
        
        # Forward
        outputs = self.model_rec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_end_positions=seq_end_positions,
            target_positions=target_positions,
        )
        
        # === Loss 计算 ===
        
        # 1. Code Loss (生成任务) - 梯度会流向 Codebook！
        code_loss = F.cross_entropy(
            outputs.logits.view(-1, self.code_num),
            labels.view(-1)
        )
        
        # 2. SIA Loss (KL 散度)
        _, _, _, _, seq_code_logits = self.model_id.rq(outputs.seq_project_latents)
        kl_loss = (
            self.compute_discrete_contrastive_loss_kl(seq_code_logits, target_code_logits) +
            self.compute_discrete_contrastive_loss_kl(target_code_logits, seq_code_logits)
        )
        
        # 3. PSA Loss (InfoNCE)
        dec_cl_loss = (
            self.compute_contrastive_loss(target_recon_embs, outputs.dec_latents) +
            self.compute_contrastive_loss(outputs.dec_latents, target_recon_embs)
        )
        
        # 总 Loss
        loss = (loss_w['code_loss'] * code_loss + 
                loss_w['kl_loss'] * kl_loss + 
                loss_w['dec_cl_loss'] * dec_cl_loss)
        
        self.accelerator.backward(loss)
        self.rec_optimizer.step()
        self.rec_lr_scheduler.step()
```

---

## 4. 5090 32GB 配置

```yaml
# config/llama_5090.yaml
model:
  base_model: "meta-llama/Llama-2-7b-hf"
precision: bf16
gradient_checkpointing: true

lora:
  r: 64
lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  batch_size_per_gpu: 2
  gradient_accumulation: 8
learning_rate: 1e-4
epochs: 50
  warmup_steps: 500

data:
  max_seq_len: 50
  code_length: 4
  code_num: 256
```

---

## 5. 启动前 Checklist

- [ ] `scid_projector` (128 → 4096) 已添加
- [ ] `output_projector` (4096 → 128) 已添加
- [ ] **移除了 `code_heads`**，改用 Codebook 点积
- [ ] SIA/PSA 使用 Last Token，移除 Mean Pooling
- [ ] DataLoader 返回 `seq_end_positions` 和 `target_positions`
- [ ] **不使用显式 SEQ_END token**，仅用位置索引
- [ ] Gradient Checkpointing 已开启
- [ ] 所有自定义层 `requires_grad_(True)`
- [ ] **Projector 初始化** (`std=0.02`)
- [ ] 运行梯度健全性检查

---

## 6. 梯度检查脚本

```python
def sanity_check(model, batch):
    """检查关键组件的梯度流，特别是 Codebook 是否收到梯度"""
    model.train()
    
    outputs = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        seq_end_positions=batch['seq_end_positions'],
        target_positions=batch['target_positions'],
    )
    
    loss = F.cross_entropy(
        outputs.logits.view(-1, 256),
        batch['labels'].view(-1)
    )
    loss.backward()
    
    print("=== Gradient Check (v3.1) ===")
    
    # ⭐ 关键：Codebook 必须有梯度！
    codebooks = model.get_codebooks()
    for i, cb in enumerate(codebooks):
        grad = cb.embedding.weight.grad
        if grad is None:
            print(f"❌ Codebook[{i}] grad is None - End-to-End 断裂!")
        elif grad.abs().sum() == 0:
            print(f"⚠️ Codebook[{i}] grad is zero")
        else:
            print(f"✅ Codebook[{i}] grad norm: {grad.norm():.6f}")
    
    # Projectors
    for name, proj in [("scid_projector", model.scid_projector), 
                       ("output_projector", model.output_projector)]:
        grad = proj.weight.grad
        if grad is not None:
            print(f"✅ {name} grad norm: {grad.norm():.6f}")
        else:
            print(f"❌ {name} grad is None")
    
    # Adapters
    enc_grad = list(model.enc_adapter.parameters())[0].grad
    dec_grad = list(model.dec_adapter.parameters())[0].grad
    print(f"✅ enc_adapter grad: {enc_grad.norm():.6f}" if enc_grad is not None else "❌ enc_adapter None")
    print(f"✅ dec_adapter grad: {dec_grad.norm():.6f}" if dec_grad is not None else "❌ dec_adapter None")
```

---

## 7. v3.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LlamaRecModel v3.1                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [Input: History Codes]                                                 │
│         │                                                               │
│         ▼                                                               │
│  ┌──────────────┐      ┌───────────────┐                               │
│  │  Codebook[i] │ ───► │ scid_projector│ ───► [128 → 4096]             │
│  │  (RQ-VAE)    │      │   (可训练)     │                               │
│  └──────────────┘      └───────────────┘                               │
│         │                     │                                         │
│         │                     ▼                                         │
│         │              ┌─────────────────┐                             │
│         │              │   LLaMA2-7B     │                             │
│         │              │   (LoRA)        │                             │
│         │              └────────┬────────┘                             │
│         │                       │                                       │
│         │           ┌───────────┼───────────┐                          │
│         │           │           │           │                          │
│         │           ▼           ▼           ▼                          │
│         │     [seq_end_pos] [seq_end_pos] [target_pos]                 │
│         │           │           │           │                          │
│         │           ▼           ▼           ▼                          │
│         │    ┌──────────┐ ┌──────────┐ ┌────────────────┐              │
│         │    │enc_adapter│ │dec_adapter│ │output_projector│             │
│         │    └────┬─────┘ └────┬─────┘ └───────┬────────┘              │
│         │         │            │               │                        │
│         │         ▼            ▼               ▼                        │
│         │    [B, 128]     [B, 256]        [B, 128]                      │
│         │         │            │               │                        │
│         │         │            │               │                        │
│         │         ▼            ▼               ▼                        │
│         │    ┌─────────┐ ┌─────────┐    ┌─────────────────┐            │
│         │    │SIA Loss │ │PSA Loss │    │ MatMul          │            │
│         │    │(KL Div) │ │(InfoNCE)│    │ (Codebook.T)    │◄─────┐     │
│         │    └─────────┘ └─────────┘    └────────┬────────┘      │     │
│         │                                        │               │     │
│         │                                        ▼               │     │
│         │                                   [B, 256]             │     │
│         │                                   (Logits)             │     │
│         │                                        │               │     │
│         │                                        ▼               │     │
│         │                                  ┌──────────┐          │     │
│         └──────────────────────────────────┤Code Loss │──────────┘     │
│                  ⭐ Weight Tying:          │(CE Loss) │                │
│                  梯度回传到 Codebook        └──────────┘                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. 实施优先级

| Phase | 任务 | 工时 |
|-------|------|------|
| 🔴 P0 | LlamaRecModel 实现 (含 Weight Tying) | 2天 |
| 🔴 P0 | DataLoader 重构 | 1.5天 |
| 🟡 P1 | Trainer 适配 | 1.5天 |
| 🟡 P1 | Generate (Beam Search) | 1.5天 |
| 🟢 P2 | 梯度调试 + 形状验证 | 1天 |
| 🟢 P2 | 评估 + 调优 | 2天 |

**总计: 9-10 天**

---

## 9. 风险点与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| 显存溢出 | 训练失败 | 降 batch，加 grad_accum，用 DeepSpeed ZeRO |
| Codebook 梯度消失 | End-to-End 失效 | 检查 output_projector 初始化，加梯度监控 |
| Logits 爆炸/NaN | 训练崩溃 | 减小初始化 std (0.02→0.01)，加 gradient clipping |
| 生成质量差 | 推荐效果下降 | 增加 beam_size，添加 prefix constraint |

---

## 10. 版本历史

| 版本 | 核心修改 |
|------|---------|
| v3.0 | 基础架构设计，SoftEmbedding + code_heads |
| v3.1 | 移除 code_heads，改用 Weight Tying；SIA 改用 Last Token |
| v3.2 | 添加 Projector 初始化；澄清不需要显式 SEQ_END token |
