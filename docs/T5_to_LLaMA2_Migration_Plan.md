# ETEGRec: T5 → LLaMA2-7B-HF 迁移方案 (Final v2.2)

> **目标设备**: RTX 5090 32GB × N 卡  
> **参考**: Align3GR, MiniOneRec, OpenOneRec

---

## 1. 核心设计决策

| 设计点 | 决策 |
|-------|------|
| **Embedding** | SoftEmbedding + Projector (128 → 4096) |
| **SIA/PSA 位置** | `[SEP]`（DataLoader 预计算） |
| **LoRA** | q/k/v/o_proj，不含 embed_tokens |
| **显存优化** | Gradient Checkpointing + bf16 |

---

## 2. 关键代码实现

### 2.1 SoftEmbedding + Projector

```python
class LlamaRecModel(nn.Module):
    def __init__(self, config, rqvae):
        super().__init__()
        
        # LLaMA 加载
        self.llama = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Gradient Checkpointing（5090 必须开启）
        self.llama.gradient_checkpointing_enable()
        
        # RQ-VAE
        self.rqvae = rqvae
        for cb in self.rqvae.codebooks:
            cb.requires_grad_(True)
        
        # ⭐ Projector: Codebook dim → LLaMA dim
        self.codebook_dim = self.rqvae.codebooks[0].embedding_dim  # 128
        self.hidden_size = self.llama.config.hidden_size  # 4096
        self.scid_projector = nn.Linear(self.codebook_dim, self.hidden_size, bias=False)
        self.scid_projector.to(torch.bfloat16)
        
        # LoRA（不含 embed_tokens）
        lora_config = LoraConfig(
            r=64, lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=[],
        )
        self.llama = get_peft_model(self.llama, lora_config)
        
        # 对齐层
        self.enc_adapter = MLPLayers([self.hidden_size, config['e_dim']])
        self.dec_adapter = MLPLayers([self.hidden_size, config['semantic_dim']])

    def get_input_embeddings(self, input_ids):
        """SoftEmbedding: SCID → Codebook → Projector → LLaMA dim"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        embeddings = torch.zeros(
            batch_size, seq_len, self.hidden_size,
            dtype=torch.bfloat16, device=device
        )
        
        # 1. Text tokens → Frozen LLaMA Embedding
        text_mask = input_ids < self.scid_token_start
        if text_mask.any():
            with torch.no_grad():
                embeddings[text_mask] = self.llama.model.model.embed_tokens(
                    input_ids[text_mask]
                )
        
        # 2. SCID tokens → Codebook → Projector
        scid_mask = ~text_mask
        if scid_mask.any():
            scid_values = input_ids.clone()
            scid_values[text_mask] = self.scid_token_start
            
            relative_ids = scid_values - self.scid_token_start
            level_idx = relative_ids // self.code_number
            code_idx = relative_ids % self.code_number
            
            for level in range(self.code_length):
                current_mask = scid_mask & (level_idx == level)
                if current_mask.any():
                    codes = code_idx[current_mask]
                    raw_embeds = self.rqvae.codebooks[level](codes)  # [N, 128]
                    proj_embeds = self.scid_projector(raw_embeds)    # [N, 4096]
                    embeddings[current_mask] = proj_embeds
        
        return embeddings

    def forward(self, input_ids, attention_mask, sep_indices, labels=None, target_item_emb=None):
        # Gradient Checkpointing 兼容性
        inputs_embeds = self.get_input_embeddings(input_ids)
        if self.training:
            inputs_embeds.requires_grad_(True)
            inputs_embeds.register_hook(lambda x: x)  # Dummy hook
        
        # LLaMA Forward
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 提取 [SEP] 位置（由 DataLoader 传入）
        last_hidden = outputs.hidden_states[-1]
        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        seq_hidden = last_hidden[batch_indices, sep_indices]
        
        # 对齐 Loss
        sia_loss = self.compute_sia(self.enc_adapter(seq_hidden), target_item_emb)
        psa_loss = self.compute_psa(self.dec_adapter(seq_hidden), target_item_emb)
        
        # 生成 Loss
        gen_loss = self.compute_gen_loss(outputs.logits, labels) if labels is not None else 0
        
        return gen_loss + 0.0001 * sia_loss + 0.0003 * psa_loss
```

### 2.2 DataLoader 返回 sep_indices

```python
# data_llama.py
class LlamaRecDataset(Dataset):
    def __getitem__(self, idx):
        # ... 构造 prompt ...
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sep_indices': len(history_tokens),  # ⭐ 预计算
            'labels': target_codes,
            'target_item_emb': item_embedding
        }
```

---

## 3. 5090 32GB 配置

```yaml
# config/llama_5090.yaml
precision: bf16
batch_size_per_gpu: 4
gradient_accumulation: 4
gradient_checkpointing: true

lora_r: 64
lora_alpha: 128

learning_rate: 1e-4
epochs: 50
```

---

## 4. 启动前 Checklist

- [ ] Projector 层已添加 (`nn.Linear(128, 4096)`)
- [ ] DataLoader 返回 `sep_indices`
- [ ] Gradient Checkpointing 已开启
- [ ] 所有层 dtype 统一为 `bfloat16`
- [ ] 运行梯度健全性检查

---

## 5. 梯度检查脚本

```python
def sanity_check(model, batch):
    model.train()
    loss = model(**batch)
    loss.backward()
    
    print("=== Gradient Check ===")
    for i, cb in enumerate(model.rqvae.codebooks):
        grad = cb.weight.grad
        if grad is None:
            print(f"❌ Codebook[{i}] grad is None")
        elif grad.abs().sum() == 0:
            print(f"⚠️ Codebook[{i}] grad is zero")
        else:
            print(f"✅ Codebook[{i}] norm: {grad.norm():.6f}")
    
    proj_grad = model.scid_projector.weight.grad
    print(f"✅ Projector grad norm: {proj_grad.norm():.6f}" if proj_grad is not None else "❌ Projector grad None")
```

---

## 6. 实施优先级

| Phase | 任务 | 工时 |
|-------|------|------|
| 🔴 P0 | SoftEmbedding + Projector + 梯度检查 | 2天 |
| 🔴 P0 | DataLoader (含 sep_indices) | 1天 |
| 🟡 P1 | 训练循环 + 多卡 | 1天 |
| 🟡 P1 | SIA/PSA 对齐 | 1天 |
| 🟢 P2 | 评估 | 1天 |

**总计: 6-7 天**
