# ETEGRec: T5 → LLaMA2-7B-HF 迁移方案 (Final v3.3)

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

### ⭐ 版本历史

| 版本 | 核心修改 |
|------|---------|
| v3.0 | 基础架构设计，SoftEmbedding + code_heads |
| v3.1 | 移除 code_heads，改用 Weight Tying；SIA 改用 Last Token |
| v3.2 | 添加 Projector 初始化；澄清不需要显式 SEQ_END token |
| **v3.3** | **实际工程实现：DDP兼容、独立Codebook、Suffix层、8-bit Adam、分批Generate** |

### ⭐ v3.3 关键修正 (vs v3.2)

1. **独立 Codebook 副本**：model_rec 持有从 rqvae 复制的独立 codebook，避免 DDP 共享参数问题
2. **Suffix Embedding**：第 4 层使用独立的 `suffix_embedding` 处理冲突计数
3. **8-bit AdamW**：LLaMA 使用 `bitsandbytes` 8-bit 优化器节省显存
4. **分批 Generate**：推理时分 chunk forward，避免 OOM
5. **Gradient Checkpointing**：使用 `use_reentrant=False` 解决 DDP 兼容性
6. **5090 补丁**：禁用 TF32，启用确定性算法

---

## 1. 核心设计决策

| 设计点 | 决策 |
|-------|------|
| **Input Embedding** | SoftEmbedding + scid_projector (128 → 4096)，按间隔查表 |
| **Output Logits** | output_projector (4096 → 128) + MatMul(Codebook.T) ⭐ Weight Tying |
| **SIA/PSA 位置** | Last Token (seq_end_pos)，不用 Mean Pooling |
| **Codebook 管理** | 独立副本 + Trainer 手动同步 (DDP 兼容) |
| **Suffix 处理** | 独立的 suffix_embedding (第4层) |
| **LoRA** | q/k/v/o_proj，显式训练 Projectors |
| **优化器** | 8-bit AdamW (LLaMA) + 普通 AdamW (RQ-VAE) |
| **显存优化** | Gradient Checkpointing (non-reentrant) + bf16 + 分批 Generate |

---

## 2. Prompt 格式设计

```
<c0_151> <c1_19> <c2_62> <c3_0> | <c0_74> <c1_44> ... | <c0_?> <c1_?> <c2_?> <c3_?>
|<--------- Item 1 --------->|   |<---- Item 2 ---->|   |<-- 目标 Item (预测) -->|
                                                    ↑
                                          seq_end_position (位置索引)
```

### ⭐ 关键说明

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

### 3.1 模型定义 (model_llama.py)

```python
class LlamaRecModel(nn.Module):
    """
    基于 LLaMA2-7B 的生成式推荐模型
    
    关键特性：
    - SoftEmbedding: SCID codes → Codebook → Projector → LLaMA
    - Weight Tying: 生成 Logits 通过点积 Codebook，保证 End-to-End 梯度流
    - Last Token: SIA/PSA 从历史序列最后一个 token 提取
    """
    
    def __init__(self, config, rqvae, llama_path="models/Llama-2-7b-hf"):
        super().__init__()
        
        # === 配置参数 ===
        self.code_length = config['code_length']  # 4
        self.code_num = config['code_num']  # 256
        self.codebook_dim = config['e_dim']  # 128
        self.semantic_dim = config['semantic_hidden_size']  # 256 or 1024 (DualSCID)
        self.n_items = config['n_items']
        self.num_beams = config.get('num_beams', 20)
        
        # === 加载 LLaMA 基座 ===
        # 注意: 多卡训练时不能用 device_map="auto"，由 accelerate 管理设备
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.hidden_size = self.llama.config.hidden_size  # 4096
        
        # 启用 Gradient Checkpointing (节省显存)
        # 使用 use_reentrant=False 解决 DDP 兼容性问题
        # (避免 "parameter marked ready twice" 错误)
        self.llama.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        
        # === ⭐ Codebook Embeddings (独立副本，避免 DDP 共享参数问题) ===
        # 从 rqvae 复制权重，model_rec 和 model_id 各持有独立的 codebook
        # 训练时需要在 trainer 中手动同步
        num_rqvae_layers = len(rqvae.rq.vq_layers)
        self.num_rqvae_layers = num_rqvae_layers  # 保存层数 (3)
        
        self.codebook_embeddings = nn.ModuleList([
            nn.Embedding(self.code_num, self.codebook_dim)
            for _ in range(num_rqvae_layers)
        ])
        # 复制权重
        for i, vq_layer in enumerate(rqvae.rq.vq_layers):
            self.codebook_embeddings[i].weight.data.copy_(vq_layer.embedding.weight.data)
        
        # === Input Projector: Codebook dim → LLaMA dim ===
        self.scid_projector = nn.Linear(self.codebook_dim, self.hidden_size, bias=False)
        
        # === Output Projector: LLaMA dim → Codebook dim (用于点积) ===
        # ⭐ Weight Tying: 不用独立的 code_heads，让梯度流向 Codebook
        self.output_projector = nn.Linear(self.hidden_size, self.codebook_dim, bias=False)
        
        # === 对齐层 ===
        self.enc_adapter = MLPLayers([self.hidden_size, self.codebook_dim])  # SIA
        self.dec_adapter = MLPLayers([self.hidden_size, self.semantic_dim])  # PSA
        
        # === 语义 Embedding (冻结，与原版 T5 一致) ===
        self.semantic_embedding = nn.Embedding(self.n_items, self.semantic_dim)
        self.semantic_embedding.requires_grad_(False)
        
        # === ⭐ Suffix Embedding (用于第 4 层，处理冲突计数) ===
        # 与原版 T5 一致，suffix 有独立的 embedding 层
        self.suffix_embedding = nn.Embedding(self.code_num, self.codebook_dim)
        self.suffix_embedding.requires_grad_(True)
        
        # === LoRA 微调 ===
        lora_config = LoraConfig(
            r=config.get('lora_r', 64),
            lora_alpha=config.get('lora_alpha', 128),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=config.get('lora_dropout', 0.05),
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llama = get_peft_model(self.llama, lora_config)
        
        # === 显式确保自定义层参与训练 ===
        self.scid_projector.requires_grad_(True)
        self.output_projector.requires_grad_(True)
        for param in self.enc_adapter.parameters():
            param.requires_grad_(True)
        for param in self.dec_adapter.parameters():
            param.requires_grad_(True)
        
        # === 初始化自定义层 (v3.2: 防止梯度爆炸/消失) ===
        self._init_custom_weights()
    
    def _init_custom_weights(self):
        """使用小方差初始化 Projector 层"""
        torch.nn.init.normal_(self.scid_projector.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.output_projector.weight, mean=0.0, std=0.02)
        
        for module in [self.enc_adapter, self.dec_adapter]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
    
    def get_codebooks(self):
        """获取码本 Embedding 层列表 (独立副本)"""
        return self.codebook_embeddings
    
    def get_input_embeddings(self, input_ids, attention_mask):
        """
        SoftEmbedding: 按间隔查表
        
        input_ids 布局: [c0, c1, c2, c3, c0, c1, c2, c3, ...]
        每隔 code_length 取一个位置，查对应层的码本
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
        
        # 按间隔查表 (Interval Slicing)
        codebooks = self.get_codebooks()
        
        for level in range(self.code_length):
            codes_at_level = input_ids_safe[:, level::self.code_length]
            
            if level < self.num_rqvae_layers:
                # 前 3 层：从对应 codebook embedding 查
                raw_embeds = codebooks[level](codes_at_level)
            else:
                # ⭐ 第 4 层 (suffix)：用独立的 suffix_embedding
                raw_embeds = self.suffix_embedding(codes_at_level)
            
            proj_embeds = self.scid_projector(raw_embeds.to(self.scid_projector.weight.dtype))
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
            seq_end_positions: [B] - 历史序列最后一个 token 的位置索引
            target_positions: [B, code_length] - 目标 code 各位置的索引
            labels: [B, code_length] - 目标 item 的真实 code
            targets: [B] - 目标 item ID
        """
        # 1. 获取输入嵌入
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
        
        if self.training:
            inputs_embeds = inputs_embeds.requires_grad_(True)
        
        # 2. LLaMA Forward
        outputs = self.llama(
            inputs_embeds=inputs_embeds.to(torch.bfloat16),
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states[-1]  # [B, L, 4096]
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        batch_indices = torch.arange(batch_size, device=device)
        
        # 3. SIA/PSA: Last Token
        last_hidden = hidden_states[batch_indices, seq_end_positions]  # [B, 4096]
        
        seq_project_latents = self.enc_adapter(last_hidden)  # [B, 128] for SIA
        dec_latents = self.dec_adapter(last_hidden)  # [B, 256] for PSA
        
        # 4. 生成 Logits: Weight Tying (点积 Codebook)
        code_logits = []
        codebooks = self.get_codebooks()
        
        for i in range(self.code_length):
            pos_i = target_positions[:, i]
            hidden_at_pos = hidden_states[batch_indices, pos_i]
            
            # Step 1: 投影回 Codebook 维度
            query_emb = self.output_projector(hidden_at_pos)
            
            # Step 2: 与对应层的权重做点积
            if i < self.num_rqvae_layers:
                codebook_weight = codebooks[i].weight.t()
            else:
                # ⭐ 第 4 层: suffix_embedding
                codebook_weight = self.suffix_embedding.weight.t()
            
            # Step 3: 计算相似度 logits
            logits = torch.matmul(query_emb, codebook_weight.to(query_emb.dtype))
            code_logits.append(logits)
        
        code_logits = torch.stack(code_logits, dim=1)  # [B, code_length, code_num]
        
        return QuantizeOutput(
            logits=code_logits,
            seq_latents=last_hidden,
            seq_project_latents=seq_project_latents,
            dec_latents=dec_latents
        )
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, num_return_sequences=10):
        """
        自回归生成目标 item 的 codes
        使用 Beam Search + 分批forward，每步用 output_projector + Codebook 点积
        
        ⭐ 优化：每步forward都分批处理，避免OOM
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        codebooks = self.get_codebooks()
        num_beams = self.num_beams
        
        # 分批forward的chunk大小（每次最多forward多少个序列）
        chunk_size = 4  # 可根据显存调整
        
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
            # === ⭐ 分批forward避免OOM ===
            total_seqs = current_embeds.size(0)
            all_hidden_states = []
            
            for chunk_start in range(0, total_seqs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_seqs)
                chunk_embeds = current_embeds[chunk_start:chunk_end]
                chunk_mask = attention_mask_expanded[chunk_start:chunk_end]
                
                outputs = self.llama(
                    inputs_embeds=chunk_embeds.to(torch.bfloat16),
                    attention_mask=chunk_mask,
                    use_cache=False,  # 不用KV Cache，简化逻辑
                    output_hidden_states=True,
                    return_dict=True
                )
                
                all_hidden_states.append(outputs.hidden_states[-1][:, -1, :])
                del outputs
                torch.cuda.empty_cache()
            
            # 合并结果
            last_hidden = torch.cat(all_hidden_states, dim=0)
            del all_hidden_states
            
            # 投影 + 点积 Codebook
            last_hidden = last_hidden.to(self.output_projector.weight.dtype)
            query_emb = self.output_projector(last_hidden)
            if code_idx < self.num_rqvae_layers:
                codebook_weight = codebooks[code_idx].weight.t()
            else:
                codebook_weight = self.suffix_embedding.weight.t()
            logits = torch.matmul(query_emb, codebook_weight.to(query_emb.dtype))
            
            # Beam Search 更新
            log_probs = F.log_softmax(logits, dim=-1)
            next_scores = log_probs + beam_scores.unsqueeze(-1)
            
            vocab_size = log_probs.size(-1)
            next_scores = next_scores.view(batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, num_beams, dim=-1)
            
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_codes = next_tokens % vocab_size
            
            beam_scores = next_scores.view(-1)
            generated_codes.append(next_codes)
            
            # 准备下一步的 embedding
            beam_idx = (next_indices + beam_idx_offset.view(batch_size, num_beams)).view(-1)
            current_embeds = current_embeds[beam_idx]
            attention_mask_expanded = attention_mask_expanded[beam_idx]
            
            # 添加新生成的 code embedding
            next_codes_flat = next_codes.view(-1)
            if code_idx < self.num_rqvae_layers:
                next_embeds = codebooks[code_idx](next_codes_flat)
            else:
                next_embeds = self.suffix_embedding(next_codes_flat)
            next_embeds = self.scid_projector(next_embeds.to(self.scid_projector.weight.dtype))
            next_embeds = next_embeds.unsqueeze(1)
            
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
            attention_mask_expanded = torch.cat([
                attention_mask_expanded,
                torch.ones(attention_mask_expanded.size(0), 1, device=device, dtype=attention_mask_expanded.dtype)
            ], dim=1)
        
        # 整理输出
        generated_codes = torch.stack(generated_codes, dim=-1)  # [B, beams, code_length]
        return generated_codes[:, :num_return_sequences, :]
```

### 3.2 DataLoader 设计 (data_llama.py)

```python
class LlamaRecDataset(Dataset):
    """
    LLaMA 推荐数据集
    将原始的 item ID 序列转换为 codes 序列，并计算关键位置索引
    """
    
    def __init__(self, inter_seq: List[List[int]], all_item_code: torch.Tensor,
                 code_length: int = 4, max_seq_len: int = 50):
        """
        Args:
            inter_seq: 交互序列列表，每个元素是 [item_id1, item_id2, ..., target_id]
            all_item_code: [n_items+1, code_length] 所有 item 的 code 表
            code_length: 每个 item 的 code 长度 (默认 4)
            max_seq_len: 最大历史序列长度 (item 数量，不是 token 数量)
        """
        self.all_item_code = all_item_code
        self.code_length = code_length
        self.max_seq_len = max_seq_len
        self.data = self._preprocess(inter_seq)
    
    def _preprocess(self, inter_seq: List[List[int]]) -> List[Dict]:
        """预处理：分离历史和目标"""
        data = []
        for seq in inter_seq:
            target = seq[-1]
            history = seq[:-1]
            data.append({'history': history, 'target': target})
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        history = item['history'][-self.max_seq_len:]
        target_item = item['target']
        
        # 构造 codes
        history_codes = []
        for item_id in history:
            item_codes = self.all_item_code[item_id].tolist()
            history_codes.extend(item_codes)
        
        target_codes = self.all_item_code[target_item].tolist()
        input_ids = history_codes + target_codes
        
        # 计算关键位置
        seq_end_position = len(history_codes) - 1
        target_positions = list(range(len(history_codes), len(history_codes) + self.code_length))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long),
            'seq_end_position': seq_end_position,
            'target_positions': torch.tensor(target_positions, dtype=torch.long),
            'labels': torch.tensor(target_codes, dtype=torch.long),
            'target_item': target_item,
        }


def collate_fn_llama(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """动态 Padding (左 Padding，LLaMA 习惯)"""
    max_len = max(len(b['input_ids']) for b in batch)
    
    input_ids = []
    attention_mask = []
    seq_end_positions = []
    target_positions = []
    labels = []
    targets = []
    
    for b in batch:
        pad_len = max_len - len(b['input_ids'])
        # ⭐ 左 Padding
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
        'seq_end_positions': torch.tensor(seq_end_positions, dtype=torch.long),
        'target_positions': torch.stack(target_positions),
        'labels': torch.stack(labels),
        'targets': torch.tensor(targets, dtype=torch.long),
    }


class LlamaCollator:
    """Collator 类封装，方便传入 DataLoader"""
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return collate_fn_llama(batch)
```

### 3.3 Trainer 核心逻辑 (trainer_llama.py)

```python
class LlamaTrainer:
    def __init__(self, config, model_rec, model_id, accelerator,
                 train_data=None, valid_data=None, test_data=None):
        # ... 配置初始化 ...
        
        # === ⭐ 优化器：LLaMA 使用 8-bit Adam 节省显存 ===
        self.rec_optimizer = self._build_optimizer(model_rec, self.lr_rec, self.weight_decay, use_8bit=True)
        self.id_optimizer = self._build_optimizer(model_id, self.lr_id, self.weight_decay, use_8bit=False)
        
        # ... Accelerate 准备 ...
    
    def _build_optimizer(self, model, lr, weight_decay, use_8bit=False):
        """构建优化器"""
        # 只优化可训练参数，避免为冻结的 7B 参数分配 optimizer state
        params = [p for p in model.parameters() if p.requires_grad]
        
        if use_8bit:
            # ⭐ 8-bit AdamW: 显存减半，用于 LLaMA 等大模型
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        return optimizer
    
    # === ⭐ Codebook 同步 (DDP 兼容) ===
    
    def _sync_codebook_to_model_id(self):
        """model_rec.codebook_embeddings → model_id.rq.vq_layers"""
        with torch.no_grad():
            rec_model = self.model_rec.module if dist.is_initialized() else self.model_rec
            id_model = self.model_id.module if dist.is_initialized() else self.model_id
            
            for i, vq_layer in enumerate(id_model.rq.vq_layers):
                vq_layer.embedding.weight.data.copy_(rec_model.codebook_embeddings[i].weight.data)
    
    def _sync_codebook_to_model_rec(self):
        """model_id.rq.vq_layers → model_rec.codebook_embeddings"""
        with torch.no_grad():
            rec_model = self.model_rec.module if dist.is_initialized() else self.model_rec
            id_model = self.model_id.module if dist.is_initialized() else self.model_id
            
            for i, vq_layer in enumerate(id_model.rq.vq_layers):
                rec_model.codebook_embeddings[i].weight.data.copy_(vq_layer.embedding.weight.data)
    
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
            if dist.is_initialized():
                target_semantic_embs = self.model_rec.module.semantic_embedding(targets)
            else:
                target_semantic_embs = self.model_rec.semantic_embedding(targets)
            
            # ⭐ model_id 在 train_rec 阶段被冻结，使用 no_grad + 直接访问 module
            with torch.no_grad():
                if dist.is_initialized():
                    target_recon_embs, _, _, _, target_code_logits = self.model_id.module(target_semantic_embs)
                else:
                    target_recon_embs, _, _, _, target_code_logits = self.model_id(target_semantic_embs)
            
            # Forward
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
            self.accelerator.clip_grad_norm_(self.model_rec.parameters(), 1.0)  # ⭐ 梯度裁剪
            self.rec_optimizer.step()
            self.rec_lr_scheduler.step()
        
        # ⭐ 同步 codebook: model_rec → model_id
        self._sync_codebook_to_model_id()
    
    def _train_epoch_id(self, epoch_idx, loss_w, verbose=True):
        """训练 Tokenizer (冻结 Recommender)"""
        # ... 类似 _train_epoch_rec ...
        
        # ⭐ 同步 codebook: model_id → model_rec
        self._sync_codebook_to_model_rec()
```

---

## 4. 5090 32GB 配置 (llama_instrument2018.yaml)

```yaml
# === 数据集配置 ===
dataset: Instrument2018_5090
seed: 2020
reproducibility: True

# DualSCID: collab + text embedding 拼接 (256 + 768 = 1024)
collab_emb_path: Instrument2018_5090_emb_256.npy
text_emb_path: Instrument2018_5090_sentence-transformer_text_768.npy
normalize: false

# === LLaMA 模型配置 ===
llama_path: models/Llama-2-7b-hf
precision: bf16
gradient_checkpointing: true

# LoRA 配置
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05

# === 推荐模型配置 ===
semantic_hidden_size: 1024  # DualSCID = 256 + 768
code_num: 256
code_length: 4  # RQ-VAE 3层 + 1层 suffix
e_dim: 128
num_beams: 20

# === 训练配置 ===
epochs: 50
lr_rec: 0.0001          # LLaMA 学习率较低
lr_id: 0.0001
weight_decay: 0.05

# Loss 权重 (训练 Tokenizer)
id_vq_loss: 1
id_code_loss: 0
id_kl_loss: 0.0001
id_dec_cl_loss: 0.0003

# Loss 权重 (训练 Recommender)
rec_vq_loss: 0
rec_code_loss: 1
rec_kl_loss: 0.0001
rec_dec_cl_loss: 0.0003

# 训练周期
cycle: 2
sim: cos
warm_epoch: 5

# 优化器
learner: AdamW
lr_scheduler_type: cosine
warmup_steps: 500

# Batch 配置 (5090 32GB)
batch_size: 2               # 每卡 batch size
gradient_accumulation_steps: 8  # 等效 batch_size = 16
eval_batch_size: 4
num_workers: 4

# 序列配置
max_seq_len: 50

# 评估配置
eval_step: 2
early_stop: 10
metrics: recall@1,recall@5,ndcg@5,recall@10,ndcg@10
valid_metric: ndcg@10
```

---

## 5. 5090 迁移专用补丁 (main_llama.py)

```python
# === 5090 迁移专用补丁 ===
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# DDP 配置：
# - find_unused_parameters: LoRA 场景下部分参数可能不参与 loss
# - 注意：不能用 static_graph=True，因为 train_id/train_rec 阶段图结构不同
ddp_kwargs = DistributedDataParallelKwargs(
    find_unused_parameters=True
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
```

---

## 6. 启动前 Checklist

- [x] `scid_projector` (128 → 4096) 已添加
- [x] `output_projector` (4096 → 128) 已添加
- [x] **移除了 `code_heads`**，改用 Codebook 点积
- [x] SIA/PSA 使用 Last Token，移除 Mean Pooling
- [x] DataLoader 返回 `seq_end_positions` 和 `target_positions`
- [x] **不使用显式 SEQ_END token**，仅用位置索引
- [x] Gradient Checkpointing 已开启 (**use_reentrant=False**)
- [x] 所有自定义层 `requires_grad_(True)`
- [x] **Projector 初始化** (`std=0.02`)
- [x] **独立 Codebook 副本** (避免 DDP 共享参数问题)
- [x] **Codebook 同步逻辑** (_sync_codebook_to_model_id/rec)
- [x] **Suffix Embedding** (第4层独立处理)
- [x] **8-bit AdamW** (节省显存)
- [x] **分批 Generate** (避免 OOM)
- [x] **5090 补丁** (禁用 TF32)
- [x] **DDP 配置** (find_unused_parameters=True)

---

## 7. 梯度检查脚本

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
    
    print("=== Gradient Check (v3.3) ===")
    
    # ⭐ 关键：Codebook 必须有梯度！
    codebooks = model.get_codebooks()
    for i, cb in enumerate(codebooks):
        grad = cb.weight.grad
        if grad is None:
            print(f"❌ Codebook[{i}] grad is None - End-to-End 断裂!")
        elif grad.abs().sum() == 0:
            print(f"⚠️ Codebook[{i}] grad is zero")
        else:
            print(f"✅ Codebook[{i}] grad norm: {grad.norm():.6f}")
    
    # ⭐ Suffix Embedding
    suffix_grad = model.suffix_embedding.weight.grad
    if suffix_grad is not None:
        print(f"✅ suffix_embedding grad norm: {suffix_grad.norm():.6f}")
    else:
        print(f"❌ suffix_embedding grad is None")
    
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

## 8. v3.3 架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LlamaRecModel v3.3                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  [Input: History Codes]                                                          │
│         │                                                                        │
│         ▼                                                                        │
│  ┌─────────────────────────────────────────┐                                    │
│  │ get_input_embeddings (Interval Slicing) │                                    │
│  │   level 0-2: codebook_embeddings[i]     │                                    │
│  │   level 3:   suffix_embedding (独立)     │                                    │
│  └────────────────┬────────────────────────┘                                    │
│                   ▼                                                              │
│           ┌───────────────┐                                                     │
│           │ scid_projector│ [128 → 4096]                                        │
│           │   (可训练)     │                                                     │
│           └───────┬───────┘                                                     │
│                   ▼                                                              │
│           ┌─────────────────────────────┐                                       │
│           │       LLaMA2-7B             │                                       │
│           │  (LoRA + Gradient Ckpt)     │                                       │
│           │  use_reentrant=False        │                                       │
│           └───────────┬─────────────────┘                                       │
│                       │                                                          │
│           ┌───────────┼───────────┐                                             │
│           │           │           │                                             │
│           ▼           ▼           ▼                                             │
│     [seq_end_pos] [seq_end_pos] [target_pos]                                    │
│           │           │           │                                             │
│           ▼           ▼           ▼                                             │
│    ┌──────────┐ ┌──────────┐ ┌────────────────┐                                 │
│    │enc_adapter│ │dec_adapter│ │output_projector│                                │
│    └────┬─────┘ └────┬─────┘ └───────┬────────┘                                 │
│         │            │               │                                           │
│         ▼            ▼               ▼                                           │
│    [B, 128]     [B, semantic]   [B, 128]                                        │
│         │            │               │                                           │
│         │            │               │                                           │
│         ▼            ▼               ▼                                           │
│    ┌─────────┐ ┌─────────┐    ┌─────────────────────────────┐                   │
│    │SIA Loss │ │PSA Loss │    │ MatMul (Weight Tying)       │                   │
│    │(KL Div) │ │(InfoNCE)│    │ level 0-2: codebook[i].T    │◄────────┐        │
│    └─────────┘ └─────────┘    │ level 3:   suffix_emb.T     │         │        │
│                               └────────────┬────────────────┘         │        │
│                                            │                          │        │
│                                            ▼                          │        │
│                                       [B, 256]                        │        │
│                                       (Logits)                        │        │
│                                            │                          │        │
│                                            ▼                          │        │
│                                      ┌──────────┐                     │        │
│                                      │Code Loss │─────────────────────┘        │
│                                      │(CE Loss) │                              │
│                                      └──────────┘                              │
│                                                                                 │
│  ⭐ 独立 codebook_embeddings + Trainer 手动同步                                  │
│  ⭐ 8-bit AdamW (LLaMA) + 普通 AdamW (RQ-VAE)                                   │
│  ⭐ 分批 Generate (避免 OOM)                                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 实施优先级

| Phase | 任务 | 状态 |
|-------|------|------|
| ✅ P0 | LlamaRecModel 实现 (含 Weight Tying + Suffix) | 完成 |
| ✅ P0 | DataLoader 重构 | 完成 |
| ✅ P1 | Trainer 适配 (含 Codebook 同步) | 完成 |
| ✅ P1 | Generate (Beam Search + 分批) | 完成 |
| 🟡 P2 | 梯度调试 + 形状验证 | 进行中 |
| 🟡 P2 | 评估 + 调优 | 进行中 |

---

## 10. 风险点与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| 显存溢出 | 训练失败 | 降 batch，加 grad_accum，用 DeepSpeed ZeRO |
| Codebook 梯度消失 | End-to-End 失效 | 检查 output_projector 初始化，加梯度监控 |
| Logits 爆炸/NaN | 训练崩溃 | 减小初始化 std (0.02→0.01)，加 gradient clipping |
| 生成质量差 | 推荐效果下降 | 增加 beam_size，添加 prefix constraint |
| DDP 参数共享冲突 | 训练报错 | ✅ 已解决：独立 codebook + 手动同步 |
| Generate OOM | 评估失败 | ✅ 已解决：分批 forward |

---

## 11. 启动命令

```bash
# 单卡
python main_llama.py --config ./config/llama_instrument2018.yaml

# 多卡 (Accelerate)
accelerate launch --config_file accelerate_config_llama.yaml main_llama.py --config ./config/llama_instrument2018.yaml

# 调试模式 (跳过 train_id)
accelerate launch main_llama.py --config ./config/llama_instrument2018.yaml --skip_id
```
