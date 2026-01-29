"""
LlamaRecModel: ETEGRec 的 LLaMA2-7B 迁移版本
基于 T5_to_LLaMA2_Migration_Plan v3.2

核心设计：
1. SoftEmbedding: 按间隔查表 (Interval Slicing)
2. Weight Tying: output_projector + MatMul(Codebook.T)
3. SIA/PSA: Last Token 提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from layers import MLPLayers


@dataclass
class QuantizeOutput(ModelOutput):
    """模型输出结构，与原 T5 版本保持一致"""
    logits: Optional[torch.FloatTensor] = None          # [B, code_length, code_num]
    seq_latents: Optional[torch.FloatTensor] = None     # [B, hidden_size] 原始 hidden
    seq_project_latents: Optional[torch.FloatTensor] = None  # [B, e_dim] for SIA
    dec_latents: Optional[torch.FloatTensor] = None     # [B, semantic_dim] for PSA


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
        self.semantic_dim = config['semantic_hidden_size']  # 256
        self.n_items = config['n_items']
        self.num_beams = config.get('num_beams', 20)
        self.generate_chunk_size = config.get('generate_chunk_size', 4)  # ⭐ 可配置
        
        # === 加载 LLaMA 基座 ===
        # 注意: 多卡训练时不能用 device_map="auto"，由 accelerate 管理设备
        # 注意: RTX 5090 (Blackwell) 暂不支持 flash_attn，使用 SDPA (PyTorch 原生)
        print(f"[LlamaRecModel] 加载 LLaMA 模型: {llama_path}")
        self.llama = AutoModelForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa"  # PyTorch 2.0 原生 SDPA，比默认实现快
        )
        self.hidden_size = self.llama.config.hidden_size  # 4096
        
        # 启用 Gradient Checkpointing (节省显存)
        # 使用 use_reentrant=False 解决 DDP 兼容性问题
        # (避免 "parameter marked ready twice" 错误)
        self.llama.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[LlamaRecModel] Gradient Checkpointing 已启用 (non-reentrant)")
        
        # === Codebook Embeddings (独立副本，避免 DDP 共享参数问题) ===
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
        print(f"[LlamaRecModel] 创建了 {num_rqvae_layers} 个独立的 codebook embeddings")
        
        # === Input Projector: Codebook dim → LLaMA dim ===
        self.scid_projector = nn.Linear(self.codebook_dim, self.hidden_size, bias=False)
        
        # === Output Projector: LLaMA dim → Codebook dim (用于点积) ===
        # ⭐ Weight Tying: 不用独立的 code_heads，让梯度流向 Codebook
        self.output_projector = nn.Linear(self.hidden_size, self.codebook_dim, bias=False)
        
        # === 对齐层 ===
        self.enc_adapter = MLPLayers([self.hidden_size, self.codebook_dim])  # SIA
        self.dec_adapter = MLPLayers([self.hidden_size, self.semantic_dim])  # PSA
        
        # === 语义 Embedding (冻结，与原版 T5 一致) ===
        # n_items 已包含 PAD 位置，无需 +1
        self.semantic_embedding = nn.Embedding(self.n_items, self.semantic_dim)
        self.semantic_embedding.requires_grad_(False)
        
        # === Suffix Embedding (用于第 4 层，处理冲突计数) ===
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
        self.llama.print_trainable_parameters()
        
        # === 显式确保自定义层参与训练 ===
        self.scid_projector.requires_grad_(True)
        self.output_projector.requires_grad_(True)
        for param in self.enc_adapter.parameters():
            param.requires_grad_(True)
        for param in self.dec_adapter.parameters():
            param.requires_grad_(True)
        
        # === 初始化自定义层 (v3.2: 防止梯度爆炸/消失) ===
        self._init_custom_weights()
        print("[LlamaRecModel] 自定义层初始化完成 (std=0.02)")
    
    def _init_custom_weights(self):
        """
        使用小方差初始化 Projector 层
        防止训练初期 Logits 过大导致 Softmax 变 one-hot (梯度消失)
        """
        # Projectors
        torch.nn.init.normal_(self.scid_projector.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.output_projector.weight, mean=0.0, std=0.02)
        
        # Adapters
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
        SoftEmbedding: 按间隔查表，与原 T5 版本逻辑一致
        
        input_ids 布局: [c0, c1, c2, c3, c0, c1, c2, c3, ...]
        每隔 code_length 取一个位置，查对应层的码本
        
        Args:
            input_ids: [B, seq_len] - codes 序列
            attention_mask: [B, seq_len]
        
        Returns:
            embeddings: [B, seq_len, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 初始化为零
        embeddings = torch.zeros(
            batch_size, seq_len, self.hidden_size,
            dtype=torch.bfloat16, device=device
        )
        
        # 处理 padding (-1 → 0)
        input_ids_safe = input_ids.clone()
        input_ids_safe[input_ids == -1] = 0
        
        # 按间隔查表 (Interval Slicing)
        # 注意: code_length=4 包含 suffix，但实际 codebook 只有 3 层
        codebooks = self.get_codebooks()
        
        for level in range(self.code_length):
            # 取每隔 code_length 的位置
            codes_at_level = input_ids_safe[:, level::self.code_length]  # [B, seq_len/K]
            
            if level < self.num_rqvae_layers:
                # 前 3 层：从对应 codebook embedding 查
                raw_embeds = codebooks[level](codes_at_level)  # [B, seq_len/K, 128]
            else:
                # 第 4 层 (suffix)：用独立的 suffix_embedding
                raw_embeds = self.suffix_embedding(codes_at_level)  # [B, seq_len/K, 128]
            
            # 投影到 LLaMA 维度
            proj_embeds = self.scid_projector(raw_embeds.to(self.scid_projector.weight.dtype))
            
            # 放回对应位置
            embeddings[:, level::self.code_length] = proj_embeds
        
        # Padding 位置置零 (attention_mask 会屏蔽)
        padding_mask = ~attention_mask.bool()
        embeddings[padding_mask] = 0
        
        return embeddings
    
    def forward(self, input_ids, attention_mask, seq_end_positions, 
                target_positions, labels=None, targets=None):
        """
        前向传播
        
        Args:
            input_ids: [B, total_len] - 历史 + 目标的 codes
            attention_mask: [B, total_len]
            seq_end_positions: [B] - 历史序列最后一个 token 的位置索引
            target_positions: [B, code_length] - 目标 code 各位置的索引
            labels: [B, code_length] - 目标 item 的真实 code (用于 code_loss)
            targets: [B] - 目标 item ID (用于 SIA/PSA)
        
        Returns:
            QuantizeOutput
        """
        # === 1. 获取输入嵌入 ===
        inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
        
        # Gradient Checkpointing 兼容
        if self.training:
            inputs_embeds = inputs_embeds.requires_grad_(True)
        
        # === 2. LLaMA Forward (确保输入是 bf16) ===
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
        
        # === 3. SIA/PSA: Last Token (不用 Mean Pooling) ===
        # Causal LM 中，最后一个 token 已经看过所有历史，信息最丰富
        last_hidden = hidden_states[batch_indices, seq_end_positions]  # [B, 4096]
        
        seq_project_latents = self.enc_adapter(last_hidden)  # [B, 128] for SIA
        dec_latents = self.dec_adapter(last_hidden)  # [B, 256] for PSA
        
        # === 4. 生成 Logits: Weight Tying (点积 Codebook) ===
        # ⭐ 关键修复：使用 target_positions[i] - 1 的 hidden state
        # 因为 Causal LM 中，位置 i 的 hidden state 已经看到了位置 i 的 token
        # 要预测位置 i 的 token，应该用位置 i-1 的 hidden state
        code_logits = []
        codebooks = self.get_codebooks()
        
        for i in range(self.code_length):
            # ⭐ 使用前一个位置的 hidden state
            if i == 0:
                # 第一个 code：使用历史序列最后一个 token 的 hidden state
                pos_i = seq_end_positions  # [B]
            else:
                # 后续 codes：使用前一个目标 code 位置的 hidden state
                pos_i = target_positions[:, i - 1]  # [B]
            
            hidden_at_pos = hidden_states[batch_indices, pos_i]  # [B, 4096]
            
            # Step 1: 投影回 Codebook 维度
            query_emb = self.output_projector(hidden_at_pos)  # [B, 128]
            
            # Step 2: 与对应层的权重做点积
            if i < self.num_rqvae_layers:
                # 前 3 层：与 Codebook 做点积 (Weight Tying!)
                codebook_weight = codebooks[i].weight.t()  # [128, 256]
            else:
                # 第 4 层 (suffix)：与 suffix_embedding 做点积
                codebook_weight = self.suffix_embedding.weight.t()  # [128, 256]
            
            # Step 3: 计算相似度 logits
            logits = torch.matmul(query_emb, codebook_weight.to(query_emb.dtype))  # [B, 256]
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
        使用 Beam Search + 分批 forward，每步用 output_projector + Codebook 点积
        
        Args:
            input_ids: [B, seq_len] - 历史序列的 codes
            attention_mask: [B, seq_len]
            num_return_sequences: 返回的候选数量
        
        Returns:
            generated_codes: [B, num_return_sequences, code_length]
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        codebooks = self.get_codebooks()
        num_beams = self.num_beams
        
        # 分批 forward 的 chunk 大小
        chunk_size = self.generate_chunk_size
        
        # === Beam Search 初始化 ===
        input_ids_expanded = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask_expanded = attention_mask.repeat_interleave(num_beams, dim=0)
        
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_scores[:, 1:] = -1e9  # 只保留第一个 beam
        beam_scores = beam_scores.view(-1)
        
        generated_codes = []
        current_embeds = self.get_input_embeddings(input_ids_expanded, attention_mask_expanded)
        
        beam_idx_offset = torch.arange(batch_size, device=device).repeat_interleave(num_beams) * num_beams
        
        for code_idx in range(self.code_length):
            # === 分批 forward 避免 OOM ===
            total_seqs = current_embeds.size(0)  # B * num_beams
            all_hidden_states = []
            
            for chunk_start in range(0, total_seqs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_seqs)
                chunk_embeds = current_embeds[chunk_start:chunk_end]
                chunk_mask = attention_mask_expanded[chunk_start:chunk_end]
                
                outputs = self.llama(
                    inputs_embeds=chunk_embeds.to(torch.bfloat16),
                    attention_mask=chunk_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # 只保留 last token 的 hidden state
                all_hidden_states.append(outputs.hidden_states[-1][:, -1, :])
                del outputs
            
            # 合并结果
            last_hidden = torch.cat(all_hidden_states, dim=0)  # [B*beams, 4096]
            del all_hidden_states
            
            # 投影 + 点积 Codebook (转换 dtype 避免 bf16 vs fp32 不匹配)
            last_hidden = last_hidden.to(self.output_projector.weight.dtype)
            query_emb = self.output_projector(last_hidden)  # [B*beams, 128]
            del last_hidden  # ⭐ 释放大张量
            if code_idx < self.num_rqvae_layers:
                codebook_weight = codebooks[code_idx].weight.t()  # [128, 256]
            else:
                codebook_weight = self.suffix_embedding.weight.t()  # [128, 256]
            logits = torch.matmul(query_emb, codebook_weight.to(query_emb.dtype))  # [B*beams, 256]
            
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
            if code_idx < self.num_rqvae_layers:
                next_embeds = codebooks[code_idx](next_codes_flat)  # [B*beams, 128]
            else:
                next_embeds = self.suffix_embedding(next_codes_flat)  # [B*beams, 128]
            next_embeds = self.scid_projector(next_embeds.to(self.scid_projector.weight.dtype))
            next_embeds = next_embeds.unsqueeze(1)  # [B*beams, 1, 4096]
            
            current_embeds = torch.cat([current_embeds, next_embeds], dim=1)
            attention_mask_expanded = torch.cat([
                attention_mask_expanded,
                torch.ones(attention_mask_expanded.size(0), 1, device=device, dtype=attention_mask_expanded.dtype)
            ], dim=1)
        
        # 整理输出: [B, beams, code_length]
        generated_codes = torch.stack(generated_codes, dim=-1)  # [B, beams, code_length]
        return generated_codes[:, :num_return_sequences, :]

