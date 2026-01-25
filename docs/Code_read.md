这份笔记主要记录了 ETEGRec 联合训练的核心流程，特别是 train_epoch_id（固定rec,训练tokenizer）和train_epoch_rec(固定tokenizer，训练rec) 函数中的数据流向、维度变化以及各个 Loss 的计算逻辑。
该笔记由我的手写版经Gemini 3pro转化成md，并人工微调。

注：上面的笔记从epoch%cycle==0,开始看起 。
注：维度字母缩写声明：
- B ：Batch大小
- L_seq:交互序列的长度（item-id版，最长序列截断为50）
- K：code_length（SID的长度，实验中为4，RQVAE生成3，后续为消除碰撞又加一个token）
- Code_num:码本的大小，即各个code的可选emb数量(实验中为256）
- H：hidden_size
- E_dim:码本向量的维度（实验中为128）
- Semantic_H:SASRec向量的维度（实验中为256）

---
ETEGRec 学习笔记
0. 前置说明：Item Code 初始化与构建逻辑 (get_code)
这是训练循环开始前（以及每个 Tokenizer 训练周期后）的关键步骤：构建全量物品的 Token 码本。
代码位置：trainer.py ->Trainer.get_code(epoch_idx, verbose)
这一步旨在生成全局的 self.all_item_code，供后续 train_epoch_rec 和 train_epoch_id 查表使用。
0.1 获取语义向量 (all_item_embs)
- 代码：all_item_embs = self.model_rec.semantic_embedding.weight.data[1:]
- 维度：[10449, 256]
- 含义：
  - 10449：实际物品数量（不含 Padding 的 0 号物品）。
  - 256：语义向量维度 (semantic_hidden_size)。
  - 作用：作为 RQ-VAE 的输入，用于生成 Token。
0.2 生成前缀 Code (all_item_prefix)
- 代码：all_item_prefix = self.model_id.get_indices(all_item_embs).detach().cpu().numpy()
- 维度：[10449, 3]
- 含义：
  - 3：RQ-VAE 输出的层级 Code 长度。
  - 作用：这是 Item 的语义部分 Token（Semantic Tokens）。例如 [151, 19, 62]。
  - 注：配置中 code_length 为 4，这里只生成了前 3 位，第 4 位留给冲突索引。
0.3 处理冲突与构建映射 (tokens2item)
- 逻辑：
tokens2item = defaultdict(list)
# 遍历每个物品的前缀
str_id = ' '.join(map(str, all_item_prefix[i])) # 例: '151 19 62'
tokens2item[str_id].append(i+1)
- 示例：
  - '151 19 62'-> 对应物品 [1]
  - '74 44 138' ->对应物品 [2]
- 含义：统计哪些物品共享相同的语义前缀，用于计算冲突位（Collision Index）。
0.4 构建最终 Token 表 (all_item_tokens)
- 代码：
all_item_tokens = [[-1, -1, -1, -1]] # 初始化 Padding 行# 循环中追加
collision_id = len(tokens2item[str_id]) - 1 # 冲突索引，从0开始
all_item_tokens.append(all_item_prefix[i] + [collision_id])
- 数据结构：
  - Row 0: [-1, -1, -1, -1] (Padding 物品的 Token)
  - Row 1: [151, 19, 62, 0] (物品 1：前缀 + 冲突位 0)
  - Row 2: [74, 44, 138, 0] (物品 2：前缀 + 冲突位 0)
- 最终 Tensor：
  - 维度：[10450, 4]
  - 10450：10449 (真实物品) + 1 (Padding)。
  - 4：3 (语义 Code) + 1 (冲突 Code)。
  - 作用：这就是训练循环中 input_ids 查表的数据源 self.all_item_code。

---
1.训练逻辑入口
条件判断：epoch % cycle == 0 (原实验设置cycle为2，即0,2,4训练tokenizer，1,3,5训练rec)
- True $$\rightarrow$$ train_epoch_id (训练 Tokenizer)
- False (else) $$\rightarrow$$ train_epoch_rec (训练 Recommender)

---
2. train_epoch_id 流程详解
① 准备数据
核心代码操作：
input_ids = self.all_item_code[input_ids].contiguous().clone().view(B, -1)
labels = self.all_item_code[targets].contiguous().clone().view(B, -1)
变量数据流向与维度解析：
1. input_ids (输入历史序列)
  - 动作：
    1. 获得 Batch 数据
    2. 把 item_id -> token序列 (查表操作)
  - 维度：[B,L_seq*K]
2. targets (预测目标的id)
  - 动作： 获得 目标物品 的 embedding
    - 注：查 model_rec 的 semantic_embedding
  - 维度：[B,1]
  - Labels:(预测目标的token 序列) [B,K]
3. attention_mask
  - 逻辑：(input_ids != -1).bool()，用于标记 Padding 部分。
② Tokenizer 的前向计算
1. 计算所有 target 的重构向量和 Logits (用于对齐 PSA Loss)
# 将当前 Batch 中的目标物品 ID（targets）展平成一维向量。
# 目的：Embedding 层通常接受一维的索引列表来进行查表操作。
target_flatten = targets.flatten() 
target_semantic_embs = self.model_rec.semantic_embedding(target_flatten) # lookup emb
target_recon_embs, _,_,_ target_code_logits = self.model_id(target_semantic_embs)
  - 输入：target_semantic_embs [B,H]
  - 输出：
    - target_recon_embs: [B,H]  target重构后的向量
    - target_code_logits: [B, Code_len, Code_Num] - RQ-VAE 预测target的Code分布
此处缺少一个model_id的讲解
2. 只对 Batch 内唯一的物品计算 Loss (优化技巧)
unq_input, unq_index = np.unique(target_flatten.cpu().numpy(), return_index=True)
unq_input = torch.tensor(unq_input).to(self.device)
unq_index = torch.tensor(unq_index).to(self.device)
unq_semantic_embs = self.model_rec.semantic_embedding(unq_input)
unq_recon_embs, commit_loss, _,_,_= self.model_id(unq_semantic_embs)
- np.unique(ar, return_index=True)：这是 NumPy 的一个函数。
  - 输入：ar 是输入数组（这里是 target_flatten，即 Batch 内所有目标物品的 code）。
  - 功能：它会找到数组中所有唯一（不重复）的元素，并对它们进行排序。
  - return_index=True：除了返回去重后的元素外，还会返回这些唯一元素在原始数组中第一次出现的位置索引。
- 输出变量：
  - unq_input：去重并排序后的 Item ID 数组。 维度: (M,)，其中 $$M \le B$$ (B为 Batch Size)。
  - unq_index：这些唯一 Item ID 在原始 target_flatten 中的索引。 维度: (M,)。
unq_recon_embs, commit_loss, _,_,_= self.model_id(unq_semantic_embs)
- 含义：将这 M 个唯一的语义向量输入 RQ-VAE。
- 输出变量：
  - unq_recon_embs: (M, H)。去重后物品的重建向量。
  - commit_loss: 这是 RQ-VAE 的量化承诺损失（Commitment Loss），用于约束 Encoder 输出不要离 Codebook 中心太远。
③ Rec 模型的前向计算
output = self.model_rec(input_ids, labels, attention_mask)
logits = outputs.logits  # (batch, code_len, code_num)
seq_project_latents = outputs.seq_project_latents  # SIA的序列表示 [Batch, hidden_size]
dec_latents = outputs.dec_latents  # PSA的解码器的表示 [Batch, hidden_size]
_, _, _, _, seq_code_logits = self.model_id.rq(seq_project_latents)
输出变量解析：
- logits: output.logits
  - 维度：[B, code_len, code_num]
  - 作用：T5 推荐物品的 Token 分布
- seq_project_latents: output.seq_project_latents
  - 维度：[B, E_dim]
  - 作用：用于 SIA (Sequence-Item Alignment) 的序列表示
- dec_latents: output.dec_latents
  - 维度：[B, Semantic_H]
  - 作用：用于 PSA (Preference-Semantic Alignment) 的用户偏好向量
后续计算：
- _, _, _, _, seq_code_logits = self.model_id.rq(seq_project_latents)
  - 维度：[B, code_len, code_num]
  - 作用：RQVAE 的量化器对序列表示预测的 Code 分布，用于L_SIA计算
model_rec的详细介绍在精读代码笔记
model_id.rq待更新
④ 损失计算
1) vq_loss
- 含义：让重构向量接近原始语义向量。
- 代码逻辑：
  - vq_loss = recon_loss(unq_recon_embs, unq_semantic_embs) + alpha * commit_loss
- 公式：
  - $$L_{SQ} = ||z - z'||^2 + \sum_{l=1}^{L} ||sg[v_l] - e_{c_l}^l||^2 + \beta ||v_l - sg[e_{c_l}^l]||^2$$
2) kl_loss (SIA)
- 含义：SIA (Sequence-Item Alignment)。让 T5 预测序列的 Code 分布与 RQVAE 给物品分配的 Code 一致。这样 Tokenizer 生成的 Code 更易被 T5 预测。
- 代码逻辑：
  - kl_loss = KL(seq_code_logits || target_code_logits) 
  - 两个logits维度均是：[B, 3, code_num]
- 公式：$$L_{SIA} = -\sum_{l=1}^{L} (D_{KL}(P_z^l || P_{z^E}^l) + D_{KL}(P_{z^E}^l || P_z^l))$$
def compute_discrete_contrastive_loss_kl(x_logits, y_logits): # kl loss
    code_num = x_logits.size(-1)
    x_logits = F.log_softmax(x_logits.view(-1, code_num), dim=-1)
    y_logits = F.log_softmax(y_logits.view(-1, code_num), dim=-1)
    loss = F.kl_div(x_logits, y_logits, reduction='batchmean', log_target=True)
    return loss
3) dec_cl_loss (PSA)
- 含义：PSA (Preference-Semantic Alignment)。拉近用户偏好向量和物品重构语义的距离。确保 Tokenizer 重构出的语义空间与用户偏好空间是对齐的。
- 代码逻辑：
  - dec_cl_loss = InfoNCE(target_recon_embs, dec_latents)
    - target_recon_embs 维度：[B, H]
    - dec_latents 维度：[B, H]
- 公式：
  - $$L_{PSA} = -\left( \log \frac{\exp(s(\bar{z}, h^D)/\tau)}{\sum_{h \in B} \exp(s(\bar{z}, h)/\tau)} + \log \frac{\exp(s(h^D, \bar{z})/\tau)}{\sum_{\hat{z} \in B} \exp(s(h^D, \hat{z})/\tau)} \right)$$
    def compute_contrastive_loss(query_embeds, semantic_embeds, temperature=0.07, sim="cos", gathered=True):
        # InfoNCE loss
        gathered_query_embeds = query_embeds
        gathered_semantic_embeds = semantic_embeds
        if sim=="cos":  # 计算相似度
            gathered_query_embeds = F.normalize(gathered_query_embeds, dim=-1)
            gathered_semantic_embeds = F.normalize(gathered_semantic_embeds, dim=-1)

        effective_bsz = gathered_query_embeds.size(0)
        labels = torch.arange(effective_bsz, dtype=torch.long, device=query_embeds.device)
        similarities = torch.matmul(gathered_query_embeds, gathered_semantic_embeds.transpose(0, 1)) / temperature

        co_loss = F.cross_entropy(similarities, labels)
        return co_loss
4) 总 Loss
- 代码逻辑：loss = vq_loss * w1 + kl_loss * w2 + dec_cl_loss * w3
- 公式：
  - $$L_{IT} = L_{SQ} + \mu L_{SIA} + \lambda L_{PSA}$$
3.train_epoch_rec
其他训练步骤基本上一样,主要是loss计算和冻结模型不同
5) code_loss (生成推荐的 Loss)
- 代码
code_loss = F.cross_entropy(logits.view(-1, self.code_num), labels.detach().reshape(-1))
- 公式：$$L_{Rec}=-\sum^L_{j=1}log P(Y_J|x,Y_{<j})$$
总loss:
- 代码逻辑：loss = code_loss * w1 + kl_loss * w2 + dec_cl_loss * w3
- 公式：
  - $$L_{rec} = L_{code} + \mu L_{SIA} + \lambda L_{PSA}$$

---
1. Model Rec 内部详细过程 (model_rec forward)
Step 1: 处理输入嵌入
- 输入：input_ids, attention_mask
  - input_ids 维度：[B, L_seq*K] - 用户历史 token 序列
- 操作：inputs_embeds = self.get_input_embeddings(input_ids, attention_mask)
- 输出：inputs_embeds
  - 维度：[B, L_seq*K, H]
Step 2: 准备解码器输入
- 目标：把真实 Code 加入 PAD。
- 输入：labels [B, 3] (目标物品真实 Code)
- 操作：decoder_input_ids = self._shift_right(labels)
  - 逻辑：labels 右移一位，前面补 pad_token_id (作为 BOS)。
  - 作用：预测第一个 token 时，作为启动符号。
  - 例：[A, B, C] ->[PAD, A, B, C] ->截断 -> [PAD, A, B]
补：_shift_right 内部细节
- torch.full(input_shape[:-1] + (1,), pad_token_id)
- 元组加法：(32,) + (1,) = (32, 1)
for i in range(min(decoder_input_ids.shape[1], self.code_length)): # 截断操作
    if i == 0:
        code_embedding = self.model.shared  # 使用 T5 共享的词嵌入else:
    else:
        code_embedding = self.token_embeddings[i-1] # 0~255，使用自定义的 embeddings
    decoder_inputs_embeds.append(code_embedding(decoder_input_ids[:, i])) # (batch, hidden_size) 根据id查表获得emb
decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1) # (batch, code_len, hidden_size)
- decoder_inputs_embeds = torch.stack(..., dim=1)
  - 维度：[B, 3, H]
  - 作用：解码器的输入向量序列，用于预测 i 位置的 token
  - 例：PAD -> A, A -> B, B -> C
补：
1. Embedding 的查表操作 (Lookup)
- decoder_input_ids[:, i] 取出整个 batch 在 i 位置的 id [B, ]。
- code_embedding(decoder_input_ids[:, i]) 根据 id 取向量。
  - code_embedding内容：一个权重矩阵 [词表大小, H]。
2. 列表->张量
- decoder_inputs_embeds = torch.stack(decoder_inputs_embeds, dim=1)
- 原列表：[[B, H], [B, H], [B, H]] 
- Stack 后维度：[B, 3, H]。
- torch.stack在指定维度插入一个新维度，堆叠起来
Step 3: 骨干网络计算 (调用 T5 模型)
model_outputs = self.model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_hidden_states=True,
            encoder_outputs=encoder_outputs
        )
- 输入：
  - inputs_embs: [B, L_seq*K, H]
    - 含义：用户历史交互序列 emb
  - attention_mask: [B, L_seq*K]
  - decoder_input_embs: [B, 3, H]
    - 含义：目标序列的 emb，使用 causal_mask 处理
  - output_hidden_states = True
    - 含义：强制返回每一层的隐状态
  - encoder_outputs: [B, L_seq*K, H]
    - 含义：Encoder 输出缓存，包含 last_hidden_state
- 输出 (Seq2SeqModelOutput)：
  1. last_hidden_state: [B, 3, H](即 decoder_hidden_states[-1]
    - 作用：Decoder 最后一层输出，用于预测下一个 token
  2. decoder_hidden_states: Tuple
    - 包含 Decoder 每一层的输出。
    - 注：output_hidden_states=True 强制返回每一层的状态。
  3. encoder_last_hidden_state ): [B, L_seq*K, H]
    - 作用：Encoder 最终输出的上下文表示
补：T5 内部过程
1. T5 Encoder Stack 
  - input_embeds+attention_mask-> Masked Self-attention -> FFN ->encoder_last_hidden_state
    - encoder_last_hidden_state ：模型对用户历史的深度理解
    - Self-attention (Bidirectional): 全向可见，可以看到 token 左边和右边。作用：捕捉序列内部的依赖关系。
    - Feed Forward Network (FFN): 标准的 MLP 层，用于特征变换。
2. T5 Decoder Stack
  - decoder_input_embeds+ encoder_last_hidden_state-> Masked Self-attention (Causal) -> Cross-Attention -> FFN -> decoder_output
    - Masked Self-attention: 加了一个 Causal Mask (因果掩码)，防止看到未来，确保生成是自回归的。
    - Cross-Attention:
      - Q: 某一层 Decoder 的输出（我现在想生成 Code，我的上下文？）
      - K/V: Encoder 输出（K是用户的历史行为特征，是被查询的索引。V是用户历史行为的内容）。
Step 4: 计算输出和对齐向量
1.1 生成任务 Logits (code_logits)
- 计算逻辑：构造一个线性分类器的权重矩阵，通过计算当前 hidden state 和每一个 code_emb 的点积（打分/度量），判断像哪一个 Code。
code_logits = []
for i in range(min(decoder_inputs_embeds.shape[1], self.code_length)):
    centroid = self.token_embeddings[i].weight.t() # (hidden_size, code_num)
    code_logits.append(torch.matmul(decoder_outputs[:, i], centroid))

code_logits = torch.stack(code_logits, dim=1) # (batch, code_len, code_num)
- 输出 code_logits:
  - 维度：[B, K, Code_num]
  - 作用：预测目标物品 K 个 token 的概率分布
1.2 序列-物品对齐向量 (SIA)
- seq_latents = model_outputs.encoder_last_hidden_state.clone()
  - 维度：[B, L_seq*K, H]
- Mask 处理：seq_latents[~attention_mask] = 0 (把 pad 位置置为 0)
- Mean Pooling: torch.sum(...) / attention_mask.sum(...)
  - 总和 / 有效长度 = 平均值。
  - 维度变化：[B, L_seq*K, H] -> [B, H]
- 投影：seq_project_latents = self.enc_adapter(seq_last_latents)
  - 维度：[B, E_dim] (Config e_dim)
  - 作用：适配 RQ-VAE 码本向量的维度，映射到 RQ-VAE 的隐空间
1.3 偏好-语义对齐向量 (PSA)
- dec_latents = model_outputs.decoder_hidden_states[-1].clone()
  - 维度：[B, K, H]
- 取首 Token：dec_latents[:, 0, :] (取第一个 token BOS 的输出)
  - 维度：[B, H]
- 投影：dec_latents = self.dec_adapter(dec_latents)
  - 维度：[B, Semantic_H]
  - 作用：还没生成具体 token 时，模型对用户下一个意图的判断（即用户偏好）。后续与 Target Item 重构后的 SASRec 向量进行 InfoNCE 学习。
Step 5: 打包返回 (QuantizeOutput)
- logits: [B, K, Code_num]
- seq_latents: [B, H] (T5 Encoder 侧原始序列表示)
- seq_project_latents: [B, E_dim] (投影后的版本，映射到 RQVAE 隐空间)
- dec_latents: [B, Semantic_H] (T5 Decoder 输出的起始状态 -> 映射到 SASRec Embedding 空间)

---
