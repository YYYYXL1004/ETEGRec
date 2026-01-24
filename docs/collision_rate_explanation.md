# RQ-VAE 碰撞率 (Collision Rate) 计算说明

本文档详细解释了在 `RQ-VAE` 训练过程中 `collision_rate` 指标的计算方法及其意义。

## 1. 什么是碰撞率？

碰撞率 (Collision Rate) 是一个用来衡量向量量化 (Vector Quantization) 效果的指标，特别是在 `RQ-VAE` 或 `VQ-VAE` 这类模型中。它反映了模型为 **不同** 输入数据生成 **相同** 离散编码表示的频率。

一个理想的量化模型应该能为不同的输入数据赋予独特、有区分度的离散编码。如果大量不同的输入都被映射到了同一个编码上，我们就称之为“碰撞”。

**碰撞率越低，代表模型的码本（codebook）利用率越高，量化效果越好。**

## 2. 计算逻辑详解

碰撞率的计算逻辑主要位于 `RQVAE/trainer.py` 文件中的 `_valid_epoch` 方法内。其核心步骤如下：

### 步骤一：获取量化索引 (Indices)

模型在评估（validation）阶段会遍历所有验证数据。对于每一个输入样本，模型会通过 `model.get_indices(data)` 方法得到其对应的码本索引序列。

例如，一个样本可能被编码为 `[10, 5, 128]` 这样的索引组合。

### 步骤二：生成唯一编码字符串

为了方便统计和比较，代码会将每个样本的索引序列转换成一个唯一的字符串。这是通过用连字符 `-` 将索引连接起来实现的。

-   索引序列 `[10, 5, 128]` 会被转换为字符串 `"10-5-128"`。
-   另一个样本的索引序列 `[2, 45, 99]` 会被转换为 `"2-45-99"`。

### 步骤三：统计不重复的编码数量

代码使用一个 Python 的 `set` (集合) 数据结构来存储所有样本生成的唯一编码字符串。`set` 的特性是它内部不允许有重复元素。

因此，遍历完所有验证样本后，`set` 的大小（`len(indices_set)`）就精确地代表了在整个验证集中，模型总共生成了多少种 **不重复的** 编码。

### 步骤四：计算碰撞率

碰撞率的计算公式如下：

```python
# num_sample: 验证集中的样本总数
# len(indices_set): 不重复编码的总数
collision_rate = (num_sample - len(indices_set)) / num_sample
```

**公式解读**：

-   `num_sample - len(indices_set)`：这个差值计算出了发生了“碰撞”的样本数量。换句话说，就是那些其编码与其他样本重复的样本总数。
-   将这个差值除以总样本数 `num_sample`，就得到了标准化的碰撞率。

## 3. 代码片段

以下是 `trainer.py` 中计算碰撞率的核心代码：

```python
@torch.no_grad()
def _valid_epoch(self, valid_data):

    self.model.eval()

    # ... (tqdm setup) ...

    indices_set = set()
    num_sample = 0
    for batch_idx, data in enumerate(iter_data):
        num_sample += len(data)
        data = data.to(self.device)
        indices = self.model.get_indices(data)
        indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = "-".join([str(int(_)) for _ in index])
            indices_set.add(code)

    collision_rate = (num_sample - len(list(indices_set)))/num_sample

    return collision_rate
```

## 4. 总结

在模型训练过程中，我们会监控 `collision_rate` 的变化。我们期望这个值能随着训练的进行而逐渐降低，这表明模型正在学习如何更有效地利用其码本，为数据生成更丰富、更多样化的离散表示。
