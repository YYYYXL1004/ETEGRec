"""
LLaMA 版本的数据加载器
基于 T5_to_LLaMA2_Migration_Plan v3.2

核心设计：
1. 将 item IDs 转换为 codes 序列
2. 计算 seq_end_position 和 target_positions
3. 左 Padding (LLaMA 习惯)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any


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
        
        # 预处理数据
        self.data = self._preprocess(inter_seq)
        print(f"[LlamaRecDataset] 加载 {len(self.data)} 条数据，"
              f"max_seq_len={max_seq_len}, code_length={code_length}")
    
    def _preprocess(self, inter_seq: List[List[int]]) -> List[Dict]:
        """预处理：分离历史和目标"""
        data = []
        for seq in inter_seq:
            target = seq[-1]
            history = seq[:-1]
            data.append({
                'history': history,
                'target': target
            })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        history = item['history']
        target_item = item['target']
        
        # === 1. 截断历史序列 ===
        history = history[-self.max_seq_len:]
        
        # === 2. 构造历史序列的 codes ===
        history_codes = []
        for item_id in history:
            item_codes = self.all_item_code[item_id].tolist()  # [code_length]
            history_codes.extend(item_codes)
        
        # === 3. 构造目标序列的 codes ===
        target_codes = self.all_item_code[target_item].tolist()  # [code_length]
        
        # === 4. 拼接: 历史 + 目标 ===
        input_ids = history_codes + target_codes
        
        # === 5. 计算关键位置 ===
        # seq_end_position: 历史序列最后一个 token 的位置
        seq_end_position = len(history_codes) - 1
        
        # target_positions: 目标 code 各位置的索引
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
    """
    动态 Padding (左 Padding，LLaMA 习惯)
    
    Args:
        batch: 一个 batch 的样本列表
    
    Returns:
        collated batch dict
    """
    max_len = max(len(b['input_ids']) for b in batch)
    
    input_ids = []
    attention_mask = []
    seq_end_positions = []
    target_positions = []
    labels = []
    targets = []
    
    for b in batch:
        cur_len = len(b['input_ids'])
        pad_len = max_len - cur_len
        
        # 左 Padding (LLaMA 习惯)
        # padding value = -1 (后续会在 get_input_embeddings 中处理)
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
    """
    Collator 类封装，方便传入 DataLoader
    """
    
    def __init__(self):
        pass
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return collate_fn_llama(batch)


# === 工具函数：创建 DataLoader ===
def create_llama_dataloader(inter_seq, all_item_code, config, shuffle=True):
    """
    创建 LLaMA 版本的 DataLoader
    
    Args:
        inter_seq: 交互序列
        all_item_code: item code 表
        config: 配置字典
        shuffle: 是否打乱
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset = LlamaRecDataset(
        inter_seq=inter_seq,
        all_item_code=all_item_code,
        code_length=config.get('code_length', 4),
        max_seq_len=config.get('max_seq_len', 50)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=shuffle,
        collate_fn=LlamaCollator(),
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return dataloader

