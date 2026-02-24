"""
ETEGRec LLaMA2-7B 版本入口文件
基于 T5_to_LLaMA2_Migration_Plan v3.2

使用方式:
    accelerate launch main_llama.py --config ./config/llama_instrument2018.yaml
"""

import yaml
import argparse
import warnings
import torch
import os
import numpy as np

# === 5090 迁移专用补丁 ===
# TF32 设置：启用可加速 ~30%，但会有微小精度损失（通常 <0.1%）
# 设置为 True 以获得更快的训练速度
ENABLE_TF32 = True  # ⭐ 改为 True 可加速
torch.backends.cuda.matmul.allow_tf32 = ENABLE_TF32
torch.backends.cudnn.allow_tf32 = ENABLE_TF32
torch.use_deterministic_algorithms(not ENABLE_TF32, warn_only=True)
if not ENABLE_TF32:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from data import load_split_data
from data_llama import LlamaRecDataset, LlamaCollator
from torch.utils.data import DataLoader
from trainer_llama import LlamaTrainer
from model_llama import LlamaRecModel
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from utils import (
    init_seed, init_logger, init_device, log, get_local_time,
    get_file_name, convert_config_dict, safe_load, parse_command_line_args
)
from vq import RQVAE
from logging import getLogger

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/llama_instrument2018.yaml")
    parser.add_argument('--skip_id', action='store_true', help='跳过 train_id 阶段，只运行 train_rec（用于调试）')
    parser.add_argument('--debug', action='store_true', help='使用小数据集快速验证（1000 train, 500 valid/test）')
    parser.add_argument('--debug_samples', type=int, default=1000, help='debug 模式下的训练样本数')
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def train(config, verbose=True, rank=0, skip_id=False, debug=False, debug_samples=1000):
    """主训练函数"""
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    logger = getLogger()
    accelerator = config['accelerator']
    
    log(f'Device: {config["device"]}', accelerator, logger)
    log(f'Config: {str(config)}', accelerator, logger)
    
    if debug:
        log(f'⚠️ DEBUG 模式: 使用 {debug_samples} 训练样本, {debug_samples//2} 验证/测试样本', accelerator, logger)
    
    # === 加载数据 ===
    item2id, num_items, train_seq, valid_seq, test_seq = load_split_data(config)
    config['n_items'] = num_items
    
    # ⭐ Debug 模式：截取小数据集
    if debug:
        train_seq = train_seq[:debug_samples]
        valid_seq = valid_seq[:debug_samples//2]
        test_seq = test_seq[:debug_samples//2]
        log(f'截取后: train={len(train_seq)}, valid={len(valid_seq)}, test={len(test_seq)}', accelerator, logger)
    
    code_num = config['code_num']
    code_length = config['code_length']
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']
    
    # === 加载语义 Embedding ===
    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, dataset)
    
    collab_emb_file = config.get("collab_emb_path")
    text_emb_file = config.get("text_emb_path")
    
    if collab_emb_file and text_emb_file:
        # Dual SCID
        collab_emb_path = os.path.join(dataset_path, collab_emb_file)
        text_emb_path = os.path.join(dataset_path, text_emb_file)
        
        logger.info(f"Loading Dual Embeddings: {collab_emb_path} + {text_emb_path}")
        
        collab_emb = np.load(collab_emb_path)
        text_emb = np.load(text_emb_path)
        assert len(collab_emb) == len(text_emb), f"Length mismatch: {len(collab_emb)} vs {len(text_emb)}"
        
        if config.get('normalize', False):
            logger.info("Normalizing Dual Embeddings...")
            collab_emb = collab_emb / (np.linalg.norm(collab_emb, axis=1, keepdims=True) + 1e-9)
            text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-9)
        
        semantic_emb = np.concatenate((collab_emb, text_emb), axis=-1)
        config['semantic_hidden_size'] = semantic_emb.shape[-1]
        logger.info(f"Dual SCID enabled. semantic_hidden_size: {config['semantic_hidden_size']}")
    else:
        # 单一 Embedding
        semantic_emb_path = os.path.join(dataset_path, config["semantic_emb_path"])
        semantic_emb = np.load(semantic_emb_path)
    
    # 注意: num_items = len(item2id) 包含 PAD (索引0)
    # semantic_emb 长度 = num_items - 1 (不含 PAD)
    # 这是正确的设计，semantic_embedding[1:] = semantic_emb
    
    accelerator.wait_for_everyone()
    
    # === 初始化 RQ-VAE ===
    model_id = RQVAE(config=config, in_dim=config['semantic_hidden_size'])
    
    rqvae_path = config.get('rqvae_path', None)
    if rqvae_path is not None:
        safe_load(model_id, rqvae_path, verbose)
        log(f"Loaded RQ-VAE from {rqvae_path}", accelerator, logger)
    
    # === 初始化 LLaMA 推荐模型 ===
    llama_path = config.get('llama_path', 'models/Llama-2-7b-hf')
    model_rec = LlamaRecModel(
        config=config,
        rqvae=model_id,
        llama_path=llama_path
    )
    
    # 加载语义 Embedding 到模型
    model_rec.semantic_embedding.weight.data = torch.tensor(semantic_emb).to(config['device'])
    
    log(f"LlamaRecModel initialized", accelerator, logger)
    log(f"RQ-VAE initialized", accelerator, logger)
    
    # === 生成初始 item code 表 ===
    log("Generating initial item codes...", accelerator, logger)
    with torch.no_grad():
        all_item_embs = model_rec.semantic_embedding.weight.data[1:]
        all_item_prefix = model_id.get_indices(all_item_embs).detach().cpu()  # [n_items-1, 3]
    
    # 构建 code 表 (RQ-VAE 3层 + suffix 1层 = 4层)
    # 初始 suffix 设为 0，后续 trainer.get_code() 会重新计算
    all_item_code = torch.full((num_items, code_length), -1, dtype=torch.long)
    all_item_code[1:, :-1] = all_item_prefix  # 前 3 列填 prefix
    all_item_code[1:, -1] = 0  # 最后一列 suffix 初始为 0
    
    log(f"Item codes shape: {all_item_code.shape}", accelerator, logger)
    
    # === 创建 Dataset ===
    train_dataset = LlamaRecDataset(
        inter_seq=train_seq,
        all_item_code=all_item_code,
        code_length=code_length,
        max_seq_len=config.get('max_seq_len', 50)
    )
    valid_dataset = LlamaRecDataset(
        inter_seq=valid_seq,
        all_item_code=all_item_code,
        code_length=code_length,
        max_seq_len=config.get('max_seq_len', 50)
    )
    test_dataset = LlamaRecDataset(
        inter_seq=test_seq,
        all_item_code=all_item_code,
        code_length=code_length,
        max_seq_len=config.get('max_seq_len', 50)
    )
    
    # === 创建 DataLoader ===
    collator = LlamaCollator()
    num_workers = config['num_workers']
    
    # ⭐ DataLoader 配置：
    # - persistent_workers=False: 允许 worker 进程在每个 epoch 后重新创建
    #   这样 trainer._sync_code_table_to_datasets() 更新 all_item_code 后，
    #   新的 worker 进程会使用更新后的 code table
    # - 如果设为 True，worker 进程会缓存旧的 all_item_code，导致评估指标为 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False  # ⭐ 必须为 False，否则 code table 更新不生效
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False
    )
    
    # === 创建 Trainer ===
    trainer = LlamaTrainer(
        config=config,
        model_rec=model_rec,
        model_id=model_id,
        accelerator=accelerator,
        train_data=train_loader,
        valid_data=valid_loader,
        test_data=test_loader
    )
    
    # === 训练 ===
    best_score = trainer.train(verbose=verbose, skip_id=config.get('skip_id', False))
    
    # === 测试 ===
    test_results = trainer._test_epoch(test_data=trainer.test_data, verbose=verbose)
    
    if accelerator.is_main_process:
        log(f"Best Validation Score: {best_score}", accelerator, logger)
        log(f"Test Results: {test_results}", accelerator, logger)
    
    return best_score, test_results


if __name__ == "__main__":
    args, unparsed_args = parse_arguments()
    command_line_configs = parse_command_line_args(unparsed_args)
    
    # 加载配置
    config = {}
    config.update(yaml.safe_load(open(args.config, 'r')))
    config.update(command_line_configs)
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    dataset = config['dataset']
    local_time = get_local_time()
    config['device'], config['use_ddp'] = init_device()
    
    # DDP 配置：
    # - find_unused_parameters: LoRA 场景下部分参数可能不参与 loss
    # - 注意：不能用 static_graph=True，因为 train_id/train_rec 阶段图结构不同
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True
    )
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # 同步时间戳
    all_run_local_time = accelerator.gather_for_metrics([local_time])
    config['run_local_time'] = all_run_local_time[0]
    
    # 设置保存路径
    ckpt_name = get_file_name(config)
    config['save_path'] = f'./myckpt/llama_{dataset}/{ckpt_name}'
    
    config = convert_config_dict(config)
    config['accelerator'] = accelerator  # ⭐ 复用同一个 Accelerator，避免重复创建
    config['skip_id'] = args.skip_id  # 传递 skip_id 参数
    
    # 开始训练
    train(config, verbose=(local_rank == 0), rank=local_rank, 
          skip_id=args.skip_id, debug=args.debug, debug_samples=args.debug_samples)

