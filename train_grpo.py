"""
GRPO Training Script for ETEGRec

Usage:
    python train_grpo.py --config ./config/beauty.yaml --sft_ckpt ./myckpt/xxx/best.pt

或者用 accelerate:
    accelerate launch train_grpo.py --config ./config/beauty.yaml --sft_ckpt ./myckpt/xxx/best.pt
"""
import os
import copy
import yaml
import json
import argparse
import warnings
import torch
import numpy as np
from data import load_split_data
from data import SequentialSplitDataset, Collator
from torch.utils.data import DataLoader
from grpo_trainer import GRPOTrainer
from transformers import T5Config, T5ForConditionalGeneration
from accelerate import Accelerator
from model import Model
from utils import *
from vq import RQVAE
from logging import getLogger

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/beauty.yaml")
    parser.add_argument('--sft_ckpt', type=str, required=True, 
                        help="Path to SFT checkpoint (.pt file)")
    parser.add_argument('--rqvae_ckpt', type=str, default=None,
                        help="Path to RQ-VAE checkpoint (.pt.rqvae file). If not provided, will try sft_ckpt.rqvae")
    parser.add_argument('--code_json', type=str, default=None,
                        help="Path to code json file. If not provided, will try sft_ckpt.code.json")
    
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def train_grpo(config, sft_ckpt, rqvae_ckpt, code_json, verbose=True, rank=0):
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    logger = getLogger()
    accelerator = config['accelerator']
    
    log(f'[GRPO] Device: {config["device"]}', accelerator, logger)
    log(f'[GRPO] SFT Checkpoint: {sft_ckpt}', accelerator, logger)
    
    # Load data
    item2id, num_items, train, valid, test = load_split_data(config)
    code_num = config['code_num']
    code_length = config['code_length']
    eos_token_id = -1
    batch_size = config.get('grpo_batch_size', config['batch_size'] // 2)  # GRPO 需要更小的 batch
    eval_batch_size = config['eval_batch_size']
    
    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, dataset)
    
    # Load semantic embeddings (same as main.py)
    collab_emb_file = config.get("collab_emb_path")
    text_emb_file = config.get("text_emb_path")
    
    if collab_emb_file and text_emb_file:
        collab_emb_path = os.path.join(dataset_path, collab_emb_file)
        text_emb_path = os.path.join(dataset_path, text_emb_file)
        collab_emb = np.load(collab_emb_path)
        text_emb = np.load(text_emb_path)
        assert len(collab_emb) == len(text_emb), \
            f"Length mismatch: collab {len(collab_emb)} vs text {len(text_emb)}"
        if config.get('normalize', False):
            collab_emb = collab_emb / (np.linalg.norm(collab_emb, axis=1, keepdims=True) + 1e-9)
            text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-9)
        semantic_emb = np.concatenate((collab_emb, text_emb), axis=-1)
        config['semantic_hidden_size'] = semantic_emb.shape[-1]
    else:
        semantic_emb_path = os.path.join(dataset_path, config["semantic_emb_path"])
        semantic_emb = np.load(semantic_emb_path)
    
    accelerator.wait_for_everyone()
    
    # Build model architecture (same as main.py)
    model_config = T5Config(
        num_layers=config['encoder_layers'],
        num_decoder_layers=config['decoder_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'],
        activation_function=config['activation_function'],
        vocab_size=1,
        pad_token_id=0,
        eos_token_id=300,
        decoder_start_token_id=0,
        feed_forward_proj=config['feed_forward_proj'],
        n_positions=config['max_length'],
    )
    
    t5 = T5ForConditionalGeneration(config=model_config)
    model_rec = Model(config=config, model=t5, n_items=num_items,
                      code_length=code_length, code_number=code_num)
    model_rec.semantic_embedding.weight.data = torch.tensor(semantic_emb).to(config['device'])
    model_id = RQVAE(config=config, in_dim=model_rec.semantic_hidden_size)
    
    # Load SFT checkpoint
    log(f'[GRPO] Loading SFT model from {sft_ckpt}', accelerator, logger)
    safe_load(model_rec, sft_ckpt, verbose)
    
    # Load RQ-VAE checkpoint
    if rqvae_ckpt is None:
        rqvae_ckpt = f"{sft_ckpt}.rqvae"
    log(f'[GRPO] Loading RQ-VAE from {rqvae_ckpt}', accelerator, logger)
    safe_load(model_id, rqvae_ckpt, verbose)
    
    # Load pre-computed codes
    if code_json is None:
        code_json = sft_ckpt.replace('.pt', '.code.json')
    if os.path.exists(code_json):
        log(f'[GRPO] Loading codes from {code_json}', accelerator, logger)
        all_item_code = json.load(open(code_json, 'r'))
    else:
        log(f'[GRPO] Code json not found, will compute codes', accelerator, logger, level='warning')
        all_item_code = None
    
    # Create ref_model (frozen copy of SFT model)
    log('[GRPO] Creating frozen reference model', accelerator, logger)
    ref_t5 = T5ForConditionalGeneration(config=model_config)
    ref_model = Model(config=config, model=ref_t5, n_items=num_items,
                      code_length=code_length, code_number=code_num)
    ref_model.semantic_embedding.weight.data = torch.tensor(semantic_emb).to(config['device'])
    safe_load(ref_model, sft_ckpt, verbose=False)
    ref_model = ref_model.to(config['device'])
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # Move models to device
    model_rec = model_rec.to(config['device'])
    model_id = model_id.to(config['device'])
    
    log(f'[GRPO] Model loaded. model_rec params: {sum(p.numel() for p in model_rec.parameters())}', 
        accelerator, logger)
    
    # Data loaders
    train_dataset = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=train)
    valid_dataset = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=valid)
    test_dataset = SequentialSplitDataset(config=config, n_items=num_items, inter_seq=test)
    
    collator = Collator(eos_token_id=eos_token_id, pad_token_id=0, max_length=config['max_length'])
    
    train_data_loader = DataLoader(train_dataset, num_workers=config["num_workers"], collate_fn=collator,
                                   batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, num_workers=config["num_workers"], collate_fn=collator,
                                   batch_size=eval_batch_size, shuffle=False, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, num_workers=config["num_workers"], collate_fn=collator,
                                  batch_size=eval_batch_size, shuffle=False, pin_memory=True)
    
    # Create GRPO Trainer
    trainer = GRPOTrainer(
        config=config,
        model_rec=model_rec,
        model_id=model_id,
        ref_model=ref_model,
        accelerator=accelerator,
        train_data=train_data_loader,
        valid_data=valid_data_loader,
        test_data=test_data_loader,
        all_item_code=all_item_code,
        eos_token_id=eos_token_id
    )
    
    # Set item codes (will build reverse index)
    if all_item_code is not None:
        trainer.set_item_codes(all_item_code)
    else:
        # Compute codes if not provided
        all_item_code = trainer.get_code(epoch_idx=-1, verbose=verbose)
        trainer.set_item_codes(all_item_code)
    
    # Start GRPO training
    best_score = trainer.train_grpo(verbose=verbose)
    
    # Final test
    test_results = trainer.test()
    
    if accelerator.is_main_process:
        log(f"[GRPO] Best Validation Score: {best_score}", accelerator, logger)
        log(f"[GRPO] Test Results: {test_results}", accelerator, logger)
    
    return best_score, test_results


if __name__ == "__main__":
    args, unparsed_args = parse_arguments()
    command_line_configs = parse_command_line_args(unparsed_args)
    
    # Load config
    config = {}
    config.update(yaml.safe_load(open(args.config, 'r')))
    config.update(command_line_configs)
    
    # GRPO default config (can be overridden by command line)
    grpo_defaults = {
        'grpo_group_size': 4,
        'grpo_kl_coeff': 0.05,
        'grpo_temperature': 1.0,
        'grpo_top_k': 50,
        'grpo_reward_alpha': 0.5,
        'grpo_clip_range': 0.2,
        'grpo_epochs': 20,
        'grpo_early_stop': 5,
        'grpo_eval_step': 1,
        'grpo_lr': 1e-5,
        'grpo_learner': 'adamw',
    }
    for k, v in grpo_defaults.items():
        if k not in config:
            config[k] = v
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    dataset = config['dataset']
    local_time = get_local_time()
    config['device'], config['use_ddp'] = init_device()
    accelerator = Accelerator()
    
    all_run_local_time = accelerator.gather_for_metrics([local_time])
    config['run_local_time'] = all_run_local_time[0]
    
    ckpt_name = get_file_name(config) + '_grpo'
    config['save_path'] = f'./myckpt/{dataset}/{ckpt_name}'
    
    config = convert_config_dict(config)
    config['accelerator'] = Accelerator()
    
    train_grpo(
        config=config,
        sft_ckpt=args.sft_ckpt,
        rqvae_ckpt=args.rqvae_ckpt,
        code_json=args.code_json,
        verbose=(local_rank == 0),
        rank=local_rank
    )
