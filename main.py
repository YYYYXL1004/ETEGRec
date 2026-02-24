import yaml
import argparse
import warnings
import torch
# ================= 5090 迁移专用补丁 (完整版) =================
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from data import load_split_data
from data import SequentialSplitDataset, Collator
from torch.utils.data import DataLoader
from trainer import Trainer
from transformers import T5Config, T5ForConditionalGeneration
from accelerate import Accelerator
from model import Model
from utils import *
from vq import RQVAE
from logging import getLogger
import numpy as np
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/beauty.yaml")
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def train(config, verbose=True, rank=0):
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    logger = getLogger()
    accelerator = config['accelerator']

    log(f'Device: {config["device"]}', accelerator, logger)
    log(f'Config: {str(config)}', accelerator, logger)

    item2id, num_items, train, valid, test = load_split_data(config)
    code_num = config['code_num']
    code_length = config['code_length']
    eos_token_id = -1
    batch_size = config['batch_size']
    eval_batch_size = config['eval_batch_size']

    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, dataset)

    cross_modal = config.get('cross_modal', False)

    if cross_modal:
        # ============================================================
        # Cross-modal 双路模式:
        # text_semantic = concat(collab, text) = 1024dim
        # image_semantic = concat(collab, image) = 1024dim
        # ============================================================
        collab_emb_file = config.get("collab_emb_path")
        text_emb_file = config.get("text_emb_path")
        image_emb_file = config.get("image_emb_path")

        assert collab_emb_file and text_emb_file and image_emb_file, \
            "cross_modal=True 需要同时提供 collab_emb_path, text_emb_path, image_emb_path"

        collab_emb = np.load(os.path.join(dataset_path, collab_emb_file))
        text_emb = np.load(os.path.join(dataset_path, text_emb_file))
        image_emb = np.load(os.path.join(dataset_path, image_emb_file))

        assert len(collab_emb) == len(text_emb) == len(image_emb), \
            f"Length mismatch: collab={len(collab_emb)}, text={len(text_emb)}, image={len(image_emb)}"

        if config.get('normalize', False):
            logger.info("Normalizing embeddings to unit sphere...")
            collab_emb = collab_emb / (np.linalg.norm(collab_emb, axis=1, keepdims=True) + 1e-9)
            text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-9)
            image_emb = image_emb / (np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-9)

        # text_semantic = collab + text, image_semantic = collab + image
        text_semantic_emb = np.concatenate([collab_emb, text_emb], axis=-1)
        image_semantic_emb = np.concatenate([collab_emb, image_emb], axis=-1)

        config['text_semantic_hidden_size'] = text_semantic_emb.shape[-1]
        config['image_semantic_hidden_size'] = image_semantic_emb.shape[-1]
        # 兼容: semantic_hidden_size 设为 text 路的维度 (两路应该相同)
        config['semantic_hidden_size'] = text_semantic_emb.shape[-1]

        logger.info(f"Cross-modal enabled. text_semantic: {text_semantic_emb.shape}, "
                     f"image_semantic: {image_semantic_emb.shape}")

    else:
        # ============================================================
        # 原始模式: 单路或多模态 concat
        # ============================================================
        collab_emb_file = config.get("collab_emb_path")
        text_emb_file = config.get("text_emb_path")
        image_emb_file = config.get("image_emb_path")

        if collab_emb_file and text_emb_file:
            collab_emb = np.load(os.path.join(dataset_path, collab_emb_file))
            text_emb = np.load(os.path.join(dataset_path, text_emb_file))
            assert len(collab_emb) == len(text_emb)

            if config.get('normalize', False):
                collab_emb = collab_emb / (np.linalg.norm(collab_emb, axis=1, keepdims=True) + 1e-9)
                text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-9)

            emb_list = [collab_emb, text_emb]

            if image_emb_file:
                image_emb = np.load(os.path.join(dataset_path, image_emb_file))
                assert len(image_emb) == len(collab_emb)
                if config.get('normalize', False):
                    image_emb = image_emb / (np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-9)
                emb_list.append(image_emb)

            semantic_emb = np.concatenate(emb_list, axis=-1)
            config['semantic_hidden_size'] = semantic_emb.shape[-1]
            modality_names = "Collab+Text" + ("+Image" if image_emb_file else "")
            logger.info(f"{modality_names} enabled. semantic_hidden_size: {config['semantic_hidden_size']}")
        else:
            semantic_emb_path = os.path.join(dataset_path, config["semantic_emb_path"])
            semantic_emb = np.load(semantic_emb_path)


    accelerator.wait_for_everyone()

    # Initialize T5 model
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

    if cross_modal:
        # 加载双路 semantic embedding (frozen)
        # embedding 文件已包含 PAD (index 0 = 零向量)，直接整体赋值
        model_rec.text_semantic_embedding.weight.data = torch.tensor(text_semantic_emb).to(config['device'])
        model_rec.image_semantic_embedding.weight.data = torch.tensor(image_semantic_emb).to(config['device'])

        # 初始化 CrossRQVAE
        from cross_vq import CrossRQVAE
        model_id = CrossRQVAE(config=config, in_dim=config['text_semantic_hidden_size'])
    else:
        # 原始单路
        # embedding 文件已包含 PAD (index 0 = 零向量)，直接整体赋值
        model_rec.semantic_embedding.weight.data = torch.tensor(semantic_emb).to(config['device'])
        model_id = RQVAE(config=config, in_dim=model_rec.semantic_hidden_size)

    log(model_rec, accelerator, logger)
    log(model_id, accelerator, logger)

    # 加载预训练 RQVAE checkpoint
    rqvae_path = config.get('rqvae_path', None)
    if rqvae_path is not None:
        safe_load(model_id, rqvae_path, verbose)

    # Datasets & DataLoaders
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

    trainer = Trainer(config=config, model_rec=model_rec, model_id=model_id, accelerator=accelerator,
                      train_data=train_data_loader, valid_data=valid_data_loader,
                      test_data=test_data_loader, eos_token_id=eos_token_id)

    best_score_pre = trainer.train(verbose=verbose)
    test_results_pre = trainer.test()

    best_score = trainer.finetune(verbose=verbose)
    test_results = trainer.test()

    if accelerator.is_main_process:
        log(f"Pre Best Validation Score: {best_score_pre}", accelerator, logger)
        log(f"Pre Test Results: {test_results_pre}", accelerator, logger)
        log(f"Best Validation Score: {best_score}", accelerator, logger)
        log(f"Test Results: {test_results}", accelerator, logger)


if __name__ == "__main__":
    args, unparsed_args = parse_arguments()
    command_line_configs = parse_command_line_args(unparsed_args)

    config = {}
    config.update(yaml.safe_load(open(args.config, 'r')))
    config.update(command_line_configs)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dataset = config['dataset']

    local_time = get_local_time()
    config['device'], config['use_ddp'] = init_device()
    accelerator = Accelerator()

    all_run_local_time = accelerator.gather_for_metrics([local_time])
    config['run_local_time'] = all_run_local_time[0]

    ckpt_name = get_file_name(config)
    config['save_path'] = f'./myckpt/{dataset}/{ckpt_name}'

    config = convert_config_dict(config)
    config['accelerator'] = Accelerator()

    train(config, verbose=local_rank == 0, rank=local_rank)
