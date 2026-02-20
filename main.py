import yaml
import argparse
import warnings
import torch
# ================= 5090 迁移专用补丁 (完整版) =================
# 1. 强制关闭 TF32，防止精度损失 (最重要)
torch.backends.cuda.matmul.allow_tf32 = False 
torch.backends.cudnn.allow_tf32 = False

# 2. 开启全局确定性算法，防止并行计算带来的随机误差
# warn_only=True 表示如果某个算子没有确定性实现，只报错警告但不中断程序
torch.use_deterministic_algorithms(True, warn_only=True)
# 3. 设置环境变量，强制 CUDA 算子确定性 (可选，加上更稳)
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
    code_length = config['code_length'] # current length of the code
    eos_token_id = -1
    batch_size=config['batch_size']
    eval_batch_size=config['eval_batch_size']
    
    data_path = config["data_path"]
    dataset = config["dataset"]
    dataset_path = os.path.join(data_path, dataset)

    # Dual SCID Logic
    collab_emb_file = config.get("collab_emb_path")
    text_emb_file = config.get("text_emb_path")
    image_emb_file = config.get("image_emb_path")
    
    if collab_emb_file and text_emb_file:
        collab_emb_path = os.path.join(dataset_path, collab_emb_file)
        text_emb_path = os.path.join(dataset_path, text_emb_file)
        
        logger.info(f"Loading Dual Embeddings: {collab_emb_path} + {text_emb_path}")
        
        collab_emb = np.load(collab_emb_path)
        text_emb = np.load(text_emb_path)
        
        # Handle PAD token mismatch (Text emb usually has PAD at index 0)
        if len(text_emb) == len(collab_emb) + 1:
            logger.info("Detected PAD token in Text embeddings. Slicing [1:] to align with Collab.")
            text_emb = text_emb[1:]
            
        assert len(collab_emb) == len(text_emb), f"Length mismatch: {len(collab_emb)} vs {len(text_emb)}"
        
        # Normalize embeddings to unit sphere if configured
        if config.get('normalize', False):
            logger.info("Normalizing Dual Embeddings to unit sphere...")
            collab_emb = collab_emb / (np.linalg.norm(collab_emb, axis=1, keepdims=True) + 1e-9)
            text_emb = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-9)
        else:
            logger.info("Skipping Normalization (using raw embeddings)...")
        
        emb_list = [collab_emb, text_emb]
        
        # 加载图像嵌入 (可选，三模态 concat)
        if image_emb_file:
            image_emb_path = os.path.join(dataset_path, image_emb_file)
            logger.info(f"Loading Image Embeddings: {image_emb_path}")
            image_emb = np.load(image_emb_path)
            
            if len(image_emb) == len(collab_emb) + 1:
                logger.info("Detected PAD token in Image embeddings. Slicing [1:] to align.")
                image_emb = image_emb[1:]
            
            assert len(image_emb) == len(collab_emb), \
                f"Image emb length mismatch: {len(image_emb)} vs {len(collab_emb)}"
            
            if config.get('normalize', False):
                image_emb = image_emb / (np.linalg.norm(image_emb, axis=1, keepdims=True) + 1e-9)
            
            emb_list.append(image_emb)
        
        semantic_emb = np.concatenate(emb_list, axis=-1)
        config['semantic_hidden_size'] = semantic_emb.shape[-1]
        modality_names = "Collab+Text" + ("+Image" if image_emb_file else "")
        logger.info(f"{modality_names} enabled. semantic_hidden_size: {config['semantic_hidden_size']}")
        
    else:
        # Original Logic
        semantic_emb_path = os.path.join(dataset_path, config["semantic_emb_path"])
        semantic_emb = np.load(semantic_emb_path)
        # config['semantic_hidden_size'] is already set in yaml (e.g. 256)

    
    accelerator.wait_for_everyone()
    # Initialize the model with the custom configuration
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
    

    # semantic_emb is already loaded above
        
    model_rec.semantic_embedding.weight.data[1:] = torch.tensor(semantic_emb).to(config['device'])
    model_id = RQVAE(config=config, in_dim=model_rec.semantic_hidden_size)
    
    log(model_rec, accelerator, logger)
    log(model_id, accelerator, logger)

    rqvae_path = config.get('rqvae_path', None)
    if rqvae_path is not None:
        safe_load(model_id, rqvae_path, verbose)

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
    
    
    trainer = Trainer(config=config, model_rec=model_rec, model_id=model_id, accelerator=accelerator, train_data=train_data_loader,
                      valid_data=valid_data_loader, test_data=test_data_loader, eos_token_id=eos_token_id)
    
    best_score_pre = trainer.train(verbose=verbose)
    test_results_pre = trainer.test()

    best_score = trainer.finetune(verbose=verbose)
    test_results = trainer.test()
    
    
    if accelerator.is_main_process:
        log(f"Pre Best Validation Score: {best_score_pre}", accelerator, logger)
        log(f"Pre Test Results: {test_results_pre}", accelerator, logger)
        log(f"Best Validation Score: {best_score}", accelerator, logger)
        log(f"Test Results: {test_results}", accelerator, logger)


if __name__=="__main__":
    args, unparsed_args = parse_arguments()
    command_line_configs = parse_command_line_args(unparsed_args)

    # Config
    config = {}
    config.update(yaml.safe_load(open(args.config, 'r')))
    config.update(command_line_configs)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dataset = config['dataset']
    
    local_time = get_local_time()
    config['device'], config['use_ddp'] = init_device()
    accelerator = Accelerator()

    # gather all the config and set the checkpoint name
    all_run_local_time = accelerator.gather_for_metrics([local_time])
    config['run_local_time'] = all_run_local_time[0]

    ckpt_name = get_file_name(config)

    config['save_path'] =f'./myckpt/{dataset}/{ckpt_name}'
    
    config = convert_config_dict(config)
    config['accelerator'] = Accelerator()
    
        
    train(config, verbose=local_rank==0, rank=local_rank)

    

    
    