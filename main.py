import yaml
import argparse
import warnings
import torch
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
    semantic_emb_path = os.path.join(dataset_path, config["semantic_emb_path"])
    
    
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
    

    semantic_emb = np.load(semantic_emb_path)
        
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
    
    # Initialize metrics to 0 or None
    best_score_pre = 0.0
    test_results_pre = {}
    best_score = 0.0
    test_results = {}

    # 解析训练阶段 stage (sft / dpo / sft_dpo)
    stage = config.get('stage', 'sft')
    
    accelerator.print(f"\n>>> Current Training Stage: {stage.upper()}")

    # === 1. SFT 阶段 (Pre-train + Finetune) ===
    if stage in ['sft', 'sft_dpo']:
        best_score_pre = trainer.train(verbose=verbose)
        test_results_pre = trainer.test()

        best_score = trainer.finetune(verbose=verbose)
        test_results = trainer.test()
    elif stage == 'dpo':
        accelerator.print(">>> Skipping SFT stage (Direct DPO).")

    # === 2. DPO 阶段 ===
    if stage in ['dpo', 'sft_dpo']:
        accelerator.print("\n>>> Starting DPO Alignment Stage...")
        
        # 确定要加载的 Checkpoint
        # 优先用刚才 finetune 出来的 best_ckpt (如果跑了的话)，如果没有则用 config 里指定的
        sft_ckpt = trainer.best_ckpt if trainer.best_ckpt else config.get('dpo_model_path')
        
        if sft_ckpt:
            accelerator.print(f"Loading SFT Checkpoint for DPO: {sft_ckpt}")
            
            # 执行 DPO 训练
            trainer.train_dpo(saved_sft_ckpt=sft_ckpt, verbose=verbose)
            
            # 测试 DPO 后的模型
            # 显式指定加载 DPO 保存的最终模型进行测试
            dpo_final_ckpt = os.path.join(trainer.save_path, "dpo_final.pt")
            if os.path.exists(dpo_final_ckpt):
                accelerator.print(f"Testing DPO model: {dpo_final_ckpt}")
                test_results_dpo = trainer.test(model_file=dpo_final_ckpt)
            else:
                accelerator.print("Warning: DPO final checkpoint not found. Testing current in-memory model.")
                test_results_dpo = trainer.test(model_file=None)

            if accelerator.is_main_process:
                log(f"DPO Stage Finished. Test Results: {test_results_dpo}", accelerator, logger)
            
            # Update global results for final logging
            test_results = test_results_dpo
            best_score = test_results.get('ndcg@10', 0.0) # Use ndcg@10 as the representative score
        else:
            accelerator.print("Error: No SFT checkpoint found for DPO. Please specify 'dpo_model_path' or run SFT first.")
    
    
    if accelerator.is_main_process:
        log(f"Pre Best Validation Score: {best_score_pre}", accelerator, logger)
        log(f"Pre Test Results: {test_results_pre}", accelerator, logger)
        log(f"Best Validation Score: {best_score}", accelerator, logger)
        log(f"Test Results: {test_results}", accelerator, logger)
    
    # Clean up distributed process group to avoid warnings
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


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

    

    
    