"""
DDP 双卡 OOM 测试脚本
用法: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test_generate_oom.py

测试内容:
1. train_rec: forward + backward + optimizer step (batch_size=16 per GPU)
2. generate: beam search (eval_batch_size=1, num_beams=20)
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
import sys
import gc
import os

sys.path.insert(0, '/data/yaoxianglin/ETEGRec')

from model_llama import LlamaRecModel
from vq import RQVAE
from utils import safe_load

def setup_ddp():
    """初始化 DDP"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    """清理 DDP"""
    dist.destroy_process_group()

def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def print_memory(prefix="", rank=0):
    if rank == 0:
        current = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[Memory] {prefix} 当前: {current:.2f} GB, 峰值: {peak:.2f} GB")

def main():
    local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    
    # === 加载配置 ===
    config_path = './config/llama_instrument2018.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # ⭐ 使用真实的 n_items (与实际训练一致)
    config['n_items'] = 10450  # Instrument2018_5090 数据集的真实大小

    if local_rank == 0:
        print("="*60)
        print("DDP 双卡 OOM 测试脚本 (v2 - 更接近真实场景)")
        print("="*60)
        print(f"n_items={config['n_items']}, semantic_dim={config['semantic_hidden_size']}")
        print(f"code_length={config['code_length']}, num_beams={config['num_beams']}")
        print(f"World size: {dist.get_world_size()}")

    # === 加载模型 ===
    if local_rank == 0:
        print("\n[1/4] 加载 RQVAE...")
    model_id = RQVAE(config=config, in_dim=config['semantic_hidden_size'])
    rqvae_path = config.get('rqvae_path', None)
    if rqvae_path is not None:
        safe_load(model_id, rqvae_path, verbose=False)
    model_id = model_id.to(device)
    model_id.eval()

    if local_rank == 0:
        print("[2/4] 加载 LlamaRecModel...")
    llama_path = config.get('llama_path', 'models/Llama-2-7b-hf')
    model_rec = LlamaRecModel(config, model_id, llama_path=llama_path)
    model_rec = model_rec.to(device)
    
    print_memory("模型加载后", local_rank)

    # === DDP 包装 ===
    if local_rank == 0:
        print("[3/4] DDP 包装...")
    model_rec = DDP(model_rec, device_ids=[local_rank], find_unused_parameters=True)
    model_id = DDP(model_id, device_ids=[local_rank])
    
    print_memory("DDP 包装后", local_rank)

    # === 创建 Optimizer ===
    if local_rank == 0:
        print("[4/4] 创建 Optimizer...")
    optimizer = torch.optim.AdamW(
        model_rec.parameters(),
        lr=config.get('lr_rec', 0.0001),
        weight_decay=config.get('weight_decay', 0.05)
    )

    # 初始化 optimizer state
    dummy_input = torch.randint(0, config['code_num'], (1, 10)).to(device)
    dummy_mask = torch.ones(1, 10, dtype=torch.long).to(device)
    dummy_seq_end = torch.tensor([5], dtype=torch.long).to(device)
    dummy_target_pos = torch.tensor([[6, 7, 8, 9]], dtype=torch.long).to(device)

    model_rec.train()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model_rec(dummy_input, dummy_mask, dummy_seq_end, dummy_target_pos)
        loss = outputs.logits.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    del outputs, loss, dummy_input, dummy_mask, dummy_seq_end, dummy_target_pos
    reset_memory()
    
    print_memory("Optimizer 初始化后", local_rank)
    baseline_mem = torch.cuda.memory_allocated() / 1024**3
    if local_rank == 0:
        print(f"\n>>> 基线显存: {baseline_mem:.2f} GB")

    dist.barrier()

    # ============================================================
    # 测试 1: train_rec (完整版，包含 SIA/PSA Loss)
    # ============================================================
    if local_rank == 0:
        print("\n" + "="*60)
        print("[测试1] train_rec: forward + backward + optimizer step")
        print("        (包含完整的 code_loss + kl_loss + dec_cl_loss)")
        print("="*60)

    train_batch_size = 16  # 每卡 batch_size
    seq_len = 210

    if local_rank == 0:
        print(f"每卡 batch_size={train_batch_size}, seq_len={seq_len}")

    input_ids = torch.randint(0, config['code_num'], (train_batch_size, seq_len)).to(device)
    attention_mask = torch.ones(train_batch_size, seq_len, dtype=torch.long).to(device)
    seq_end_positions = torch.full((train_batch_size,), seq_len - config['code_length'] - 1, dtype=torch.long).to(device)
    target_positions = torch.stack([
        torch.arange(seq_len - config['code_length'], seq_len) for _ in range(train_batch_size)
    ]).to(device)
    labels = torch.randint(0, config['code_num'], (train_batch_size, config['code_length'])).to(device)
    targets = torch.randint(1, config['n_items'], (train_batch_size,)).to(device)

    # ⭐ 完整的 train_rec 流程（与 trainer_llama.py 一致）
    model_rec.train()
    model_id.eval()
    
    torch.cuda.reset_peak_memory_stats()

    train_success = False
    try:
        # 获取目标 item 的语义 embedding
        target_semantic_embs = model_rec.module.semantic_embedding(targets)
        
        # model_id forward (冻结状态)
        with torch.no_grad():
            target_recon_embs, _, _, _, target_code_logits = model_id.module(target_semantic_embs)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # model_rec forward
            outputs = model_rec(
                input_ids=input_ids,
                attention_mask=attention_mask,
                seq_end_positions=seq_end_positions,
                target_positions=target_positions,
            )
            print_memory("Forward 后", local_rank)
            
            # 1. Code Loss
            code_loss = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, config['code_num']),
                labels.view(-1)
            )
            
            # 2. SIA Loss (KL 散度) - 需要 model_id.rq forward
            _, _, _, _, seq_code_logits = model_id.module.rq(outputs.seq_project_latents)
            
            x_logits = torch.nn.functional.log_softmax(seq_code_logits.view(-1, config['code_num']), dim=-1)
            y_logits = torch.nn.functional.log_softmax(target_code_logits.view(-1, config['code_num']), dim=-1)
            kl_loss = torch.nn.functional.kl_div(x_logits, y_logits, reduction='batchmean', log_target=True)
            
            # 3. PSA Loss (InfoNCE)
            query_embeds = torch.nn.functional.normalize(outputs.dec_latents, dim=-1)
            semantic_embeds = torch.nn.functional.normalize(target_recon_embs, dim=-1)
            similarities = torch.matmul(query_embeds, semantic_embeds.t()) / 0.07
            dec_cl_loss = torch.nn.functional.cross_entropy(
                similarities, 
                torch.arange(train_batch_size, device=device)
            )
            
            # 总 Loss
            loss = code_loss + 0.0001 * kl_loss + 0.0003 * dec_cl_loss
        
        loss.backward()
        print_memory("Backward 后", local_rank)
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print_memory("Optimizer step 后", local_rank)
        
        train_peak = torch.cuda.max_memory_allocated() / 1024**3
        if local_rank == 0:
            print(f"[测试1] ✓ train_rec 成功! 峰值显存: {train_peak:.2f} GB")
        train_success = True
        
    except torch.cuda.OutOfMemoryError as e:
        if local_rank == 0:
            print(f"[测试1] ✗ OOM! {e}")
    except Exception as e:
        if local_rank == 0:
            print(f"[测试1] ✗ 其他错误: {e}")
            import traceback
            traceback.print_exc()

    # 清理
    try:
        del outputs, loss, code_loss, kl_loss, dec_cl_loss
        del seq_code_logits, x_logits, y_logits, query_embeds, semantic_embeds, similarities
    except:
        pass
    del input_ids, attention_mask, seq_end_positions, target_positions, labels, targets
    del target_semantic_embs, target_recon_embs, target_code_logits
    model_rec.zero_grad(set_to_none=True)
    reset_memory()

    dist.barrier()

    # ============================================================
    # 测试 2: generate (多 batch 连续评估)
    # ============================================================
    if local_rank == 0:
        print("\n" + "="*60)
        print("[测试2] generate: beam search evaluate (连续多 batch)")
        print("="*60)

    eval_batch_size = config.get('eval_batch_size', 4)  # 使用配置中的值
    eval_seq_len = 210
    num_eval_batches = 5  # 连续评估多个 batch

    if local_rank == 0:
        print(f"eval_batch_size={eval_batch_size}, seq_len={eval_seq_len}, num_beams={config['num_beams']}")
        print(f"连续评估 {num_eval_batches} 个 batch")

    model_rec.eval()
    torch.cuda.reset_peak_memory_stats()

    generate_success = False
    try:
        with torch.no_grad():
            for batch_idx in range(num_eval_batches):
                input_ids = torch.randint(0, config['code_num'], (eval_batch_size, eval_seq_len)).to(device)
                attention_mask = torch.ones(eval_batch_size, eval_seq_len, dtype=torch.long).to(device)
                
                preds = model_rec.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_return_sequences=10
                )
                
                # 模拟 _test_epoch 中的变量释放
                del preds, input_ids, attention_mask
                
                if local_rank == 0 and batch_idx == 0:
                    print_memory(f"Generate batch {batch_idx+1} 后", local_rank)
        
        gen_peak = torch.cuda.max_memory_allocated() / 1024**3
        if local_rank == 0:
            print(f"[测试2] ✓ generate 成功! 峰值显存: {gen_peak:.2f} GB")
        generate_success = True
        
    except torch.cuda.OutOfMemoryError as e:
        if local_rank == 0:
            print(f"[测试2] ✗ OOM! {e}")
    except Exception as e:
        if local_rank == 0:
            print(f"[测试2] ✗ 其他错误: {e}")
            import traceback
            traceback.print_exc()

    dist.barrier()

    # ============================================================
    # 测试 3: train_rec → generate → train_id 切换 (完整版)
    # ============================================================
    if local_rank == 0:
        print("\n" + "="*60)
        print("[测试3] 模拟真实场景: train_rec → generate → train_id")
        print("        (完整 Loss 计算，与 trainer_llama.py 一致)")
        print("="*60)
    
    # 清理显存
    reset_memory()
    
    # --- 3.1 train_rec (完整版) ---
    if local_rank == 0:
        print("\n[3.1] train_rec (完整 Loss)...")
    
    train_batch_size = 16
    seq_len = 210
    
    input_ids = torch.randint(0, config['code_num'], (train_batch_size, seq_len)).to(device)
    attention_mask = torch.ones(train_batch_size, seq_len, dtype=torch.long).to(device)
    seq_end_positions = torch.full((train_batch_size,), seq_len - config['code_length'] - 1, dtype=torch.long).to(device)
    target_positions = torch.stack([
        torch.arange(seq_len - config['code_length'], seq_len) for _ in range(train_batch_size)
    ]).to(device)
    labels = torch.randint(0, config['code_num'], (train_batch_size, config['code_length'])).to(device)
    targets = torch.randint(1, config['n_items'], (train_batch_size,)).to(device)
    
    model_rec.train()
    model_id.eval()
    
    # 完整的 train_rec forward + backward
    target_semantic_embs = model_rec.module.semantic_embedding(targets)
    with torch.no_grad():
        target_recon_embs, _, _, _, target_code_logits = model_id.module(target_semantic_embs)
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model_rec(input_ids, attention_mask, seq_end_positions, target_positions)
        code_loss = torch.nn.functional.cross_entropy(outputs.logits.view(-1, config['code_num']), labels.view(-1))
        _, _, _, _, seq_code_logits = model_id.module.rq(outputs.seq_project_latents)
        x_logits = torch.nn.functional.log_softmax(seq_code_logits.view(-1, config['code_num']), dim=-1)
        y_logits = torch.nn.functional.log_softmax(target_code_logits.view(-1, config['code_num']), dim=-1)
        kl_loss = torch.nn.functional.kl_div(x_logits, y_logits, reduction='batchmean', log_target=True)
        query_embeds = torch.nn.functional.normalize(outputs.dec_latents, dim=-1)
        semantic_embeds = torch.nn.functional.normalize(target_recon_embs, dim=-1)
        similarities = torch.matmul(query_embeds, semantic_embeds.t()) / 0.07
        dec_cl_loss = torch.nn.functional.cross_entropy(similarities, torch.arange(train_batch_size, device=device))
        loss = code_loss + 0.0001 * kl_loss + 0.0003 * dec_cl_loss
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    # 清理
    del outputs, loss, code_loss, kl_loss, dec_cl_loss, seq_code_logits
    del x_logits, y_logits, query_embeds, semantic_embeds, similarities
    del input_ids, attention_mask, seq_end_positions, target_positions, labels
    del target_semantic_embs, target_recon_embs, target_code_logits, targets
    
    print_memory("train_rec 后", local_rank)
    
    # ⭐ 关键：训练后清理显存
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("清理后", local_rank)
    
    # --- 3.2 generate (evaluate) - 多 batch ---
    if local_rank == 0:
        print("\n[3.2] generate (evaluate, 5 batches)...")
    
    eval_batch_size = config.get('eval_batch_size', 4)
    eval_seq_len = 210
    
    model_rec.eval()
    
    with torch.no_grad():
        for batch_idx in range(5):
            input_ids = torch.randint(0, config['code_num'], (eval_batch_size, eval_seq_len)).to(device)
            attention_mask = torch.ones(eval_batch_size, eval_seq_len, dtype=torch.long).to(device)
            preds = model_rec.module.generate(input_ids, attention_mask, num_return_sequences=10)
            del preds, input_ids, attention_mask
    
    print_memory("generate 后", local_rank)
    
    # ⭐ 关键：评估后清理显存
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print_memory("清理后", local_rank)
    
    # --- 3.3 train_id (完整版) ---
    if local_rank == 0:
        print("\n[3.3] train_id (完整 Loss)...")
    
    train_id_success = False
    try:
        # ⭐ 再次清理（模拟 trainer 的逻辑）
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        model_id.train()
        model_rec.eval()
        
        # 创建 train_id 的 optimizer
        id_optimizer = torch.optim.AdamW(model_id.parameters(), lr=config.get('lr_id', 0.0001))
        
        # 完整的 train_id forward + backward
        input_ids = torch.randint(0, config['code_num'], (train_batch_size, seq_len)).to(device)
        attention_mask = torch.ones(train_batch_size, seq_len, dtype=torch.long).to(device)
        seq_end_positions = torch.full((train_batch_size,), seq_len - config['code_length'] - 1, dtype=torch.long).to(device)
        target_positions = torch.stack([
            torch.arange(seq_len - config['code_length'], seq_len) for _ in range(train_batch_size)
        ]).to(device)
        targets = torch.randint(1, config['n_items'], (train_batch_size,)).to(device)
        
        target_semantic_embs = model_rec.module.semantic_embedding(targets)
        target_recon_embs, commit_loss, _, _, target_code_logits = model_id(target_semantic_embs)
        
        # model_rec forward (冻结)
        with torch.no_grad():
            outputs = model_rec.module(input_ids, attention_mask, seq_end_positions, target_positions)
        
        # VQ Loss
        vq_loss = torch.nn.functional.mse_loss(target_recon_embs, target_semantic_embs) + commit_loss
        
        # SIA Loss
        _, _, _, _, seq_code_logits = model_id.module.rq(outputs.seq_project_latents)
        x_logits = torch.nn.functional.log_softmax(seq_code_logits.view(-1, config['code_num']), dim=-1)
        y_logits = torch.nn.functional.log_softmax(target_code_logits.view(-1, config['code_num']), dim=-1)
        kl_loss = torch.nn.functional.kl_div(x_logits, y_logits, reduction='batchmean', log_target=True)
        
        # PSA Loss
        query_embeds = torch.nn.functional.normalize(outputs.dec_latents, dim=-1)
        semantic_embeds = torch.nn.functional.normalize(target_recon_embs, dim=-1)
        similarities = torch.matmul(query_embeds, semantic_embeds.t()) / 0.07
        dec_cl_loss = torch.nn.functional.cross_entropy(similarities, torch.arange(train_batch_size, device=device))
        
        loss = vq_loss + 0.0001 * kl_loss + 0.0003 * dec_cl_loss
        loss.backward()
        id_optimizer.step()
        id_optimizer.zero_grad(set_to_none=True)
        
        print_memory("train_id 后", local_rank)
        
        # 清理
        del target_semantic_embs, target_recon_embs, commit_loss, target_code_logits
        del outputs, vq_loss, kl_loss, dec_cl_loss, seq_code_logits
        del x_logits, y_logits, query_embeds, semantic_embeds, similarities
        del loss, targets, id_optimizer
        del input_ids, attention_mask, seq_end_positions, target_positions
        
        if local_rank == 0:
            print("[测试3] ✓ train_rec → generate → train_id 切换成功!")
        train_id_success = True
        
    except torch.cuda.OutOfMemoryError as e:
        if local_rank == 0:
            print(f"[测试3] ✗ train_id OOM! {e}")
    except Exception as e:
        if local_rank == 0:
            print(f"[测试3] ✗ 其他错误: {e}")
            import traceback
            traceback.print_exc()

    dist.barrier()

    # ============================================================
    # 总结
    # ============================================================
    if local_rank == 0:
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"基线显存: {baseline_mem:.2f} GB")
        
        all_success = train_success and generate_success and train_id_success
        
        if all_success:
            print(f"\n✓ 所有测试都通过!")
            print(f"  - train_rec 峰值: {train_peak:.2f} GB")
            print(f"  - generate 峰值: {gen_peak:.2f} GB")
            print(f"  - train_rec → generate → train_id 切换: ✓")
            
            if max(train_peak, gen_peak) < 30:
                print("\n✓ 峰值 < 30GB，可以在 32GB 显卡上运行")
            else:
                print("\n⚠ 峰值 >= 30GB，可能需要调整参数")
        else:
            print("\n✗ 有测试失败:")
            print(f"  - train_rec: {'✓' if train_success else '✗'}")
            print(f"  - generate: {'✓' if generate_success else '✗'}")
            print(f"  - train_id 切换: {'✓' if train_id_success else '✗'}")

    cleanup_ddp()

if __name__ == "__main__":
    main()
