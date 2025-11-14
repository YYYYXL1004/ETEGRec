from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
import torch
import numpy as np
import json
import os

def train_and_extract_embeddings():
    """
    训练 SASRec 并提取物品嵌入
    """
    print("=" * 70)
    print("🎵 SASRec 训练 - Musical Instruments 2023 (优化版)")
    print("=" * 70)
    
    # ============ 配置 ============
    config_dict = {
        # 数据集配置
        'model': 'SASRec',
        'dataset': 'Instruments2023',
        'data_path': './dataset/',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp']
        },
        
        # 数据划分策略
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'group_by': 'user',
            'mode': 'full'
        },
        
        # 🔧 SASRec 模型参数（与作者对齐）
        'hidden_size': 256,          # 嵌入维度
        'inner_size': 256,
        'n_layers': 2,
        'n_heads': 2,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        'loss_type': 'CE',
        'max_seq_length': 50,        # 🔧 限制为50（与作者一致）
        
        # 🔧 修复：禁用负采样
        'train_neg_sample_args': None,
        
        # 训练参数
        'epochs': 300,
        'train_batch_size': 2048,
        'eval_batch_size': 2048,
        'learner': 'adam',
        'learning_rate': 0.001,
        
        # 评估参数
        'eval_step': 1,
        'stopping_step': 10,
        'metrics': ['Recall', 'NDCG', 'Hit', 'MRR'],
        'topk': [5, 10, 20],
        'valid_metric': 'NDCG@10',
        'metric_decimal_place': 4,
        
        # GPU 配置
        'gpu_id': '3',
        'use_gpu': True,
        
        # 保存配置
        'checkpoint_dir': './saved/SASRec',
        'show_progress': True,
    }
    
    try:
        # ============ 加载数据 ============
        print("\n🔧 正在加载数据集...")
        config = Config(model='SASRec', dataset='Instruments2023', config_dict=config_dict)
        dataset = create_dataset(config)
        
        print(f"✅ 数据集加载成功!")
        print(f"   用户数: {dataset.user_num:,}")
        print(f"   物品数: {dataset.item_num:,}")
        print(f"   交互数: {dataset.inter_num:,}")
        
        train_data, valid_data, test_data = data_preparation(config, dataset)
        
        # ============ 创建模型 ============
        print("\n🤖 正在创建 SASRec 模型...")
        model = SASRec(config, train_data.dataset).to(config['device'])
        print(f"   设备: {config['device']}")
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # ============ 训练 ============
        print("\n🚀 开始训练...")
        print(f"   总轮数: {config['epochs']}")
        print(f"   Batch Size: {config['train_batch_size']}")
        print(f"   学习率: {config['learning_rate']}")
        print(f"   早停步数: {config['stopping_step']}")
        print(f"   最大序列长度: {config['max_seq_length']}")
        
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data,
            saved=True,
            show_progress=config['show_progress']
        )
        
        print(f"\n✅ 训练完成!")
        print(f"   最佳验证 {config['valid_metric']}: {best_valid_score:.4f}")
        
        # ============ 测试 ============
        print("\n📊 在测试集上评估...")
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=True)
        print(f"   测试结果:")
        for metric, value in test_result.items():
            print(f"      {metric}: {value:.4f}")
        
        # ============ 提取嵌入 ============
        print("\n💾 正在提取物品嵌入...")
        
        # 获取训练好的 item embedding
        item_embedding = model.item_embedding.weight.data.cpu().numpy()
        print(f"   原始嵌入形状: {item_embedding.shape}")
        
        # 去掉 padding token (索引 0)
        item_embedding_no_pad = item_embedding[1:]
        print(f"   去除 padding 后: {item_embedding_no_pad.shape}")
        
        # ============ 保存文件 ============
        output_dir = './dataset/Instruments2023'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 保存 .npy 嵌入文件
        npy_path = os.path.join(output_dir, 'Instruments2023_emb_256.npy')
        np.save(npy_path, item_embedding_no_pad)
        print(f"\n✅ 嵌入文件已保存: {npy_path}")
        print(f"   形状: {item_embedding_no_pad.shape}")
        print(f"   大小: {item_embedding_no_pad.nbytes / 1024 / 1024:.2f} MB")
        
        # 2. 生成 item2id 映射 (ETEGRec 格式)
        # 🔧 与作者格式一致：包含 [PAD] token
        item_token2id = dataset.field2token_id['item_id']
        
        # 创建映射（包含 [PAD]）
        item2id_etegrec = {}
        item2id_etegrec['[PAD]'] = 0  # 🔧 添加 PAD token
        
        for token, idx in item_token2id.items():
            if idx > 0:  # 跳过 RecBole 的 padding (idx=0)
                item2id_etegrec[str(token)] = int(idx)
        
        # 保存为 .emb_map.json
        map_path = os.path.join(output_dir, 'Instruments2023.emb_map.json')
        with open(map_path, 'w') as f:
            json.dump(item2id_etegrec, f, indent=2)
        print(f"✅ Item2ID 映射已保存: {map_path}")
        print(f"   映射条目数: {len(item2id_etegrec)} (包含 [PAD])")
        print(f"   物品数: {len(item2id_etegrec) - 1} (不含 [PAD])")
        
        # ============ 验证 ============
        print("\n🔍 验证生成的文件...")
        
        # 验证嵌入文件
        loaded_emb = np.load(npy_path)
        assert loaded_emb.shape == item_embedding_no_pad.shape, "嵌入形状不匹配!"
        
        # 验证映射文件
        with open(map_path, 'r') as f:
            loaded_map = json.load(f)
        
        # 🔧 映射数量应该是嵌入数量 + 1 ([PAD])
        expected_map_size = loaded_emb.shape[0] + 1
        assert len(loaded_map) == expected_map_size, \
            f"映射数量 ({len(loaded_map)}) 应该是 {expected_map_size} (嵌入数 + PAD)!"
        
        assert '[PAD]' in loaded_map and loaded_map['[PAD]'] == 0, \
            "映射应包含 [PAD] token，且索引为 0!"
        
        print("✅ 所有验证通过!")
        print(f"   映射格式: {{'[PAD]': 0, ...}}")
        print(f"   映射数量与嵌入匹配")
        
        # ============ 总结 ============
        print("\n" + "=" * 70)
        print("🎉 训练和提取完成!")
        print("=" * 70)
        print(f"\n📁 生成的文件:")
        print(f"   1. {npy_path}")
        print(f"      - 形状: {loaded_emb.shape}")
        print(f"      - 用途: ETEGRec 的 semantic_emb_path")
        print(f"\n   2. {map_path}")
        print(f"      - 条目数: {len(loaded_map)} (含 [PAD])")
        print(f"      - 物品数: {len(loaded_map) - 1}")
        print(f"      - 用途: ETEGRec 的 map_path")
        
        print(f"\n📊 模型性能:")
        print(f"   验证集 {config['valid_metric']}: {best_valid_score:.4f}")
        for metric, value in test_result.items():
            print(f"   测试集 {metric}: {value:.4f}")
        
        print(f"\n✨ 与作者数据集对齐:")
        print(f"   ✅ 最大序列长度: {config['max_seq_length']}")
        print(f"   ✅ 映射包含 [PAD] token")
        print(f"   ✅ 嵌入维度: 256")
        
        print("\n" + "=" * 70)
        
        return model, dataset, item_embedding_no_pad, test_result
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

if __name__ == '__main__':
    model, dataset, embeddings, test_result = train_and_extract_embeddings()
    
    if model is not None:
        print("\n✨ 下一步: 准备 ETEGRec 的训练数据!")
        print("   运行: python prepare_etegrec_data.py")