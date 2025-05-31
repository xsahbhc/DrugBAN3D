import os
import argparse
import torch
import numpy as np
import pandas as pd
from yacs.config import CfgNode
from configs import get_cfg_defaults
from drugban_3d import DrugBAN3D
from models import DrugBAN
from dataloader_3d import ProteinLigandDataset, filter_none_collate_fn
from torch.utils.data import DataLoader
import dgl


def parse_args():
    parser = argparse.ArgumentParser(description="DrugBAN推理")
    # 配置文件
    parser.add_argument("--cfg", dest="cfg_file", help="配置文件路径", required=True, type=str)
    # 模型权重路径
    parser.add_argument("--model", dest="model_path", help="模型权重路径", required=True, type=str)
    # 输入数据
    parser.add_argument("--input", dest="input_data", help="输入数据文件/目录", required=True, type=str)
    # 输出目录
    parser.add_argument("--output", dest="output_file", help="输出结果文件", required=True, type=str)
    # 3D数据
    parser.add_argument("--use_3d", action="store_true", help="使用3D数据和模型")
    parser.add_argument("--data_3d_root", type=str, help="3D数据根目录")
    
    return parser.parse_args()


def read_config(args):
    """读取配置文件并设置默认值"""
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    
    # 设置3D数据参数
    if args.use_3d:
        cfg.MODEL_TYPE = "DrugBAN3D"
        if args.data_3d_root:
            cfg.DATA_3D.ROOT_DIR = args.data_3d_root
    
    return cfg


def load_model(model_path, config, device):
    """加载模型"""
    if config.MODEL_TYPE == "DrugBAN":
        model = DrugBAN(**config).to(device)
    else:  # DrugBAN3D
        model = DrugBAN3D(**config).to(device)
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def prepare_data(args, config):
    """准备数据"""
    if config.MODEL_TYPE == "DrugBAN3D":
        # 对于3D数据
        if os.path.isfile(args.input_data) and args.input_data.endswith('.csv'):
            # 从CSV文件加载complex_id列表
            df = pd.read_csv(args.input_data)
            dataset = ProteinLigandDataset(
                root_dir=config.DATA_3D.ROOT_DIR,
                is_test=True,
                dis_threshold=config.DATA_3D.DIS_THRESHOLD
            )
            dataset.complex_ids = df['complex_id'].values
            if 'label' in df.columns:
                dataset.labels = df['label'].values
            else:
                # 如果没有标签，默认填充0
                dataset.labels = np.zeros(len(dataset.complex_ids))
        else:
            # 直接从目录加载
            dataset = ProteinLigandDataset(
                root_dir=args.input_data,
                is_test=True,
                dis_threshold=config.DATA_3D.DIS_THRESHOLD
            )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=filter_none_collate_fn
        )
    else:
        # 为DrugBAN实现数据准备逻辑
        raise NotImplementedError("暂不支持原始DrugBAN模型的推理")
    
    return dataloader


def run_inference(model, dataloader, device):
    """运行推理"""
    model.eval()
    results = []
    complex_ids = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            try:
                if batch is None or (isinstance(batch, tuple) and None in batch):
                    continue
                
                # 处理不同类型的批处理数据
                if len(batch) == 3:  # 原始DrugBAN的数据格式: (v_d, v_p, labels)
                    v_d, v_p, _ = batch
                    v_d, v_p = v_d.to(device), v_p.to(device)
                    _, _, _, score = model(v_d, v_p, mode="train")
                elif len(batch) == 2:  # DrugBAN3D的数据格式: (bg, labels)
                    bg, _ = batch
                    if bg is None:
                        continue
                    
                    bg = bg.to(device)
                    _, _, score, _ = model(bg, mode="eval")
                else:
                    continue
                
                # 应用sigmoid获取概率
                probs = torch.sigmoid(score).squeeze().cpu().numpy()
                
                # 保存结果
                if isinstance(batch, tuple) and len(batch) >= 2:
                    # 获取当前批次的complex_ids
                    batch_complex_ids = dataloader.dataset.complex_ids[i*dataloader.batch_size:i*dataloader.batch_size+len(probs)]
                    complex_ids.extend(batch_complex_ids)
                    results.extend(probs.tolist() if isinstance(probs, np.ndarray) else [probs])
            except Exception as e:
                print(f"处理批次 {i} 时出错: {str(e)}")
                continue
    
    return complex_ids, results


def save_results(complex_ids, predictions, output_file):
    """保存结果"""
    df = pd.DataFrame({
        'complex_id': complex_ids,
        'prediction': predictions
    })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"预测结果已保存到 {output_file}")


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 读取配置
    cfg = read_config(args)
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.model_path, cfg, device)
    print(f"模型已加载: {args.model_path}")
    
    # 准备数据
    dataloader = prepare_data(args, cfg)
    print(f"数据已准备: 共 {len(dataloader.dataset)} 个样本")
    
    # 运行推理
    complex_ids, predictions = run_inference(model, dataloader, device)
    print(f"推理完成: 共处理 {len(predictions)} 个样本")
    
    # 保存结果
    save_results(complex_ids, predictions, args.output_file)
    
    print("推理完成")


if __name__ == '__main__':
    main() 