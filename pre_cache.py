"""用于预处理和缓存图数据的脚本"""
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import argparse
import time
import dgl
import sys
from glob import glob

# 导入项目模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from graph_constructor_3d import create_heterograph_from_files, load_molecule_from_file

def find_files(complex_id, root_dir):
    """查找给定复合物ID的配体和口袋文件"""
    # 先检查平铺结构
    flat_ligand_candidates = [
        os.path.join(root_dir, f"{complex_id}_ligand.sdf"),
        os.path.join(root_dir, f"{complex_id}_ligand.pdb"),
        os.path.join(root_dir, f"{complex_id}_ligand.mol2")
    ]
    
    flat_pocket_candidates = [
        os.path.join(root_dir, f"{complex_id}_pocket.pdb"),
        os.path.join(root_dir, f"{complex_id}_protein.pdb")
    ]
    
    # 寻找配体文件
    ligand_file = None
    for candidate in flat_ligand_candidates:
        if os.path.exists(candidate):
            ligand_file = candidate
            break
    
    # 寻找口袋文件
    pocket_file = None
    for candidate in flat_pocket_candidates:
        if os.path.exists(candidate):
            pocket_file = candidate
            break
    
    # 如果平铺结构没找到，尝试层次结构
    if ligand_file is None or pocket_file is None:
        protein_id = '_'.join(complex_id.split('_')[:2])  # 提取蛋白质ID
        
        # 构建文件路径
        protein_folder = os.path.join(root_dir, protein_id)
        complex_folder = os.path.join(protein_folder, complex_id)
        
        if os.path.exists(complex_folder):
            # 配体文件
            ligand_candidates = [
                os.path.join(complex_folder, f"{complex_id}_ligand.sdf"),
                os.path.join(complex_folder, f"{complex_id}_ligand.pdb"),
                os.path.join(complex_folder, f"{complex_id}_ligand.mol2")
            ]
            
            for candidate in ligand_candidates:
                if os.path.exists(candidate):
                    ligand_file = candidate
                    break
            
            # 口袋文件
            pocket_candidates = [
                os.path.join(complex_folder, f"{complex_id}_pocket_5.0A.pdb"),
                os.path.join(complex_folder, f"{complex_id}_pocket.pdb"),
                os.path.join(complex_folder, f"{complex_id}_protein.pdb"),
                os.path.join(protein_folder, f"{protein_id}.pdb")
            ]
            
            for candidate in pocket_candidates:
                if os.path.exists(candidate):
                    pocket_file = candidate
                    break
    
    return ligand_file, pocket_file

def process_complex(args):
    """处理单个复合物并保存缓存"""
    complex_id, root_dir, output_dir, dis_threshold, label = args
    cache_file = os.path.join(output_dir, f"{complex_id}.pt")
    
    try:
        # 如果缓存已存在，跳过
        if os.path.exists(cache_file):
            return complex_id, True, "缓存已存在"
        
        # 查找文件
        ligand_file, pocket_file = find_files(complex_id, root_dir)
        
        if ligand_file is None or pocket_file is None:
            return complex_id, False, "文件未找到"
        
        # 构建异构图
        graph = create_heterograph_from_files(ligand_file, pocket_file, dis_threshold)
        
        if graph is None:
            return complex_id, False, "图构建失败"
        
        # 保存处理后的图和标签
        torch.save({'graph': graph, 'label': label}, cache_file)
        return complex_id, True, "成功"
    
    except Exception as e:
        return complex_id, False, f"错误: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="预处理并缓存分子图数据")
    parser.add_argument("--root_dir", type=str, required=True, help="3D结构数据根目录")
    parser.add_argument("--label_file", type=str, required=True, help="标签文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="缓存输出目录")
    parser.add_argument("--dis_threshold", type=float, default=5.0, help="原子间距离阈值")
    parser.add_argument("--num_workers", type=int, default=8, help="并行处理的工作进程数")
    parser.add_argument("--batch_size", type=int, default=1000, help="处理的批次大小")
    parser.add_argument("--start_idx", type=int, default=0, help="起始索引")
    parser.add_argument("--end_idx", type=int, default=None, help="结束索引")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取标签文件
    print(f"读取标签文件: {args.label_file}")
    labels_df = pd.read_csv(args.label_file)
    
    # 获取复合物ID和标签
    complex_ids = labels_df['complex_id'].values
    labels = labels_df['label'].values if 'label' in labels_df.columns else np.zeros(len(complex_ids))
    
    # 设置处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(complex_ids)
    complex_ids = complex_ids[start_idx:end_idx]
    labels = labels[start_idx:end_idx]
    
    print(f"预处理 {len(complex_ids)} 个复合物 (索引 {start_idx} 到 {end_idx-1})")
    print(f"输出目录: {args.output_dir}")
    print(f"使用 {args.num_workers} 个工作进程")
    
    # 准备任务参数
    tasks = [(complex_ids[i], args.root_dir, args.output_dir, args.dis_threshold, float(labels[i])) 
             for i in range(len(complex_ids))]
    
    # 启动多进程池
    start_time = time.time()
    with mp.Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_complex, tasks, chunksize=max(1, len(tasks)//args.num_workers//10)), 
                           total=len(tasks), desc="预处理进度"))
    
    # 统计结果
    success = sum(1 for _, status, _ in results if status)
    print(f"\n预处理完成! 耗时: {time.time() - start_time:.2f}秒")
    print(f"成功: {success}/{len(tasks)} ({100.0*success/len(tasks):.2f}%)")
    
    # 记录失败的复合物
    failures = [(cid, msg) for cid, status, msg in results if not status]
    if failures:
        with open(os.path.join(args.output_dir, "failures.txt"), "w") as f:
            for cid, msg in failures:
                f.write(f"{cid}: {msg}\n")
        print(f"失败记录已保存到 {os.path.join(args.output_dir, 'failures.txt')}")

if __name__ == "__main__":
    main()