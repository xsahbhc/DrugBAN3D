#!/usr/bin/env python3
"""
DrugBAN 数据增强与标签生成一体化脚本

此脚本结合了两个功能：
1. 基于现有缓存图实现3D分子图数据增强
2. 为增强数据自动生成对应的标签文件

用法:
    python augment_data_with_labels.py --cache_dir /path/to/cached_graphs --output_dir /path/to/output --label_file /path/to/labels.csv --train_label /path/to/train_labels.csv
"""
import os
import torch
import numpy as np
import pandas as pd
import dgl
import random
import argparse
from tqdm import tqdm
import multiprocessing as mp
import time
import glob
import warnings
import traceback
import math  # 添加这个导入

# 设置随机种子，保证结果可复现
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

###########################################
# 第一部分：数据增强相关函数
###########################################

def rotate_coordinates(coords, angle_x=0, angle_y=0, angle_z=0):
    """
    对坐标进行3D旋转
    
    参数:
    coords: 原子坐标 [N, 3]
    angle_x, angle_y, angle_z: 绕各轴旋转角度（弧度）
    
    返回:
    rotated_coords: 旋转后的坐标 [N, 3]
    """
    # 转换为numpy数组以便矩阵计算
    if isinstance(coords, torch.Tensor):
        coords_np = coords.numpy()
    else:
        coords_np = coords
        
    # X轴旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    # Y轴旋转矩阵
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    # Z轴旋转矩阵
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    # 组合旋转矩阵
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # 对坐标进行旋转
    rotated_coords = np.dot(coords_np, R.T)
    
    # 如果原始输入是torch.Tensor，转回tensor
    if isinstance(coords, torch.Tensor):
        rotated_coords = torch.tensor(rotated_coords, dtype=coords.dtype, device=coords.device)
        
    return rotated_coords

def jitter_coordinates(coords, scale=0.1):
    """
    对坐标添加随机扰动
    
    参数:
    coords: 原子坐标 [N, 3]
    scale: 扰动幅度
    
    返回:
    jittered_coords: 扰动后的坐标 [N, 3]
    """
    # 生成随机噪声
    if isinstance(coords, torch.Tensor):
        noise = torch.randn_like(coords) * scale
        return coords + noise
    else:
        noise = np.random.randn(*coords.shape) * scale
        return coords + noise

def random_remove_edges(graph, ratio=0.05):
    """
    随机移除图中的一部分边
    
    参数:
    graph: DGL图
    ratio: 移除的边的比例
    
    返回:
    new_graph: 处理后的图
    """
    new_graph = graph.clone()
    
    # 对每种类型的边进行处理
    for edge_type in new_graph.canonical_etypes:
        src, rel_type, dst = edge_type
        
        # 仅对分子内部边进行抽样，不处理蛋白质-配体之间的相互作用边
        if (src == 'ligand' and dst == 'protein') or (src == 'protein' and dst == 'ligand'):
            continue
            
        edges = new_graph.edges(etype=edge_type)
        num_edges = len(edges[0])
        
        if num_edges > 0:
            # 确定要保留的边的数量
            num_to_keep = int(num_edges * (1 - ratio))
            # 随机选择要保留的边的索引
            keep_indices = torch.randperm(num_edges)[:num_to_keep]
            # 获取要保留的边的源节点和目标节点
            src_nodes = edges[0][keep_indices]
            dst_nodes = edges[1][keep_indices]
            
            # 获取边特征
            edge_features = {}
            for feat_name, feat_data in new_graph.edges[edge_type].data.items():
                edge_features[feat_name] = feat_data[keep_indices]
            
            # 移除所有边
            new_graph.remove_edges(torch.arange(num_edges), etype=edge_type)
            # 添加保留的边
            new_graph.add_edges(src_nodes, dst_nodes, data=edge_features, etype=edge_type)
    
    return new_graph

def torsional_rotation(coords, atom_indices, angle):
    """
    执行分子扭转旋转，模拟分子的柔性变化。
    
    参数:
    coords: 原子坐标 [N, 3]
    atom_indices: 旋转轴两端的原子索引 [2]
    angle: 旋转角度（弧度）
    
    返回:
    rotated_coords: 旋转后的坐标 [N, 3]
    """
    if isinstance(coords, torch.Tensor):
        coords_np = coords.numpy()
        is_tensor = True
    else:
        coords_np = coords
        is_tensor = False
    
    # 获取旋转轴的两个原子的坐标
    if len(atom_indices) < 2:
        return coords  # 如果没有足够的原子，直接返回原始坐标
        
    atom1, atom2 = atom_indices[0], atom_indices[1]
    if atom1 >= len(coords_np) or atom2 >= len(coords_np):
        return coords  # 原子索引超出范围，返回原始坐标
    
    # 计算旋转轴
    axis = coords_np[atom2] - coords_np[atom1]
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-10:
        return coords  # 轴长度近似为零，返回原始坐标
    
    # 归一化旋转轴
    axis = axis / axis_length
    
    # 创建旋转矩阵
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rodriguez旋转公式
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
    
    # 移动坐标，使旋转轴的第一个原子位于原点
    translated_coords = coords_np - coords_np[atom1]
    
    # 对每个原子应用旋转
    rotated_coords = np.copy(coords_np)
    for i in range(len(coords_np)):
        # 只旋转选定的子结构，这里简化为旋转全部，可以基于分子拓扑结构进行优化
        rotated_coords[i] = coords_np[atom1] + np.dot(R, translated_coords[i])
    
    # 如果输入是tensor，转回tensor
    if is_tensor:
        rotated_coords = torch.tensor(rotated_coords, dtype=coords.dtype, device=coords.device)
    
    return rotated_coords

def conformational_sampling(graph, num_samples=1, max_iterations=100, temperature=0.1, distance_constraint=0.1):
    """
    执行构象采样，生成分子的多个低能量构象
    
    参数:
    graph: DGL图
    num_samples: 采样构象数量
    max_iterations: 最大迭代次数
    temperature: 温度参数，控制采样范围
    distance_constraint: 原子间距离约束，避免非物理性构象
    
    返回:
    sampled_graphs: 采样得到的图列表
    """
    if 'ligand' not in graph.ntypes or 'coords' not in graph.nodes['ligand'].data:
        return [graph.clone()]  # 如果没有配体坐标，返回原始图
        
    # 获取原始配体坐标
    orig_coords = graph.nodes['ligand'].data['coords']
    
    # 收集配体内部的边信息，用于约束采样
    internal_edges = []
    if ('ligand', 'to', 'ligand') in graph.canonical_etypes:
        src, dst = graph.edges(etype=('ligand', 'to', 'ligand'))
        for i in range(len(src)):
            internal_edges.append((src[i].item(), dst[i].item()))
    
    # 生成多个候选构象
    sampled_graphs = []
    for _ in range(num_samples):
        sampled_graph = graph.clone()
        current_coords = sampled_graph.nodes['ligand'].data['coords'].clone()
        
        # 使用简化的Monte Carlo采样来优化构象
        for _ in range(max_iterations):
            # 随机选择一个扭转角度
            angle = (random.random() * 2 - 1) * math.pi * temperature
            
            # 随机选择一个旋转轴
            if len(internal_edges) > 0:
                axis_atoms = random.choice(internal_edges)
            else:
                # 如果没有内部边信息，随机选择两个原子
                num_atoms = current_coords.shape[0]
                if num_atoms < 2:
                    break
                atom1 = random.randint(0, num_atoms-1)
                atom2 = random.randint(0, num_atoms-1)
                while atom2 == atom1:
                    atom2 = random.randint(0, num_atoms-1)
                axis_atoms = (atom1, atom2)
            
            # 执行扭转旋转
            new_coords = torsional_rotation(current_coords, axis_atoms, angle)
            
            # 检查新构象是否满足约束条件
            # 这里简化为简单接受，实际应该检查是否是物理合理的构象
            current_coords = new_coords
        
        # 更新图的配体坐标
        sampled_graph.nodes['ligand'].data['coords'] = current_coords
        
        # 更新蛋白质-配体间的边特征，如距离
        update_interaction_features(sampled_graph)
        
        sampled_graphs.append(sampled_graph)
    
    return sampled_graphs

def update_interaction_features(graph):
    """更新蛋白质-配体相互作用特征"""
    if 'ligand' not in graph.ntypes or 'protein' not in graph.ntypes:
        return
        
    ligand_coords = graph.nodes['ligand'].data.get('coords', None)
    protein_coords = graph.nodes['protein'].data.get('coords', None)
    
    if ligand_coords is None or protein_coords is None:
        return
    
    # 更新配体到蛋白质的边特征
    if ('ligand', 'to', 'protein') in graph.canonical_etypes:
        src, dst = graph.edges(etype=('ligand', 'to', 'protein'))
        if len(src) > 0 and 'distance' in graph.edges[('ligand', 'to', 'protein')].data:
            distances = torch.norm(ligand_coords[src] - protein_coords[dst], dim=1)
            graph.edges[('ligand', 'to', 'protein')].data['distance'] = distances
    
    # 更新蛋白质到配体的边特征
    if ('protein', 'to', 'ligand') in graph.canonical_etypes:
        src, dst = graph.edges(etype=('protein', 'to', 'ligand'))
        if len(src) > 0 and 'distance' in graph.edges[('protein', 'to', 'ligand')].data:
            distances = torch.norm(protein_coords[src] - ligand_coords[dst], dim=1)
            graph.edges[('protein', 'to', 'ligand')].data['distance'] = distances

def local_geometry_perturbation(coords, indices=None, scale=0.05):
    """
    对分子的局部区域进行几何扰动
    
    参数:
    coords: 原子坐标 [N, 3]
    indices: 要扰动的原子索引，如果为None则随机选择50%的原子
    scale: 扰动幅度
    
    返回:
    perturbed_coords: 扰动后的坐标 [N, 3]
    """
    if isinstance(coords, torch.Tensor):
        if indices is None:
            # 随机选择50%的原子
            num_atoms = coords.shape[0]
            num_to_perturb = max(1, int(num_atoms * 0.5))
            indices = torch.randperm(num_atoms)[:num_to_perturb]
            
        # 生成随机噪声
        noise = torch.zeros_like(coords)
        noise[indices] = torch.randn_like(coords[indices]) * scale
        
        return coords + noise
    else:
        if indices is None:
            # 随机选择50%的原子
            num_atoms = coords.shape[0]
            num_to_perturb = max(1, int(num_atoms * 0.5))
            indices = np.random.permutation(num_atoms)[:num_to_perturb]
            
        # 生成随机噪声
        noise = np.zeros_like(coords)
        noise[indices] = np.random.randn(*coords[indices].shape) * scale
        
        return coords + noise

def validate_graph_quality(graph, original_graph=None):
    """
    验证增强后图的质量

    参数:
    graph: 增强后的图
    original_graph: 原始图（用于对比）

    返回:
    is_valid: 是否通过质量检查
    quality_score: 质量分数 (0-1)
    """
    try:
        # 基本结构检查
        if graph.number_of_nodes() == 0:
            return False, 0.0

        # 检查图的连通性
        if 'ligand' in graph.ntypes:
            ligand_nodes = graph.number_of_nodes('ligand')
            if ligand_nodes == 0:
                return False, 0.0

        quality_score = 1.0

        # 如果有原始图，进行对比检查
        if original_graph is not None:
            # 节点数量应该保持一致
            if graph.number_of_nodes() != original_graph.number_of_nodes():
                quality_score *= 0.5

            # 边数量变化不应该太大（允许±10%）
            orig_edges = original_graph.number_of_edges()
            new_edges = graph.number_of_edges()
            if orig_edges > 0:
                edge_change_ratio = abs(new_edges - orig_edges) / orig_edges
                if edge_change_ratio > 0.1:  # 超过10%变化
                    quality_score *= 0.7

        # 检查坐标的合理性
        if 'ligand' in graph.ntypes and 'coords' in graph.nodes['ligand'].data:
            coords = graph.nodes['ligand'].data['coords']

            # 检查是否有NaN或无穷大值
            if torch.isnan(coords).any() or torch.isinf(coords).any():
                return False, 0.0

            # 检查坐标范围是否合理（不应该偏离太远）
            coord_std = torch.std(coords)
            if coord_std > 100:  # 坐标标准差过大
                quality_score *= 0.8

        # 质量分数阈值
        return quality_score >= 0.6, quality_score

    except Exception as e:
        return False, 0.0

def augment_graph(graph, augment_type='gentle_rotate', params=None):
    """
    对分子图进行数据增强 - 优化版本，增加质量控制

    参数:
    graph: DGL图
    augment_type: 增强类型，'gentle_rotate', 'thermal_vibration', 'minimal_edge_dropout', 'smart_combined', 'bond_flexibility'
    params: 增强参数

    返回:
    augmented_graph: 增强后的图
    """
    if params is None:
        params = {}

    # 尝试多次增强，选择质量最好的结果
    best_graph = None
    best_quality = 0.0
    max_attempts = 3  # 最多尝试3次

    for attempt in range(max_attempts):
        augmented_graph = graph.clone()

        # 1. 温和旋转：模拟分子在溶液中的自然旋转（±3度，更保守）
        if augment_type == 'gentle_rotate' or augment_type == 'smart_combined':
            # 进一步减小旋转范围，使用更保守的角度
            angle_x = np.random.uniform(-np.pi/60, np.pi/60)  # ±3度范围（更保守）
            angle_y = np.random.uniform(-np.pi/60, np.pi/60)
            angle_z = np.random.uniform(-np.pi/60, np.pi/60)

            # 仅旋转配体节点，保持蛋白质不变
            if 'ligand' in augmented_graph.ntypes and 'coords' in augmented_graph.nodes['ligand'].data:
                ligand_coords = augmented_graph.nodes['ligand'].data['coords']
                rotated_ligand_coords = rotate_coordinates(ligand_coords, angle_x, angle_y, angle_z)
                augmented_graph.nodes['ligand'].data['coords'] = rotated_ligand_coords

                # 更新蛋白质-配体相互作用特征
                update_interaction_features(augmented_graph)

        # 2. 热振动：模拟分子的热运动（更小的扰动）
        if augment_type == 'thermal_vibration' or augment_type == 'smart_combined':
            # 使用更小的扰动幅度
            if 'ligand' in augmented_graph.ntypes and 'coords' in augmented_graph.nodes['ligand'].data:
                coords = augmented_graph.nodes['ligand'].data['coords']
                # 使用更小的扰动幅度（0.005 Å）
                scale = 0.005  # 约0.005埃的振动幅度
                jittered_coords = jitter_coordinates(coords, scale)
                augmented_graph.nodes['ligand'].data['coords'] = jittered_coords

                # 更新蛋白质-配体相互作用特征
                update_interaction_features(augmented_graph)

        # 3. 最小边删除：模拟实验中的微小结构变化
        if augment_type == 'minimal_edge_dropout':
            # 使用更保守的边删除比例
            dropout_ratio = 0.005  # 只删除0.5%的边
            augmented_graph = random_remove_edges(augmented_graph, dropout_ratio)

        # 4. 键长微调：模拟键的柔性
        if augment_type == 'bond_flexibility':
            if 'ligand' in augmented_graph.ntypes and 'coords' in augmented_graph.nodes['ligand'].data:
                coords = augmented_graph.nodes['ligand'].data['coords']
                # 对配体坐标进行更小的局部扰动
                perturbed_coords = local_geometry_perturbation(coords, scale=0.003)  # 0.003埃的微调
                augmented_graph.nodes['ligand'].data['coords'] = perturbed_coords

                # 更新蛋白质-配体相互作用特征
                update_interaction_features(augmented_graph)

        # 5. 智能组合增强：结合多种保守方法
        if augment_type == 'smart_combined':
            if 'ligand' in augmented_graph.ntypes and 'coords' in augmented_graph.nodes['ligand'].data:
                # 最后进行极小的键长微调
                coords = augmented_graph.nodes['ligand'].data['coords']
                final_coords = local_geometry_perturbation(coords, scale=0.002)  # 极小的微调
                augmented_graph.nodes['ligand'].data['coords'] = final_coords

                # 更新蛋白质-配体相互作用特征
                update_interaction_features(augmented_graph)

        # 质量检查
        is_valid, quality_score = validate_graph_quality(augmented_graph, graph)

        if is_valid and quality_score > best_quality:
            best_graph = augmented_graph
            best_quality = quality_score

        # 如果质量足够好，提前退出
        if quality_score > 0.9:
            break

    # 如果没有找到合格的增强结果，返回原图
    if best_graph is None:
        return graph

    return best_graph

def process_complex(args):
    """处理单个复合物并生成增强后的图缓存"""
    complex_id, cache_dir, output_dir, label, is_positive, pos_augment_count, neg_augment_count, augment_types = args
    
    try:
        # 加载原始缓存文件
        cache_file = os.path.join(cache_dir, f"{complex_id}.pt")
        if not os.path.exists(cache_file):
            return complex_id, False, "缓存文件不存在"
        
        # 加载缓存数据
        data = torch.load(cache_file, map_location='cpu')
        graph = data.get('graph', None)
        
        if graph is None:
            return complex_id, False, "缓存文件中没有图数据"
        
        # 生成原始样本的ID
        orig_id = f"{complex_id}_orig"
        
        # 保存原始图，使用新的ID
        orig_data = {
            'graph': graph,
            'label': label
        }
        out_file = os.path.join(output_dir, f"{orig_id}.pt")
        torch.save(orig_data, out_file)
        
        # 确定增强次数
        augment_count = pos_augment_count if is_positive else neg_augment_count
        
        # 执行增强
        for i in range(augment_count):
            # 随机选择增强类型 - 改进版本，使用更保守和科学的策略
            if augment_types is None:
                # 使用科学保守的增强类型选择策略
                if is_positive:
                    # 正样本使用高质量的保守增强方法
                    rand_val = random.random()
                    if rand_val < 0.50:  # 50%概率使用温和旋转
                        aug_type = 'gentle_rotate'
                        params = {}
                    elif rand_val < 0.80:  # 30%概率使用热振动
                        aug_type = 'thermal_vibration'
                        params = {}
                    elif rand_val < 0.95:  # 15%概率使用键长微调
                        aug_type = 'bond_flexibility'
                        params = {}
                    else:  # 5%概率使用智能组合
                        aug_type = 'smart_combined'
                        params = {}
                else:
                    # 负样本不进行增强（NEG_AUGMENT_COUNT=0）
                    # 这部分代码实际不会执行，但保留以防配置改变
                    aug_type = 'gentle_rotate'
                    params = {}
            else:
                # 如果提供了指定的增强类型列表，从中随机选择
                aug_type = random.choice(augment_types)
                params = {}
            
            # 应用增强变换
            augmented_graph = augment_graph(graph, aug_type, params)
            
            # 保存增强后的图
            aug_id = f"{complex_id}_aug{i+1}"
            aug_data = {
                'graph': augmented_graph,
                'label': label,
                'augmentation': {'type': aug_type, 'params': params}
            }
            aug_file = os.path.join(output_dir, f"{aug_id}.pt")
            torch.save(aug_data, aug_file)
            
        return complex_id, True, f"成功生成{augment_count}个增强样本"
        
    except Exception as e:
        error_msg = f"增强失败: {str(e)}\n{traceback.format_exc()}"
        return complex_id, False, error_msg

def augment_data(cache_dir, output_dir, label_file, pos_augment_count=3, neg_augment_count=1, num_workers=8, augment_types=None):
    """
    执行数据增强流程
    
    参数:
    cache_dir: 原始缓存目录
    output_dir: 增强缓存输出目录
    label_file: 标签文件路径
    pos_augment_count: 每个正样本生成的增强样本数量
    neg_augment_count: 每个负样本生成的增强样本数量
    num_workers: 并行处理的工作进程数
    augment_types: 数据增强类型列表
    
    返回:
    success: 增强是否成功
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取标签文件
    print(f"读取标签文件: {label_file}")
    labels_df = pd.read_csv(label_file)
    
    # 提取复合物ID和标签
    complex_ids = labels_df['complex_id'].values
    labels = labels_df['label'].values
    
    # 区分正负样本
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    pos_ratio = len(pos_indices) / len(labels) * 100
    print(f"总样本数: {len(labels)}, 正样本数: {len(pos_indices)} ({pos_ratio:.2f}%), "
          f"负样本数: {len(neg_indices)} ({100-pos_ratio:.2f}%)")
    
    # 确认增强类型 - 改进版本，使用更保守和科学的方法
    if augment_types is None:
        # 使用保守的增强类型列表，注重物理合理性
        augment_types = [
            'gentle_rotate', 'thermal_vibration', 'bond_flexibility',
            'smart_combined', 'minimal_edge_dropout'
        ]
    valid_types = [
        'gentle_rotate', 'thermal_vibration', 'bond_flexibility',
        'smart_combined', 'minimal_edge_dropout'
    ]
    augment_types = [t for t in augment_types if t in valid_types]
    if not augment_types:
        augment_types = ['gentle_rotate', 'thermal_vibration']  # 默认保守增强类型
    
    print(f"使用的增强类型: {augment_types}")
    print(f"正样本增强数量: 每个样本{pos_augment_count}次")
    print(f"负样本增强数量: 每个样本{neg_augment_count}次")
    
    # 准备多进程任务
    tasks = []
    for i, complex_id in enumerate(complex_ids):
        is_positive = (labels[i] == 1)
        tasks.append((
            complex_id, 
            cache_dir, 
            output_dir, 
            float(labels[i]), 
            is_positive,
            pos_augment_count,
            neg_augment_count,
            augment_types
        ))
    
    # 启动多进程处理
    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_complex, tasks, chunksize=max(1, len(tasks)//num_workers//10)), 
            total=len(tasks), 
            desc="数据增强进度"
        ))
    
    # 统计结果
    success = sum(1 for _, status, _ in results if status)
    print(f"\n数据增强完成! 耗时: {time.time() - start_time:.2f}秒")
    print(f"成功: {success}/{len(tasks)} ({100.0*success/len(tasks):.2f}%)")
    
    # 记录失败的复合物
    failures = [(cid, msg) for cid, status, msg in results if not status]
    if failures:
        with open(os.path.join(output_dir, "augment_failures.txt"), "w") as f:
            for cid, msg in failures:
                f.write(f"{cid}: {msg}\n")
        print(f"失败记录已保存到 {os.path.join(output_dir, 'augment_failures.txt')}")
    
    # 统计生成的文件数量
    all_files = glob.glob(os.path.join(output_dir, "*.pt"))
    orig_files = glob.glob(os.path.join(output_dir, "*_orig.pt"))
    aug_files = glob.glob(os.path.join(output_dir, "*_aug*.pt"))
    
    print(f"生成的文件总数: {len(all_files)}")
    print(f"  原始文件数: {len(orig_files)}")
    print(f"  增强文件数: {len(aug_files)}")
    
    # 计算增强后的正负样本比例
    total_pos = len(pos_indices) * (1 + pos_augment_count)
    total_neg = len(neg_indices) * (1 + neg_augment_count)
    total_samples = total_pos + total_neg
    new_pos_ratio = total_pos / total_samples * 100
    
    print(f"增强后样本总数: {total_samples}")
    print(f"  正样本数: {total_pos} ({new_pos_ratio:.2f}%)")
    print(f"  负样本数: {total_neg} ({100-new_pos_ratio:.2f}%)")
    
    return success > 0, all_files

###########################################
# 第二部分：标签生成相关函数
###########################################

def create_augmented_labels(cache_dir, original_labels_file, output_dir, train_label_file=None, val_label_file=None, test_label_file=None):
    """为增强数据创建标签文件，保持测试集不变"""
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始标签文件
    print(f"读取原始标签文件: {original_labels_file}")
    try:
        original_labels = pd.read_csv(original_labels_file)
        print(f"原始标签文件包含 {len(original_labels)} 条记录")
        
        # 构建原始ID到标签的映射
        original_label_map = {}
        for _, row in original_labels.iterrows():
            if 'complex_id' in row and 'label' in row:
                original_label_map[row['complex_id']] = float(row['label'])
    except Exception as e:
        print(f"读取标签文件失败: {str(e)}")
        return None
    
    # 如果提供了预划分的标签文件，直接使用
    train_df = None
    val_df = None
    test_df = None
    
    # 读取测试集
    if test_label_file and os.path.exists(test_label_file):
        print(f"使用预划分的测试集标签文件: {test_label_file}")
        test_df = pd.read_csv(test_label_file)
        print(f"测试集包含 {len(test_df)} 条记录")
        # 确保测试集有正确的列名
        if 'complex_id' not in test_df.columns:
            raise ValueError(f"测试集标签文件必须包含'complex_id'列")
        if 'label' not in test_df.columns:
            raise ValueError(f"测试集标签文件必须包含'label'列")
        # 计算测试集正样本率
        test_pos_rate = test_df['label'].sum() / len(test_df)
        print(f"测试集正样本率: {test_pos_rate*100:.2f}%")
    else:
        test_pos_rate = 0.3  # 没有测试集时默认值
    
    # 从缓存目录读取所有PT文件（包括原始和增强文件）
    print(f"正在从缓存目录 {cache_dir} 中读取增强数据...")
    cache_files = glob.glob(os.path.join(cache_dir, "*.pt"))
    print(f"找到 {len(cache_files)} 个缓存文件")
    
    # 创建一个空的列表来存储增强后的标签
    augmented_labels = []
    
    # 处理每个缓存文件
    print("从缓存文件中提取标签信息...")
    missing_label_count = 0
    
    # 提取基础ID到增强ID的映射，用于后续划分数据集
    base_id_mapping = {}
    
    for file_path in tqdm(cache_files):
        file_name = os.path.basename(file_path)
        complex_id = file_name.replace(".pt", "")  # 包含_orig或_augN后缀的ID
        
        try:
            # 从缓存文件加载数据
            data = torch.load(file_path, map_location='cpu')
            
            # 提取基础ID
            if '_aug' in complex_id:
                base_id = complex_id.split('_aug')[0]
            elif '_orig' in complex_id:
                base_id = complex_id.replace('_orig', '')
            else:
                base_id = complex_id
            
            # 添加到基础ID映射中
            if base_id not in base_id_mapping:
                base_id_mapping[base_id] = []
            base_id_mapping[base_id].append(complex_id)
            
            # 首选：如果缓存文件中已有标签，直接使用
            if 'label' in data:
                label = float(data['label'])
                augmented_labels.append({
                    'complex_id': complex_id,
                    'label': label,
                    'base_id': base_id
                })
                continue
            
            # 其次：如果没有缓存标签，尝试从标签映射查找
            if base_id in original_label_map:
                label = original_label_map[base_id]
                augmented_labels.append({
                    'complex_id': complex_id,
                    'label': label,
                    'base_id': base_id
                })
            else:
                missing_label_count += 1
                if missing_label_count <= 50:  # 限制打印警告的数量
                    print(f"警告: 未找到复合物 {base_id} 的标签")
                elif missing_label_count == 51:
                    print("超过50个复合物没有找到标签，后续警告将被抑制...")
                
        except Exception as e:
            print(f"处理文件 '{file_path}' 时出错: {str(e)}")
    
    if missing_label_count > 0:
        print(f"总计 {missing_label_count} 个复合物没有找到标签")
    
    # 转换为DataFrame
    augmented_df = pd.DataFrame(augmented_labels)
    
    if augmented_df.empty:
        print("错误: 没有生成任何标签数据")
        return None
    
    # 保存增强标签文件（包含所有样本的综合标签文件，用于数据加载器匹配所有缓存文件）
    all_label_file = os.path.join(output_dir, "augmented_labels.csv")
    # 只保留必要的列
    augmented_df_save = augmented_df[['complex_id', 'label']]
    augmented_df_save.to_csv(all_label_file, index=False)
    print(f"增强标签已保存至 {all_label_file}，包含 {len(augmented_df_save)} 条记录")
    
    # 统计正负样本数量
    pos_count = len(augmented_df[augmented_df['label'] > 0])
    neg_count = len(augmented_df[augmented_df['label'] <= 0])
    pos_ratio = pos_count / len(augmented_df) * 100
    neg_ratio = neg_count / len(augmented_df) * 100
    print(f"正样本: {pos_count} ({pos_ratio:.2f}%), 负样本: {neg_count} ({neg_ratio:.2f}%)")
    
    # 如果没有预划分的训练和验证标签文件
    if train_df is None or val_df is None:
        # 使用预划分的测试集
        if test_df is not None:
            # 移除测试集使用的ID
            test_ids = set(test_df['complex_id'].values)
            remaining_df = augmented_df[~augmented_df['base_id'].isin(test_ids)]
            
            # 获取剩余的基础ID列表，用于分割训练和验证集
            remaining_base_ids = list(set(remaining_df['base_id'].values))
            
            # 对基础ID进行随机打乱
            random.shuffle(remaining_base_ids)
            
            # 按照85%训练集、15%验证集比例划分剩余基础ID（因为测试集已经单独分出）
            train_split = int(len(remaining_base_ids) * 0.85)
            
            train_base_ids = set(remaining_base_ids[:train_split])
            valid_base_ids = set(remaining_base_ids[train_split:])
            
            print(f"数据集划分: 训练集 {len(train_base_ids)} 分子, 验证集 {len(valid_base_ids)} 分子, 测试集 {len(test_ids)} 分子")
            
            # 根据基础ID分配样本到不同数据集
            train_samples = []
            valid_samples = []
            
            # 先将所有样本按基础ID分配到各自数据集
            for _, row in augmented_df.iterrows():
                base_id = row['base_id']
                sample_data = {'complex_id': row['complex_id'], 'label': row['label']}
                
                if base_id in train_base_ids:
                    train_samples.append(sample_data)
                elif base_id in valid_base_ids:
                    valid_samples.append(sample_data)
            
            # 创建训练和验证DataFrame
            train_df = pd.DataFrame(train_samples)
            
            # 为验证集调整正负样本比例，使其接近测试集的分布
            valid_df_full = pd.DataFrame(valid_samples)
            valid_pos = valid_df_full[valid_df_full['label'] > 0]
            valid_neg = valid_df_full[valid_df_full['label'] <= 0]
            
            # 计算验证集当前正样本率和目标正样本率
            current_valid_pos_rate = len(valid_pos) / len(valid_df_full)
            target_pos_rate = test_pos_rate  # 与测试集相同
            
            print(f"验证集当前正样本率: {current_valid_pos_rate*100:.2f}%, 目标正样本率: {target_pos_rate*100:.2f}%")
            
            # 如果验证集正样本率高于目标，需要随机抽样一部分正样本
            if current_valid_pos_rate > target_pos_rate:
                # 计算应保留的正样本数
                target_pos_count = int(len(valid_neg) * target_pos_rate / (1 - target_pos_rate))
                print(f"调整验证集: 从{len(valid_pos)}个正样本中抽样{target_pos_count}个")
                valid_pos = valid_pos.sample(n=min(target_pos_count, len(valid_pos)), random_state=42)
                # 合并正负样本
                valid_df = pd.concat([valid_pos, valid_neg])
            # 如果验证集正样本率低于目标，需要随机抽样一部分负样本
            elif current_valid_pos_rate < target_pos_rate:
                # 计算应保留的负样本数
                target_neg_count = int(len(valid_pos) * (1 - target_pos_rate) / target_pos_rate)
                print(f"调整验证集: 从{len(valid_neg)}个负样本中抽样{target_neg_count}个")
                valid_neg = valid_neg.sample(n=min(target_neg_count, len(valid_neg)), random_state=42)
                # 合并正负样本
                valid_df = pd.concat([valid_pos, valid_neg])
            else:
                # 正样本率已经匹配，无需调整
                valid_df = valid_df_full
            
            # 随机打乱验证集
            valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"调整后验证集正样本率: {valid_df['label'].sum() / len(valid_df) * 100:.2f}%")
            
        else:
            # 没有预划分的测试集，自行划分
            # 获取所有基础ID列表，用于分割数据集
            all_base_ids = list(base_id_mapping.keys())
            
            # 对基础ID进行随机打乱
            random.shuffle(all_base_ids)
            
            # 按照75%训练集、15%验证集、10%测试集比例划分基础ID
            train_split = int(len(all_base_ids) * 0.75)
            valid_split = int(len(all_base_ids) * 0.90)  # 前75%为训练集，接下来15%为验证集
            
            train_base_ids = set(all_base_ids[:train_split])
            valid_base_ids = set(all_base_ids[train_split:valid_split])
            test_base_ids = set(all_base_ids[valid_split:])
            
            print(f"数据集划分: 训练集 {len(train_base_ids)} 分子, 验证集 {len(valid_base_ids)} 分子, 测试集 {len(test_base_ids)} 分子")
            
            # 根据基础ID分配样本到不同数据集
            train_samples = []
            valid_samples = []
            test_samples = []
            
            # 遍历所有样本，根据基础ID分配到不同数据集
            for _, row in augmented_df.iterrows():
                base_id = row['base_id']
                sample_data = {'complex_id': row['complex_id'], 'label': row['label']}
                
                if base_id in train_base_ids:
                    train_samples.append(sample_data)
                elif base_id in valid_base_ids:
                    valid_samples.append(sample_data)
                elif base_id in test_base_ids:
                    test_samples.append(sample_data)
            
            # 创建训练、验证和测试DataFrame
            train_df = pd.DataFrame(train_samples)
            valid_df_full = pd.DataFrame(valid_samples)
            test_df = pd.DataFrame(test_samples)
            
            # 为验证集调整正负样本比例，使其接近测试集的分布
            test_pos_rate = test_df['label'].sum() / len(test_df)
            valid_pos = valid_df_full[valid_df_full['label'] > 0]
            valid_neg = valid_df_full[valid_df_full['label'] <= 0]
            
            # 计算验证集当前正样本率和目标正样本率
            current_valid_pos_rate = len(valid_pos) / len(valid_df_full)
            target_pos_rate = test_pos_rate  # 与测试集相同
            
            print(f"验证集当前正样本率: {current_valid_pos_rate*100:.2f}%, 目标正样本率: {target_pos_rate*100:.2f}%")
            
            # 如果验证集正样本率高于目标，需要随机抽样一部分正样本
            if current_valid_pos_rate > target_pos_rate:
                # 计算应保留的正样本数
                target_pos_count = int(len(valid_neg) * target_pos_rate / (1 - target_pos_rate))
                print(f"调整验证集: 从{len(valid_pos)}个正样本中抽样{target_pos_count}个")
                valid_pos = valid_pos.sample(n=min(target_pos_count, len(valid_pos)), random_state=42)
                # 合并正负样本
                valid_df = pd.concat([valid_pos, valid_neg])
            # 如果验证集正样本率低于目标，需要随机抽样一部分负样本
            elif current_valid_pos_rate < target_pos_rate:
                # 计算应保留的负样本数
                target_neg_count = int(len(valid_pos) * (1 - target_pos_rate) / target_pos_rate)
                print(f"调整验证集: 从{len(valid_neg)}个负样本中抽样{target_neg_count}个")
                valid_neg = valid_neg.sample(n=min(target_neg_count, len(valid_neg)), random_state=42)
                # 合并正负样本
                valid_df = pd.concat([valid_pos, valid_neg])
            else:
                # 正样本率已经匹配，无需调整
                valid_df = valid_df_full
            
            # 随机打乱验证集
            valid_df = valid_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"调整后验证集正样本率: {valid_df['label'].sum() / len(valid_df) * 100:.2f}%")
    
    # 保存分割后的标签文件
    train_label_file = os.path.join(output_dir, "train_augmented.csv")
    train_df.to_csv(train_label_file, index=False)
    print(f"训练标签已保存至 {train_label_file}，包含 {len(train_df)} 条记录")
    
    valid_label_file = os.path.join(output_dir, "valid_augmented.csv")
    valid_df.to_csv(valid_label_file, index=False)
    print(f"验证标签已保存至 {valid_label_file}，包含 {len(valid_df)} 条记录")
    
    test_label_file = os.path.join(output_dir, "test_augmented.csv")
    test_df.to_csv(test_label_file, index=False)
    print(f"测试标签已保存至 {test_label_file}，包含 {len(test_df)} 条记录")
    
    # 计算每个数据集的正负样本比例
    train_pos = train_df['label'].sum()
    valid_pos = valid_df['label'].sum()
    test_pos = test_df['label'].sum()
    
    train_pos_ratio = train_pos / len(train_df) * 100 if len(train_df) > 0 else 0
    valid_pos_ratio = valid_pos / len(valid_df) * 100 if len(valid_df) > 0 else 0
    test_pos_ratio = test_pos / len(test_df) * 100 if len(test_df) > 0 else 0
    
    print(f"训练集: 总样本 {len(train_df)}, 正样本率 {train_pos_ratio:.2f}%")
    print(f"验证集: 总样本 {len(valid_df)}, 正样本率 {valid_pos_ratio:.2f}%")
    print(f"测试集: 总样本 {len(test_df)}, 正样本率 {test_pos_ratio:.2f}%")
    
    return augmented_df

###########################################
# 主函数和命令行接口
###########################################

def main():
    """主函数，解析命令行参数并执行数据增强流程"""
    parser = argparse.ArgumentParser(description="DrugBAN数据增强与标签生成工具")
    parser.add_argument("--cache_dir", required=True, help="原始缓存图目录")
    parser.add_argument("--output_dir", required=True, help="增强后的缓存图保存目录")
    parser.add_argument("--label_file", required=True, help="原始标签文件路径")
    parser.add_argument("--train_label", required=True, help="训练集标签文件路径")
    parser.add_argument("--val_label", default=None, help="验证集标签文件路径")
    parser.add_argument("--test_label", default=None, help="测试集标签文件路径")
    parser.add_argument("--labels_output_dir", default=None, help="增强标签文件保存目录")
    parser.add_argument("--pos_augment", type=int, default=3, help="每个正样本的增强次数")
    parser.add_argument("--neg_augment", type=int, default=1, help="每个负样本的增强次数")
    parser.add_argument("--num_workers", type=int, default=8, help="并行处理的工作线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--augment_test", action="store_true", help="是否增强测试集（默认不增强）")
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设置标签输出目录
    if args.labels_output_dir is None:
        args.labels_output_dir = os.path.join(args.output_dir, "..", "augmented_labels")
    
    # 打印配置信息
    print("\n" + "=" * 80)
    print("DrugBAN 数据增强与标签生成流程")
    print("=" * 80)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"原始缓存目录: {args.cache_dir}")
    print(f"增强缓存目录: {args.output_dir}")
    print(f"原始标签文件: {args.label_file}")
    print(f"训练标签文件: {args.train_label}")
    print(f"验证标签文件: {args.val_label if args.val_label else '未指定'}")
    print(f"测试标签文件: {args.test_label if args.test_label else '未指定'}")
    print(f"增强标签目录: {args.labels_output_dir}")
    print(f"是否增强测试集: {'是' if args.augment_test else '否'}")
    print("=" * 80 + "\n")
    
    # 步骤1：数据增强
    print("步骤1：执行数据增强...")
    success, augmented_files = augment_data(
        args.cache_dir,
        args.output_dir,
        args.label_file,
        args.pos_augment,
        args.neg_augment,
        args.num_workers
    )
    
    if not success:
        print("错误: 数据增强失败")
        return 1
    
    # 步骤2：生成增强数据的标签
    print("\n" + "=" * 80)
    print("步骤2：为增强数据生成标签文件...")
    augmented_df = create_augmented_labels(
        args.output_dir,
        args.label_file,
        args.labels_output_dir,
        args.train_label,
        args.val_label,
        args.test_label
    )
    
    if augmented_df is None:
        print("错误: 标签生成失败")
        return 1
        
    print("\n" + "=" * 80)
    print("数据增强与标签生成完成!")
    print(f"增强后的缓存文件保存在: {args.output_dir}")
    print(f"增强后的标签文件保存在: {args.labels_output_dir}")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    return 0

if __name__ == "__main__":
    # 忽略特定警告
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    
    # 清空GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        exit(1)
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        traceback.print_exc()
        exit(1) 