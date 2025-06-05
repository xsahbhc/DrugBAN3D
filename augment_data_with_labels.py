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

def augment_graph(graph, augment_type='rotate', params=None):
    """
    对分子图进行数据增强
    
    参数:
    graph: DGL图
    augment_type: 增强类型，'rotate', 'jitter', 'edge_dropout' 或 'combined'
    params: 增强参数
    
    返回:
    augmented_graph: 增强后的图
    """
    if params is None:
        params = {}
        
    augmented_graph = graph.clone()
    
    if augment_type == 'rotate' or augment_type == 'combined':
        # 减小旋转范围，使增强更温和
        angle_x = np.random.uniform(-np.pi/9, np.pi/9)  # ±20度范围，更加温和
        angle_y = np.random.uniform(-np.pi/9, np.pi/9)
        angle_z = np.random.uniform(-np.pi/9, np.pi/9)
        
        # 仅旋转配体节点，保持蛋白质不变
        if 'ligand' in augmented_graph.ntypes and 'coords' in augmented_graph.nodes['ligand'].data:
            ligand_coords = augmented_graph.nodes['ligand'].data['coords']
            rotated_ligand_coords = rotate_coordinates(ligand_coords, angle_x, angle_y, angle_z)
            augmented_graph.nodes['ligand'].data['coords'] = rotated_ligand_coords
            
            # 重新计算边特征（如果边特征包含空间信息）
            if ('ligand', 'to', 'protein') in augmented_graph.canonical_etypes:
                src_nodes, dst_nodes = augmented_graph.edges(etype=('ligand', 'to', 'protein'))
                if len(src_nodes) > 0 and 'distance' in augmented_graph.edges[('ligand', 'to', 'protein')].data:
                    protein_coords = augmented_graph.nodes['protein'].data['coords']
                    distances = torch.norm(rotated_ligand_coords[src_nodes] - protein_coords[dst_nodes], dim=1)
                    augmented_graph.edges[('ligand', 'to', 'protein')].data['distance'] = distances
            
            if ('protein', 'to', 'ligand') in augmented_graph.canonical_etypes:
                src_nodes, dst_nodes = augmented_graph.edges(etype=('protein', 'to', 'ligand'))
                if len(src_nodes) > 0 and 'distance' in augmented_graph.edges[('protein', 'to', 'ligand')].data:
                    protein_coords = augmented_graph.nodes['protein'].data['coords']
                    distances = torch.norm(protein_coords[src_nodes] - rotated_ligand_coords[dst_nodes], dim=1)
                    augmented_graph.edges[('protein', 'to', 'ligand')].data['distance'] = distances
    
    if augment_type == 'jitter' or augment_type == 'combined':
        # 减小扰动幅度
        scale = params.get('jitter_scale', 0.03)  # 默认值降低到0.03
        
        # 仅对配体节点添加扰动，不扰动蛋白质
        if 'ligand' in augmented_graph.ntypes and 'coords' in augmented_graph.nodes['ligand'].data:
            coords = augmented_graph.nodes['ligand'].data['coords']
            jittered_coords = jitter_coordinates(coords, scale)
            augmented_graph.nodes['ligand'].data['coords'] = jittered_coords
            
            # 更新边特征（如果必要）
            if ('ligand', 'to', 'protein') in augmented_graph.canonical_etypes:
                if 'protein' in augmented_graph.ntypes:
                    if 'coords' in augmented_graph.nodes['protein'].data:
                        src_nodes, dst_nodes = augmented_graph.edges(etype=('ligand', 'to', 'protein'))
                        if len(src_nodes) > 0 and 'distance' in augmented_graph.edges[('ligand', 'to', 'protein')].data:
                            protein_coords = augmented_graph.nodes['protein'].data['coords']
                            distances = torch.norm(jittered_coords[src_nodes] - protein_coords[dst_nodes], dim=1)
                            augmented_graph.edges[('ligand', 'to', 'protein')].data['distance'] = distances
            
            if ('protein', 'to', 'ligand') in augmented_graph.canonical_etypes:
                if 'protein' in augmented_graph.ntypes:
                    if 'coords' in augmented_graph.nodes['protein'].data:
                        src_nodes, dst_nodes = augmented_graph.edges(etype=('protein', 'to', 'ligand'))
                        if len(src_nodes) > 0 and 'distance' in augmented_graph.edges[('protein', 'to', 'ligand')].data:
                            protein_coords = augmented_graph.nodes['protein'].data['coords']
                            distances = torch.norm(protein_coords[src_nodes] - jittered_coords[dst_nodes], dim=1)
                            augmented_graph.edges[('protein', 'to', 'ligand')].data['distance'] = distances
    
    if augment_type == 'edge_dropout' or augment_type == 'combined':
        # 减小边抽样比例
        edge_dropout_ratio = params.get('edge_dropout_ratio', 0.01)  # 减小默认比例到0.01
        augmented_graph = random_remove_edges(augmented_graph, ratio=edge_dropout_ratio)
    
    return augmented_graph

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
            # 随机选择增强类型
            if augment_types is None:
                # 根据任务特点，配置更适合的增强类型概率分布
                # 对于药物-蛋白质相互作用，分子的空间构象尤为重要，因此以旋转为主
                rand_val = random.random()
                if rand_val < 0.5:  # 50%概率使用旋转
                    aug_type = 'rotate'
                    params = {'rotation_strength': 0.7}  # 控制旋转幅度
                elif rand_val < 0.8:  # 30%概率使用轻微扰动
                    aug_type = 'jitter'
                    params = {'jitter_scale': 0.03}  # 小幅度扰动
                elif rand_val < 0.9:  # 10%概率使用边删除
                    aug_type = 'edge_dropout'
                    params = {'edge_dropout_ratio': 0.01}  # 很小的边删除率
                else:  # 10%概率使用组合增强
                    aug_type = 'combined'
                    params = {
                        'jitter_scale': 0.02,  # 组合时进一步降低扰动幅度
                        'edge_dropout_ratio': 0.01
                    }
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
    
    # 确认增强类型
    if augment_types is None:
        augment_types = ['rotate', 'jitter', 'edge_dropout', 'combined']
    valid_types = ['rotate', 'jitter', 'edge_dropout', 'combined']
    augment_types = [t for t in augment_types if t in valid_types]
    if not augment_types:
        augment_types = ['rotate', 'combined']  # 默认增强类型
    
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