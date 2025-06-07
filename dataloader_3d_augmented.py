"""
支持加载增强数据的3D蛋白质-配体数据加载器
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import dgl
from graph_constructor_3d import create_heterograph_from_files, collate_fn
import glob
import random
import traceback  # 添加traceback模块用于详细错误信息

class AugmentedProteinLigandDataset(Dataset):
    """处理增强后的蛋白质-配体3D结构数据的数据集"""
    
    def __init__(self, root_dir, label_file=None, transform=None, is_test=False, dis_threshold=5.0, cache_dir=None):
        """
        初始化增强后的蛋白质-配体数据集
        
        参数:
        root_dir (str): 包含蛋白质-配体复合物数据的根目录
        label_file (str): 标签文件的路径，包含复合物ID和标签
        transform (callable, optional): 应用于样本的可选转换
        is_test (bool): 是否为测试模式，如果是则不需要标签
        dis_threshold (float): 构建分子间相互作用边的距离阈值
        cache_dir (str): 包含增强后缓存的目录
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.dis_threshold = dis_threshold
        self.cache_dir = cache_dir
        
        # 缓存统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        
        # 加载标签数据
        if not is_test and label_file is not None:
            # 检查标签文件是否为增强后的格式
            self.label_data = pd.read_csv(label_file)
            print(f"加载标签文件: {label_file}, 包含 {len(self.label_data)} 条记录")
            
            # 如果是增强标签文件，complex_id已经包含了_orig或_augX后缀
            # 创建文件名到标签的直接映射
            self.label_map = {}
            for _, row in self.label_data.iterrows():
                complex_id = row['complex_id']
                label = float(row['label'])
                self.label_map[complex_id] = label
            
            # 从缓存目录读取所有文件，但只保留标签文件中存在的ID
            if self.cache_dir is not None and os.path.exists(self.cache_dir):
                # 创建缓存文件路径列表，只包含标签文件中存在的ID
                self.file_paths = []
                self.complex_ids = []
                
                # 只扫描与当前标签文件中出现的complex_id对应的缓存文件
                for complex_id in self.label_map.keys():
                    file_path = os.path.join(cache_dir, f"{complex_id}.pt")
                    if os.path.exists(file_path):
                        self.file_paths.append(file_path)
                        self.complex_ids.append(complex_id)
                
                print(f"找到 {len(self.file_paths)}/{len(self.label_map)} 个对应的缓存文件")
                if len(self.file_paths) < len(self.label_map):
                    missing_ids = set(self.label_map.keys()) - set(self.complex_ids)
                    print(f"警告: 有 {len(missing_ids)} 个complex_id在缓存中找不到，例如：{list(missing_ids)[:5]}")
            else:
                # 如果没有缓存目录，从标签文件获取复合物ID
                self.complex_ids = self.label_data['complex_id'].values.tolist()
                self.file_paths = [None] * len(self.complex_ids)
                if self.cache_dir is not None:
                    print(f"警告: 缓存目录 '{self.cache_dir}' 不存在")
                else:
                    print("未指定缓存目录，将直接从原始文件加载数据")
            
        else:
            # 如果是测试模式，检查并处理原始测试数据
            self.label_map = {}
            
            # 如果提供了测试标签文件，读取测试标签
            if is_test and label_file is not None and os.path.exists(label_file):
                try:
                    test_df = pd.read_csv(label_file)
                    print(f"读取测试标签文件: {label_file}, 包含 {len(test_df)} 条记录")
                    
                    # 检查列名并尝试修复
                    if 'complex_id' not in test_df.columns or 'label' not in test_df.columns:
                        required_columns = ['complex_id', 'label']
                        present_columns = [col for col in required_columns if col in test_df.columns]
                        missing_columns = [col for col in required_columns if col not in test_df.columns]
                        print(f"警告: 测试标签文件缺少必要列: {missing_columns}")
                        print(f"可用列: {list(test_df.columns)}")
                        
                        # 尝试自动修复常见的列名问题
                        if 'id' in test_df.columns and 'complex_id' not in test_df.columns:
                            test_df['complex_id'] = test_df['id']
                            print("自动将'id'列映射为'complex_id'")
                            
                        if 'y' in test_df.columns and 'label' not in test_df.columns:
                            test_df['label'] = test_df['y']
                            print("自动将'y'列映射为'label'")
                    
                    # 创建标签映射
                    for _, row in test_df.iterrows():
                        if 'complex_id' in row and 'label' in row:
                            complex_id = str(row['complex_id'])
                            label = float(row['label'])
                            self.label_map[complex_id] = label
                            
                            # 检查是否还需要添加_orig后缀的映射（兼容性）
                            if not complex_id.endswith('_orig'):
                                self.label_map[f"{complex_id}_orig"] = label
                    
                    print(f"成功创建测试标签映射，包含 {len(self.label_map)} 个条目")
                    
                except Exception as e:
                    print(f"读取测试标签文件出错: {str(e)}")
                    traceback.print_exc()
            
            # 获取复合物ID列表
            self.complex_ids = self._get_complex_ids()
            self.file_paths = [None] * len(self.complex_ids)
            
            # 如果提供了缓存目录，尝试构建缓存文件路径
            if self.cache_dir is not None and os.path.exists(self.cache_dir) and self.complex_ids:
                # 更新文件路径，检查是否存在原始或特殊缓存文件
                updated_paths = []
                updated_ids = []

                # 如果是测试模式且有标签映射，只处理标签文件中的样本
                if self.is_test and self.label_map:
                    print(f"测试模式：只处理标签文件中的 {len(self.label_map)} 个样本")
                    for complex_id in self.label_map.keys():
                        # 尝试几种可能的文件名格式
                        candidates = [
                            f"{complex_id}.pt",               # 标准格式
                            f"{complex_id}_orig.pt",          # 原始文件后缀
                            f"{complex_id}_test.pt"           # 测试专用后缀
                        ]

                        # 检查每种可能的文件是否存在
                        for candidate in candidates:
                            file_path = os.path.join(self.cache_dir, candidate)
                            if os.path.exists(file_path):
                                updated_paths.append(file_path)
                                updated_ids.append(complex_id)
                                break

                    # 更新complex_ids为只包含标签文件中的ID
                    self.complex_ids = updated_ids
                    self.file_paths = updated_paths
                    print(f"在缓存目录中找到 {len(updated_paths)}/{len(self.label_map)} 个测试样本缓存文件")
                else:
                    # 非测试模式或没有标签映射，使用原来的逻辑
                    for i, complex_id in enumerate(self.complex_ids):
                        # 尝试几种可能的文件名格式
                        candidates = [
                            f"{complex_id}.pt",               # 标准格式
                            f"{complex_id}_orig.pt",          # 原始文件后缀
                            f"{complex_id}_test.pt"           # 测试专用后缀
                        ]

                        # 检查每种可能的文件是否存在
                        for candidate in candidates:
                            file_path = os.path.join(self.cache_dir, candidate)
                            if os.path.exists(file_path):
                                updated_paths.append(file_path)
                                updated_ids.append(complex_id)
                                break

                    if updated_paths:
                        print(f"在缓存目录中找到 {len(updated_paths)}/{len(self.complex_ids)} 个样本缓存文件")
                        self.file_paths = updated_paths
                        self.complex_ids = updated_ids
        
        print(f"数据集初始化完成，总样本数: {len(self.complex_ids)}")
            
    def _get_complex_ids(self):
        """获取所有复合物ID"""
        complex_ids = []

        # 如果是测试模式且有标签映射，直接使用标签文件中的ID
        if self.is_test and self.label_map:
            complex_ids = list(self.label_map.keys())
            print(f"测试模式：从标签文件获取 {len(complex_ids)} 个复合物ID")
            return complex_ids

        # 如果提供了缓存目录，从_orig.pt文件中提取复合物ID
        if self.cache_dir is not None and os.path.exists(self.cache_dir):
            if self.is_test:
                # 测试模式但没有标签映射，尝试查找测试相关的文件
                all_files = glob.glob(os.path.join(self.cache_dir, "*.pt"))

                # 按优先级查找: _test.pt, _orig.pt
                test_files = [f for f in all_files if "_test.pt" in f]
                if test_files:
                    print(f"从缓存目录中找到 {len(test_files)} 个测试专用缓存文件")
                    for file_path in test_files:
                        file_name = os.path.basename(file_path)
                        complex_id = file_name.replace("_test.pt", "")
                        complex_ids.append(complex_id)
                    return complex_ids

                orig_files = [f for f in all_files if "_orig.pt" in f]
                if orig_files:
                    print(f"警告：测试模式但没有标签映射，将使用所有 {len(orig_files)} 个原始缓存文件")
                    for file_path in orig_files:
                        file_name = os.path.basename(file_path)
                        complex_id = file_name.replace("_orig.pt", "")
                        complex_ids.append(complex_id)
                    return complex_ids

                # 如果没有特殊标记的文件，警告并返回空列表
                print(f"警告：测试模式但没有标签映射，且缓存目录中没有找到合适的测试文件")
                return []
            else:
                # 非测试模式，只查找原始文件
                orig_files = glob.glob(os.path.join(self.cache_dir, "*_orig.pt"))
                for file_path in orig_files:
                    file_name = os.path.basename(file_path)
                    complex_id = file_name.replace("_orig.pt", "")
                    complex_ids.append(complex_id)
                
                if complex_ids:
                    print(f"从缓存目录中找到 {len(complex_ids)} 个复合物ID")
                    return complex_ids
        
        # 如果没有从缓存目录找到，尝试从根目录获取
        try:
            # 检查根目录是否存在
            if not os.path.exists(self.root_dir):
                print(f"警告: 指定的数据根目录 '{self.root_dir}' 不存在")
                return complex_ids
                
            # 检查是否有复合物直接放在根目录下（平铺结构）
            direct_complex_files = False
            for file in os.listdir(self.root_dir):
                if file.endswith('.pdb') or file.endswith('.sdf') or file.endswith('.mol2'):
                    # 找到直接放置的复合物文件
                    direct_complex_files = True
                    break
                    
            # 如果找到直接的复合物文件，尝试从文件名提取复合物ID
            if direct_complex_files:
                print(f"检测到平铺的数据结构: 复合物文件直接放在根目录 '{self.root_dir}'")
                for file in os.listdir(self.root_dir):
                    if file.endswith('_ligand.pdb') or file.endswith('_ligand.sdf') or file.endswith('_ligand.mol2'):
                        # 从配体文件名中提取复合物ID
                        complex_id = file.replace('_ligand.pdb', '').replace('_ligand.sdf', '').replace('_ligand.mol2', '')
                        complex_ids.append(complex_id)
                print(f"从平铺结构中找到 {len(complex_ids)} 个复合物")
                return complex_ids
                
            # 按常规结构查找蛋白质文件夹
            protein_folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]
            if len(protein_folders) == 0:
                print(f"警告: 在根目录 '{self.root_dir}' 中未找到任何蛋白质文件夹")
                return complex_ids
                
            # 遍历蛋白质文件夹
            for protein_folder in protein_folders:
                protein_path = os.path.join(self.root_dir, protein_folder)
                if os.path.isdir(protein_path):
                    # 检查复合物文件夹
                    complex_folders = [f for f in os.listdir(protein_path) if os.path.isdir(os.path.join(protein_path, f))]
                    for complex_folder in complex_folders:
                        complex_ids.append(complex_folder)
                
            print(f"从层次结构中找到 {len(complex_ids)} 个复合物")
            
        except Exception as e:
            print(f"获取复合物ID时出错: {str(e)}")
            traceback.print_exc()
            
        return complex_ids
    
    def __len__(self):
        return len(self.complex_ids)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            complex_id = self.complex_ids[idx]
            file_path = self.file_paths[idx]
            
            # 如果有预处理缓存文件路径，直接加载
            if file_path is not None and os.path.exists(file_path):
                try:
                    # 从缓存加载数据
                    data = torch.load(file_path, map_location='cpu')
                    self.cache_hits += 1
                    
                    # 减少缓存统计输出频率（仅在详细模式下每500次打印一次）
                    if hasattr(self, 'verbose') and self.verbose and self.cache_hits % 500 == 0:
                        total = self.cache_hits + self.cache_misses
                        hit_rate = self.cache_hits / total * 100 if total > 0 else 0
                        print(f"缓存统计 - 命中: {self.cache_hits}, 未命中: {self.cache_misses}, 错误: {self.errors}, 命中率: {hit_rate:.1f}%")
                    
                    # 验证图结构是否有效（特别是在测试模式下）
                    if 'graph' in data:
                        graph = data['graph']
                        
                        # 验证图的基本属性
                        if not isinstance(graph, dgl.DGLGraph):
                            print(f"警告: 文件 {file_path} 中的图不是有效的DGLGraph")
                            self.errors += 1
                            return None, None
                            
                        # 检查图是否为空
                        if graph.number_of_nodes() == 0:
                            print(f"警告: 文件 {file_path} 中的图没有节点")
                            self.errors += 1
                            return None, None
                            
                        # 在测试模式下，减少图结构检查输出（仅在详细模式下且频率更低）
                        if (self.is_test and hasattr(self, 'verbose') and self.verbose and
                            idx < 3):  # 只检查前3个样本
                            node_types = graph.ntypes
                            edge_types = graph.etypes
                            print(f"图结构检查 [{idx}]: 节点类型={node_types}, 边类型={edge_types}, "
                                  f"节点数={graph.number_of_nodes()}, 边数={graph.number_of_edges()}")
                    else:
                        print(f"警告: 文件 {file_path} 中没有找到'graph'字段")
                        self.errors += 1
                        return None, None
                    
                    # 检查标签文件中是否有对应的标签
                    if not self.is_test:
                        # 从映射表中获取标签
                        if complex_id in self.label_map:
                            # 使用标签文件中的标签（优先）
                            label = self.label_map[complex_id]
                        elif 'label' in data:
                            # 如果标签文件中没有，但缓存中有标签，使用缓存中的标签
                            label = data['label']
                        else:
                            print(f"警告: 无法找到'{complex_id}'的标签")
                            self.errors += 1
                            return None, None
                            
                        return data['graph'], label
                    else:
                        # 测试模式，首先尝试从标签映射获取标签
                        if complex_id in self.label_map:
                            label = self.label_map[complex_id]
                            return data['graph'], label
                        # 如果标签映射没有，但缓存中有标签，使用缓存中的标签
                        elif 'label' in data:
                            label = data['label']
                            return data['graph'], label
                        # 否则使用0占位
                        else:
                            return data['graph'], 0.0
                    
                except Exception as e:
                    print(f"加载缓存 '{file_path}' 失败: {str(e)}")
                    self.errors += 1
                    self.cache_misses += 1
                    return None, None
            
            # 如果是测试模式且没有缓存文件，尝试更多的候选缓存文件
            if self.is_test and self.cache_dir is not None and os.path.exists(self.cache_dir):
                # 尝试几种可能的文件名格式
                candidates = [
                    f"{complex_id}.pt",               # 标准格式
                    f"{complex_id}_orig.pt",          # 原始文件后缀
                    f"{complex_id}_test.pt"           # 测试专用后缀
                ]
                
                for candidate in candidates:
                    candidate_path = os.path.join(self.cache_dir, candidate)
                    if os.path.exists(candidate_path):
                        try:
                            data = torch.load(candidate_path, map_location='cpu')
                            if 'graph' in data:
                                # 获取标签
                                label = 0.0  # 默认值
                                if complex_id in self.label_map:
                                    label = self.label_map[complex_id]
                                elif 'label' in data:
                                    label = data['label']
                                    
                                # 更新文件路径以便下次直接使用
                                self.file_paths[idx] = candidate_path
                                return data['graph'], label
                        except Exception as e:
                            print(f"尝试加载备用缓存 '{candidate_path}' 失败: {str(e)}")
                            continue
                
                # 如果所有候选文件都失败，记录未命中
                self.cache_misses += 1
                print(f"警告: 在缓存目录中未找到样本 '{complex_id}' 的有效缓存文件")
                    
            # 如果没有缓存文件或加载失败，则返回None
            self.cache_misses += 1
            return None, None
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            self.errors += 1
            return None, None

def filter_none_collate_fn(batch):
    """过滤掉None值并进行批处理"""
    # 首先过滤掉无效的样本
    valid_batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    
    # 如果批次中没有有效样本，返回None
    if len(valid_batch) == 0:
        return None, None
    
    # 分离图和标签
    graphs, labels = map(list, zip(*valid_batch))
    
    # 验证所有图是否有效
    valid_graphs = []
    valid_labels = []
    for i, (g, l) in enumerate(zip(graphs, labels)):
        if isinstance(g, dgl.DGLGraph) and g.number_of_nodes() > 0:
            valid_graphs.append(g)
            valid_labels.append(l)
    
    # 如果没有有效的图，返回None
    if len(valid_graphs) == 0:
        return None, None
    
    # 如果有效样本数少于原始批次大小，打印警告
    if len(valid_graphs) < len(batch):
        print(f"警告: 批次中只有 {len(valid_graphs)}/{len(batch)} 个有效样本")
    
    # 批处理有效的图和标签
    try:
        batched_graph = dgl.batch(valid_graphs)
        labels = torch.tensor(valid_labels, dtype=torch.float32)
        return batched_graph, labels
    except Exception as e:
        print(f"批处理图失败: {str(e)}")
        traceback.print_exc()
        return None, None

def get_augmented_dataloader(root_dir, label_file=None, batch_size=32, shuffle=True, num_workers=4, 
                         is_test=False, dis_threshold=5.0, cache_dir=None):
    """
    获取处理增强数据的数据加载器
    
    参数:
    root_dir (str): 数据根目录
    label_file (str): 标签文件路径
    batch_size (int): 批次大小
    shuffle (bool): 是否打乱数据
    num_workers (int): 数据加载线程数
    is_test (bool): 是否为测试模式
    dis_threshold (float): 构建分子间相互作用边的距离阈值
    cache_dir (str): 缓存目录路径
    
    返回:
    DataLoader: 数据加载器
    """
    # 特殊处理测试数据
    if is_test:
        print(f"测试模式: {'使用原始测试数据' if not label_file.endswith('_augmented.csv') else '使用增强测试数据'}")
        print(f"测试标签文件: {label_file}")
        print(f"缓存目录: {cache_dir}")
        
        # 检查标签文件是否存在
        if not os.path.exists(label_file):
            print(f"错误: 测试标签文件 {label_file} 不存在!")
            return DataLoader([], batch_size=batch_size, shuffle=False, collate_fn=filter_none_collate_fn)
            
        # 读取测试标签文件
        try:
            test_df = pd.read_csv(label_file)
            print(f"测试标签文件包含 {len(test_df)} 条记录")
            
            # 检查标签文件格式
            if 'complex_id' not in test_df.columns or 'label' not in test_df.columns:
                required_columns = ['complex_id', 'label']
                present_columns = [col for col in required_columns if col in test_df.columns]
                missing_columns = [col for col in required_columns if col not in test_df.columns]
                print(f"警告: 测试标签文件缺少必要列: {missing_columns}")
                print(f"可用列: {list(test_df.columns)}")
                
                # 尝试自动修复常见的列名问题
                if 'id' in test_df.columns and 'complex_id' not in test_df.columns:
                    test_df['complex_id'] = test_df['id']
                    print("自动将'id'列映射为'complex_id'")
                    
                if 'y' in test_df.columns and 'label' not in test_df.columns:
                    test_df['label'] = test_df['y']
                    print("自动将'y'列映射为'label'")
            
            # 统计测试集标签分布
            if 'label' in test_df.columns:
                pos_count = test_df['label'].sum()
                neg_count = len(test_df) - pos_count
                pos_ratio = pos_count / len(test_df) * 100
                print(f"测试集标签分布: 正样本 {pos_count} ({pos_ratio:.2f}%), 负样本 {neg_count} ({100-pos_ratio:.2f}%)")
        except Exception as e:
            print(f"读取测试标签文件出错: {str(e)}")
            traceback.print_exc()
            print("将创建空的测试数据加载器")
            return DataLoader([], batch_size=batch_size, shuffle=False, collate_fn=filter_none_collate_fn)
    
    # 初始化数据集
    dataset = AugmentedProteinLigandDataset(
        root_dir=root_dir,
        label_file=label_file,
        is_test=is_test,
        dis_threshold=dis_threshold,
        cache_dir=cache_dir
    )
    
    # 输出数据集大小信息
    print(f"{'测试' if is_test else '训练/验证'} 数据集初始化完成: {len(dataset)} 个样本")
    
    # 数据集过滤
    if len(dataset) == 0:
        print("警告: 数据集为空，将返回空的数据加载器")
        return DataLoader([], batch_size=batch_size, shuffle=False, collate_fn=filter_none_collate_fn)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not is_test,  # 测试模式不打乱数据
        num_workers=num_workers,
        collate_fn=filter_none_collate_fn,
        drop_last=False  # 测试时不丢弃最后一个批次
    )
    
    # 验证数据加载器是否包含预期的批次数
    expected_batches = (len(dataset) + batch_size - 1) // batch_size
    print(f"创建数据加载器成功: 预期 {expected_batches} 批次，批次大小 {batch_size}")
    
    return dataloader