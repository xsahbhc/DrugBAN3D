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
            if self.cache_dir is not None:
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
            else:
                # 如果没有缓存目录，从标签文件获取复合物ID
                self.complex_ids = self.label_data['complex_id'].values.tolist()
                self.file_paths = [None] * len(self.complex_ids)
            
        else:
            # 如果是测试模式，不考虑增强，只加载原始复合物ID
            self.label_map = {}
            self.complex_ids = self._get_complex_ids()
            self.file_paths = [None] * len(self.complex_ids)
        
        print(f"数据集初始化完成，总样本数: {len(self.complex_ids)}")
            
    def _get_complex_ids(self):
        """获取所有复合物ID"""
        complex_ids = []
        
        # 如果提供了缓存目录，从_orig.pt文件中提取复合物ID
        if self.cache_dir is not None:
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
                    
                    # 每100次缓存命中打印一次统计信息
                    if self.cache_hits % 100 == 0:
                        total = self.cache_hits + self.cache_misses
                        hit_rate = self.cache_hits / total * 100 if total > 0 else 0
                        print(f"缓存统计 - 命中: {self.cache_hits}, 未命中: {self.cache_misses}, 命中率: {hit_rate:.1f}%")
                    
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
                            return None, None
                            
                        return data['graph'], label
                    else:
                        # 测试模式，如果缓存中有标签就使用，否则用0占位
                        label = data.get('label', 0.0)
                        return data['graph'], label
                    
                except Exception as e:
                    print(f"加载缓存 '{file_path}' 失败: {str(e)}")
                    
            # 如果没有缓存文件或加载失败，则返回None
            return None, None
            
        except Exception as e:
            print(f"获取样本 {idx} 时出错: {str(e)}")
            return None, None

def filter_none_collate_fn(batch):
    """过滤掉None值并进行批处理"""
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if len(batch) == 0:
        return None, None
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels

def get_augmented_dataloader(root_dir, label_file=None, batch_size=32, shuffle=True, num_workers=4, 
                             is_test=False, dis_threshold=5.0, cache_dir=None):
    """获取支持增强数据的蛋白质-配体数据加载器"""
    dataset = AugmentedProteinLigandDataset(
        root_dir=root_dir,
        label_file=label_file,
        is_test=is_test,
        dis_threshold=dis_threshold,
        cache_dir=cache_dir
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=filter_none_collate_fn
    )