import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import dgl
from graph_constructor_3d import create_heterograph_from_files, collate_fn
import glob

class ProteinLigandDataset(Dataset):
    """处理蛋白质-配体3D结构数据的数据集"""
    
    def __init__(self, root_dir, label_file=None, transform=None, is_test=False, dis_threshold=5.0, cache_dir=None):
        """
        初始化蛋白质-配体数据集
        
        参数:
        root_dir (str): 包含蛋白质-配体复合物数据的根目录
        label_file (str): 标签文件的路径，包含复合物ID和标签
        transform (callable, optional): 应用于样本的可选转换
        is_test (bool): 是否为测试模式，如果是则不需要标签
        dis_threshold (float): 构建分子间相互作用边的距离阈值
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.dis_threshold = dis_threshold
        self.cache_dir = cache_dir
        
        # 缓存统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 加载标签数据（如果需要）
        if not is_test and label_file is not None:
            self.label_data = pd.read_csv(label_file)
            self.complex_ids = self.label_data['complex_id'].values
            self.labels = self.label_data['label'].values
        else:
            # 如果是测试模式或没有标签文件，则只加载复合物ID
            self.complex_ids = self._get_complex_ids()
            self.labels = None
            
        # 如果提供了缓存目录，检查有哪些复合物已缓存
        if self.cache_dir is not None and os.path.exists(self.cache_dir):
            cached_files = glob.glob(os.path.join(self.cache_dir, "*.pt"))
            self.cached_complexes = set([os.path.basename(f).replace(".pt", "") for f in cached_files])
            print(f"找到 {len(self.cached_complexes)} 个预处理缓存文件")
        else:
            self.cached_complexes = set()
            if self.cache_dir is not None:
                print(f"缓存目录 '{self.cache_dir}' 不存在或未提供")
        
    def _get_complex_ids(self):
        """获取所有复合物ID"""
        complex_ids = []
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
            
            # 首先检查是否有预处理缓存
            if self.cache_dir is not None and complex_id in self.cached_complexes:
                cache_file = os.path.join(self.cache_dir, f"{complex_id}.pt")
                try:
                    # 从缓存加载数据
                    data = torch.load(cache_file, map_location='cpu')
                    self.cache_hits += 1
                    # 每100次缓存命中打印一次统计信息
                    if self.cache_hits % 100 == 0:
                        total = self.cache_hits + self.cache_misses
                        hit_rate = self.cache_hits / total * 100 if total > 0 else 0
                        print(f"缓存统计 - 命中: {self.cache_hits}, 未命中: {self.cache_misses}, 命中率: {hit_rate:.1f}%")
                    return data['graph'], data['label']
                except Exception as e:
                    # 缓存加载失败，回退到标准处理
                    print(f"加载缓存 '{complex_id}' 失败: {str(e)}")
            
            # 缓存未命中，记录统计
            if self.cache_dir is not None:
                self.cache_misses += 1
                
            protein_id = '_'.join(complex_id.split('_')[:2])  # 尝试提取蛋白质ID（前两部分）
            
            # 首先检查平铺结构的数据
            # 构建可能的配体文件路径（根目录下的平铺结构）
            flat_ligand_candidates = [
                os.path.join(self.root_dir, f"{complex_id}_ligand.sdf"),
                os.path.join(self.root_dir, f"{complex_id}_ligand.pdb"),
                os.path.join(self.root_dir, f"{complex_id}_ligand.mol2")
            ]
            
            # 检查是否有平铺结构的配体文件
            ligand_file = None
            for candidate in flat_ligand_candidates:
                if os.path.exists(candidate):
                    ligand_file = candidate
                    break
                    
            # 检查是否有平铺结构的口袋文件
            flat_pocket_candidates = [
                os.path.join(self.root_dir, f"{complex_id}_pocket.pdb"),
                os.path.join(self.root_dir, f"{complex_id}_protein.pdb")
            ]
            
            pocket_file = None
            for candidate in flat_pocket_candidates:
                if os.path.exists(candidate):
                    pocket_file = candidate
                    break
                    
            # 如果找到了平铺结构的文件，直接处理
            if ligand_file is not None and pocket_file is not None:
                # 创建异构图
                graph = create_heterograph_from_files(ligand_file, pocket_file, self.dis_threshold)
                
                # 获取标签（如果不是测试模式）
                if not self.is_test and self.labels is not None:
                    label = self.labels[idx]
                    return graph, label
                else:
                    # 测试模式下没有标签，返回0作为占位符
                    return graph, 0.0
            
            # 如果没有找到平铺结构的文件，尝试层次结构
            # 构建文件路径
            protein_folder = os.path.join(self.root_dir, protein_id)
            complex_folder = os.path.join(protein_folder, complex_id)
            
            # 检查复合物文件夹是否存在
            if not os.path.exists(complex_folder):
                if idx < 5:  # 只对前几个样本打印详细信息，避免日志过多
                    print(f"警告: 复合物文件夹 '{complex_folder}' 不存在")
                return None, None
            
            # 首先尝试查找预计算的DGL图 - 暂时禁用这个功能，因为它可能导致DGL版本兼容性问题
            dgl_file = os.path.join(complex_folder, f"{complex_id}.dgl")
            if False and os.path.exists(dgl_file):  # 暂时禁用预计算图加载
                try:
                    graph = dgl.load_graphs(dgl_file)[0][0]
                    if self.is_test or self.labels is None:
                        return graph, 0.0
                    else:
                        label = self.labels[idx]
                        return graph, label
                except Exception as e:
                    # 如果预计算图加载失败，继续尝试从原始文件构建
                    pass
            
            # 构建可能的配体文件路径（尝试不同的扩展名）
            ligand_candidates = [
                os.path.join(complex_folder, f"{complex_id}_ligand.sdf"),
                os.path.join(complex_folder, f"{complex_id}_ligand.pdb"),
                os.path.join(complex_folder, f"{complex_id}_ligand.mol2")
            ]
            
            # 找到第一个存在的配体文件
            ligand_file = None
            for candidate in ligand_candidates:
                if os.path.exists(candidate):
                    ligand_file = candidate
                    break
            
            if ligand_file is None:
                if idx < 5:  # 只对前几个样本打印详细信息
                    print(f"警告: 无法找到复合物 '{complex_id}' 的配体文件")
                return None, None
            
            # 构建可能的口袋文件路径
            pocket_candidates = [
                os.path.join(complex_folder, f"{complex_id}_pocket_5.0A.pdb"),
                os.path.join(complex_folder, f"{complex_id}_pocket.pdb"),
                os.path.join(complex_folder, f"{complex_id}_protein.pdb"),
                os.path.join(protein_folder, f"{protein_id}.pdb")  # 尝试使用蛋白质PDB文件
            ]
            
            # 找到第一个存在的口袋文件
            pocket_file = None
            for candidate in pocket_candidates:
                if os.path.exists(candidate):
                    pocket_file = candidate
                    break
            
            if pocket_file is None:
                # 如果还是找不到，尝试自动从完整蛋白质生成口袋
                full_protein_path = os.path.join(protein_folder, f"{protein_id}.pdb")
                if os.path.exists(full_protein_path):
                    try:
                        # 创建目标口袋文件路径
                        pocket_file = os.path.join(complex_folder, f"{complex_id}_pocket_5.0A.pdb")
                        
                        # 创建简单的口袋文件（复制完整蛋白质文件）
                        import shutil
                        shutil.copy(full_protein_path, pocket_file)
                    except Exception as e:
                        pocket_file = None
                
            if pocket_file is None:
                if idx < 5:  # 只对前几个样本打印详细信息
                    print(f"警告: 无法找到复合物 '{complex_id}' 的口袋文件")
                return None, None
            
            # 创建异构图
            try:
                # 尝试从文件创建异构图
                graph = create_heterograph_from_files(ligand_file, pocket_file, self.dis_threshold)
                
                # 验证图是否有效
                if graph is None or not isinstance(graph, dgl.DGLHeteroGraph):
                    if idx < 5:  # 只对前几个样本打印详细信息
                        print(f"警告: 为复合物 '{complex_id}' 创建的图无效")
                    return None, None
                    
                # 验证图是否包含所需的节点和边类型
                if 'ligand' not in graph.ntypes or 'pocket' not in graph.ntypes:
                    if idx < 5:  # 只对前几个样本打印详细信息
                        print(f"警告: 复合物 '{complex_id}' 的图缺少必要的节点类型")
                    return None, None
                    
                # 暂时禁用保存功能，防止DGL版本兼容性问题
                if False:  # 禁用保存功能
                    try:
                        dgl.save_graphs(dgl_file, [graph])
                    except Exception as e:
                        pass
                    
                # 应用转换（如果需要）
                if self.transform:
                    graph = self.transform(graph)
                    
                # 获取标签（如果不是测试模式）
                if not self.is_test and self.labels is not None:
                    label = self.labels[idx]
                    return graph, label
                else:
                    # 测试模式下没有标签，返回0作为占位符
                    return graph, 0.0
                    
            except Exception as e:
                # 出错时返回None，在DataLoader中需要处理这种情况
                if idx < 5:  # 只对前几个样本打印详细信息
                    print(f"处理复合物 '{complex_id}' 时出错: {str(e)}")
                return None, None
        except Exception as e:
            # 捕获所有可能的异常
            if idx < 5:  # 只对前几个样本打印详细信息
                print(f"处理索引 {idx} 时出错: {str(e)}")
            return None, None

class ProteinLigandGraphDataset(Dataset):
    """处理预计算的异构图数据的数据集"""
    
    def __init__(self, data_dir, df, transform=None):
        """
        初始化预计算的图数据集
        
        参数:
        data_dir (str): 包含预计算图文件的目录
        df (DataFrame): 包含复合物ID和标签的DataFrame
        transform (callable, optional): 应用于样本的可选转换
        """
        self.data_dir = data_dir
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        row = self.df.iloc[idx]
        complex_id = row['complex_id']
        label = row['label']
        
        # 加载预计算的DGL图文件
        graph_file = os.path.join(self.data_dir, f"{complex_id}.dgl")
        
        try:
            graph = dgl.load_graphs(graph_file)[0][0]
            
            # 应用转换（如果需要）
            if self.transform:
                graph = self.transform(graph)
                
            return graph, label
                
        except Exception as e:
            # 出错时返回None，在DataLoader中需要处理这种情况
            return None, None

def filter_none_collate_fn(batch):
    """过滤None样本的批处理函数"""
    # 过滤掉None样本
    batch = [(g, l) for g, l in batch if g is not None and l is not None]
    if len(batch) == 0:
        return None, None
    return collate_fn(batch)

def get_protein_ligand_dataloader(root_dir, label_file=None, batch_size=32, shuffle=True, num_workers=4, is_test=False, dis_threshold=5.0, cache_dir=None):
    """创建蛋白质-配体数据加载器"""
    dataset = ProteinLigandDataset(
        root_dir=root_dir,
        label_file=label_file,
        is_test=is_test,
        dis_threshold=dis_threshold,
        cache_dir=cache_dir
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=filter_none_collate_fn
    )
    
    return dataloader

def get_precomputed_graph_dataloader(data_dir, label_file, batch_size=32, shuffle=True, num_workers=4):
    """创建预计算图数据的加载器"""
    # 加载标签数据
    df = pd.read_csv(label_file)
    
    dataset = ProteinLigandGraphDataset(
        data_dir=data_dir,
        df=df
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=filter_none_collate_fn
    )
    
    return dataloader

def get_loader(root_dir, label_file=None, batch_size=32, shuffle=True, num_workers=4, dis_threshold=5.0, cache_dir=None):
    """
    获取蛋白质-配体3D数据加载器的统一接口，便于在主程序中调用
    
    参数:
    root_dir: 数据根目录
    label_file: 标签文件路径
    batch_size: 批处理大小
    shuffle: 是否打乱数据
    num_workers: 数据加载的工作线程数
    dis_threshold: 构建分子间相互作用边的距离阈值
    cache_dir: 预处理缓存目录，如果提供则优先使用缓存
    
    返回:
    训练、验证和测试数据加载器
    """
    # 如果标签文件不为None，读取标签数据
    if label_file is not None:
        df = pd.read_csv(label_file)
        
        # 按8:1:1的比例划分训练、验证和测试集
        train_size = int(0.8 * len(df))
        val_size = int(0.1 * len(df))
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size+val_size]
        test_df = df[train_size+val_size:]
        
        # 创建训练集数据加载器
        train_dataset = ProteinLigandDataset(
            root_dir=root_dir,
            label_file=None,  # 已经划分好了数据，不需要再次读取标签文件
            is_test=False,
            dis_threshold=dis_threshold,
            cache_dir=cache_dir
        )
        train_dataset.complex_ids = train_df['complex_id'].values
        train_dataset.labels = train_df['label'].values
        
        # 创建验证集数据加载器
        val_dataset = ProteinLigandDataset(
            root_dir=root_dir,
            label_file=None,
            is_test=False,
            dis_threshold=dis_threshold,
            cache_dir=cache_dir
        )
        val_dataset.complex_ids = val_df['complex_id'].values
        val_dataset.labels = val_df['label'].values
        
        # 创建测试集数据加载器
        test_dataset = ProteinLigandDataset(
            root_dir=root_dir,
            label_file=None,
            is_test=False,
            dis_threshold=dis_threshold,
            cache_dir=cache_dir
        )
        test_dataset.complex_ids = test_df['complex_id'].values
        test_dataset.labels = test_df['label'].values
    else:
        # 无标签文件时，使用默认数据集
        print("没有提供标签文件，将使用所有数据作为训练集和测试集")
        dataset = ProteinLigandDataset(
            root_dir=root_dir,
            is_test=False,
            dis_threshold=dis_threshold,
            cache_dir=cache_dir
        )
        
        # 划分数据集
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        # 创建数据集的副本
        train_dataset = ProteinLigandDataset(
            root_dir=root_dir,
            is_test=False,
            dis_threshold=dis_threshold,
            cache_dir=cache_dir
        )
        train_dataset.complex_ids = [dataset.complex_ids[i] for i in train_indices]
        
        val_dataset = ProteinLigandDataset(
            root_dir=root_dir,
            is_test=False,
            dis_threshold=dis_threshold,
            cache_dir=cache_dir
        )
        val_dataset.complex_ids = [dataset.complex_ids[i] for i in val_indices]
        
        test_dataset = ProteinLigandDataset(
            root_dir=root_dir,
            is_test=False,
            dis_threshold=dis_threshold,
            cache_dir=cache_dir
        )
        test_dataset.complex_ids = [dataset.complex_ids[i] for i in test_indices]
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=filter_none_collate_fn
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=filter_none_collate_fn
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=filter_none_collate_fn
    )
    
    return train_loader, val_loader, test_loader