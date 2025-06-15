"""
多模态数据加载器 - 同时加载3D结构数据和对应的1D/2D序列数据
确保数据完全对应，不使用任何数据增强
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import dgl
from dataloader_3d import get_loader as get_3d_loader
from dataloader import DTIDataset
from rdkit import Chem
from rdkit.Chem import Descriptors
import traceback


class MultimodalDataset(Dataset):
    """多模态数据集，同时包含3D结构数据和1D/2D序列数据"""
    
    def __init__(self, data_3d_root, data_1d2d_root, label_file, seqid_mapping_file, 
                 dis_threshold=5.0, cache_dir=None, is_test=False):
        """
        初始化多模态数据集
        
        参数:
        data_3d_root: 3D数据根目录
        data_1d2d_root: 1D/2D数据根目录
        label_file: 标签文件路径
        seqid_mapping_file: seqid映射文件路径
        dis_threshold: 3D数据距离阈值
        cache_dir: 3D数据缓存目录
        is_test: 是否为测试模式
        """
        self.data_3d_root = data_3d_root
        self.data_1d2d_root = data_1d2d_root
        self.dis_threshold = dis_threshold
        self.cache_dir = cache_dir
        self.is_test = is_test
        
        # 加载标签数据
        self.label_df = pd.read_csv(label_file)
        print(f"加载标签文件: {label_file}, 包含 {len(self.label_df)} 条记录")
        
        # 加载seqid映射数据
        self.seqid_mapping = pd.read_csv(seqid_mapping_file)
        print(f"加载seqid映射文件: {seqid_mapping_file}, 包含 {len(self.seqid_mapping)} 条记录")
        
        # 创建complex_id到seqid的映射
        self.complex_to_seqid = {}
        for _, row in self.label_df.iterrows():
            complex_id = row['complex_id']
            # 从complex_id中提取seqid（假设格式为 PDB_CHAIN_SEQID）
            parts = complex_id.split('_')
            if len(parts) >= 3:
                seqid = parts[2]  # 提取seqid部分
                self.complex_to_seqid[complex_id] = int(seqid)
        
        # 创建seqid到1D/2D数据的映射
        self.seqid_to_1d2d = {}
        for _, row in self.seqid_mapping.iterrows():
            seqid = row['seqid']
            self.seqid_to_1d2d[seqid] = {
                'SMILES': row['SMILES'],
                'Protein': row['Protein'],
                'Y': row['Y']
            }
        
        # 过滤出有对应1D/2D数据的样本
        self.valid_samples = []
        for _, row in self.label_df.iterrows():
            complex_id = row['complex_id']
            if complex_id in self.complex_to_seqid:
                seqid = self.complex_to_seqid[complex_id]
                if seqid in self.seqid_to_1d2d:
                    self.valid_samples.append({
                        'complex_id': complex_id,
                        'seqid': seqid,
                        'label': row['label']
                    })
        
        print(f"找到 {len(self.valid_samples)}/{len(self.label_df)} 个有对应1D/2D数据的样本")
        
        # 初始化3D数据加载器组件
        self._init_3d_loader()
        
        # 初始化1D/2D数据加载器组件
        self._init_1d2d_loader()
    
    def _init_3d_loader(self):
        """初始化3D数据加载组件"""
        # 这里需要实现3D数据的加载逻辑
        # 可以复用现有的3D数据加载代码，但要确保不使用增强
        pass
    
    def _init_1d2d_loader(self):
        """初始化1D/2D数据加载组件"""
        # 创建1D/2D数据的DataFrame
        data_1d2d = []
        for sample in self.valid_samples:
            seqid = sample['seqid']
            if seqid in self.seqid_to_1d2d:
                data_1d2d.append({
                    'SMILES': self.seqid_to_1d2d[seqid]['SMILES'],
                    'Protein': self.seqid_to_1d2d[seqid]['Protein'],
                    'Y': sample['label']  # 使用3D标签文件中的标签
                })
        
        self.df_1d2d = pd.DataFrame(data_1d2d)
        print(f"创建1D/2D数据DataFrame，包含 {len(self.df_1d2d)} 条记录")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        """获取单个多模态样本"""
        try:
            sample = self.valid_samples[idx]
            complex_id = sample['complex_id']
            seqid = sample['seqid']
            label = sample['label']
            
            # 加载3D数据
            graph_3d = self._load_3d_data(complex_id)
            if graph_3d is None:
                return None, None, None
            
            # 加载1D/2D数据
            data_1d2d = self._load_1d2d_data(seqid)
            if data_1d2d is None:
                return None, None, None
            
            return graph_3d, data_1d2d, label
            
        except Exception as e:
            print(f"加载样本 {idx} 时出错: {str(e)}")
            traceback.print_exc()
            return None, None, None
    
    def _load_3d_data(self, complex_id):
        """加载3D图数据"""
        # 检查缓存
        if self.cache_dir and os.path.exists(self.cache_dir):
            cache_file = os.path.join(self.cache_dir, f"{complex_id}.pt")
            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file, map_location='cpu')
                    if 'graph' in data:
                        return data['graph']
                except Exception as e:
                    print(f"加载3D缓存失败: {str(e)}")

        # 如果缓存不存在，从原始文件构建
        try:
            from graph_constructor_3d import create_heterograph_from_files

            # 构建文件路径
            protein_id = '_'.join(complex_id.split('_')[:2])  # 提取蛋白质ID

            # 尝试平铺结构
            ligand_candidates = [
                os.path.join(self.data_3d_root, f"{complex_id}_ligand.sdf"),
                os.path.join(self.data_3d_root, f"{complex_id}_ligand.pdb"),
                os.path.join(self.data_3d_root, f"{complex_id}_ligand.mol2")
            ]

            pocket_candidates = [
                os.path.join(self.data_3d_root, f"{complex_id}_pocket.pdb"),
                os.path.join(self.data_3d_root, f"{complex_id}_protein.pdb")
            ]

            # 找到存在的文件
            ligand_file = None
            for candidate in ligand_candidates:
                if os.path.exists(candidate):
                    ligand_file = candidate
                    break

            pocket_file = None
            for candidate in pocket_candidates:
                if os.path.exists(candidate):
                    pocket_file = candidate
                    break

            # 如果平铺结构不存在，尝试层次结构
            if ligand_file is None or pocket_file is None:
                protein_folder = os.path.join(self.data_3d_root, protein_id)
                complex_folder = os.path.join(protein_folder, complex_id)

                if os.path.exists(complex_folder):
                    ligand_candidates = [
                        os.path.join(complex_folder, f"{complex_id}_ligand.sdf"),
                        os.path.join(complex_folder, f"{complex_id}_ligand.pdb"),
                        os.path.join(complex_folder, f"{complex_id}_ligand.mol2")
                    ]

                    pocket_candidates = [
                        os.path.join(complex_folder, f"{complex_id}_pocket_5.0A.pdb"),
                        os.path.join(complex_folder, f"{complex_id}_pocket.pdb"),
                        os.path.join(complex_folder, f"{complex_id}_protein.pdb"),
                        os.path.join(protein_folder, f"{protein_id}.pdb")
                    ]

                    for candidate in ligand_candidates:
                        if os.path.exists(candidate):
                            ligand_file = candidate
                            break

                    for candidate in pocket_candidates:
                        if os.path.exists(candidate):
                            pocket_file = candidate
                            break

            if ligand_file and pocket_file:
                graph = create_heterograph_from_files(ligand_file, pocket_file, self.dis_threshold)
                return graph
            else:
                print(f"无法找到复合物 {complex_id} 的3D结构文件")
                return None

        except Exception as e:
            print(f"构建3D图失败 (complex_id={complex_id}): {str(e)}")
            return None
    
    def _load_1d2d_data(self, seqid):
        """加载1D/2D序列数据并转换为DrugBAN格式"""
        if seqid in self.seqid_to_1d2d:
            data = self.seqid_to_1d2d[seqid]
            smiles = data['SMILES']
            protein_seq = data['Protein']

            try:
                # 导入必要的模块
                from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
                from utils import integer_label_protein
                from functools import partial

                # 初始化特征提取器
                atom_featurizer = CanonicalAtomFeaturizer()
                bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
                fc = partial(smiles_to_bigraph, add_self_loop=True)

                # 将SMILES转换为分子图（DrugBAN格式）
                mol_graph = fc(smiles=smiles,
                              node_featurizer=atom_featurizer,
                              edge_featurizer=bond_featurizer)

                # 处理分子图的节点特征（添加虚拟节点标记）
                actual_node_feats = mol_graph.ndata.pop('h')
                num_actual_nodes = actual_node_feats.shape[0]
                max_drug_nodes = 290  # DrugBAN的默认最大节点数
                num_virtual_nodes = max_drug_nodes - num_actual_nodes

                if num_virtual_nodes > 0:
                    # 添加虚拟节点标记
                    virtual_node_bit = torch.zeros([num_actual_nodes, 1])
                    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
                    mol_graph.ndata['h'] = actual_node_feats

                    # 添加虚拟节点
                    virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74),
                                                  torch.ones(num_virtual_nodes, 1)), 1)
                    mol_graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
                else:
                    # 如果节点数已经达到最大值，只添加虚拟节点标记
                    virtual_node_bit = torch.zeros([num_actual_nodes, 1])
                    actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
                    mol_graph.ndata['h'] = actual_node_feats

                mol_graph = mol_graph.add_self_loop()

                # 将蛋白质序列转换为整数编码（DrugBAN格式）
                max_protein_len = 1000  # DrugBAN的默认最大蛋白质长度
                if len(protein_seq) > max_protein_len:
                    protein_seq = protein_seq[:max_protein_len]

                protein_encoded = integer_label_protein(protein_seq, max_protein_len)
                protein_tensor = torch.tensor(protein_encoded, dtype=torch.long)

                return {
                    'mol_graph': mol_graph,
                    'protein_seq': protein_tensor,
                    'smiles': smiles,
                    'protein_seq_raw': protein_seq
                }
            except Exception as e:
                print(f"转换1D/2D数据失败 (seqid={seqid}): {str(e)}")
                traceback.print_exc()
                return None
        return None


def multimodal_collate_fn(batch):
    """多模态数据的批处理函数"""
    # 过滤掉无效样本
    valid_batch = [item for item in batch if all(x is not None for x in item)]

    if len(valid_batch) == 0:
        return None, None, None

    # 分离不同模态的数据
    graphs_3d, data_1d2d, labels = zip(*valid_batch)

    # 批处理3D图数据
    try:
        batched_graph_3d = dgl.batch(graphs_3d)
    except Exception as e:
        print(f"批处理3D图失败: {str(e)}")
        return None, None, None

    # 批处理1D/2D数据
    try:
        # 分离分子图和蛋白质序列
        mol_graphs = [item['mol_graph'] for item in data_1d2d if item is not None]
        protein_seqs = [item['protein_seq'] for item in data_1d2d if item is not None]

        # 批处理分子图
        batched_mol_graphs = dgl.batch(mol_graphs)

        # 批处理蛋白质序列（堆叠为张量）
        batched_protein_seqs = torch.stack(protein_seqs)

        batched_data_1d2d = {
            'mol_graph': batched_mol_graphs,
            'protein_seq': batched_protein_seqs,
            'smiles': [item['smiles'] for item in data_1d2d if item is not None],
            'protein_seq_raw': [item['protein_seq_raw'] for item in data_1d2d if item is not None]
        }
    except Exception as e:
        print(f"批处理1D/2D数据失败: {str(e)}")
        traceback.print_exc()
        return None, None, None

    # 批处理标签
    batched_labels = torch.tensor(labels, dtype=torch.float32)

    return batched_graph_3d, batched_data_1d2d, batched_labels


def get_multimodal_dataloader(data_3d_root, data_1d2d_root, label_file, seqid_mapping_file,
                             batch_size=32, shuffle=True, num_workers=4, 
                             dis_threshold=5.0, cache_dir=None, is_test=False):
    """
    获取多模态数据加载器
    
    参数:
    data_3d_root: 3D数据根目录
    data_1d2d_root: 1D/2D数据根目录  
    label_file: 标签文件路径
    seqid_mapping_file: seqid映射文件路径
    batch_size: 批次大小
    shuffle: 是否打乱数据
    num_workers: 工作线程数
    dis_threshold: 3D数据距离阈值
    cache_dir: 3D数据缓存目录
    is_test: 是否为测试模式
    
    返回:
    DataLoader: 多模态数据加载器
    """
    dataset = MultimodalDataset(
        data_3d_root=data_3d_root,
        data_1d2d_root=data_1d2d_root,
        label_file=label_file,
        seqid_mapping_file=seqid_mapping_file,
        dis_threshold=dis_threshold,
        cache_dir=cache_dir,
        is_test=is_test
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=multimodal_collate_fn,
        drop_last=not is_test  # 测试时不丢弃最后一个不完整的批次
    )
    
    return dataloader
