import pandas as pd
import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein, heterograph_collate_func
import dgl
import os
from rdkit import Chem
import pickle
import logging
from graph_constructor_3d import construct_heterograph, load_molecule_from_file, create_heterograph_from_files

class DTIDataset(data.Dataset):
    """原始DrugBAN的数据集 - 保留用于向后兼容"""
    def __init__(self, list_IDs, df, max_drug_nodes=290):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer, edge_featurizer=self.bond_featurizer)
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
        v_d = v_d.add_self_loop()

        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]["Y"]
        return v_d, v_p, y


class DrugBAN3DDataset(data.Dataset):
    """适用于DrugBAN3D的3D蛋白质-配体数据集"""
    def __init__(self, data_dir, df=None, is_test=False, dis_threshold=5.0):
        """
        初始化DrugBAN3D数据集
        
        参数:
        data_dir (str): 包含蛋白质-配体复合物数据的根目录
        df (DataFrame): 包含复合物ID和标签的DataFrame
        is_test (bool): 是否为测试模式，如果是则不需要标签
        dis_threshold (float): 构建分子间相互作用边的距离阈值
        """
        self.data_dir = data_dir
        self.is_test = is_test
        self.dis_threshold = dis_threshold
        
        # 处理数据框
        self.df = df
        if df is not None:
            self.complex_ids = df['complex_id'].values if 'complex_id' in df.columns else df.index.values
            self.labels = df['label'].values if 'label' in df.columns else np.zeros(len(df))
        else:
            # 如果没有提供数据框，尝试从目录结构获取复合物IDs
            self.complex_ids = self._get_complex_ids()
            self.labels = np.zeros(len(self.complex_ids))  # 默认标签为0
            
    def _get_complex_ids(self):
        """从目录结构获取复合物IDs"""
        complex_ids = []
        try:
            # 检查平铺结构（直接在根目录下的文件）
            flat_structure = any(f.endswith(('_ligand.pdb', '_ligand.sdf', '_ligand.mol2')) 
                                for f in os.listdir(self.data_dir))
            
            if flat_structure:
                for file in os.listdir(self.data_dir):
                    if file.endswith('_ligand.pdb') or file.endswith('_ligand.sdf') or file.endswith('_ligand.mol2'):
                        complex_id = file.replace('_ligand.pdb', '').replace('_ligand.sdf', '').replace('_ligand.mol2', '')
                        complex_ids.append(complex_id)
            else:
                # 检查层次结构（蛋白质文件夹/复合物文件夹）
                for protein_folder in os.listdir(self.data_dir):
                    protein_path = os.path.join(self.data_dir, protein_folder)
                    if os.path.isdir(protein_path):
                        for complex_folder in os.listdir(protein_path):
                            complex_path = os.path.join(protein_path, complex_folder)
                            if os.path.isdir(complex_path):
                                complex_ids.append(complex_folder)
                
        except Exception as e:
            logging.warning(f"获取复合物ID时出错: {str(e)}")
            
        return complex_ids
    
    def __len__(self):
        return len(self.complex_ids)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        try:
            complex_id = self.complex_ids[idx]
            
            # 首先检查是否存在预计算的异构图
            graph = self._load_cached_graph(complex_id)
            if graph is not None:
                label = self.labels[idx] if not self.is_test else 0.0
                return graph, label
            
            # 尝试从文件构建异构图
            ligand_file, pocket_file = self._find_files(complex_id)
            if ligand_file and pocket_file:
                graph = create_heterograph_from_files(ligand_file, pocket_file, self.dis_threshold)
                label = self.labels[idx] if not self.is_test else 0.0
                return graph, label
                
            logging.warning(f"无法找到复合物 {complex_id} 的文件")
            return None, self.labels[idx] if not self.is_test else 0.0
            
        except Exception as e:
            logging.error(f"加载复合物 {idx} (ID: {self.complex_ids[idx]}) 时出错: {str(e)}")
            return None, self.labels[idx] if not self.is_test else 0.0
            
    def _load_cached_graph(self, complex_id):
        """尝试加载预计算的图"""
        # 构建可能的缓存文件路径
        protein_id = '_'.join(complex_id.split('_')[:2])
        
        # 检查平铺结构
        dgl_file = os.path.join(self.data_dir, f"{complex_id}.dgl")
        if os.path.exists(dgl_file):
            try:
                return dgl.load_graphs(dgl_file)[0][0]
            except:
                pass
                
        # 检查层次结构
        dgl_file = os.path.join(self.data_dir, protein_id, complex_id, f"{complex_id}.dgl")
        if os.path.exists(dgl_file):
            try:
                return dgl.load_graphs(dgl_file)[0][0]
            except:
                pass
                
        return None
        
    def _find_files(self, complex_id):
        """查找配体和口袋文件"""
        protein_id = '_'.join(complex_id.split('_')[:2])
        
        # 检查平铺结构
        flat_ligand_candidates = [
            os.path.join(self.data_dir, f"{complex_id}_ligand.sdf"),
            os.path.join(self.data_dir, f"{complex_id}_ligand.pdb"),
            os.path.join(self.data_dir, f"{complex_id}_ligand.mol2")
        ]
        
        flat_pocket_candidates = [
            os.path.join(self.data_dir, f"{complex_id}_pocket.pdb"),
            os.path.join(self.data_dir, f"{complex_id}_protein.pdb")
        ]
        
        # 检查平铺结构的文件
        for l_file in flat_ligand_candidates:
            if os.path.exists(l_file):
                for p_file in flat_pocket_candidates:
                    if os.path.exists(p_file):
                        return l_file, p_file
        
        # 检查层次结构
        complex_folder = os.path.join(self.data_dir, protein_id, complex_id)
        if os.path.exists(complex_folder):
            ligand_candidates = [
                os.path.join(complex_folder, f"{complex_id}_ligand.sdf"),
                os.path.join(complex_folder, f"{complex_id}_ligand.pdb"),
                os.path.join(complex_folder, f"{complex_id}_ligand.mol2")
            ]
            
            pocket_candidates = [
                os.path.join(complex_folder, f"{complex_id}_pocket.pdb"),
                os.path.join(complex_folder, f"{complex_id}_protein.pdb"),
                os.path.join(self.data_dir, protein_id, f"{protein_id}.pdb")
            ]
            
            for l_file in ligand_candidates:
                if os.path.exists(l_file):
                    for p_file in pocket_candidates:
                        if os.path.exists(p_file):
                            return l_file, p_file
        
        return None, None


class MultiDataLoader(object):
    """多数据加载器，用于处理多个数据源 - 保留用于向后兼容"""
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError("n_batches should be > 0")
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders[di])
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches


# 辅助函数：过滤无效的数据批次
def filter_none_collate_fn(batch):
    """过滤掉None值的批处理函数"""
    valid_batch = [(g, l) for g, l in batch if g is not None]
    if len(valid_batch) == 0:
        return None, None
    return heterograph_collate_func(valid_batch)


def get_drugban3d_dataloader(data_dir, df=None, batch_size=32, shuffle=True, 
                            num_workers=4, is_test=False, dis_threshold=5.0):
    """获取DrugBAN3D数据加载器"""
    dataset = DrugBAN3DDataset(
        data_dir=data_dir,
        df=df,
        is_test=is_test,
        dis_threshold=dis_threshold
    )
    
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=filter_none_collate_fn,
        drop_last=False
    )
