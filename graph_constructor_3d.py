import os
import numpy as np
import pickle
import torch
from scipy.spatial import distance_matrix
import networkx as nx
import dgl
from rdkit import Chem
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# 从EHIGN移植的工具函数
def cal_dist(a, b, ord=2):
    """计算两点间距离"""
    return np.linalg.norm(a - b, ord=ord)

def area_triangle(v1, v2):
    """计算三角形面积"""
    return 0.5 * np.linalg.norm(np.cross(v1, v2))

def angle(v1, v2):
    """计算两个向量之间的角度"""
    return np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))

def one_of_k_encoding(k, possible_values):
    """One-hot编码"""
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]

def one_of_k_encoding_unk(x, allowable_set):
    """带有未知类别的One-hot编码"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):
    """提取分子中原子的特征"""
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
                    Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # 处理显式氢原子
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def edge_features(mol, graph):
    """提取分子中化学键的特征及几何特征"""
    geom = mol.GetConformers()[0].GetPositions()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
            k = neighbor.GetIdx() 
            if mol.GetBondBetweenAtoms(j, k) is not None and i != k:
                vector1 = geom[j] - geom[i]
                vector2 = geom[k] - geom[i]

                angles_ijk.append(angle(vector1, vector2))
                areas_ijk.append(area_triangle(vector1, vector2))
                dists_ik.append(cal_dist(geom[i], geom[k]))

        angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
        areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
        dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
        dist_ij1 = cal_dist(geom[i], geom[j], ord=1)
        dist_ij2 = cal_dist(geom[i], geom[j], ord=2)

        # 几何特征 - 11维
        geom_feats = [
            angles_ijk.max()*0.1,
            angles_ijk.sum()*0.01,
            angles_ijk.mean()*0.1,
            areas_ijk.max()*0.1,
            areas_ijk.sum()*0.01,
            areas_ijk.mean()*0.1,
            dists_ik.max()*0.1,
            dists_ik.sum()*0.01,
            dists_ik.mean()*0.1,
            dist_ij1*0.1,
            dist_ij2*0.1,
        ]

        # 化学键特征
        bond_type = bond.GetBondType()
        basic_feats = [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ]

        graph.add_edge(i, j, feats=torch.tensor(basic_feats+geom_feats).float())

def mol2graph(mol):
    """将分子转换为图数据结构"""
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_features(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph.edges(data=True)]).T
    edge_attr = torch.stack([feats['feats'] for u, v, feats in graph.edges(data=True)])

    return x, edge_index, edge_attr

def geom_feat(pos_i, pos_j, pos_k, angles_ijk, areas_ijk, dists_ik):
    """计算几何特征"""
    vector1 = pos_j - pos_i
    vector2 = pos_k - pos_i
    angles_ijk.append(angle(vector1, vector2))
    areas_ijk.append(area_triangle(vector1, vector2))
    dists_ik.append(cal_dist(pos_i, pos_k))

def geom_feats(pos_i, pos_j, angles_ijk, areas_ijk, dists_ik):
    """计算完整的几何特征集"""
    angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
    areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
    dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
    dist_ij1 = cal_dist(pos_i, pos_j, ord=1)
    dist_ij2 = cal_dist(pos_i, pos_j, ord=2)
    
    # 11维几何特征
    geom = [
        angles_ijk.max()*0.1,
        angles_ijk.sum()*0.01,
        angles_ijk.mean()*0.1,
        areas_ijk.max()*0.1,
        areas_ijk.sum()*0.01,
        areas_ijk.mean()*0.1,
        dists_ik.max()*0.1,
        dists_ik.sum()*0.01,
        dists_ik.mean()*0.1,
        dist_ij1*0.1,
        dist_ij2*0.1,
    ]

    return geom

def inter_graph(ligand, pocket, dis_threshold=5.0):
    """构建分子间相互作用图"""
    graph_l2p = nx.DiGraph()
    graph_p2l = nx.DiGraph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    
    # 计算所有原子间距离矩阵
    dis_matrix = distance_matrix(pos_l, pos_p)
    # 找出距离小于阈值的原子对
    node_idx = np.where(dis_matrix < dis_threshold)
    
    # 构建配体-蛋白质相互作用边
    for i, j in zip(node_idx[0], node_idx[1]):
        # 配体->蛋白质边
        ks = node_idx[0][node_idx[1] == j]
        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for k in ks:
            if k != i:
                geom_feat(pos_l[i], pos_p[j], pos_l[k], angles_ijk, areas_ijk, dists_ik)
        geom = geom_feats(pos_l[i], pos_p[j], angles_ijk, areas_ijk, dists_ik)
        bond_feats = torch.FloatTensor(geom)
        graph_l2p.add_edge(i, j, feats=bond_feats)
        
        # 蛋白质->配体边
        ks = node_idx[1][node_idx[0] == i]
        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for k in ks:
            if k != j:
                geom_feat(pos_p[j], pos_l[i], pos_p[k], angles_ijk, areas_ijk, dists_ik)     
        geom = geom_feats(pos_p[j], pos_l[i], angles_ijk, areas_ijk, dists_ik)
        bond_feats = torch.FloatTensor(geom)
        graph_p2l.add_edge(j, i, feats=bond_feats)
    
    # 转换为张量格式
    edge_index_l2p = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph_l2p.edges(data=True)]).T if graph_l2p.edges() else torch.zeros((2, 0), dtype=torch.long)
    edge_attr_l2p = torch.stack([feats['feats'] for u, v, feats in graph_l2p.edges(data=True)]) if graph_l2p.edges() else torch.zeros((0, 11), dtype=torch.float)

    edge_index_p2l = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph_p2l.edges(data=True)]).T if graph_p2l.edges() else torch.zeros((2, 0), dtype=torch.long)
    edge_attr_p2l = torch.stack([feats['feats'] for u, v, feats in graph_p2l.edges(data=True)]) if graph_p2l.edges() else torch.zeros((0, 11), dtype=torch.float)

    return (edge_index_l2p, edge_attr_l2p), (edge_index_p2l, edge_attr_p2l)

def construct_heterograph(ligand, pocket, dis_threshold=5.0):
    """构建异构图"""
    # 获取配体分子图
    x_l, edge_index_l, edge_attr_l = mol2graph(ligand)
    # 获取蛋白质口袋分子图
    x_p, edge_index_p, edge_attr_p = mol2graph(pocket)
    # 获取分子间相互作用图
    (edge_index_l2p, edge_attr_l2p), (edge_index_p2l, edge_attr_p2l) = inter_graph(ligand, pocket, dis_threshold)
    
    # 构建DGL异构图
    graph_data = {
        ('ligand', 'intra_l', 'ligand'): (edge_index_l[0], edge_index_l[1]),
        ('pocket', 'intra_p', 'pocket'): (edge_index_p[0], edge_index_p[1]),
        ('ligand', 'inter_l2p', 'pocket'): (edge_index_l2p[0], edge_index_l2p[1]),
        ('pocket', 'inter_p2l', 'ligand'): (edge_index_p2l[0], edge_index_p2l[1])
    }
    
    # 创建异构图
    g = dgl.heterograph(graph_data)
    
    # 设置节点特征
    g.nodes['ligand'].data['h'] = x_l
    g.nodes['pocket'].data['h'] = x_p
    
    # 设置边特征
    g.edges['intra_l'].data['e'] = edge_attr_l
    g.edges['intra_p'].data['e'] = edge_attr_p
    g.edges['inter_l2p'].data['e'] = edge_attr_l2p
    g.edges['inter_p2l'].data['e'] = edge_attr_p2l
    
    return g

def load_molecule_from_file(ligand_file, pocket_file):
    """从文件加载分子，返回RDKit分子对象"""
    # 根据文件扩展名选择适当的加载方式
    ligand_ext = os.path.splitext(ligand_file)[1].lower()
    pocket_ext = os.path.splitext(pocket_file)[1].lower()
    
    # 配体文件载入
    ligand = None
    if ligand_ext == '.sdf':
        # SDF文件载入
        try:
            supplier = Chem.SDMolSupplier(ligand_file, removeHs=False, sanitize=True)
            ligand = supplier[0]
        except:
            return None, None
    elif ligand_ext == '.mol2':
        # MOL2文件载入
        try:
            ligand = Chem.MolFromMol2File(ligand_file, removeHs=False, sanitize=True)
        except:
            return None, None
    elif ligand_ext == '.pdb':
        # PDB文件载入
        try:
            ligand = Chem.MolFromPDBFile(ligand_file, removeHs=False, sanitize=True)
        except:
            return None, None
    
    # 口袋文件载入（通常是PDB格式）
    pocket = None
    if pocket_ext == '.pdb':
        try:
            pocket = Chem.MolFromPDBFile(pocket_file, removeHs=False, sanitize=False)
        except:
            return None, None
    
    # 验证是否成功载入
    if ligand is None or pocket is None:
        return None, None
    
    # 验证分子是否有坐标
    if ligand.GetNumConformers() == 0 or pocket.GetNumConformers() == 0:
        return None, None
    
    return ligand, pocket

def create_heterograph_from_files(ligand_file, pocket_file, dis_threshold=5.0):
    """从文件创建异构图"""
    ligand, pocket = load_molecule_from_file(ligand_file, pocket_file)
    if ligand is None or pocket is None:
        return None
    
    return construct_heterograph(ligand, pocket, dis_threshold)

def create_heterograph_from_complex(complex_path, dis_threshold=5.0):
    """从复合物文件创建异构图"""
    ligand_file = os.path.join(complex_path, "ligand.sdf")
    pocket_file = os.path.join(complex_path, "pocket.pdb")
    return create_heterograph_from_files(ligand_file, pocket_file, dis_threshold)

def collate_fn(data_batch):
    """批处理数据的折叠函数，处理None值"""
    # 过滤None值
    data_batch = list(filter(lambda x: x is not None, data_batch))
    if len(data_batch) == 0:
        return None
    
    # 正常的批处理折叠
    graphs, labels = map(list, zip(*data_batch))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels) 