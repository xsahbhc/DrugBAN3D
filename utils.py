import os
import random
import numpy as np
import torch
import dgl
import logging
from scipy.spatial import distance_matrix

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25

# 3D几何相关的辅助函数
def cal_dist(a, b, ord=2):
    """计算两点之间的距离"""
    return np.linalg.norm(a - b, ord=ord)

def area_triangle(v1, v2):
    """计算由两个向量形成的三角形面积"""
    return 0.5 * np.linalg.norm(np.cross(v1, v2))

def angle(v1, v2):
    """计算两个向量之间的角度"""
    return np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))

def set_seed(seed=1000):
    """设置随机种子以确保结果可复现"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def graph_collate_func(x):
    """原始DrugBAN的图批处理函数 - 保留用于向后兼容"""
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)


def heterograph_collate_func(data_batch):
    """用于异构图的批处理函数"""
    graphs, labels = map(list, zip(*data_batch))
    # 过滤掉None值
    valid_pairs = [(g, l) for g, l in zip(graphs, labels) if g is not None]
    if len(valid_pairs) == 0:
        return None, None
    
    # 分离图和标签
    valid_graphs, valid_labels = map(list, zip(*valid_pairs))
    
    # 批处理异构图
    batched_graph = dgl.batch(valid_graphs)
    
    # 将标签转换为张量
    labels_tensor = torch.tensor(valid_labels, dtype=torch.float32)
    
    return batched_graph, labels_tensor


def mkdir(path):
    """创建目录，如果目录不存在"""
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


def integer_label_protein(sequence, max_length=1200):
    """氨基酸序列的整数编码"""
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]
        except KeyError:
            logging.warning(f"字符 {letter} 不存在于序列类别编码中，跳过并视为填充。")
    return encoding
