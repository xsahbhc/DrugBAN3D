import torch
import torch.nn as nn
from dgl.nn.pytorch import edge_softmax
import dgl

class BiasCorrectionLigandPocket(nn.Module):
    """配体到蛋白质的偏置校正模块"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(BiasCorrectionLigandPocket, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        
        # 参照EHIGN_PLA项目的实现方式
        # 将节点特征投影到隐藏层维度
        self.prj_src = nn.Linear(node_feat_dim, hidden_dim)
        self.prj_dst = nn.Linear(node_feat_dim, hidden_dim)
        self.prj_edge = nn.Linear(edge_feat_dim, hidden_dim)
        
        # 投影权重层
        self.w_src = nn.Linear(node_feat_dim, hidden_dim)
        self.w_dst = nn.Linear(node_feat_dim, hidden_dim)
        self.w_edge = nn.Linear(edge_feat_dim, hidden_dim)
        
        # 注意力权重层
        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出层 - 使用全连接网络
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def get_weight(self, edges):
        """计算边的注意力权重"""
        w = edges.src['h'] + edges.dst['h'] + edges.data['e']
        w = self.lin_att(w)
        return {'w': w}
    
    def apply_scores(self, edges):
        """应用注意力分数"""
        return {'l': edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        """
        前向传播
        
        参数:
        g: DGL异构图
        
        返回:
        偏置校正值
        """
        with g.local_scope():
            # 获取节点特征
            node_ligand_feats = g.nodes['ligand'].data.get('h')
            node_pocket_feats = g.nodes['pocket'].data.get('h')
            
            # 获取设备
            device = next(self.prj_src.parameters()).device
                
            # 检查特征是否存在
            if node_ligand_feats is None or node_pocket_feats is None:
                return torch.zeros(1, device=device)
            
            if g.number_of_edges(('ligand', 'inter_l2p', 'pocket')) == 0:
                return torch.zeros(1, device=device)
            
            # 获取边特征
            edge_feat = g.edges['inter_l2p'].data.get('e')
            if edge_feat is None:
                # 如果边特征不存在，尝试创建一个全为0的张量
                num_edges = g.number_of_edges(('ligand', 'inter_l2p', 'pocket'))
                edge_feat = torch.zeros(num_edges, self.edge_feat_dim, device=device)
            
            # 计算注意力权重
            g.nodes['ligand'].data['h'] = self.prj_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.prj_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.prj_edge(edge_feat)
            g.apply_edges(self.get_weight, etype='inter_l2p')
            scores = edge_softmax(g['inter_l2p'], g.edges['inter_l2p'].data['w'])
            
            # 应用注意力分数
            g.edges['inter_l2p'].data['a'] = scores
            g.nodes['ligand'].data['h'] = self.w_src(node_ligand_feats)
            g.nodes['pocket'].data['h'] = self.w_dst(node_pocket_feats)
            g.edges['inter_l2p'].data['e'] = self.w_edge(edge_feat)
            g.apply_edges(self.apply_scores, etype='inter_l2p')
            
            # 汇总边特征
            bias = self.fc(dgl.sum_edges(g, 'l', etype='inter_l2p'))
            
            return bias

class BiasCorrectionPocketLigand(nn.Module):
    """蛋白质到配体的偏置校正模块"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(BiasCorrectionPocketLigand, self).__init__()
        self.hidden_dim = hidden_dim
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        
        # 参照EHIGN_PLA项目的实现方式
        # 将节点特征投影到隐藏层维度
        self.prj_src = nn.Linear(node_feat_dim, hidden_dim)
        self.prj_dst = nn.Linear(node_feat_dim, hidden_dim)
        self.prj_edge = nn.Linear(edge_feat_dim, hidden_dim)
        
        # 投影权重层
        self.w_src = nn.Linear(node_feat_dim, hidden_dim)
        self.w_dst = nn.Linear(node_feat_dim, hidden_dim)
        self.w_edge = nn.Linear(edge_feat_dim, hidden_dim)
        
        # 注意力权重层
        self.lin_att = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出层 - 使用全连接网络
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def get_weight(self, edges):
        """计算边的注意力权重"""
        w = edges.src['h'] + edges.dst['h'] + edges.data['e']
        w = self.lin_att(w)
        return {'w': w}
    
    def apply_scores(self, edges):
        """应用注意力分数"""
        return {'l': edges.data['a'] * edges.data['e'] * edges.src['h'] * edges.dst['h']}

    def forward(self, g):
        """
        前向传播
        
        参数:
        g: DGL异构图
        
        返回:
        偏置校正值
        """
        with g.local_scope():
            # 获取节点特征
            node_ligand_feats = g.nodes['ligand'].data.get('h')
            node_pocket_feats = g.nodes['pocket'].data.get('h')
            
            # 获取设备
            device = next(self.prj_src.parameters()).device
                
            # 检查特征是否存在
            if node_ligand_feats is None or node_pocket_feats is None:
                return torch.zeros(1, device=device)
            
            if g.number_of_edges(('pocket', 'inter_p2l', 'ligand')) == 0:
                return torch.zeros(1, device=device)
            
            # 获取边特征
            edge_feat = g.edges['inter_p2l'].data.get('e')
            if edge_feat is None:
                # 如果边特征不存在，尝试创建一个全为0的张量
                num_edges = g.number_of_edges(('pocket', 'inter_p2l', 'ligand'))
                edge_feat = torch.zeros(num_edges, self.edge_feat_dim, device=device)
            
            # 计算注意力权重
            g.nodes['pocket'].data['h'] = self.prj_src(node_pocket_feats)
            g.nodes['ligand'].data['h'] = self.prj_dst(node_ligand_feats)
            g.edges['inter_p2l'].data['e'] = self.prj_edge(edge_feat)
            g.apply_edges(self.get_weight, etype='inter_p2l')
            scores = edge_softmax(g['inter_p2l'], g.edges['inter_p2l'].data['w'])
            
            # 应用注意力分数
            g.edges['inter_p2l'].data['a'] = scores
            g.nodes['pocket'].data['h'] = self.w_src(node_pocket_feats)
            g.nodes['ligand'].data['h'] = self.w_dst(node_ligand_feats)
            g.edges['inter_p2l'].data['e'] = self.w_edge(edge_feat)
            g.apply_edges(self.apply_scores, etype='inter_p2l')
            
            # 汇总边特征
            bias = self.fc(dgl.sum_edges(g, 'l', etype='inter_p2l'))
            
            return bias 