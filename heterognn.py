import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import edge_softmax
import math

class HeteroGraphConv(nn.Module):
    """异构图卷积层，处理不同类型的边"""
    
    def __init__(self, mods, aggregate='sum'):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        
        # 设置聚合函数
        if aggregate == 'sum':
            self.agg_fn = self._sum_agg
        elif aggregate == 'mean':
            self.agg_fn = self._mean_agg
        elif aggregate == 'max':
            self.agg_fn = self._max_agg
        else:
            raise ValueError(f"不支持的聚合方法: {aggregate}")
    
    def _sum_agg(self, tensors):
        """求和聚合"""
        return torch.stack(tensors, dim=0).sum(dim=0)
    
    def _mean_agg(self, tensors):
        """平均聚合"""
        return torch.stack(tensors, dim=0).mean(dim=0)
    
    def _max_agg(self, tensors):
        """最大值聚合"""
        return torch.stack(tensors, dim=0).max(dim=0)[0]
    
    def forward(self, g, feat_dict, edge_feat_dict=None):
        """
        前向传播
        
        参数:
        g: DGL异构图
        feat_dict: 节点特征字典，键为节点类型，值为特征张量
        edge_feat_dict: 边特征字典，键为边类型，值为特征张量
        
        返回:
        节点特征字典，键为节点类型，值为更新后的特征张量
        """
        outputs = {ntype: [] for ntype in g.dsttypes}
        
        # 对每种类型的边应用相应的卷积
        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            
            if rel_graph.number_of_edges() == 0:
                continue
            
            if stype not in feat_dict or dtype not in feat_dict:
                continue
            
            # 获取边特征
            edge_feat = None
            if edge_feat_dict is not None and etype in edge_feat_dict:
                edge_feat = edge_feat_dict[etype]
            elif hasattr(rel_graph, 'edata') and 'e' in rel_graph.edata:
                edge_feat = rel_graph.edata['e']
            
            # 应用对应的卷积
            dstdata = self.mods[etype](
                rel_graph,
                (feat_dict[stype], feat_dict[dtype]),
                edge_feat
            )
            
            outputs[dtype].append(dstdata)
        
        # 聚合各类型边的结果
        results = {}
        for ntype, ntype_outputs in outputs.items():
            if len(ntype_outputs) > 0:
                results[ntype] = self.agg_fn(ntype_outputs)
            else:
                # 如果没有更新，保持原特征
                results[ntype] = feat_dict[ntype]
        
        return results

class CIGConv(nn.Module):
    """分子内部相互作用卷积（Chemical Interaction Graph Convolution）"""
    
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(CIGConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = 17  # 6(基础特征) + 11(几何特征)
        
        # 特征变换
        self.linear_src = nn.Linear(in_dim, out_dim)
        self.linear_dst = nn.Linear(in_dim, out_dim)
        self.linear_edge = nn.Linear(self.edge_dim, out_dim)
        
        # 处理高维边特征的适配器
        self.edge_adapter = nn.Sequential(
            nn.Linear(256, 64),   # 先降维到64
            nn.ReLU(),
            nn.Linear(64, self.edge_dim)  # 再降维到17
        )
        
        # 激活与规范化
        self.activation = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def edge_attention(self, edges):
        """边注意力机制"""
        # 获取边特征
        edge_feat = edges.data['e']
        src_feat = self.linear_src(edges.src['h'])
        dst_feat = self.linear_dst(edges.dst['h'])
        
        # 处理边特征
        try:
            if edge_feat.shape[1] == self.edge_dim:
                # 正常情况：边特征维度匹配预期
                edge_feat_processed = edge_feat
            elif edge_feat.shape[1] == 256:
                # 边特征维度为256：使用adapter网络降维
                edge_feat_processed = self.edge_adapter(edge_feat)
            elif edge_feat.shape[1] > self.edge_dim:
                # 边特征维度大于预期但不是256：截断到正确维度
                edge_feat_processed = edge_feat[:, :self.edge_dim]
            else:
                # 边特征维度小于预期：填充到正确维度
                padded = torch.zeros((edge_feat.shape[0], self.edge_dim), device=edge_feat.device)
                padded[:, :edge_feat.shape[1]] = edge_feat
                edge_feat_processed = padded
                
            # 投影到隐藏维度
            edge_proj = self.linear_edge(edge_feat_processed)
            
        except Exception as e:
            # 处理任何可能的错误
            print(f"CIGConv处理边特征时出错: {str(e)}，使用零向量替代")
            edge_proj = torch.zeros_like(src_feat)
        
        # 结合源节点、目标节点和边特征
        z = src_feat + dst_feat + edge_proj
        z = self.activation(z)
        return {'z': z}
    
    def forward(self, g, node_feats, edge_feats=None):
        """
        前向传播
        
        参数:
        g: DGL图或边类型图
        node_feats: 源节点和目标节点特征的元组(src_feat, dst_feat)
        edge_feats: 边特征
        
        返回:
        更新后的节点特征
        """
        with g.local_scope():
            src_feat, dst_feat = node_feats
            g.srcdata['h'] = src_feat
            g.dstdata['h'] = dst_feat
            
            # 设置边特征
            if edge_feats is not None:
                g.edata['e'] = edge_feats
            
            # 应用边注意力
            g.apply_edges(self.edge_attention)
            
            # 消息传递与聚合
            g.update_all(
                message_func=lambda edges: {'m': edges.data['z']},
                reduce_func=lambda nodes: {'h_new': torch.sum(nodes.mailbox['m'], dim=1)}
            )
            
            # 获取更新后的特征
            h_new = g.dstdata['h_new']
            
            # 应用批归一化、激活和dropout
            h_new = self.bn(h_new)
            h_new = self.activation(h_new)
            h_new = self.dropout(h_new)
            
            return h_new

class NIGConv(nn.Module):
    """分子间相互作用卷积（Non-covalent Interaction Graph Convolution）"""
    
    def __init__(self, in_dim, out_dim, feat_drop=0.1):
        super(NIGConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # 特征变换
        self.linear_src = nn.Linear(in_dim, out_dim)
        self.linear_dst = nn.Linear(in_dim, out_dim)
        
        # 边特征投影，根据构建方式，分子间边的特征维度为11
        self.edge_dim = 11  # 分子间相互作用边的几何特征维度
        self.linear_edge = nn.Linear(self.edge_dim, out_dim)
        
        # 处理可能存在的高维边特征
        self.edge_adapter = nn.Sequential(
            nn.Linear(256, 64),   # 先降维到64
            nn.ReLU(),
            nn.Linear(64, self.edge_dim)  # 再降维到11
        )
        
        # 注意力
        self.attn_l = nn.Linear(out_dim * 3, 1)
        
        # 激活与规范化
        self.activation = nn.PReLU()
        self.bn = nn.BatchNorm1d(out_dim)
        self.feat_drop = nn.Dropout(feat_drop)
    
    def edge_attention(self, edges):
        """计算边注意力权重"""
        # 获取边特征
        edge_feat = edges.data['e']
        
        # 投影源节点、目标节点
        src_feat = self.linear_src(edges.src['h'])
        dst_feat = self.linear_dst(edges.dst['h'])
        
        # 处理边特征
        try:
            if edge_feat.shape[1] == self.edge_dim:
                # 正常情况：边特征维度匹配预期
                edge_feat_processed = edge_feat
            elif edge_feat.shape[1] == 256:
                # 边特征维度为256：使用adapter网络降维
                edge_feat_processed = self.edge_adapter(edge_feat)
            elif edge_feat.shape[1] > self.edge_dim:
                # 边特征维度大于预期但不是256：截断到正确维度
                edge_feat_processed = edge_feat[:, :self.edge_dim]
            else:
                # 边特征维度小于预期：填充到正确维度
                padded = torch.zeros((edge_feat.shape[0], self.edge_dim), device=edge_feat.device)
                padded[:, :edge_feat.shape[1]] = edge_feat
                edge_feat_processed = padded
                
            # 投影到隐藏维度
            edge_proj = self.linear_edge(edge_feat_processed)
            
        except Exception as e:
            # 处理任何可能的错误
            print(f"处理边特征时出错: {str(e)}，使用零向量替代")
            edge_proj = torch.zeros_like(src_feat)
        
        # 结合源节点、目标节点和边特征
        z2 = torch.cat([src_feat, dst_feat, edge_proj], dim=1)
        a = self.attn_l(z2)
        return {'a': a, 'z': z2}
    
    def forward(self, g, node_feats, edge_feats=None):
        """
        前向传播
        
        参数:
        g: DGL图或边类型图
        node_feats: 源节点和目标节点特征的元组(src_feat, dst_feat)
        edge_feats: 边特征
        
        返回:
        更新后的节点特征
        """
        with g.local_scope():
            src_feat, dst_feat = node_feats
            g.srcdata['h'] = src_feat
            g.dstdata['h'] = dst_feat
            
            # 设置边特征
            if edge_feats is not None:
                g.edata['e'] = edge_feats
            
            # 应用边注意力
            g.apply_edges(self.edge_attention)
            
            # 计算注意力权重
            attn_weights = edge_softmax(g, g.edata['a'])
            g.edata['a'] = attn_weights
            
            # 消息传递与聚合
            g.update_all(
                message_func=lambda edges: {'m': edges.data['a'] * self.linear_src(edges.src['h'])},
                reduce_func=lambda nodes: {'h_new': torch.sum(nodes.mailbox['m'], dim=1)}
            )
            
            # 获取更新后的特征
            h_new = g.dstdata['h_new']
            
            # 应用批归一化、激活和dropout
            h_new = self.bn(h_new)
            h_new = self.activation(h_new)
            h_new = self.feat_drop(h_new)
            
            return h_new

class HeteroGNN(nn.Module):
    """异构图神经网络，处理不同类型的节点和边"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers=3, dropout=0.2):
        super(HeteroGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 输入特征转换
        self.node_projs = nn.ModuleDict({
            ntype: nn.Linear(dim, hidden_dim)
            for ntype, dim in node_feat_dim.items()
        })
        
        # 创建多层异构图卷积
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 第一层使用原始edge_feat_dim
            # 后续层使用统一的hidden_dim作为边特征维度
            current_edge_dim = edge_feat_dim if i == 0 else {k: hidden_dim for k in edge_feat_dim.keys()}
            
            # 为每种类型的边创建对应的卷积层
            layer_dict = {}
            for etype in edge_feat_dim.keys():
                if etype.startswith('intra'):
                    # 分子内部边使用CIGConv
                    layer_dict[etype] = CIGConv(hidden_dim, hidden_dim, dropout=dropout)
                else:
                    # 分子间边使用NIGConv
                    layer_dict[etype] = NIGConv(hidden_dim, hidden_dim, feat_drop=dropout)
            
            # 创建异构图卷积层
            self.layers.append(HeteroGraphConv(layer_dict, aggregate='sum'))
        
        # 添加层归一化
        self.layer_norms = nn.ModuleDict({
            ntype: nn.LayerNorm(hidden_dim)
            for ntype in node_feat_dim.keys()
        })
        
        # 批归一化和激活函数
        self.batch_norms = nn.ModuleDict({
            ntype: nn.BatchNorm1d(hidden_dim)
            for ntype in node_feat_dim.keys()
        })
        
        self.dropouts = nn.ModuleDict({
            ntype: nn.Dropout(dropout)
            for ntype in node_feat_dim.keys()
        })
        
        self.activation = nn.PReLU()
    
    def forward(self, g):
        """
        前向传播
        
        参数:
        g: DGL异构图
        
        返回:
        节点特征字典，键为节点类型，值为更新后的特征张量
        """
        # 获取初始节点特征
        h_dict = {}
        for ntype in g.ntypes:
            h_dict[ntype] = g.nodes[ntype].data.get('h', None)
            
            # 如果特征不存在，初始化为全零向量
            if h_dict[ntype] is None:
                print(f"警告: 节点类型 {ntype} 没有特征，使用零向量代替")
                num_nodes = g.number_of_nodes(ntype)
                h_dict[ntype] = torch.zeros((num_nodes, 1), device=g.device)
        
        # 投影到统一的隐藏维度
        for ntype, proj in self.node_projs.items():
            if ntype in h_dict and h_dict[ntype] is not None:
                h_dict[ntype] = proj(h_dict[ntype])
        
        # 获取边特征
        edge_feat_dict = {}
        for stype, etype, dtype in g.canonical_etypes:
            if g.number_of_edges((stype, etype, dtype)) > 0:
                edge_feat = g.edges[stype, etype, dtype].data.get('e', None)
                if edge_feat is not None:
                    edge_feat_dict[etype] = edge_feat
        
        # 应用多层异构图卷积
        h_final = {k: v for k, v in h_dict.items()}  # 创建副本用于残差连接
        
        for i, layer in enumerate(self.layers):
            # 应用图卷积层
            h_dict = layer(g, h_dict, edge_feat_dict)
            
            # 应用残差连接、层归一化、批归一化和dropout
            for ntype in h_dict.keys():
                if ntype in h_final:
                    # 残差连接
                    h_dict[ntype] = h_dict[ntype] + h_final[ntype]
                
                # 层归一化
                if ntype in self.layer_norms:
                    h_dict[ntype] = self.layer_norms[ntype](h_dict[ntype])
                
                # 激活函数
                h_dict[ntype] = self.activation(h_dict[ntype])
                
                # 批归一化
                if ntype in self.batch_norms:
                    h_dict[ntype] = self.batch_norms[ntype](h_dict[ntype])
                
                # Dropout
                if ntype in self.dropouts:
                    h_dict[ntype] = self.dropouts[ntype](h_dict[ntype])
            
            # 更新残差连接
            h_final = {k: v for k, v in h_dict.items()}
        
        return h_dict

class AtomAtomAffinities(nn.Module):
    """计算原子-原子亲和力"""
    
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim):
        super(AtomAtomAffinities, self).__init__()
        
        self.prj_l2p_src = nn.Linear(node_feat_dim, hidden_dim)
        self.prj_l2p_dst = nn.Linear(node_feat_dim, hidden_dim)
        self.prj_l2p_edge = nn.Linear(edge_feat_dim, hidden_dim)
        
        self.prj_p2l_src = nn.Linear(node_feat_dim, hidden_dim)
        self.prj_p2l_dst = nn.Linear(node_feat_dim, hidden_dim)
        self.prj_p2l_edge = nn.Linear(edge_feat_dim, hidden_dim)
        
        self.fc_l2p = nn.Linear(hidden_dim, 1)
        self.fc_p2l = nn.Linear(hidden_dim, 1)
    
    def apply_interactions(self, edges):
        """计算边的相互作用特征"""
        return {'i': edges.data['e'] * edges.src['h'] * edges.dst['h']}
    
    def forward(self, g, h_dict):
        """
        前向传播
        
        参数:
        g: DGL异构图
        h_dict: 节点特征字典
        
        返回:
        配体到蛋白质和蛋白质到配体的亲和力得分
        """
        with g.local_scope():
            # 获取节点特征
            node_ligand_feats = h_dict['ligand']
            node_pocket_feats = h_dict['pocket']
            
            # 计算配体到蛋白质的亲和力
            if g.number_of_edges(('ligand', 'inter_l2p', 'pocket')) > 0:
                edge_l2p_feat = g.edges['inter_l2p'].data['e']
                
                g.nodes['ligand'].data['h'] = self.prj_l2p_src(node_ligand_feats)
                g.nodes['pocket'].data['h'] = self.prj_l2p_dst(node_pocket_feats)
                g.edges['inter_l2p'].data['e'] = self.prj_l2p_edge(edge_l2p_feat)
                
                g.apply_edges(self.apply_interactions, etype='inter_l2p')
                logit_l2p = self.fc_l2p(g.edges['inter_l2p'].data['i'])
                g.edges['inter_l2p'].data['logit_l2p'] = logit_l2p
                logit_l2p = dgl.sum_edges(g, 'logit_l2p', etype='inter_l2p')
            else:
                # 如果没有边，返回零张量
                logit_l2p = torch.zeros(1, 1, device=node_ligand_feats.device)
            
            # 计算蛋白质到配体的亲和力
            if g.number_of_edges(('pocket', 'inter_p2l', 'ligand')) > 0:
                edge_p2l_feat = g.edges['inter_p2l'].data['e']
                
                g.nodes['ligand'].data['h'] = self.prj_p2l_dst(node_ligand_feats)
                g.nodes['pocket'].data['h'] = self.prj_p2l_src(node_pocket_feats)
                g.edges['inter_p2l'].data['e'] = self.prj_p2l_edge(edge_p2l_feat)
                
                g.apply_edges(self.apply_interactions, etype='inter_p2l')
                logit_p2l = self.fc_p2l(g.edges['inter_p2l'].data['i'])
                g.edges['inter_p2l'].data['logit_p2l'] = logit_p2l
                logit_p2l = dgl.sum_edges(g, 'logit_p2l', etype='inter_p2l')
            else:
                # 如果没有边，返回零张量
                logit_p2l = torch.zeros(1, 1, device=node_ligand_feats.device)
            
            return logit_l2p, logit_p2l 