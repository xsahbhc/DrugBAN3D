import torch
import torch.nn as nn
import torch.nn.functional as F
from heterognn import HeteroGNN, AtomAtomAffinities
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import dgl
from bias_correction import BiasCorrectionLigandPocket, BiasCorrectionPocketLigand
import math


class Spatial3DFeatureFusion(nn.Module):
    """
    3D空间感知的特征融合模块
    结合空间距离信息进行更好的特征融合
    """
    def __init__(self, ligand_dim, pocket_dim, hidden_dim, dropout=0.1):
        super(Spatial3DFeatureFusion, self).__init__()
        self.ligand_dim = ligand_dim
        self.pocket_dim = pocket_dim
        self.hidden_dim = hidden_dim

        # 特征投影层
        self.ligand_proj = nn.Linear(ligand_dim, hidden_dim)
        self.pocket_proj = nn.Linear(pocket_dim, hidden_dim)

        # 空间距离编码器
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )

        # 门控融合机制
        self.gate_ligand = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.gate_pocket = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 层归一化
        self.ln_ligand = nn.LayerNorm(hidden_dim)
        self.ln_pocket = nn.LayerNorm(hidden_dim)

        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, ligand_features, pocket_features, bg):
        """
        Args:
            ligand_features: [num_ligand_nodes, ligand_dim]
            pocket_features: [num_pocket_nodes, pocket_dim]
            bg: 异构图，包含空间坐标信息
        """
        # 1. 特征投影到统一维度
        ligand_proj = self.ligand_proj(ligand_features)  # [num_ligand, hidden_dim]
        pocket_proj = self.pocket_proj(pocket_features)  # [num_pocket, hidden_dim]

        # 2. 提取3D空间坐标
        try:
            ligand_coords = bg.nodes['ligand'].data.get('coords', None)
            pocket_coords = bg.nodes['pocket'].data.get('coords', None)

            if ligand_coords is not None and pocket_coords is not None:
                # 计算配体-口袋原子间的距离矩阵
                distances = self._compute_distance_matrix(ligand_coords, pocket_coords)

                # 3. 距离编码
                distance_encoding = self.distance_encoder(distances.unsqueeze(-1))  # [num_ligand, num_pocket, hidden_dim]

                # 4. 空间感知的特征增强
                ligand_enhanced = self._spatial_enhance_features(
                    ligand_proj, pocket_proj, distance_encoding, mode='ligand'
                )
                pocket_enhanced = self._spatial_enhance_features(
                    pocket_proj, ligand_proj, distance_encoding.transpose(0, 1), mode='pocket'
                )
            else:
                # 如果没有坐标信息，使用原始特征
                ligand_enhanced = ligand_proj
                pocket_enhanced = pocket_proj

        except Exception as e:
            print(f"空间特征融合失败，使用原始特征: {e}")
            ligand_enhanced = ligand_proj
            pocket_enhanced = pocket_proj

        # 5. 交叉注意力机制
        ligand_attended, _ = self.cross_attention(
            ligand_enhanced.unsqueeze(0),  # [1, num_ligand, hidden_dim]
            pocket_enhanced.unsqueeze(0),  # [1, num_pocket, hidden_dim]
            pocket_enhanced.unsqueeze(0)   # [1, num_pocket, hidden_dim]
        )
        ligand_attended = ligand_attended.squeeze(0)  # [num_ligand, hidden_dim]

        pocket_attended, _ = self.cross_attention(
            pocket_enhanced.unsqueeze(0),  # [1, num_pocket, hidden_dim]
            ligand_enhanced.unsqueeze(0),  # [1, num_ligand, hidden_dim]
            ligand_enhanced.unsqueeze(0)   # [1, num_ligand, hidden_dim]
        )
        pocket_attended = pocket_attended.squeeze(0)  # [num_pocket, hidden_dim]

        # 6. 残差连接和层归一化
        ligand_fused = self.ln_ligand(ligand_enhanced + ligand_attended)
        pocket_fused = self.ln_pocket(pocket_enhanced + pocket_attended)

        return ligand_fused, pocket_fused

    def _compute_distance_matrix(self, coords1, coords2):
        """计算两组坐标间的距离矩阵"""
        # coords1: [num1, 3], coords2: [num2, 3]
        # 返回: [num1, num2]
        diff = coords1.unsqueeze(1) - coords2.unsqueeze(0)  # [num1, num2, 3]
        distances = torch.norm(diff, dim=-1)  # [num1, num2]
        return distances

    def _spatial_enhance_features(self, query_features, key_features, distance_encoding, mode='ligand'):
        """基于空间距离增强特征"""
        # query_features: [num_query, hidden_dim]
        # key_features: [num_key, hidden_dim]
        # distance_encoding: [num_query, num_key, hidden_dim]

        # 计算空间权重
        spatial_weights = torch.softmax(-distance_encoding.mean(dim=-1), dim=-1)  # [num_query, num_key]

        # 加权聚合邻居特征
        aggregated_features = torch.bmm(
            spatial_weights.unsqueeze(1),  # [num_query, 1, num_key]
            key_features.unsqueeze(0).expand(spatial_weights.size(0), -1, -1)  # [num_query, num_key, hidden_dim]
        ).squeeze(1)  # [num_query, hidden_dim]

        # 门控融合
        combined = torch.cat([query_features, aggregated_features], dim=-1)  # [num_query, hidden_dim*2]

        if mode == 'ligand':
            gate = self.gate_ligand(combined)
        else:
            gate = self.gate_pocket(combined)

        enhanced_features = gate * query_features + (1 - gate) * aggregated_features

        return enhanced_features


class MultiScale3DExtractor(nn.Module):
    """
    多尺度3D特征提取器
    在不同的空间尺度上提取相互作用特征
    """
    def __init__(self, hidden_dim, scales=[2.0, 5.0, 8.0], dropout=0.1):
        super(MultiScale3DExtractor, self).__init__()
        self.scales = scales
        self.hidden_dim = hidden_dim

        # 为每个尺度创建特征提取器
        self.scale_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in scales
        ])

        # 尺度融合层
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 注意力权重
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim * len(scales), len(scales)),
            nn.Softmax(dim=-1)
        )

    def forward(self, features, bg, node_type):
        """
        Args:
            features: [num_nodes, hidden_dim]
            bg: 异构图
            node_type: 'ligand' 或 'pocket'
        """
        try:
            coords = bg.nodes[node_type].data.get('coords', None)
            if coords is None:
                # 如果没有坐标信息，直接返回原特征
                return features

            scale_features = []

            # 在不同尺度上提取特征
            for i, scale in enumerate(self.scales):
                # 基于距离阈值构建邻接矩阵
                distances = torch.cdist(coords, coords)  # [num_nodes, num_nodes]
                adjacency = (distances <= scale).float()

                # 邻域聚合
                neighbor_features = torch.mm(adjacency, features) / (adjacency.sum(dim=-1, keepdim=True) + 1e-8)

                # 特征变换
                scale_feature = self.scale_extractors[i](neighbor_features)
                scale_features.append(scale_feature)

            # 拼接所有尺度的特征
            concatenated = torch.cat(scale_features, dim=-1)  # [num_nodes, hidden_dim * num_scales]

            # 计算注意力权重
            attention_weights = self.scale_attention(concatenated)  # [num_nodes, num_scales]

            # 加权融合
            weighted_features = torch.stack(scale_features, dim=-1)  # [num_nodes, hidden_dim, num_scales]
            fused_features = torch.sum(
                weighted_features * attention_weights.unsqueeze(1),
                dim=-1
            )  # [num_nodes, hidden_dim]

            # 残差连接
            output = features + fused_features

            return output

        except Exception as e:
            print(f"多尺度特征提取失败，使用原始特征: {e}")
            return features


class DrugBAN3D(nn.Module):
    """
    DrugBAN 3D版本，整合了异构图神经网络和双线性注意力机制
    """
    def __init__(self, **config):
        super(DrugBAN3D, self).__init__()
        # 配置参数
        ligand_node_dim = config['DRUG']['NODE_IN_FEATS']
        pocket_node_dim = config['PROTEIN']['NODE_IN_FEATS']
        edge_dim = config['EDGE_FEATS']
        gnn_hidden_dim = config['GNN']['HIDDEN_DIM']
        gnn_layers = config['GNN']['NUM_LAYERS']
        mlp_in_dim = config['DECODER']['IN_DIM']
        mlp_hidden_dim = config['DECODER']['HIDDEN_DIM']
        mlp_out_dim = config['DECODER']['OUT_DIM']
        out_binary = config['DECODER']['BINARY']
        ban_heads = config['BCN']['HEADS']
        
        # 获取DECODER中的dropout率，如果未定义则使用默认值0.5
        dropout_rate = config['DECODER'].get('DROPOUT', 0.5)
        
        # 偏差校正配置 - 默认启用
        self.use_bias_correction = config.get('USE_BIAS_CORRECTION', True)
        print(f"\n{'='*50}")
        print(f"DrugBAN3D初始化 - 偏差校正模块: {'启用' if self.use_bias_correction else '禁用'}")
        print(f"Dropout率: {dropout_rate}")  # 打印dropout率
        print(f"GNN层数: {gnn_layers}")  # 打印GNN层数
        print(f"{'='*50}\n")
        
        # 异构图神经网络进行特征提取
        self.hetero_gnn = HeteroGNN(
            node_feat_dim={
                'ligand': ligand_node_dim,
                'pocket': pocket_node_dim
            },
            edge_feat_dim={
                'intra_l': edge_dim['intra_l'],
                'intra_p': edge_dim['intra_p'],
                'inter_l2p': edge_dim['inter_l2p'],
                'inter_p2l': edge_dim['inter_p2l']
            },
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_layers
        )
        
        # 原子-原子亲和力计算
        self.atom_atom_affinity = AtomAtomAffinities(
            node_feat_dim=gnn_hidden_dim,
            edge_feat_dim=edge_dim['inter_l2p'],  # 使用配置中定义的维度
            hidden_dim=gnn_hidden_dim
        )
        
        # 添加偏置校正模块 - 从EHIGN融合
        self.bias_ligandpocket = BiasCorrectionLigandPocket(
            node_feat_dim=gnn_hidden_dim,  # 使用GNN隐藏层维度作为节点特征维度
            edge_feat_dim=edge_dim['inter_l2p'],  # 使用原始边特征维度
            hidden_dim=gnn_hidden_dim  # 使用GNN隐藏层维度作为隐藏层维度
        )
        
        self.bias_pocketligand = BiasCorrectionPocketLigand(
            node_feat_dim=gnn_hidden_dim,  # 使用GNN隐藏层维度作为节点特征维度
            edge_feat_dim=edge_dim['inter_p2l'],  # 使用原始边特征维度
            hidden_dim=gnn_hidden_dim  # 使用GNN隐藏层维度作为隐藏层维度
        )

        # ===== 新增的改进模块 =====
        # 3D空间感知的特征融合模块
        self.spatial_fusion = Spatial3DFeatureFusion(
            ligand_dim=gnn_hidden_dim,
            pocket_dim=gnn_hidden_dim,
            hidden_dim=gnn_hidden_dim,
            dropout=0.1
        )

        # 多尺度3D特征提取器
        self.multi_scale_extractor = MultiScale3DExtractor(
            hidden_dim=gnn_hidden_dim,
            scales=[2.0, 5.0, 8.0],  # 不同的距离阈值(Å)
            dropout=0.1
        )

        # 双线性注意力网络
        self.bcn = weight_norm(
            BANLayer(
                v_dim=gnn_hidden_dim,  # 药物特征维度
                q_dim=gnn_hidden_dim,  # 蛋白质特征维度
                h_dim=mlp_in_dim,      # 注意力隐藏层维度
                h_out=ban_heads        # 注意力头数
            ),
            name='h_mat',
            dim=None
        )
        
        # MLP分类器 - 传入dropout率，增加更多特征维度
        self.mlp_classifier = MLPDecoder(
            mlp_in_dim + 6,  # 增加6个维度：l2p_signal, p2l_signal, spatial_similarity, interaction_strength, distance_stats, contact_density
            mlp_hidden_dim,
            mlp_out_dim,
            binary=out_binary,
            dropout=dropout_rate  # 使用配置中定义的dropout率
        )
        
    def forward(self, bg, mode="train"):
        """
        前向传播
        
        参数:
        bg: DGL异构图批处理
        mode: 运行模式，"train"或"eval"
        
        返回:
        根据mode返回不同的结果
        """
        # 异构图特征提取
        h_dict = self.hetero_gnn(bg)

        # 获取药物和蛋白质节点表示
        v_d_raw = h_dict['ligand']  # 形状: [total_num_ligand_nodes, hidden_dim]
        v_p_raw = h_dict['pocket']  # 形状: [total_num_pocket_nodes, hidden_dim]

        # ===== 应用新的特征融合机制 =====
        try:
            # 1. 3D空间感知特征融合
            v_d_spatial, v_p_spatial = self.spatial_fusion(v_d_raw, v_p_raw, bg)

            # 2. 多尺度3D特征提取
            v_d = self.multi_scale_extractor(v_d_spatial, bg, 'ligand')
            v_p = self.multi_scale_extractor(v_p_spatial, bg, 'pocket')

        except Exception as e:
            print(f"特征融合失败，使用原始特征: {e}")
            v_d = v_d_raw
            v_p = v_p_raw
        
        # 计算原子-原子亲和力
        try:
            affinity_l2p, affinity_p2l = self.atom_atom_affinity(bg, h_dict)
        except Exception as e:
            # 如果计算亲和力失败，使用零张量替代
            device = v_d.device
            affinity_l2p = torch.zeros(1, 1, device=device)
            affinity_p2l = torch.zeros(1, 1, device=device)
            print(f"计算原子-原子亲和力时出错: {str(e)}")
            
        # 计算偏置校正
        device = v_d.device
        bias_l2p = torch.zeros(1, 1, device=device)
        bias_p2l = torch.zeros(1, 1, device=device)
        
        if self.use_bias_correction:
            # 保存原始特征，以便后续恢复
            orig_data = {}
            
            if 'h' in bg.nodes['ligand'].data:
                orig_data['ligand_h'] = bg.nodes['ligand'].data['h']
            if 'h' in bg.nodes['pocket'].data:
                orig_data['pocket_h'] = bg.nodes['pocket'].data['h']
            if 'e' in bg.edges['inter_l2p'].data:
                orig_data['inter_l2p_e'] = bg.edges['inter_l2p'].data['e']
            if 'e' in bg.edges['inter_p2l'].data:
                orig_data['inter_p2l_e'] = bg.edges['inter_p2l'].data['e']
            
            try:
                # 设置GNN输出的节点特征
                bg.nodes['ligand'].data['h'] = h_dict['ligand']
                bg.nodes['pocket'].data['h'] = h_dict['pocket']
                
                # 按照EHIGN_PLA的方式处理：这里不做维度转换
                # 偏置校正模块应该与GNN隐藏层使用相同的维度初始化
                
                # 调用偏置校正模块 - 使用原始边特征
                bias_l2p = self.bias_ligandpocket(bg)
                bias_p2l = self.bias_pocketligand(bg)
            
            finally:
                # 恢复原始特征
                for key, value in orig_data.items():
                    if key == 'ligand_h':
                        bg.nodes['ligand'].data['h'] = value
                    elif key == 'pocket_h':
                        bg.nodes['pocket'].data['h'] = value
                    elif key == 'inter_l2p_e':
                        bg.edges['inter_l2p'].data['e'] = value
                    elif key == 'inter_p2l_e':
                        bg.edges['inter_p2l'].data['e'] = value
        
        # 重塑张量形状以适应BCN要求
        # 需要将节点特征重组为批次形式 [batch_size, num_nodes, hidden_dim]
        try:
            # 对于DGL 0.6.1及以上版本
            batch_num_nodes_l = bg.batch_num_nodes('ligand')  # 每个图中配体节点数量
            batch_num_nodes_p = bg.batch_num_nodes('pocket')  # 每个图中口袋节点数量
        except:
            try:
                # 对于较老版本的DGL
                batch_num_nodes_l = bg.batch_num_nodes(ntype='ligand')  # 每个图中配体节点数量
                batch_num_nodes_p = bg.batch_num_nodes(ntype='pocket')  # 每个图中口袋节点数量
            except Exception as e:
                # 如果获取batch_num_nodes失败，打印错误并使用替代方法
                print(f"获取batch_num_nodes失败: {str(e)}")
                # 尝试通过其他方式估计每个图的节点数量
                batch_size = bg.batch_size
                batch_num_nodes_l = torch.ones(batch_size, dtype=torch.int64, device=v_d.device) * (v_d.shape[0] // batch_size)
                batch_num_nodes_p = torch.ones(batch_size, dtype=torch.int64, device=v_p.device) * (v_p.shape[0] // batch_size)
        
        # 计算批处理大小
        batch_size = len(batch_num_nodes_l)
        
        # 计算每个批次的最大节点数
        max_num_nodes_l = max(batch_num_nodes_l).item() if len(batch_num_nodes_l) > 0 else 1
        max_num_nodes_p = max(batch_num_nodes_p).item() if len(batch_num_nodes_p) > 0 else 1
        
        # 创建填充张量
        batched_v_d = torch.zeros(batch_size, max_num_nodes_l, v_d.shape[1], device=v_d.device)
        batched_v_p = torch.zeros(batch_size, max_num_nodes_p, v_p.shape[1], device=v_p.device)
        
        # 填充张量
        node_offset_l = 0
        node_offset_p = 0
        for i in range(batch_size):
            # 确保不会超出边界
            if i < len(batch_num_nodes_l):
                num_nodes_l = batch_num_nodes_l[i].item()
                if node_offset_l + num_nodes_l <= v_d.shape[0]:  # 确保索引有效
                    batched_v_d[i, :num_nodes_l] = v_d[node_offset_l:node_offset_l+num_nodes_l]
                node_offset_l += num_nodes_l
            
            if i < len(batch_num_nodes_p):
                num_nodes_p = batch_num_nodes_p[i].item()
                if node_offset_p + num_nodes_p <= v_p.shape[0]:  # 确保索引有效
                    batched_v_p[i, :num_nodes_p] = v_p[node_offset_p:node_offset_p+num_nodes_p]
                node_offset_p += num_nodes_p
        
        # 应用双线性注意力
        f, att = self.bcn(batched_v_d, batched_v_p)
        
        # 整合原子-原子亲和力和偏置校正的信息
        l2p_signal = (affinity_l2p - bias_l2p).view(-1, 1)
        p2l_signal = (affinity_p2l - bias_p2l).view(-1, 1)

        # 确保维度匹配
        if l2p_signal.shape[0] == 1 and f.shape[0] > 1:
            l2p_signal = l2p_signal.expand(f.shape[0], -1)
        if p2l_signal.shape[0] == 1 and f.shape[0] > 1:
            p2l_signal = p2l_signal.expand(f.shape[0], -1)

        # ===== 计算额外的3D空间特征 =====
        try:
            # 计算空间相似性特征
            spatial_similarity = self._compute_spatial_similarity(batched_v_d, batched_v_p, bg)

            # 计算相互作用强度
            interaction_strength = self._compute_interaction_strength(batched_v_d, batched_v_p, bg)

            # 计算距离统计特征
            distance_stats = self._compute_distance_statistics(bg)

            # 计算接触密度
            contact_density = self._compute_contact_density(bg)

        except Exception as e:
            print(f"3D特征计算失败，使用零特征: {e}")
            batch_size = f.shape[0]
            device = f.device
            spatial_similarity = torch.zeros(batch_size, 1, device=device)
            interaction_strength = torch.zeros(batch_size, 1, device=device)
            distance_stats = torch.zeros(batch_size, 1, device=device)
            contact_density = torch.zeros(batch_size, 1, device=device)

        # 确保所有特征都在同一设备上
        device = f.device
        l2p_signal = l2p_signal.to(device)
        p2l_signal = p2l_signal.to(device)
        spatial_similarity = spatial_similarity.to(device)
        interaction_strength = interaction_strength.to(device)
        distance_stats = distance_stats.to(device)
        contact_density = contact_density.to(device)

        # 组合所有特征
        combined_f = torch.cat([
            f, l2p_signal, p2l_signal,
            spatial_similarity, interaction_strength,
            distance_stats, contact_density
        ], dim=1)
        
        # 分类预测
        score = self.mlp_classifier(combined_f)
        
        # 根据模式返回不同结果
        if mode == "train":
            return v_d, v_p, combined_f, score
        elif mode == "eval":
            return v_d, v_p, score, att
        elif mode == "detailed":
            # 返回更详细的信息，包括原子-原子亲和力和偏置校正
            return v_d, v_p, combined_f, score, affinity_l2p, affinity_p2l, bias_l2p, bias_p2l

    def _compute_spatial_similarity(self, batched_v_d, batched_v_p, bg):
        """计算配体和口袋的空间相似性特征"""
        try:
            # 计算每个样本的平均特征向量
            ligand_mean = batched_v_d.mean(dim=1)  # [batch_size, hidden_dim]
            pocket_mean = batched_v_p.mean(dim=1)  # [batch_size, hidden_dim]

            # 计算余弦相似度
            similarity = F.cosine_similarity(ligand_mean, pocket_mean, dim=1)  # [batch_size]

            return similarity.unsqueeze(1)  # [batch_size, 1]

        except Exception as e:
            print(f"计算空间相似性失败: {e}")
            batch_size = batched_v_d.shape[0]
            return torch.zeros(batch_size, 1, device=batched_v_d.device)

    def _compute_interaction_strength(self, batched_v_d, batched_v_p, bg):
        """计算相互作用强度"""
        try:
            # 计算配体和口袋特征向量的L2距离
            ligand_mean = batched_v_d.mean(dim=1)  # [batch_size, hidden_dim]
            pocket_mean = batched_v_p.mean(dim=1)  # [batch_size, hidden_dim]

            # 计算欧几里得距离的倒数作为相互作用强度
            distance = torch.norm(ligand_mean - pocket_mean, dim=1)  # [batch_size]
            strength = 1.0 / (1.0 + distance)  # 归一化的相互作用强度

            return strength.unsqueeze(1)  # [batch_size, 1]

        except Exception as e:
            print(f"计算相互作用强度失败: {e}")
            batch_size = batched_v_d.shape[0]
            return torch.zeros(batch_size, 1, device=batched_v_d.device)

    def _compute_distance_statistics(self, bg):
        """计算距离统计特征"""
        try:
            # 尝试获取配体和口袋的坐标
            ligand_coords = bg.nodes['ligand'].data.get('coords', None)
            pocket_coords = bg.nodes['pocket'].data.get('coords', None)

            if ligand_coords is not None and pocket_coords is not None:
                # 确保坐标在正确的设备上
                device = ligand_coords.device

                # 计算配体-口袋原子间的最小距离
                distances = torch.cdist(ligand_coords, pocket_coords)  # [num_ligand, num_pocket]
                min_distance = distances.min()

                # 按批次处理
                try:
                    batch_num_nodes_l = bg.batch_num_nodes('ligand')
                    batch_size = len(batch_num_nodes_l)

                    # 为每个批次计算最小距离
                    batch_min_distances = []
                    node_offset_l = 0
                    node_offset_p = 0

                    for i in range(batch_size):
                        num_nodes_l = batch_num_nodes_l[i].item()
                        num_nodes_p = bg.batch_num_nodes('pocket')[i].item()

                        batch_ligand_coords = ligand_coords[node_offset_l:node_offset_l+num_nodes_l]
                        batch_pocket_coords = pocket_coords[node_offset_p:node_offset_p+num_nodes_p]

                        if batch_ligand_coords.shape[0] > 0 and batch_pocket_coords.shape[0] > 0:
                            batch_distances = torch.cdist(batch_ligand_coords, batch_pocket_coords)
                            batch_min_dist = batch_distances.min().item()
                        else:
                            batch_min_dist = 10.0  # 默认值

                        batch_min_distances.append(batch_min_dist)
                        node_offset_l += num_nodes_l
                        node_offset_p += num_nodes_p

                    distance_stats = torch.tensor(batch_min_distances, device=device, dtype=torch.float32).unsqueeze(1)

                except:
                    # 如果批次处理失败，使用全局最小距离
                    batch_size = bg.batch_size
                    distance_stats = min_distance.expand(batch_size, 1).to(device)

                return distance_stats
            else:
                # 如果没有坐标信息，返回默认值
                batch_size = bg.batch_size
                # 获取正确的设备
                device = next(iter(bg.nodes['ligand'].data.values())).device if bg.nodes['ligand'].data else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.ones(batch_size, 1, device=device, dtype=torch.float32) * 5.0

        except Exception as e:
            print(f"计算距离统计失败: {e}")
            batch_size = bg.batch_size
            # 获取正确的设备
            device = next(iter(bg.nodes['ligand'].data.values())).device if bg.nodes['ligand'].data else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.ones(batch_size, 1, device=device, dtype=torch.float32) * 5.0

    def _compute_contact_density(self, bg):
        """计算接触密度"""
        try:
            # 计算配体-口袋相互作用边的数量
            if 'inter_l2p' in bg.etypes:
                num_interactions = bg.num_edges('inter_l2p')
                num_ligand_nodes = bg.num_nodes('ligand')
                num_pocket_nodes = bg.num_nodes('pocket')

                # 计算接触密度
                max_possible_contacts = num_ligand_nodes * num_pocket_nodes
                contact_density = num_interactions / max(max_possible_contacts, 1)

                # 按批次处理
                try:
                    batch_num_nodes_l = bg.batch_num_nodes('ligand')
                    batch_size = len(batch_num_nodes_l)

                    # 获取正确的设备
                    device = next(iter(bg.nodes['ligand'].data.values())).device if bg.nodes['ligand'].data else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    # 简化处理：使用全局接触密度
                    density_stats = torch.tensor([contact_density] * batch_size,
                                                device=device, dtype=torch.float32).unsqueeze(1)

                except:
                    batch_size = bg.batch_size
                    device = next(iter(bg.nodes['ligand'].data.values())).device if bg.nodes['ligand'].data else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    density_stats = torch.tensor([contact_density] * batch_size, device=device, dtype=torch.float32).unsqueeze(1)

                return density_stats
            else:
                batch_size = bg.batch_size
                device = next(iter(bg.nodes['ligand'].data.values())).device if bg.nodes['ligand'].data else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                return torch.zeros(batch_size, 1, device=device, dtype=torch.float32)

        except Exception as e:
            print(f"计算接触密度失败: {e}")
            batch_size = bg.batch_size
            device = next(iter(bg.nodes['ligand'].data.values())).device if bg.nodes['ligand'].data else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.zeros(batch_size, 1, device=device, dtype=torch.float32)


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout=0.5):
        super(MLPDecoder, self).__init__()
        self.dropout = dropout
        
        # 第一层
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二层 - 带残差连接
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # 第三层 - 带残差连接
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.dropout3 = nn.Dropout(dropout)
        
        # 输出层
        self.fc4 = nn.Linear(out_dim, binary)
        
        # 如果维度不同，需要投影残差连接
        self.has_proj = (hidden_dim != in_dim)
        if self.has_proj:
            self.proj1 = nn.Linear(in_dim, hidden_dim)
        
        self.has_proj2 = (out_dim != hidden_dim)
        if self.has_proj2:
            self.proj2 = nn.Linear(hidden_dim, out_dim)
            
        # 初始化权重，减少过拟合
        self._init_weights()
        
    def _init_weights(self):
        """使用He初始化权重以减少过拟合并提高训练效率"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 第一层
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout1(x)
        
        # 残差连接1
        if self.has_proj:
            identity = self.proj1(identity)
        x = x + identity * 0.1
        
        # 第二层
        identity = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout2(x)
        
        # 残差连接2
        x = x + identity * 0.1
        
        # 第三层
        identity = x
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout3(x)
        
        # 残差连接3
        if self.has_proj2:
            identity = self.proj2(identity)
        x = x + identity * 0.1
        
        # 输出层
        x = self.fc4(x)
        
        return x


def binary_cross_entropy(pred_output, labels, label_smoothing=0.0):
    """
    二元交叉熵损失，支持标签平滑和focal loss
    
    参数:
    - pred_output: 模型预测输出
    - labels: 目标标签
    - label_smoothing: 标签平滑系数，默认为0（不使用标签平滑）
    
    返回:
    - n: 经过sigmoid处理的预测概率
    - loss: 损失值 (标量)
    """
    # 首先校正输入形状
    if pred_output.dim() > 2:
        pred_output = pred_output.view(-1, pred_output.size(-1))
    
    if labels.dim() > 1:
        labels = labels.view(-1)
    
    # 确保标签是浮点型
    if labels.dtype != torch.float:
        labels = labels.float()
    
    # 应用标签平滑
    if label_smoothing > 0:
        # 应用标签平滑: 正例标签变为1-label_smoothing，负例标签变为label_smoothing*0.5
        smooth_labels = labels.clone()
        smooth_labels = smooth_labels * (1.0 - label_smoothing) + (1.0 - smooth_labels) * label_smoothing * 0.5
    else:
        smooth_labels = labels
    
    # 确保logits和标签具有相同的形状
    if pred_output.shape[0] != smooth_labels.shape[0]:
        if pred_output.dim() > smooth_labels.dim():
            smooth_labels = smooth_labels.unsqueeze(-1)
        elif pred_output.dim() < smooth_labels.dim():
            pred_output = pred_output.unsqueeze(-1)
    
    # 计算标准BCELoss
    pred_output = pred_output.squeeze()
    bce_loss = F.binary_cross_entropy_with_logits(
        pred_output, 
        smooth_labels,
        reduction='none'  # 修改为'none'，返回每个样本的损失
    )
    
    # 计算sigmoid激活值用于返回和focal loss计算
    n = torch.sigmoid(pred_output)
    
    # 计算Focal Loss成分，用于处理类别不平衡
    gamma = 1.0  # 降低gamma参数，使焦点损失更温和
    # 置信度因子：预测正确的样本权重较小，预测错误的样本权重较大
    p_t = n * smooth_labels + (1 - n) * (1 - smooth_labels)
    focal_weight = (1 - p_t) ** gamma
    
    # 添加alpha平衡因子，增强对少数类的学习
    alpha = 0.6  # 降低alpha值，避免过度关注正样本
    alpha_weight = alpha * smooth_labels + (1 - alpha) * (1 - smooth_labels)
    
    # 计算每个样本的加权focal loss
    focal_bce_loss = bce_loss * focal_weight * alpha_weight
    
    # 最终损失是两者的加权组合 - 减少焦点损失的权重
    loss = bce_loss * 0.7 + focal_bce_loss * 0.3  # 更偏向于标准BCE损失
    
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss 