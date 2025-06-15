"""
多模态DrugBAN模型 - 融合3D结构信息和1D/2D序列信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from drugban_3d import DrugBAN3D
from models import DrugBAN
import dgl
import numpy as np


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, dim_3d, dim_1d2d, num_heads=8, dropout=0.1):
        super(CrossModalAttention, self).__init__()

        # 自动调整注意力头数，确保维度可以被头数整除
        def find_valid_heads(dim, max_heads=16):
            for heads in range(min(max_heads, dim), 0, -1):
                if dim % heads == 0:
                    return heads
            return 1  # 最坏情况下使用1个头

        self.num_heads_3d = find_valid_heads(dim_3d, num_heads)
        self.num_heads_1d2d = find_valid_heads(dim_1d2d, num_heads)
        self.dim_3d = dim_3d
        self.dim_1d2d = dim_1d2d

        print(f"跨模态注意力配置: 3D维度={dim_3d}, 头数={self.num_heads_3d}; 1D/2D维度={dim_1d2d}, 头数={self.num_heads_1d2d}")

        self.head_dim_3d = dim_3d // self.num_heads_3d
        self.head_dim_1d2d = dim_1d2d // self.num_heads_1d2d

        # 3D到1D/2D的注意力
        self.q_3d = nn.Linear(dim_3d, dim_3d)
        self.k_1d2d = nn.Linear(dim_1d2d, dim_3d)  # 投影到3D维度
        self.v_1d2d = nn.Linear(dim_1d2d, dim_3d)

        # 1D/2D到3D的注意力
        self.q_1d2d = nn.Linear(dim_1d2d, dim_1d2d)
        self.k_3d = nn.Linear(dim_3d, dim_1d2d)  # 投影到1D/2D维度
        self.v_3d = nn.Linear(dim_3d, dim_1d2d)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_3d = nn.LayerNorm(dim_3d)
        self.layer_norm_1d2d = nn.LayerNorm(dim_1d2d)
        
    def forward(self, feat_3d, feat_1d2d):
        """
        前向传播
        
        参数:
        feat_3d: 3D特征 [batch_size, dim_3d]
        feat_1d2d: 1D/2D特征 [batch_size, dim_1d2d]
        
        返回:
        enhanced_3d: 增强的3D特征
        enhanced_1d2d: 增强的1D/2D特征
        """
        batch_size = feat_3d.size(0)
        
        # 3D特征通过1D/2D特征增强
        q_3d = self.q_3d(feat_3d).view(batch_size, self.num_heads_3d, self.head_dim_3d)
        k_1d2d = self.k_1d2d(feat_1d2d).view(batch_size, self.num_heads_3d, self.head_dim_3d)
        v_1d2d = self.v_1d2d(feat_1d2d).view(batch_size, self.num_heads_3d, self.head_dim_3d)

        # 计算注意力权重
        attn_3d = torch.matmul(q_3d, k_1d2d.transpose(-2, -1)) / np.sqrt(self.head_dim_3d)
        attn_3d = F.softmax(attn_3d, dim=-1)
        attn_3d = self.dropout(attn_3d)

        # 应用注意力
        enhanced_3d = torch.matmul(attn_3d, v_1d2d).view(batch_size, self.dim_3d)
        enhanced_3d = self.layer_norm_3d(feat_3d + enhanced_3d)

        # 1D/2D特征通过3D特征增强
        q_1d2d = self.q_1d2d(feat_1d2d).view(batch_size, self.num_heads_1d2d, self.head_dim_1d2d)
        k_3d = self.k_3d(feat_3d).view(batch_size, self.num_heads_1d2d, self.head_dim_1d2d)
        v_3d = self.v_3d(feat_3d).view(batch_size, self.num_heads_1d2d, self.head_dim_1d2d)

        # 计算注意力权重
        attn_1d2d = torch.matmul(q_1d2d, k_3d.transpose(-2, -1)) / np.sqrt(self.head_dim_1d2d)
        attn_1d2d = F.softmax(attn_1d2d, dim=-1)
        attn_1d2d = self.dropout(attn_1d2d)

        # 应用注意力
        enhanced_1d2d = torch.matmul(attn_1d2d, v_3d).view(batch_size, self.dim_1d2d)
        enhanced_1d2d = self.layer_norm_1d2d(feat_1d2d + enhanced_1d2d)
        
        return enhanced_3d, enhanced_1d2d


class ModalityFusion(nn.Module):
    """模态融合模块"""
    
    def __init__(self, dim_3d, dim_1d2d, fusion_dim, fusion_type="concat", dropout=0.2, weight_net_hidden=256):
        super(ModalityFusion, self).__init__()
        self.fusion_type = fusion_type
        self.dropout = nn.Dropout(dropout)
        self.weight_net_hidden = weight_net_hidden
        
        if fusion_type == "concat":
            # 简单拼接
            self.fusion_layer = nn.Linear(dim_3d + dim_1d2d, fusion_dim)
        elif fusion_type == "weighted":
            # 动态权重融合 - 让模型自动学习最优权重分配
            # 使用注意力机制动态计算权重
            self.weight_net = nn.Sequential(
                nn.Linear(dim_3d + dim_1d2d, self.weight_net_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.weight_net_hidden, 2),  # 输出两个权重
                nn.Softmax(dim=-1)
            )
            # 将不同维度的特征投影到统一的融合维度
            self.proj_3d = nn.Linear(dim_3d, fusion_dim)
            self.proj_1d2d = nn.Linear(dim_1d2d, fusion_dim)
        elif fusion_type == "gated":
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(dim_3d + dim_1d2d, fusion_dim),
                nn.Sigmoid()
            )
            self.fusion_layer = nn.Linear(dim_3d + dim_1d2d, fusion_dim)
        
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
    def forward(self, feat_3d, feat_1d2d):
        """
        融合两种模态的特征
        
        参数:
        feat_3d: 3D特征
        feat_1d2d: 1D/2D特征
        
        返回:
        fused_feat: 融合后的特征
        """
        if self.fusion_type == "concat":
            # 简单拼接
            combined = torch.cat([feat_3d, feat_1d2d], dim=-1)
            fused = self.fusion_layer(combined)
        elif self.fusion_type == "weighted":
            # 动态权重融合
            # 将特征投影到统一维度
            feat_3d_proj = self.proj_3d(feat_3d)
            feat_1d2d_proj = self.proj_1d2d(feat_1d2d)

            # 动态计算权重
            combined_for_weight = torch.cat([feat_3d, feat_1d2d], dim=-1)
            weights = self.weight_net(combined_for_weight)  # [batch_size, 2]

            # 应用动态权重进行融合
            fused = weights[:, 0:1] * feat_3d_proj + weights[:, 1:2] * feat_1d2d_proj
        elif self.fusion_type == "gated":
            # 门控融合
            combined = torch.cat([feat_3d, feat_1d2d], dim=-1)
            gate = self.gate(combined)
            fused = gate * self.fusion_layer(combined)
        
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused


class DrugBANMultimodal(nn.Module):
    """多模态DrugBAN模型"""
    
    def __init__(self, **config):
        super(DrugBANMultimodal, self).__init__()
        self.config = config
        
        # 初始化3D分支
        self.drugban_3d = DrugBAN3D(**config)
        
        # 初始化1D/2D分支（需要适配原DrugBAN的配置）
        drugban_1d2d_config = self._adapt_config_for_1d2d(config)
        self.drugban_1d2d = DrugBAN(**drugban_1d2d_config)
        
        # 多模态融合配置
        multimodal_config = config.get('MULTIMODAL', {})
        self.fusion_type = multimodal_config.get('FUSION_TYPE', 'hierarchical')
        self.cross_attention_heads = multimodal_config.get('CROSS_ATTENTION_HEADS', 8)
        self.fusion_dropout = multimodal_config.get('FUSION_DROPOUT', 0.2)

        # 延迟初始化跨模态注意力和融合模块
        # 将在第一次前向传播时根据实际特征维度初始化
        self.cross_attention = None
        self.modality_fusion = None
        self.classifier = None

        # 保存配置用于延迟初始化
        self.fusion_dim = config.get('DECODER', {}).get('HIDDEN_DIM', 512)
        self.weight_net_hidden = multimodal_config.get('WEIGHT_NET_HIDDEN_DIM', 256)
        
        # 分类器将在动态初始化时创建
        # 这里不初始化，因为我们需要等待实际特征维度
        
    def _adapt_config_for_1d2d(self, config):
        """适配1D/2D DrugBAN的配置"""
        # 创建适配的配置字典
        adapted_config = {}

        # 复制基本配置，但使用DRUG_1D2D替换DRUG
        for key in ['PROTEIN', 'DECODER', 'BCN', 'GNN']:
            if key in config:
                adapted_config[key] = config[key]

        # 使用DRUG_1D2D配置替换DRUG配置
        if 'DRUG_1D2D' in config:
            adapted_config['DRUG'] = config['DRUG_1D2D']
        else:
            # 如果没有DRUG_1D2D配置，使用默认的1D/2D配置
            adapted_config['DRUG'] = {
                'NODE_IN_FEATS': 75,  # CanonicalAtomFeaturizer的特征维度
                'NODE_IN_EMBEDDING': 128,
                'PADDING': True,
                'HIDDEN_LAYERS': [128, 128, 128]  # 原始DrugBAN配置
            }

        # 添加必要的配置项
        adapted_config['SOLVER'] = config.get('SOLVER', {})
        adapted_config['TRAIN'] = config.get('TRAIN', {})

        return adapted_config
        
    def _initialize_modules(self, feat_3d, feat_1d2d):
        """根据实际特征维度初始化模块"""
        dim_3d = feat_3d.shape[-1]
        dim_1d2d = feat_1d2d.shape[-1]

        print(f"动态初始化多模态模块: 3D特征维度={dim_3d}, 1D/2D特征维度={dim_1d2d}")

        # 初始化跨模态注意力
        self.cross_attention = CrossModalAttention(
            dim_3d=dim_3d,
            dim_1d2d=dim_1d2d,
            num_heads=self.cross_attention_heads,
            dropout=self.fusion_dropout
        ).to(feat_3d.device)

        # 初始化模态融合
        self.modality_fusion = ModalityFusion(
            dim_3d=dim_3d,
            dim_1d2d=dim_1d2d,
            fusion_dim=self.fusion_dim,
            fusion_type="weighted",
            dropout=self.fusion_dropout,
            weight_net_hidden=self.weight_net_hidden
        ).to(feat_3d.device)

        # 初始化分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.fusion_dropout),
            nn.Linear(self.fusion_dim // 2, 1),
            nn.Sigmoid()
        ).to(feat_3d.device)

    def forward(self, graph_3d, data_1d2d=None):
        """
        前向传播

        参数:
        graph_3d: 3D图数据
        data_1d2d: 1D/2D序列数据字典，包含mol_graph和protein_seq

        返回:
        v_d, v_p, f, score: 与其他DrugBAN模型一致的返回格式
        """
        # 3D分支特征提取 - 从DrugBAN3D的forward方法中提取特征
        v_d_3d, v_p_3d, feat_3d, _ = self.drugban_3d(graph_3d)

        # 1D/2D分支特征提取 - 使用原DrugBAN
        mol_graph = data_1d2d['mol_graph']
        protein_seq = data_1d2d['protein_seq']

        # 调用原DrugBAN进行特征提取
        v_d_1d2d, v_p_1d2d, feat_1d2d, _ = self.drugban_1d2d(mol_graph, protein_seq)

        # 如果模块未初始化，则根据实际特征维度初始化
        if self.cross_attention is None:
            self._initialize_modules(feat_3d, feat_1d2d)

        # 跨模态注意力增强
        enhanced_3d, enhanced_1d2d = self.cross_attention(feat_3d, feat_1d2d)

        # 模态融合
        fused_features = self.modality_fusion(enhanced_3d, enhanced_1d2d)

        # 最终预测
        prediction = self.classifier(fused_features)

        # 返回与其他DrugBAN模型一致的格式
        # v_d: 药物特征, v_p: 蛋白质特征, f: 融合特征, score: 预测分数
        return v_d_3d, v_p_3d, fused_features, prediction

    def get_dynamic_weights(self, graph_3d, data_1d2d):
        """获取当前样本的动态权重（用于分析）"""
        with torch.no_grad():
            # 3D分支特征提取
            _, _, feat_3d, _ = self.drugban_3d(graph_3d, mode="train")

            # 1D/2D分支特征提取
            mol_graph = data_1d2d['mol_graph']
            protein_seq = data_1d2d['protein_seq']
            _, _, feat_1d2d, _ = self.drugban_1d2d(mol_graph, protein_seq, mode="train")

            # 跨模态注意力增强
            enhanced_3d, enhanced_1d2d = self.cross_attention(feat_3d, feat_1d2d)

            # 获取动态权重
            if self.modality_fusion.fusion_type == "weighted":
                combined_for_weight = torch.cat([enhanced_3d, enhanced_1d2d], dim=-1)
                weights = self.modality_fusion.weight_net(combined_for_weight)
                return weights  # [batch_size, 2] - [3D权重, 1D/2D权重]
            else:
                return None
    
    def extract_features(self, graph_3d, data_1d2d):
        """提取融合后的特征（用于分析）"""
        # 3D分支特征提取
        v_d_3d, v_p_3d, feat_3d, _ = self.drugban_3d(graph_3d, mode="train")

        # 1D/2D分支特征提取
        mol_graph = data_1d2d['mol_graph']
        protein_seq = data_1d2d['protein_seq']
        v_d_1d2d, v_p_1d2d, feat_1d2d, _ = self.drugban_1d2d(mol_graph, protein_seq, mode="train")

        # 跨模态注意力增强
        enhanced_3d, enhanced_1d2d = self.cross_attention(feat_3d, feat_1d2d)

        # 模态融合
        fused_features = self.modality_fusion(enhanced_3d, enhanced_1d2d)

        return fused_features
