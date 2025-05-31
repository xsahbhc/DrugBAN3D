import torch
import torch.nn as nn
import torch.nn.functional as F
from heterognn import HeteroGNN, AtomAtomAffinities
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import dgl
from dgl.nn.pytorch import edge_softmax

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
        
        # MLP分类器
        self.mlp_classifier = MLPDecoder(
            mlp_in_dim,
            mlp_hidden_dim,
            mlp_out_dim,
            binary=out_binary
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
        v_d = h_dict['ligand']  # 形状: [total_num_ligand_nodes, hidden_dim]
        v_p = h_dict['pocket']  # 形状: [total_num_pocket_nodes, hidden_dim]
        
        # 计算原子-原子亲和力（可选）
        try:
            affinity_l2p, affinity_p2l = self.atom_atom_affinity(bg, h_dict)
        except Exception as e:
            # 如果计算亲和力失败，使用零张量替代
            device = v_d.device
            affinity_l2p = torch.zeros(1, 1, device=device)
            affinity_p2l = torch.zeros(1, 1, device=device)
            print(f"计算原子-原子亲和力时出错: {str(e)}")
        
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
        
        # 分类预测
        score = self.mlp_classifier(f)
        
        # 根据模式返回不同结果
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att
        elif mode == "detailed":
            # 返回更详细的信息，包括原子-原子亲和力
            return v_d, v_p, f, score, affinity_l2p, affinity_p2l


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
        reduction='mean'
    )
    
    # 计算sigmoid激活值用于返回和focal loss计算
    n = torch.sigmoid(pred_output)
    
    # 计算Focal Loss (gamma=2)成分，用于处理类别不平衡
    gamma = 2.0
    # 置信度因子：预测正确的样本权重较小，预测错误的样本权重较大
    p_t = n * smooth_labels + (1 - n) * (1 - smooth_labels)
    focal_weight = (1 - p_t) ** gamma
    
    # 计算每个样本的加权focal loss
    # 使用binary_cross_entropy_with_logits的方式计算
    alpha = 0.25  # 类别权重因子
    focal_bce_loss = F.binary_cross_entropy_with_logits(
        pred_output,
        smooth_labels,
        reduction='none'
    )
    weighted_focal_loss = torch.mean(focal_weight * focal_bce_loss)
    
    # 最终损失是两者的加权组合
    loss = bce_loss * 0.7 + weighted_focal_loss * 0.3
    
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