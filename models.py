import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from dgllife.model.gnn import GCN
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm
import dgl


def binary_cross_entropy(pred_output, labels, label_smoothing=0.0):
    """带标签平滑的二元交叉熵损失"""
    if label_smoothing > 0:
        # 应用标签平滑
        target = labels * (1 - label_smoothing) + 0.5 * label_smoothing
    else:
        target = labels
        
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    # 获取预测输出的形状
    pred_shape = pred_output.shape
    
    # 将预测输出转换为sigmoid激活的概率值
    n = m(pred_output)
    
    # 确保n和target的形状匹配 - 如果预测是[batch_size, 1]，而标签是[batch_size]，则对标签进行扩展
    if len(pred_shape) > 1 and pred_shape[1] == 1 and len(target.shape) == 1:
        target = target.unsqueeze(1)
    # 如果预测是[batch_size, 1]，而我们需要扁平化的输出，则挤压预测
    elif len(pred_shape) > 1 and pred_shape[1] == 1 and len(target.shape) > 1 and target.shape[1] == 1:
        n = n.squeeze(1)
        target = target.squeeze(1)
    
    loss = loss_fct(n, target)
    
    # 确保返回的n是一维张量，方便后续计算AUC
    if len(n.shape) > 1:
        n = n.squeeze(-1)
    
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    """计算交叉熵损失"""
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # 获取最大对数概率的索引
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    """计算熵损失"""
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class DrugBAN(nn.Module):
    """
    原始DrugBAN模型 - 保留用于向后兼容
    """
    def __init__(self, **config):
        super(DrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]
        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1], h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_extractor(v_p)
        f, att = self.bcn(v_d, v_p)
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class MolecularGCN(nn.Module):
    """药物分子图卷积网络"""
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinCNN(nn.Module):
    """蛋白质序列卷积网络"""
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    """MLP分类器，用于预测结合可能性"""
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout=0.2):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(out_dim, binary)
        
        # 添加残差连接支持
        self.has_proj = (hidden_dim != in_dim)
        if self.has_proj:
            self.proj = nn.Linear(in_dim, hidden_dim)
            
        self.has_proj2 = (out_dim != hidden_dim)
        if self.has_proj2:
            self.proj2 = nn.Linear(hidden_dim, out_dim)
        
        # 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        """使用He初始化权重"""
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
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.dropout1(x)
        
        # 残差连接1
        if self.has_proj:
            identity = self.proj(identity)
        x = x + identity * 0.1
        
        # 第二层
        identity = x
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.dropout2(x)
        
        # 残差连接2
        x = x + identity * 0.1
        
        # 第三层
        identity = x
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.dropout3(x)
        
        # 残差连接3
        if self.has_proj2:
            identity = self.proj2(identity)
        x = x + identity * 0.1
        
        # 输出层
        x = self.fc4(x)
        return x


class SimpleClassifier(nn.Module):
    """简单的分类器模型，用于领域适应"""
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


class RandomLayer(nn.Module):
    """用于领域适应的随机层"""
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        device = input_list[0].device
        return_list = [torch.mm(input_list[i], self.random_matrix[i].to(device)) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self, device = None):
        super(RandomLayer, self).cuda(device)
        self.random_matrix = [val.cuda(device) for val in self.random_matrix]
