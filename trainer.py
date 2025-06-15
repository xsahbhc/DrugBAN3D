import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import time
import traceback


class Trainer(object):
    def __init__(self, model, train_dataloader, val_dataloader=None, test_dataloader=None, config=None):
        """初始化训练器"""
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"运行设备: {self.device}")
        
        # 将模型移至设备
        self.model = self.model.to(self.device)
        
        # 设置域适应标志
        self.is_da = False
        if config and hasattr(config, 'DA') and hasattr(config.DA, 'USE'):
            self.is_da = config.DA.USE
        
        # 设置超参数
        if config:
            self.max_epoch = config.SOLVER.MAX_EPOCH
            self.batch_size = config.SOLVER.BATCH_SIZE if hasattr(config.SOLVER, 'BATCH_SIZE') else 32
            self.n_class = config.DECODER.BINARY if hasattr(config, 'DECODER') and hasattr(config.DECODER, 'BINARY') else 1
            
            # 学习率和权重衰减
            self.learning_rate = config.TRAIN.G_LEARNING_RATE if hasattr(config.TRAIN, 'G_LEARNING_RATE') else 0.001
            self.weight_decay = config.TRAIN.WEIGHT_DECAY if hasattr(config.TRAIN, 'WEIGHT_DECAY') else 0.0001
            
            # 标签平滑和类别权重
            self.label_smoothing = config.TRAIN.LABEL_SMOOTHING if hasattr(config.TRAIN, 'LABEL_SMOOTHING') else 0.0
            self.pos_weight = torch.tensor(config.TRAIN.POS_WEIGHT).to(self.device) if hasattr(config.TRAIN, 'POS_WEIGHT') else None
            
            # 梯度裁剪
            self.clip_grad_norm = config.TRAIN.GRADIENT_CLIP_NORM if hasattr(config.TRAIN, 'GRADIENT_CLIP_NORM') else 0.0
            
            # 学习率调度器
            self.use_lr_scheduler = config.SOLVER.LR_SCHEDULER if hasattr(config.SOLVER, 'LR_SCHEDULER') else False
            self.lr_scheduler_type = config.SOLVER.LR_SCHEDULER_TYPE if hasattr(config.SOLVER, 'LR_SCHEDULER_TYPE') else 'plateau'
            
            # 早停
            self.use_early_stopping = config.USE_EARLY_STOPPING if hasattr(config, 'USE_EARLY_STOPPING') else False
            self.early_stopping_patience = config.EARLY_STOPPING_PATIENCE if hasattr(config, 'EARLY_STOPPING_PATIENCE') else 10
        else:
            # 默认值
            self.max_epoch = 100
            self.batch_size = 32
            self.n_class = 1
            self.learning_rate = 0.001
            self.weight_decay = 0.0001
            self.label_smoothing = 0.0
            self.pos_weight = None
            self.clip_grad_norm = 0.0
            self.use_lr_scheduler = False
            self.lr_scheduler_type = 'plateau'
            self.use_early_stopping = False
            self.early_stopping_patience = 10
        
        # 创建优化器
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 创建域适应优化器（如果需要）
        self.optim_da = None
        if self.is_da and hasattr(self.model, 'domain_dmm'):
            self.optim_da = torch.optim.Adam(
                self.model.domain_dmm.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # 当前轮次和步数
        self.current_epoch = 0
        self.step = 0
        
        # 记录最佳验证结果
        self.best_val_auroc = 0.0
        self.best_val_auprc = 0.0
        self.best_model_state = None
        
        # 早停计数器
        self.early_stopping_counter = 0

        # 配置损失函数
        self.configure_loss_fn()
        
        # 配置学习率调度器
        if self.use_lr_scheduler:
            self.configure_lr_scheduler()
        else:
            self.scheduler = None

        # 初始化属性
        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]
        
        # 设置实验记录
        self.experiment = None
        
        # 域适应相关参数初始化
        if self.is_da:
            self.da_init_epoch = config["DA"]["INIT_EPOCH"]
            self.init_lamb_da = config["DA"]["LAMB_DA"]
            self.da_method = config["DA"]["METHOD"]
            self.use_da_entropy = config["DA"]["USE_ENTROPY"]
            self.epochs = self.max_epoch
            
            # 初始化域适应组件
            self.domain_dmm = Discriminator(config["ADAPT"]["RANDOM_DIM"] if config["DA"]["RANDOM_LAYER"] else 
                                          config.DECODER.OUT_DIM * self.n_class).to(self.device)
            
            # 初始化随机层
            self.random_layer = None
            if config["DA"]["RANDOM_LAYER"]:
                self.random_layer = RandomLayer([config.DECODER.OUT_DIM, self.n_class], 
                                               config["ADAPT"]["RANDOM_DIM"]).to(self.device)
                
            # 域适应alpha参数
            self.alpha = 0.0  # 初始化为0，后续会根据轮次调整
        else:
            # 域适应默认参数
            self.da_init_epoch = 0
            self.init_lamb_da = 0.1
            self.da_method = None
            self.use_da_entropy = False
            self.domain_dmm = None
            self.random_layer = None
            self.alpha = 0.0
        
        # 是否每轮保存结果
        self.save_each_epoch = config["RESULT"].get("SAVE_EACH_EPOCH", False)
        self.save_best_only = config["RESULT"].get("SAVE_BEST_ONLY", False)

        # 添加verbose属性
        self.verbose = False  # 默认关闭详细输出
            
        # 确保DA配置存在
        if hasattr(config, "DA") and hasattr(config["DA"], "ORIGINAL_RANDOM"):
            self.original_random = config["DA"]["ORIGINAL_RANDOM"]
        else:
            self.original_random = False

    def load_model(self, checkpoint_path):
        """加载模型权重"""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        self.model.load_state_dict(state_dict)
        print(f"从 {checkpoint_path} 加载模型权重")
        
    def get_best_model(self):
        """获取最佳模型"""
        # 创建和当前模型相同架构的模型实例
        best_model = copy.deepcopy(self.model)
        # 加载最佳权重
        best_model.load_state_dict(self.best_model_state)
        return best_model

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (
                non_init_epoch * self.nb_training
        )
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        
        # 检查并创建子目录结构
        logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # 创建训练日志文件
        log_file = os.path.join(logs_dir, "training_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {self.config['MODEL_TYPE']}\n")
            f.write(f"总训练轮数: {self.max_epoch}\n\n")
            f.write("轮次\t训练损失\t验证损失\t验证AUROC\t验证AUPRC\n")
            f.write("-" * 70 + "\n")
        
        for i in range(self.max_epoch):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss,
                                                                                        epoch_lamb, da_loss]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)
            
            # 简化：移除表格添加
            # self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
            
            # 简化：移除表格添加
            # val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            # self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            
            # 更新学习率调度器
            if self.use_lr_scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # 对于ReduceLROnPlateau，传递验证集AUROC作为监控指标
                    self.scheduler.step(auroc)
                    # 记录当前学习率
                    current_lr = self.optim.param_groups[0]['lr']
                    print(f"当前学习率: {current_lr:.6f}")
                else:
                    # 对于其他调度器，正常step
                    self.scheduler.step()
                
            # 检查是否是新的最佳模型
            if auroc >= self.best_val_auroc:
                # 保存模型state_dict而不是整个模型对象
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_val_auroc = auroc
                self.best_val_auprc = auprc
                self.best_epoch = self.current_epoch
                # 保存最佳模型
                if self.save_best_only:
                    # 确保输出目录存在
                    os.makedirs(self.output_dir, exist_ok=True)
                    best_model_path = os.path.join(self.output_dir, f"best_model.pth")
                    torch.save(self.best_model_state, best_model_path)
                    print(f"保存最佳模型（轮次 {self.best_epoch}）到 {best_model_path}")
                # 重置早停计数器
                self.early_stopping_counter = 0
            else:
                # 增加早停计数器
                self.early_stopping_counter += 1
                if self.use_early_stopping and self.early_stopping_counter >= self.early_stopping_patience:
                    print(f"早停触发！连续{self.early_stopping_patience}轮验证集性能未提升")
                    break
                
            # 只输出简洁的训练状态信息
            print(f'Epoch {self.current_epoch}/{self.max_epoch} - 训练损失: {train_loss:.4f} - 验证AUROC: {auroc:.4f} - 验证AUPRC: {auprc:.4f}')
            
            # 简化的训练日志记录
            if not hasattr(self, '_log_header_written'):
                with open(os.path.join(self.output_dir, "train.log"), 'w') as f:
                    f.write("epoch\ttrain_loss\tval_loss\tauroc\tauprc\n")
                self._log_header_written = True

            with open(os.path.join(self.output_dir, "train.log"), 'a') as f:
                f.write(f"{self.current_epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{auroc:.4f}\t{auprc:.4f}\n")
            
            # 检查是否需要保存当前轮次的结果
            if self.save_each_epoch:
                self.save_epoch_result()
        
        # 确保在测试评估前已保存了最佳模型文件
        if self.best_model_state is not None:
            best_model_path = os.path.join(self.output_dir, "best_model.pth")
            if not os.path.exists(best_model_path):
                torch.save(self.best_model_state, best_model_path)
                print(f"训练结束时保存最佳模型（轮次 {self.best_epoch}）到 {best_model_path}")
                
        # 测试评估时使用最佳模型的权重
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(dataloader="test")
        
        # 简洁的测试结果输出
        print(f'训练完成！最佳模型来自Epoch {self.best_epoch} - AUROC: {auroc:.4f} - F1: {f1:.4f}')
        
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_val_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])

        # 简洁的完成信息
        print(f"结果已保存到: {self.output_dir}")
        return self.test_metrics

    def save_epoch_result(self):
        """每个epoch结束后保存验证结果和最佳模型（不保存每轮模型）"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 检查并创建子目录结构
        models_dir = os.path.join(self.output_dir, "models")
        metrics_dir = os.path.join(self.output_dir, "metrics")
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        
        # 创建必要的子目录
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 创建本轮结果目录
        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{self.current_epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        # 保存验证指标为JSON格式
        metrics = {
            "epoch": self.current_epoch,
            "train_loss": self.train_loss_epoch[-1] if len(self.train_loss_epoch) > 0 else None,
            "val_loss": self.val_loss_epoch[-1] if len(self.val_loss_epoch) > 0 else None,
            "val_auroc": self.val_auroc_epoch[-1] if len(self.val_auroc_epoch) > 0 else None,
            "is_best": False
        }
        
        # 将指标保存为JSON文件
        import json
        with open(os.path.join(epoch_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        # 只在验证性能提升时保存模型
        if self.val_auroc_epoch[-1] > self.best_val_auroc:
            self.best_val_auroc = self.val_auroc_epoch[-1]
            self.best_val_auprc = self.val_auprc_epoch[-1]  # 修复此处的变量错误
            self.best_epoch = self.current_epoch
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            # 更新最佳指标标志
            metrics["is_best"] = True
            with open(os.path.join(epoch_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        
            # 保存最佳模型
            if self.config["RESULT"]["SAVE_MODEL"]:
                best_model_path = os.path.join(models_dir, "best_model.pth")
                
                if hasattr(self, 'use_state_dict') and self.use_state_dict:
                    torch.save({
                        'state_dict': self.best_model_state,
                        'epoch': self.best_epoch,
                        'auroc': self.best_val_auroc,
                        'auprc': self.best_val_auprc
                    }, best_model_path)
                else:
                    torch.save(self.best_model_state, best_model_path)
                
                print(f"保存最佳模型（轮次 {self.best_epoch}）到 {best_model_path}")
                
                # 同时在checkpoint目录中保存一份
                checkpoint_best_path = os.path.join(epoch_dir, f"best_model_epoch_{self.best_epoch}.pth")
                torch.save(self.best_model_state, checkpoint_best_path)
                
            # 重置早停计数器
            self.early_stopping_counter = 0
        else:
            # 增加早停计数器
            self.early_stopping_counter += 1

    def save_result(self):
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 使用时间戳创建唯一标识
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if self.config["RESULT"]["SAVE_MODEL"]:
            # 确保最佳模型状态已定义
            if not hasattr(self, 'best_model_state') or self.best_model_state is None:
                self.best_model_state = self.model.state_dict()

            # 只保存一个最佳模型文件到主目录
            best_model_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(self.best_model_state, best_model_path)

        # 保存配置文件备份
        config_path = os.path.join(self.output_dir, "config.yaml")
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(dict(self.config), f, default_flow_style=False)

        # 创建简洁的结果摘要文件
        results_file = os.path.join(self.output_dir, "results.txt")
        best_epoch = self.best_epoch if hasattr(self, 'best_epoch') else 0
        with open(results_file, 'w') as f:
            f.write(f"训练完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"最佳轮次: {best_epoch}\n")
            f.write(f"AUROC: {self.test_metrics['auroc']:.4f}\n")
            f.write(f"AUPRC: {self.test_metrics['auprc']:.4f}\n")
            f.write(f"F1分数: {self.test_metrics['F1']:.4f}\n")
            f.write(f"敏感度: {self.test_metrics['sensitivity']:.4f}\n")
            f.write(f"特异度: {self.test_metrics['specificity']:.4f}\n")
            f.write(f"准确率: {self.test_metrics['accuracy']:.4f}\n")
            f.write(f"精确度: {self.test_metrics['Precision']:.4f}\n")

        # 保存详细的训练日志（如果需要调试）
        if hasattr(self, 'save_detailed_logs') and self.save_detailed_logs:
            train_log_path = os.path.join(self.output_dir, "train.log")
            state = {
                "train_epoch_loss": self.train_loss_epoch,
                "val_epoch_loss": self.val_loss_epoch,
                "test_metrics": self.test_metrics,
                "best_epoch": best_epoch,
            }
            torch.save(state, train_log_path)

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        losses = []
        steps = len(self.train_dataloader)
        
        # 设置简单的进度条，只显示进度百分比
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}/{self.max_epoch}", ncols=100)
        
        # 增加标签统计
        pos_count = 0
        neg_count = 0
        # 记录没有标签的样本数
        none_labels = 0
        
        for i, batch in enumerate(pbar):
            # 检查batch是否为None（数据加载错误）
            if batch is None:
                print("警告: 遇到空批次，跳过")
                continue
                
            try:
                # 处理不同类型的批处理数据
                if len(batch) == 3:  # 多模态数据格式: (bg_3d, data_1d2d, labels) 或原始DrugBAN格式: (v_d, v_p, labels)
                    if hasattr(self.model, 'drugban_3d') and hasattr(self.model, 'drugban_1d2d'):
                        # 多模态DrugBAN数据格式
                        bg_3d, data_1d2d, labels = batch

                        # 检查是否有空数据
                        if bg_3d is None or data_1d2d is None or labels is None:
                            none_labels += 1
                            continue

                        # 将数据移到设备上
                        bg_3d = bg_3d.to(self.device)
                        labels = labels.float().to(self.device)

                        # 处理1D/2D数据，将其移到设备上
                        mol_graph = data_1d2d['mol_graph'].to(self.device)  # 确保DGL图移动到正确设备
                        protein_seq = data_1d2d['protein_seq'].to(self.device)

                        data_1d2d_device = {
                            'mol_graph': mol_graph,
                            'protein_seq': protein_seq
                        }

                        # 前向传播（多模态）
                        v_d, v_p, f, score = self.model(bg_3d, data_1d2d_device)
                    else:
                        # 原始DrugBAN数据格式
                        v_d, v_p, labels = batch
                        v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                        v_d, v_p, f, score = self.model(v_d, v_p)

                elif len(batch) == 2:  # DrugBAN3D数据格式: (bg, labels)
                    bg, labels = batch

                    # 检查是否有空标签
                    if labels is None:
                        none_labels += 1
                        continue

                    # 将图和标签移到设备上
                    bg = bg.to(self.device)
                    labels = labels.float().to(self.device)

                    # 前向传播
                    v_d, v_p, f, score = self.model(bg)
                else:
                    print(f"警告: 批次 {i} 的元素数量 ({len(batch)}) 不符合预期，跳过")
                    continue

                # 每100个批次打印一次标签统计信息
                if i % 100 == 0:
                    batch_pos = (labels > 0).sum().item()
                    batch_neg = (labels <= 0).sum().item()
                    pos_count += batch_pos
                    neg_count += batch_neg
                    print(f"批次 {i}: 正样本 {batch_pos}, 负样本 {batch_neg}")

                # 确保标签形状正确
                if len(labels.shape) == 0:
                    labels = labels.unsqueeze(0)

                # 清除梯度
                self.optim.zero_grad()

                # 计算损失
                # 调整预测值形状，确保与标签形状匹配
                score = score.squeeze()  # 将形状从[batch_size, 1]变为[batch_size]
                loss = self.loss_fn(score, labels)

                # 反向传播
                loss.backward()

                # 梯度裁剪（如果启用）
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                # 优化器步进
                self.optim.step()

                # 记录损失
                losses.append(loss.item())
                
            except Exception as e:
                print(f"训练批次处理出错: {str(e)}")
                traceback.print_exc()
        
        # 打印标签统计信息
        if pos_count + neg_count > 0:
            pos_ratio = pos_count / (pos_count + neg_count) * 100
            print(f"本轮训练: 正样本 {pos_count} ({pos_ratio:.2f}%), 负样本 {neg_count} ({100-pos_ratio:.2f}%)")
        
        if none_labels > 0:
            print(f"警告: 训练中有 {none_labels} 个批次的标签为空")
        
        # 返回平均损失
        if losses:
            return sum(losses) / len(losses)
        else:
            print("警告: 本轮训练中没有计算任何损失")
            return float('inf')

    def train_da_epoch(self):
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0
        if self.current_epoch >= self.da_init_epoch:
            # epoch_lamb_da = self.da_lambda_decay()
            epoch_lamb_da = 1
            if self.experiment:
                self.experiment.log_metric("DA loss lambda", epoch_lamb_da, epoch=self.current_epoch)
        num_batches = len(self.train_dataloader)
        
        # 添加进度条
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}/{self.max_epoch}", ncols=100)
        
        for i, (batch_s, batch_t) in enumerate(pbar):
            self.step += 1
            v_d, v_p, labels = batch_s[0].to(self.device), batch_s[1].to(self.device), batch_s[2].float().to(
                self.device)
            v_d_t, v_p_t = batch_t[0].to(self.device), batch_t[1].to(self.device)
            self.optim.zero_grad()
            self.optim_da.zero_grad()
            v_d, v_p, f, score = self.model(v_d, v_p)
            if self.n_class == 1:
                n, model_loss = binary_cross_entropy(score, labels, self.label_smoothing)
            else:
                n, model_loss = cross_entropy_logits(score, labels)
            if self.current_epoch >= self.da_init_epoch:
                v_d_t, v_p_t, f_t, t_score = self.model(v_d_t, v_p_t)
                if self.da_method == "CDAN":
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    softmax_output = torch.nn.Softmax(dim=1)(score)
                    softmax_output = softmax_output.detach()
                    # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
                    if self.original_random:
                        self.random_layer = self.random_layer.to(v_d.device)
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_src_score = self.domain_dmm(random_out)
                        else:
                            adv_output_src_score = self.domain_dmm(feature)

                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                    softmax_output_t = softmax_output_t.detach()
                    # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
                    if self.original_random:
                        random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    else:
                        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_t = self.random_layer.forward(feature_t)
                            adv_output_tgt_score = self.domain_dmm(random_out_t)
                        else:
                            adv_output_tgt_score = self.domain_dmm(feature_t)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score)
                        entropy_tgt = self._compute_entropy_weights(t_score)
                        src_weight = entropy_src / torch.sum(entropy_src)
                        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        src_weight = None
                        tgt_weight = None

                    n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(self.batch_size).to(self.device),
                                                                src_weight)
                    n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(self.batch_size).to(self.device),
                                                                tgt_weight)
                    da_loss = loss_cdan_src + loss_cdan_tgt
                else:
                    raise ValueError(f"The da method {self.da_method} is not supported")
                loss = model_loss + da_loss
            else:
                loss = model_loss
            loss.backward()
            self.optim.step()
            self.optim_da.step()
            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
            if self.current_epoch >= self.da_init_epoch:
                da_loss_epoch += da_loss.item()
                if self.experiment:
                    self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)
                    
        total_loss_epoch = total_loss_epoch / num_batches
        model_loss_epoch = model_loss_epoch / num_batches
        da_loss_epoch = da_loss_epoch / num_batches
        if self.current_epoch < self.da_init_epoch:
            print('Training at Epoch ' + str(self.current_epoch) + ' with model training loss ' + str(total_loss_epoch))
        else:
            print('Training at Epoch ' + str(self.current_epoch) + ' model training loss ' + str(model_loss_epoch)
                  + ", da loss " + str(da_loss_epoch) + ", total training loss " + str(total_loss_epoch) + ", DA lambda " +
                  str(epoch_lamb_da))
        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
            # 使用最佳模型的权重
            prev_state = copy.deepcopy(self.model.state_dict())
            
            # 检查测试数据加载器是否正确初始化
            if data_loader is None:
                print("错误: 测试数据加载器未初始化!")
                return 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, float('inf'), 0.5, 0.0
                
            # 检查测试数据加载器长度
            print(f"测试数据集大小: {len(data_loader)} 批次")
            
            # 增加检查最佳模型状态是否存在
            if self.best_model_state is None:
                print("警告: 没有保存的最佳模型状态。尝试从磁盘加载最佳模型文件。")
                best_model_path = os.path.join(self.output_dir, "best_model.pth")
                if os.path.exists(best_model_path):
                    try:
                        self.best_model_state = torch.load(best_model_path, map_location=self.device)
                        if isinstance(self.best_model_state, dict) and 'state_dict' in self.best_model_state:
                            self.best_model_state = self.best_model_state['state_dict']
                        print(f"从磁盘成功加载最佳模型: {best_model_path}")
                    except Exception as e:
                        print(f"加载最佳模型失败: {str(e)}")
                        print("将使用当前模型进行测试")
                        self.best_model_state = self.model.state_dict()
                else:
                    print(f"警告: 找不到最佳模型文件 {best_model_path}")
                    print("将使用当前模型进行测试")
                    self.best_model_state = self.model.state_dict()
            
            # 加载最佳模型状态
            try:
                self.model.load_state_dict(self.best_model_state)
                print(f"成功加载最佳模型状态（来自Epoch {self.best_epoch if hasattr(self, 'best_epoch') else 'unknown'}）")
            except Exception as e:
                print(f"加载最佳模型状态失败: {str(e)}")
                print("将使用当前模型进行测试")
                # 保持当前模型状态
        elif dataloader == "val":
            data_loader = self.val_dataloader
            print(f"验证数据集大小: {len(data_loader)} 批次")
        else:
            raise ValueError(f"Error key value {dataloader}")
        
        num_batches = 0
        processed_samples = 0
        error_batches = 0
        
        with torch.no_grad():
            self.model.eval()
            # 添加进度条
            print(f"开始进行{dataloader}评估...")
            for i, batch in enumerate(data_loader):
                # 检查批处理数据是否为None
                if batch is None:
                    print(f"警告: 批次 {i} 为空，跳过")
                    error_batches += 1
                    continue
                
                if isinstance(batch, tuple) and None in batch:
                    print(f"警告: 批次 {i} 包含None值，跳过。批次结构: {[type(x) if x is not None else None for x in batch]}")
                    error_batches += 1
                    continue
                
                try:
                    # 处理不同类型的批处理数据
                    if len(batch) == 3:  # 多模态数据格式: (bg_3d, data_1d2d, labels) 或原始DrugBAN格式: (v_d, v_p, labels)
                        if hasattr(self.model, 'drugban_3d') and hasattr(self.model, 'drugban_1d2d'):
                            # 多模态DrugBAN数据格式
                            bg_3d, data_1d2d, labels = batch

                            if bg_3d is None or data_1d2d is None or labels is None:
                                print(f"警告: 批次 {i} 包含None值，跳过")
                                error_batches += 1
                                continue

                            batch_size = labels.shape[0]

                            # 将数据移到设备上
                            bg_3d = bg_3d.to(self.device)
                            labels = labels.float().to(self.device)

                            # 处理1D/2D数据，将其移到设备上
                            mol_graph = data_1d2d['mol_graph'].to(self.device)  # 确保DGL图移动到正确设备
                            protein_seq = data_1d2d['protein_seq'].to(self.device)

                            data_1d2d_device = {
                                'mol_graph': mol_graph,
                                'protein_seq': protein_seq
                            }

                            # 前向传播（多模态）
                            v_d, v_p, f, score = self.model(bg_3d, data_1d2d_device)
                        else:
                            # 原始DrugBAN数据格式
                            v_d, v_p, labels = batch
                            v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                            v_d, v_p, f, score = self.model(v_d, v_p)
                            batch_size = labels.shape[0]

                    elif len(batch) == 2:  # DrugBAN3D的数据格式: (bg, labels)
                        bg, labels = batch
                        if bg is None or labels is None:
                            print(f"警告: 批次 {i} 的图或标签为None，跳过")
                            error_batches += 1
                            continue

                        batch_size = labels.shape[0]
                        # 简化的批次信息输出（仅在详细模式下）
                        if self.verbose and (i < 3 or i % 50 == 0):  # 减少输出频率
                            print(f"处理批次 {i}/{len(dataloader)}, 样本数: {batch_size}")

                        bg, labels = bg.to(self.device), labels.float().to(self.device)
                        v_d, v_p, f, score = self.model(bg)

                    else:
                        print(f"警告: 批次 {i} 的元素数量 ({len(batch)}) 不符合预期，跳过")
                        error_batches += 1
                        continue

                    # 调整预测值形状，确保与标签形状匹配
                    score = score.squeeze()
                    
                    if self.n_class == 1:
                        # 添加调试信息和形状检查
                        if i < 5:  # 只在前几个批次打印调试信息
                            print(f"批次 {i}: score形状={score.shape}, labels形状={labels.shape}")

                        # 确保score和labels都是有效的张量
                        if score.numel() == 0 or labels.numel() == 0:
                            print(f"警告: 批次 {i} 包含空张量，跳过")
                            error_batches += 1
                            continue

                        n, loss = binary_cross_entropy(score, labels, self.label_smoothing)
                    else:
                        n, loss = cross_entropy_logits(score, labels)
                    test_loss += loss.item()
                    
                    # 添加详细的批次预测信息
                    batch_labels = labels.to("cpu").tolist()
                    batch_preds = n.to("cpu").tolist()
                    y_label.extend(batch_labels)
                    y_pred.extend(batch_preds)
                    
                    processed_samples += batch_size
                    num_batches += 1
                    
                    # 打印批次处理进度
                    if i < 5 or i % 100 == 0:
                        print(f"处理批次 {i}/{len(data_loader)}, 累计样本数: {processed_samples}, 批次损失: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"处理批次 {i} 时出错: {str(e)}")
                    traceback.print_exc()  # 打印完整的错误堆栈
                    error_batches += 1
                    continue
        
        # 打印处理统计信息
        print(f"测试完成: 处理了 {num_batches}/{len(data_loader)} 批次, {processed_samples} 个样本")
        print(f"错误批次数: {error_batches}")
        print(f"预测标签统计: 正样本数={sum([1 for x in y_label if x > 0.5])}, 负样本数={sum([1 for x in y_label if x <= 0.5])}")
        
        # 如果是测试集，恢复原来的模型权重
        if dataloader == "test":
            self.model.load_state_dict(prev_state)
            
        # 防止没有成功处理任何批次的情况
        if len(y_label) == 0 or len(y_pred) == 0:
            print("警告: 没有成功处理任何样本，无法计算评估指标!")
            if dataloader == "test":
                return 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, float('inf'), 0.5, 0.0
            else:
                return 0.5, 0.5, float('inf')
        
        # 计算评估指标
        try:
            auroc = roc_auc_score(y_label, y_pred)
            auprc = average_precision_score(y_label, y_pred)
            print(f"成功计算评估指标: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
        except Exception as e:
            print(f"计算评估指标时出错: {str(e)}")
            auroc, auprc = 0.5, 0.5
            
        # 防止除零错误
        test_loss = test_loss / max(1, num_batches)

        if dataloader == "test":
            try:
                fpr, tpr, thresholds = roc_curve(y_label, y_pred)
                prec, recall, _ = precision_recall_curve(y_label, y_pred)
                precision = tpr / (tpr + fpr + 1e-10)  # 添加小的epsilon防止除零
                f1 = 2 * precision * tpr / (tpr + precision + 1e-10)
                
                # 确保f1数组不为空
                if len(f1) <= 5:
                    print("警告: F1分数数组过短，使用默认阈值0.5")
                    thred_optim = 0.5
                    max_f1 = 0.0
                else:
                    max_f1 = np.max(f1[5:])
                    thred_optim = thresholds[5:][np.argmax(f1[5:])]
                
                print(f"最佳阈值: {thred_optim:.4f}, 对应的F1分数: {max_f1:.4f}")
                
                y_pred_s = [1 if i >= thred_optim else 0 for i in y_pred]
                cm1 = confusion_matrix(y_label, y_pred_s)
                print(f"混淆矩阵:\n{cm1}")
                
                accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
                sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1] + 1e-10)
                specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1] + 1e-10)
                
                if self.experiment:
                    self.experiment.log_curve("test_roc curve", fpr, tpr)
                    self.experiment.log_curve("test_pr curve", recall, prec)
                
                precision1 = precision_score(y_label, y_pred_s)
                
                print(f"详细测试结果: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, F1={max_f1:.4f}, "
                      f"敏感度={sensitivity:.4f}, 特异度={specificity:.4f}, 准确率={accuracy:.4f}, "
                      f"精确度={precision1:.4f}")
                
                return auroc, auprc, max_f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
            except Exception as e:
                print(f"计算详细测试指标时出错: {str(e)}")
                traceback.print_exc()
                return 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, test_loss, 0.5, 0.0
        else:
            return auroc, auprc, test_loss

    def configure_loss_fn(self):
        """配置损失函数"""
        if self.n_class == 1:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            if self.label_smoothing > 0:
                print(f"启用标签平滑，参数值: {self.label_smoothing}")
            if self.pos_weight is not None:
                print(f"启用类别加权，正样本权重: {self.pos_weight.item()}")
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            print(f"使用多分类交叉熵损失函数，类别数: {self.n_class}")

    def configure_lr_scheduler(self):
        """配置学习率调度器"""
        if self.lr_scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(self.optim, T_max=self.max_epoch)
            print(f"使用余弦退火学习率调度器，T_max={self.max_epoch}")
        elif self.lr_scheduler_type == "plateau":
            # 增强ReduceLROnPlateau的参数设置
            self.scheduler = ReduceLROnPlateau(
                self.optim, 
                mode='max',          # 监控指标增大时(AUROC)表示改进
                factor=0.5,          # 学习率降低为之前的一半
                patience=4,          # 4轮不改善才降低学习率（批次增大，轮次减少）
                verbose=True,        # 打印学习率变化
                threshold=0.001,     # 指标改善阈值增大，适应更大批次
                min_lr=1e-6          # 最小学习率限制
            )
            print("使用ReduceLROnPlateau学习率调度器，监控验证集AUROC")
        elif self.lr_scheduler_type == "one_cycle":
            steps_per_epoch = len(self.train_dataloader)
            self.scheduler = OneCycleLR(
                self.optim, 
                max_lr=self.learning_rate * 10, 
                epochs=self.max_epoch,
                steps_per_epoch=steps_per_epoch
            )
            print("使用OneCycleLR学习率调度器")
