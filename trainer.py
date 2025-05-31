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


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]
        
        # 标签平滑参数
        self.label_smoothing = config["TRAIN"].get("LABEL_SMOOTHING", 0.0)
        if self.label_smoothing > 0:
            print(f"启用标签平滑，参数值: {self.label_smoothing}")
        
        # 添加类别加权参数（正样本权重）
        # 从配置中读取POS_WEIGHT，如果未定义则使用默认值2.125
        pos_weight_value = config["TRAIN"].get("POS_WEIGHT", 2.125)
        self.pos_weight = torch.tensor([pos_weight_value]).to(self.device)
        print(f"启用类别加权，正样本权重: {self.pos_weight.item()}")
        
        # 添加早停参数
        self.patience = config.get("EARLY_STOPPING_PATIENCE", 5)  # 默认5轮不提升则早停
        self.early_stopping_counter = 0  # 计数器，记录连续未改善的轮数
        self.early_stopping = config.get("USE_EARLY_STOPPING", True)  # 默认启用早停
        
        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(in_features=config["DECODER"]["IN_DIM"]*self.n_class, out_features=config["DA"]
                ["RANDOM_DIM"], bias=False).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
                self.random_layer = self.random_layer.to(self.device)
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False
        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model_state = None  # 使用state_dict而不是完整模型
        self.best_epoch = None
        self.best_auroc = 0

        # 梯度裁剪设置
        self.clip_grad_norm = config["TRAIN"].get("GRADIENT_CLIP_NORM", 0.0)
        if self.clip_grad_norm > 0:
            print(f"启用梯度裁剪，阈值：{self.clip_grad_norm}")
            
        # 使用state_dict保存模型
        self.use_state_dict = config["RESULT"].get("USE_STATE_DICT", True)
        if self.use_state_dict:
            print("使用state_dict保存模型，避免完整对象复制")

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]
        
        # 是否每轮保存结果
        self.save_each_epoch = config["RESULT"].get("SAVE_EACH_EPOCH", False)
        self.save_best_only = config["RESULT"].get("SAVE_BEST_ONLY", False)
            
        # 设置学习率调度器
        self.use_lr_scheduler = config["SOLVER"].get("LR_SCHEDULER", False)
        if self.use_lr_scheduler:
            lr_scheduler_type = config["SOLVER"].get("LR_SCHEDULER_TYPE", "cosine")
            warmup_epochs = config["SOLVER"].get("LR_WARMUP_EPOCHS", 0)
            
            if lr_scheduler_type == "cosine":
                self.scheduler = CosineAnnealingLR(self.optim, T_max=self.epochs)
                print(f"使用余弦退火学习率调度器，T_max={self.epochs}")
            elif lr_scheduler_type == "plateau":
                # 增强ReduceLROnPlateau的参数设置
                self.scheduler = ReduceLROnPlateau(
                    self.optim, 
                    mode='max',          # 监控指标增大时(AUROC)表示改进
                    factor=0.5,          # 学习率降低为之前的一半
                    patience=5,          # 5轮不改善才降低学习率
                    verbose=True,        # 打印学习率变化
                    threshold=0.0005,    # 指标改善阈值
                    min_lr=1e-6          # 最小学习率限制
                )
                print("使用ReduceLROnPlateau学习率调度器，监控验证集AUROC")
            elif lr_scheduler_type == "one_cycle":
                steps_per_epoch = len(self.train_dataloader)
                self.scheduler = OneCycleLR(
                    self.optim, 
                    max_lr=config["TRAIN"]["G_LEARNING_RATE"] * 10, 
                    epochs=self.epochs,
                    steps_per_epoch=steps_per_epoch
                )
                print("使用OneCycleLR学习率调度器")

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        if not self.is_da:
            train_metric_header = ["# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

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
        
        # 创建训练日志文件
        log_file = os.path.join(self.output_dir, "training_log.txt")
        with open(log_file, 'w') as f:
            f.write(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"模型: {self.config['MODEL_TYPE']}\n")
            f.write(f"总训练轮数: {self.epochs}\n\n")
            f.write("轮次\t训练损失\t验证损失\t验证AUROC\t验证AUPRC\n")
            f.write("-" * 70 + "\n")
        
        for i in range(self.epochs):
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
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
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
            if auroc >= self.best_auroc:
                # 保存模型state_dict而不是整个模型对象
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
                # 保存最佳模型
                if self.save_best_only:
                    torch.save(self.best_model_state,
                           os.path.join(self.output_dir, f"best_model.pth"))
                # 重置早停计数器
                self.early_stopping_counter = 0
            else:
                # 增加早停计数器
                self.early_stopping_counter += 1
                if self.early_stopping and self.early_stopping_counter >= self.patience:
                    print(f"早停触发！连续{self.patience}轮验证集性能未提升")
                    break
                
            # 只输出简洁的训练状态信息
            print(f'Epoch {self.current_epoch}/{self.epochs} - 训练损失: {train_loss:.4f} - 验证AUROC: {auroc:.4f} - 验证AUPRC: {auprc:.4f}')
            
            # 更新并记录训练日志
            with open(os.path.join(self.output_dir, "training_log.txt"), 'a') as f:
                f.write(f"{self.current_epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{auroc:.4f}\t{auprc:.4f}\n")
                
            # 保存进度CSV文件，方便实时查看训练进度
            progress_csv = os.path.join(self.output_dir, "training_progress.csv")
            with open(progress_csv, 'w') as fp:
                # 写入表头
                header = "epoch,train_loss,val_loss,val_auroc,val_auprc"
                if self.is_da:
                    header += ",model_loss,da_loss"
                fp.write(header + "\n")
                
                # 写入已完成轮次的指标
                for j in range(len(self.train_loss_epoch)):
                    epoch_num = j + 1
                    line = f"{epoch_num},{self.train_loss_epoch[j]:.6f},{self.val_loss_epoch[j]:.6f}"
                    if j < len(self.val_auroc_epoch):
                        line += f",{self.val_auroc_epoch[j]:.6f}"
                        # 从验证表格中提取AUPRC值
                        try:
                            if j < len(self.val_table._rows):
                                auprc = float(self.val_table._rows[j][2])  # AUPRC是表格的第三列
                                line += f",{auprc:.6f}"
                            else:
                                line += ",NA" 
                        except:
                            line += ",NA"
                    else:
                        line += ",NA,NA"
                        
                    if self.is_da and j < len(self.train_model_loss_epoch) and j < len(self.train_da_loss_epoch):
                        line += f",{self.train_model_loss_epoch[j]:.6f},{self.train_da_loss_epoch[j]:.6f}"
                    
                    fp.write(line + "\n")
            
            # 检查是否需要保存当前轮次的结果
            if self.save_each_epoch:
                self.save_epoch_result()
                
        # 测试评估时使用最佳模型的权重
        self.model.load_state_dict(self.best_model_state)
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim, test_loss]))
        self.test_table.add_row(test_lst)
        print('测试结果 (最佳模型来自Epoch ' + str(self.best_epoch) + '):')
        print(f'AUROC: {auroc:.4f} - AUPRC: {auprc:.4f} - F1: {f1:.4f} - 敏感度: {sensitivity:.4f} - 特异度: {specificity:.4f} - 准确率: {accuracy:.4f}')
        
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
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
            self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_threshold", self.test_metrics["thred_optim"])
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
            self.experiment.log_metric("test_precision", self.test_metrics["Precision"])
        return self.test_metrics

    def save_epoch_result(self):
        """每个epoch结束后保存验证结果和最佳模型（不保存每轮模型）"""
        # 创建本轮结果目录
        epoch_dir = os.path.join(self.output_dir, f"epoch_{self.current_epoch}")
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
        if self.val_auroc_epoch[-1] > self.best_auroc:
            self.best_auroc = self.val_auroc_epoch[-1]
            self.best_epoch = self.current_epoch
            self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            # 更新最佳指标标志
            metrics["is_best"] = True
            with open(os.path.join(epoch_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        
            # 保存最佳模型
            if self.config["RESULT"]["SAVE_MODEL"]:
                best_model_path = os.path.join(self.output_dir, "best_model.pth")
                if self.use_state_dict:
                    torch.save({
                        'state_dict': self.best_model_state,
                        'epoch': self.best_epoch,
                        'auroc': self.best_auroc
                    }, best_model_path)
                else:
                    torch.save(self.model, best_model_path)
                
                print(f"保存最佳模型（轮次 {self.best_epoch}）到 {best_model_path}")
                
            # 重置早停计数器
            self.early_stopping_counter = 0
        else:
            # 增加早停计数器
            self.early_stopping_counter += 1

    def save_result(self):
        # 使用时间戳创建唯一的结果目录
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.output_dir, f"final_result_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model_state, os.path.join(result_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(result_dir, f"final_model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(result_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(result_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(result_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(result_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())
            
        # 保存CSV格式的训练日志，方便绘图和分析
        metrics_log_file = os.path.join(result_dir, "training_metrics.csv")
        with open(metrics_log_file, 'w') as fp:
            # 写入表头
            header = "epoch,train_loss,val_loss,val_auroc,val_auprc"
            if self.is_da:
                header += ",model_loss,da_loss"
            fp.write(header + "\n")
            
            # 写入每轮指标
            for i in range(len(self.train_loss_epoch)):
                epoch_num = i + 1
                line = f"{epoch_num},{self.train_loss_epoch[i]:.6f},{self.val_loss_epoch[i]:.6f}"
                if i < len(self.val_auroc_epoch):
                    line += f",{self.val_auroc_epoch[i]:.6f}"
                    # 没有直接存储val_auprc，尝试从表格中提取
                    try:
                        # 从验证表格中提取对应行的AUPRC值
                        if i < len(self.val_table._rows):
                            auprc = float(self.val_table._rows[i][2])  # AUPRC是表格的第三列
                            line += f",{auprc:.6f}"
                        else:
                            line += ",NA"
                    except:
                        line += ",NA"
                else:
                    line += ",NA,NA"
                    
                if self.is_da and i < len(self.train_model_loss_epoch) and i < len(self.train_da_loss_epoch):
                    line += f",{self.train_model_loss_epoch[i]:.6f},{self.train_da_loss_epoch[i]:.6f}"
                
                fp.write(line + "\n")
                
        print(f"训练指标日志已保存到 {metrics_log_file}")
        
        # 保存最终测试结果的单行CSV，方便比较不同实验
        test_summary_file = os.path.join(result_dir, "test_summary.csv") 
        with open(test_summary_file, 'w') as fp:
            # 写入表头
            header = "best_epoch,auroc,auprc,f1,sensitivity,specificity,accuracy"
            fp.write(header + "\n")
            
            # 写入测试结果
            line = f"{self.best_epoch},{self.test_metrics['auroc']:.6f},{self.test_metrics['auprc']:.6f}"
            line += f",{self.test_metrics['F1']:.6f},{self.test_metrics['sensitivity']:.6f}"
            line += f",{self.test_metrics['specificity']:.6f},{self.test_metrics['accuracy']:.6f}"
            fp.write(line + "\n")
            
        print(f"测试指标摘要已保存到 {test_summary_file}")
        print(f"所有结果已保存到目录: {result_dir}")

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = 0
        
        # 使用tqdm创建进度条，但不显示加载信息
        for i, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}/{self.epochs}", ncols=100)):
            self.step += 1
            
            # 检查批处理数据是否为None
            if batch is None or (isinstance(batch, tuple) and None in batch):
                continue
                
            # 处理不同类型的批处理数据
            try:
                if len(batch) == 3:  # 原始DrugBAN的数据格式: (v_d, v_p, labels)
                    v_d, v_p, labels = batch
                    v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                    v_d, v_p, f, score = self.model(v_d, v_p)
                elif len(batch) == 2:  # DrugBAN3D的数据格式: (bg, labels)
                    bg, labels = batch
                    if bg is None or labels is None:
                        continue
                    bg, labels = bg.to(self.device), labels.float().to(self.device)
                    v_d, v_p, f, score = self.model(bg)
                else:
                    continue
                
                # 使用类别加权的损失函数
                if self.n_class == 1:
                    # 使用binary_cross_entropy函数(支持焦点损失和标签平滑)
                    n, loss = binary_cross_entropy(
                        score,
                        labels,
                        label_smoothing=self.label_smoothing
                    )
                    
                    # 应用类别权重
                    # 获取正样本和负样本的索引
                    pos_indices = (labels > 0.5).float()
                    neg_indices = (labels <= 0.5).float()
                    
                    # 计算每个样本的加权损失
                    weighted_loss = loss * (pos_indices * self.pos_weight.item() + neg_indices)
                    loss = weighted_loss.mean()
                else:
                    n, loss = cross_entropy_logits(score, labels)
                
                # 反向传播和优化
                self.optim.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                    
                self.optim.step()
                
                loss_epoch += loss.item()
                num_batches += 1
                
                if self.experiment:
                    self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
            except Exception as e:
                print(f"训练批次错误，跳过: {str(e)}")
                continue
                
        # 防止没有成功处理任何批次的情况
        if num_batches == 0:
            return 0
            
        loss_epoch = loss_epoch / num_batches
        return loss_epoch

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
        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
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
            self.model.load_state_dict(self.best_model_state)
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        
        num_batches = 0
        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(data_loader):
                # 检查批处理数据是否为None
                if batch is None or (isinstance(batch, tuple) and None in batch):
                    continue
                
                try:
                    # 处理不同类型的批处理数据
                    if len(batch) == 3:  # 原始DrugBAN的数据格式: (v_d, v_p, labels)
                        v_d, v_p, labels = batch
                        v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
                        v_d, v_p, f, score = self.model(v_d, v_p)
                    elif len(batch) == 2:  # DrugBAN3D的数据格式: (bg, labels)
                        bg, labels = batch
                        if bg is None or labels is None:
                            continue
                        
                        bg, labels = bg.to(self.device), labels.float().to(self.device)
                        v_d, v_p, f, score = self.model(bg)
                    else:
                        continue
                    
                    if self.n_class == 1:
                        n, loss = binary_cross_entropy(score, labels, self.label_smoothing)
                    else:
                        n, loss = cross_entropy_logits(score, labels)
                    test_loss += loss.item()
                    y_label = y_label + labels.to("cpu").tolist()
                    y_pred = y_pred + n.to("cpu").tolist()
                    num_batches += 1
                except Exception as e:
                    continue
                    
        # 如果是测试集，恢复原来的模型权重
        if dataloader == "test":
            self.model.load_state_dict(prev_state)
            
        # 防止没有成功处理任何批次的情况
        if len(y_label) == 0 or len(y_pred) == 0:
            if dataloader == "test":
                return 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, float('inf'), 0.5, 0.0
            else:
                return 0.5, 0.5, float('inf')
        
        # 计算评估指标
        try:
            auroc = roc_auc_score(y_label, y_pred)
            auprc = average_precision_score(y_label, y_pred)
        except Exception as e:
            auroc, auprc = 0.5, 0.5
            
        # 防止除零错误
        test_loss = test_loss / max(1, num_batches)

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr + 1e-10)  # 添加小的epsilon防止除零
            f1 = 2 * precision * tpr / (tpr + precision + 1e-10)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1] + 1e-10)
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1] + 1e-10)
            if self.experiment:
                self.experiment.log_curve("test_roc curve", fpr, tpr)
                self.experiment.log_curve("test_pr curve", recall, prec)
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss
