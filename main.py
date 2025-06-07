# Comet ML支持 - 必须在其他导入之前
comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False

import os
import argparse
import torch
import numpy as np
import random
import pandas as pd
import warnings
from time import time
import time as time_module
from yacs.config import CfgNode
from configs import get_cfg_defaults
from trainer import Trainer
from dataloader import DTIDataset, MultiDataLoader
from dataloader_3d import get_protein_ligand_dataloader, get_precomputed_graph_dataloader, get_loader
from dataloader_3d_augmented import get_augmented_dataloader  # 导入增强数据加载器
from models import DrugBAN
from drugban_3d import DrugBAN3D
from torch.utils.data import DataLoader
from utils import set_seed, graph_collate_func, mkdir
from dataloader_3d import ProteinLigandDataset, filter_none_collate_fn
from domain_adaptator import Discriminator


def parse_args():
    parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
    # 配置文件
    parser.add_argument("--cfg", dest="cfg_file", help="配置文件路径", required=True, type=str)
    # 数据集
    parser.add_argument("--data", dest="data", help="数据集名称: ['biosnap', 'bindingdb', 'human']",
                        required=True, type=str)
    # 分割方式
    parser.add_argument("--split", dest="split", help="分割方式: ['random', 'cold', 'cluster', 'stratified']",
                        required=True, type=str, choices=['random', 'cold', 'cluster', 'stratified'])
    # 模型权重路径
    parser.add_argument("--load", dest="load", help="从检查点加载", default=None, type=str)
    # 输出目录
    parser.add_argument("--output-dir", dest="output_dir", help="输出目录", type=str, default=None)
    # 随机种子
    parser.add_argument("--seed", type=int, default=42, help='random seed')
    # 标签
    parser.add_argument("--tag", type=str, help='experiment tag')
    
    # 3D数据相关参数
    parser.add_argument("--use_3d", action="store_true", help="使用3D数据和模型")
    parser.add_argument("--data_3d_root", type=str, help="3D数据根目录")
    parser.add_argument("--data_3d_label", type=str, help="3D数据标签文件")
    parser.add_argument("--use_augmented", action="store_true", help="使用增强后的数据")
    
    # 交叉验证参数
    parser.add_argument("--cv_fold", type=int, default=0, help="当前交叉验证折数(1-based)")
    parser.add_argument("--cv_total_folds", type=int, default=0, help="交叉验证总折数")
    
    # 测试相关参数
    parser.add_argument("--test_only", action="store_true", help="仅执行测试，不进行训练")
    parser.add_argument("--verbose", action="store_true", help="输出详细日志信息")
    parser.add_argument("--save_predictions", action="store_true", help="保存测试预测结果")
    
    return parser.parse_args()


def read_config(args):
    """读取配置文件并设置默认值"""
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)
    
    # 设置输出目录 - 优先使用命令行参数
    if args.output_dir is not None:
        cfg.RESULT.OUTPUT_DIR = args.output_dir
        print(f"使用命令行指定的输出目录: {args.output_dir}")
    elif not cfg.RESULT.OUTPUT_DIR or cfg.RESULT.OUTPUT_DIR == "":
        # 如果配置文件中没有设置OUTPUT_DIR或为空，则使用时间戳创建目录
        timestamp = time_module.strftime("%Y%m%d_%H%M%S")
        default_output_dir = f"result/DrugBAN3D_run_{timestamp}"
        cfg.RESULT.OUTPUT_DIR = default_output_dir
        print(f"未指定输出目录，使用默认时间戳目录: {default_output_dir}")
    else:
        print(f"使用配置文件中的输出目录: {cfg.RESULT.OUTPUT_DIR}")
    
    # 设置标签 - 更安全的方式处理tag参数
    if args.tag is not None:
        if not hasattr(cfg.COMET, 'TAG'):
            cfg.COMET.TAG = args.tag
        else:
            cfg.COMET.TAG = args.tag
    
    # 设置3D数据参数
    if args.use_3d:
        cfg.MODEL_TYPE = "DrugBAN3D"
        if args.data_3d_root:
            cfg.DATA_3D.ROOT_DIR = args.data_3d_root
        if args.data_3d_label:
            cfg.DATA_3D.LABEL_FILE = args.data_3d_label
    
    # 添加测试相关配置
    if args.test_only:
        cfg.TEST_ONLY = True
    if args.verbose:
        cfg.VERBOSE = True
    if args.save_predictions:
        cfg.SAVE_PREDICTIONS = True
    
    # 检查并创建必要的目录结构
    ensure_output_dir_structure(cfg.RESULT.OUTPUT_DIR)
        
    return cfg


def ensure_output_dir_structure(output_dir):
    """确保输出目录结构存在"""
    # 创建主输出目录
    mkdir(output_dir)
    
    # 创建子目录
    subdirs = ["models", "metrics", "logs", "config", "predictions"]
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        mkdir(subdir_path)
        
    return output_dir


def set_random_seed(seed):
    """设置随机种子以便复现结果"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 读取配置
    cfg = read_config(args)
    
    # 创建输出目录
    mkdir(cfg.RESULT.OUTPUT_DIR)
    
    # 保存当前配置到输出目录
    config_save_path = os.path.join(cfg.RESULT.OUTPUT_DIR, "config", "config_used.yaml")
    with open(config_save_path, 'w') as f:
        f.write(str(cfg))
    print(f"已保存当前配置到 {config_save_path}")
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置日志
    suffix = str(int(time() * 1000))[6:]
    experiment = None
    
    print(f"配置文件: {args.cfg_file}")
    print(f"输出目录: {cfg.RESULT.OUTPUT_DIR}")
    print(f"运行设备: {device}")
    
    # 判断是否仅执行测试
    test_only = args.test_only
    if test_only:
        print("测试模式: 仅执行测试，不进行训练")
        if args.load is None:
            print("错误: 测试模式需要指定模型路径，使用 --load 参数")
            return None
        
        print(f"将使用模型: {args.load}")
        if not os.path.exists(args.load):
            print(f"错误: 指定的模型文件不存在: {args.load}")
            return None
    
    # 交叉验证支持
    cv_fold = args.cv_fold
    cv_total_folds = args.cv_total_folds
    use_cv = cv_fold > 0 and cv_total_folds > 0
    
    if use_cv:
        print(f"使用交叉验证: 当前第{cv_fold}折，共{cv_total_folds}折")
    
    # 准备数据
    training_generator = None
    val_generator = None
    test_generator = None
    
    if cfg.MODEL_TYPE == "DrugBAN":
        # 使用原始DrugBAN数据加载方式
        dataFolder = f'{cfg.PATH.DATA_DIR}/{args.data}'
        dataFolder = os.path.join(dataFolder, str(args.split))
        
        train_path = os.path.join(dataFolder, 'train.csv')
        val_path = os.path.join(dataFolder, "val.csv")
        test_path = os.path.join(dataFolder, "test.csv")
        
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)
        
        train_samples = len(df_train)
        val_samples = len(df_val)
        test_samples = len(df_test)
        
        if args.split == 'stratified':
            # 计算正样本比例
            train_pos = df_train['label'].sum()
            val_pos = df_val['label'].sum()
            test_pos = df_test['label'].sum()
            train_pos_ratio = train_pos / train_samples * 100
            val_pos_ratio = val_pos / val_samples * 100
            test_pos_ratio = test_pos / test_samples * 100
            print(f"分层数据集: 训练集 {train_samples} 样本, 验证集 {val_samples} 样本, 测试集 {test_samples} 样本")
            print(f"正样本比例: 训练集 {train_pos_ratio:.2f}%, 验证集 {val_pos_ratio:.2f}%, 测试集 {test_pos_ratio:.2f}%")

        train_dataset = DTIDataset(df_train.index.values, df_train)
        val_dataset = DTIDataset(df_val.index.values, df_val)
        test_dataset = DTIDataset(df_test.index.values, df_test)
        
        params = {
            'batch_size': cfg.TRAIN.BATCH_SIZE, 
            'shuffle': True, 
            'num_workers': 4,
            'drop_last': True, 
            'collate_fn': graph_collate_func
        }
        
        training_generator = DataLoader(train_dataset, **params)
        params['shuffle'] = False
        params['drop_last'] = False
        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)
    else:  # DrugBAN3D
        # 获取工作线程数
        num_workers = cfg.DATA_3D.NUM_WORKERS if hasattr(cfg.DATA_3D, 'NUM_WORKERS') else 2
        print(f"数据加载器使用 {num_workers} 个工作线程")
        
        # 使用3D数据加载器
        try:
            # 优先使用分层采样数据集
            if hasattr(cfg, 'DATA') and hasattr(cfg.DATA, 'TRAIN_FILE') and \
               hasattr(cfg.DATA, 'VAL_FILE') and hasattr(cfg.DATA, 'TEST_FILE'):
                print("使用分层采样的预划分数据集")
                
                # 读取分层采样的数据集
                if use_cv:
                    # 对于交叉验证，读取完整训练集，然后自行划分
                    all_data_df = pd.concat([
                        pd.read_csv(cfg.DATA.TRAIN_FILE),
                        pd.read_csv(cfg.DATA.VAL_FILE)
                    ]).reset_index(drop=True)
                    
                    # 打乱数据
                    all_data_df = all_data_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
                    
                    # 计算每折的大小
                    fold_size = len(all_data_df) // cv_total_folds
                    
                    # 划分当前折的验证集
                    val_start = (cv_fold - 1) * fold_size
                    val_end = cv_fold * fold_size if cv_fold < cv_total_folds else len(all_data_df)
                    
                    # 划分训练集和验证集
                    val_df = all_data_df.iloc[val_start:val_end].reset_index(drop=True)
                    train_df = pd.concat([all_data_df.iloc[:val_start], all_data_df.iloc[val_end:]]).reset_index(drop=True)
                    
                    # 测试集保持不变
                    test_df = pd.read_csv(cfg.DATA.TEST_FILE)
                else:
                    # 常规训练，直接使用预划分的数据
                    train_df = pd.read_csv(cfg.DATA.TRAIN_FILE)
                    val_df = pd.read_csv(cfg.DATA.VAL_FILE)
                    test_df = pd.read_csv(cfg.DATA.TEST_FILE)
                
                # 打印数据集统计信息
                train_samples = len(train_df)
                val_samples = len(val_df)
                test_samples = len(test_df)
                
                # 计算正样本比例
                train_pos = train_df['label'].sum()
                val_pos = val_df['label'].sum()
                test_pos = test_df['label'].sum()
                
                train_pos_ratio = train_pos / train_samples * 100
                val_pos_ratio = val_pos / val_samples * 100
                test_pos_ratio = test_pos / test_samples * 100
                
                print(f"数据集: 训练集 {train_samples} 样本, 验证集 {val_samples} 样本, 测试集 {test_samples} 样本")
                print(f"正样本比例: 训练集 {train_pos_ratio:.2f}%, 验证集 {val_pos_ratio:.2f}%, 测试集 {test_pos_ratio:.2f}%")
                
                # 检查是否使用增强数据集
                if args.use_augmented or hasattr(cfg.PATH, 'CACHE_DIR') and cfg.PATH.CACHE_DIR and 'augmented' in cfg.PATH.CACHE_DIR:
                    print("使用增强数据集进行训练...")
                    # 加载增强数据
                    training_generator = get_augmented_dataloader(
                        root_dir=cfg.DATA_3D.ROOT_DIR,
                        label_file=cfg.DATA.TRAIN_FILE,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=True,
                        num_workers=num_workers,
                        is_test=False,
                    cache_dir=cfg.PATH.CACHE_DIR
                )
                    
                    val_generator = get_augmented_dataloader(
                        root_dir=cfg.DATA_3D.ROOT_DIR,
                        label_file=cfg.DATA.VAL_FILE,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=False,
                        num_workers=num_workers,
                        is_test=False,
                    cache_dir=cfg.PATH.CACHE_DIR
                )
                
                    # 测试集使用原始缓存目录，不使用增强缓存
                    test_generator = get_augmented_dataloader(
                        root_dir=cfg.DATA_3D.ROOT_DIR,
                        label_file=cfg.DATA.TEST_FILE,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=False,
                        num_workers=num_workers,
                        is_test=True,
                        cache_dir="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs"  # 使用原始缓存
                    )
                else:
                    # 使用标准数据加载器
                    print("使用标准数据集进行训练...")
                    training_generator = get_loader(
                        cfg.DATA_3D.ROOT_DIR,
                        label_file=cfg.DATA.TRAIN_FILE,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    shuffle=True,
                    num_workers=num_workers,
                        dis_threshold=cfg.DATA_3D.DIS_THRESHOLD,
                        cache_dir=cfg.PATH.CACHE_DIR
                )
                
                    val_generator = get_loader(
                        cfg.DATA_3D.ROOT_DIR,
                        label_file=cfg.DATA.VAL_FILE,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    shuffle=False,
                    num_workers=num_workers,
                        dis_threshold=cfg.DATA_3D.DIS_THRESHOLD,
                        cache_dir=cfg.PATH.CACHE_DIR
                )
                
                    test_generator = get_loader(
                        cfg.DATA_3D.ROOT_DIR,
                        label_file=cfg.DATA.TEST_FILE,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    shuffle=False,
                    num_workers=num_workers,
                        dis_threshold=cfg.DATA_3D.DIS_THRESHOLD,
                        cache_dir=cfg.PATH.CACHE_DIR
                )
                
            else:
                # 首先尝试使用get_loader函数，它会自动处理训练/验证/测试集划分
                training_generator, val_generator, test_generator = get_loader(
                    root_dir=cfg.DATA_3D.ROOT_DIR,
                    label_file=cfg.DATA_3D.LABEL_FILE,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    shuffle=True,
                    num_workers=num_workers,
                    dis_threshold=cfg.DATA_3D.DIS_THRESHOLD,
                    cache_dir=cfg.PATH.CACHE_DIR
                )
                print("成功使用get_loader函数加载数据并划分训练/验证/测试集")
            
        except Exception as e:
            print(f"使用预划分数据集失败: {str(e)}，尝试备用方法")
            
            # 备用方法 - 直接加载标签文件
            if cfg.DATA_3D.LABEL_FILE:
                # 如果有标签文件，使用预计算图数据
                training_generator = get_precomputed_graph_dataloader(
                    cfg.DATA_3D.ROOT_DIR, 
                    cfg.DATA_3D.LABEL_FILE,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    shuffle=True,
                    num_workers=num_workers
                )
            else:
                # 直接从3D结构数据创建
                training_generator = get_protein_ligand_dataloader(
                    cfg.DATA_3D.ROOT_DIR,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    shuffle=True,
                    num_workers=num_workers,
                    dis_threshold=cfg.DATA_3D.DIS_THRESHOLD,
                    cache_dir=cfg.PATH.CACHE_DIR
                )
                
            # 简化实现，使用相同的数据集作为验证集和测试集
            val_generator = training_generator
            test_generator = training_generator
            print("使用备用方法加载数据（未划分训练/验证/测试集）")
    
    # 初始化Comet ML
    if cfg.COMET.USE and comet_support:
        # 检查API密钥
        api_key = os.environ.get("COMET_API_KEY")
        if not api_key:
            print("警告: 未找到COMET_API_KEY环境变量。Comet ML可能无法正常工作。")
            print("请设置环境变量: export COMET_API_KEY='您的密钥'")
        
        try:
            experiment = Experiment(
                api_key=api_key,  # 添加API密钥参数
                project_name=cfg.COMET.PROJECT,
                workspace=cfg.COMET.WORKSPACE,
                auto_output_logging="simple",
                log_graph=True,
                log_code=False,
                log_git_metadata=False,
                log_git_patch=False,
                auto_param_logging=False,
                auto_metric_logging=False
            )
            
            # 记录超参数
            experiment.log_parameters(dict(cfg))
            
            # 添加标签
            if cfg.COMET.TAG is not None:
                experiment.add_tag(cfg.COMET.TAG)
            
            # 设置实验名称
            experiment.set_name(f"{args.data}_{suffix}")
            
            print(f"Comet ML初始化成功: 项目 '{cfg.COMET.PROJECT}', 工作区 '{cfg.COMET.WORKSPACE}'")
        except Exception as e:
            print(f"Comet ML初始化失败: {str(e)}")
            print("将继续训练，但不会记录到Comet平台")
            experiment = None
    else:
        experiment = None
    
    # 创建模型
    if cfg.MODEL_TYPE == "DrugBAN":
        model = DrugBAN(**cfg).to(device)
    else:  # DrugBAN3D
        model = DrugBAN3D(**cfg).to(device)
    
    # 创建优化器
    weight_decay = cfg.TRAIN.WEIGHT_DECAY if hasattr(cfg.TRAIN, 'WEIGHT_DECAY') else 0.0
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=cfg.TRAIN.G_LEARNING_RATE,
                                weight_decay=weight_decay)
    
    # 记录优化器配置
    print(f"优化器: Adam, 学习率: {cfg.TRAIN.G_LEARNING_RATE}, L2正则化强度: {weight_decay}")
    
    # 标签平滑
    label_smoothing = cfg.TRAIN.LABEL_SMOOTHING if hasattr(cfg.TRAIN, 'LABEL_SMOOTHING') else 0.0
    if label_smoothing > 0:
        print(f"启用标签平滑，参数值: {label_smoothing}")
    
    # 类别权重
    pos_weight = None
    if hasattr(cfg.TRAIN, 'CLASS_WEIGHT') and cfg.TRAIN.CLASS_WEIGHT:
        # 计算数据集中的正负样本比例
        train_labels = np.array([sample[1] for batch in training_generator for sample in batch])
        pos_ratio = train_labels.mean()
        neg_ratio = 1 - pos_ratio
        pos_weight = torch.tensor([neg_ratio / pos_ratio * cfg.TRAIN.CLASS_WEIGHT]).to(device)
        print(f"启用类别加权，正样本权重: {pos_weight.item():.3f}")
    elif hasattr(cfg.TRAIN, 'POS_WEIGHT') and cfg.TRAIN.POS_WEIGHT:
        pos_weight = torch.tensor([cfg.TRAIN.POS_WEIGHT]).to(device)
        print(f"启用类别加权，正样本权重: {pos_weight.item():.3f}")
    else:
        # 分析数据集计算权重
        try:
            if hasattr(train_df, 'label'):
                pos_ratio = train_df['label'].mean() 
                if pos_ratio < 0.5:
                    pos_weight = torch.tensor([2.125]).to(device)  # 根据数据集不平衡程度设置
                    print(f"启用类别加权，正样本权重: {pos_weight.item():.3f}")
        except:
            pass
    
    # 确保结果目录正确设置
    if args.output_dir:
        cfg.RESULT.OUTPUT_DIR = args.output_dir
        # 确保目录结构存在
        ensure_output_dir_structure(cfg.RESULT.OUTPUT_DIR)
        
    # 梯度裁剪
    if hasattr(cfg.TRAIN, 'GRADIENT_CLIP_NORM') and cfg.TRAIN.GRADIENT_CLIP_NORM > 0:
        print(f"启用梯度裁剪，阈值：{cfg.TRAIN.GRADIENT_CLIP_NORM}")
        
    # 模型保存方式
    if hasattr(cfg.RESULT, 'USE_STATE_DICT') and cfg.RESULT.USE_STATE_DICT:
        print("使用state_dict保存模型，避免完整对象复制")
    
    # 学习率调度器
    if hasattr(cfg.SOLVER, 'LR_SCHEDULER') and cfg.SOLVER.LR_SCHEDULER:
        print(f"使用{cfg.SOLVER.LR_SCHEDULER_TYPE}学习率调度器")
    
    # 创建训练器
    trainer = Trainer(
        model=model, 
        train_dataloader=training_generator,
        val_dataloader=val_generator,
        test_dataloader=test_generator,
        config=cfg
    )
    
    # 设置experiment
    trainer.experiment = experiment
    
    # 加载模型权重（如果指定了）
    if args.load is not None:
        print(f"从 {args.load} 加载模型权重")
        trainer.load_model(args.load)
    
    # 根据模式执行训练或测试
    if test_only:
        print("执行测试...")
        try:
            # 测试集评估
            auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, threshold, precision = trainer.test(dataloader="test")
            
            # 输出测试结果
            print("\n======== 测试结果 ========")
            print(f"AUROC: {auroc:.4f}")
            print(f"AUPRC: {auprc:.4f}")
            print(f"F1分数: {f1:.4f}")
            print(f"敏感度: {sensitivity:.4f}")
            print(f"特异度: {specificity:.4f}")
            print(f"准确率: {accuracy:.4f}")
            print(f"精确度: {precision:.4f}")
            print(f"最佳阈值: {threshold:.4f}")
            print(f"测试损失: {test_loss:.4f}")
            
            # 将详细测试结果写入文件
            detailed_results_file = os.path.join(cfg.RESULT.OUTPUT_DIR, "detailed_test_results.txt")
            with open(detailed_results_file, "w") as f:
                f.write(f"测试时间: {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型文件: {args.load}\n\n")
                f.write("======== 测试结果 ========\n")
                f.write(f"AUROC: {auroc:.6f}\n")
                f.write(f"AUPRC: {auprc:.6f}\n")
                f.write(f"F1分数: {f1:.6f}\n")
                f.write(f"敏感度: {sensitivity:.6f}\n")
                f.write(f"特异度: {specificity:.6f}\n")
                f.write(f"准确率: {accuracy:.6f}\n")
                f.write(f"精确度: {precision:.6f}\n")
                f.write(f"最佳阈值: {threshold:.6f}\n")
                f.write(f"测试损失: {test_loss:.6f}\n")
            
            print(f"详细测试结果已保存到: {detailed_results_file}")
            
            # 如果配置了Comet ML，记录测试指标
            if experiment:
                experiment.log_metric("test_auroc", auroc)
                experiment.log_metric("test_auprc", auprc)
                experiment.log_metric("test_f1", f1)
                experiment.log_metric("test_sensitivity", sensitivity)
                experiment.log_metric("test_specificity", specificity)
                experiment.log_metric("test_accuracy", accuracy)
                experiment.log_metric("test_precision", precision)
                experiment.log_metric("test_threshold", threshold)
                experiment.log_metric("test_loss", test_loss)
                
            return {
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "precision": precision,
                "threshold": threshold,
                "test_loss": test_loss
            }
            
        except Exception as e:
            print(f"测试过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"开始训练...")
        # 执行训练并获取测试结果
        test_metrics = trainer.train()
        
        return test_metrics


if __name__ == "__main__":
    main()
