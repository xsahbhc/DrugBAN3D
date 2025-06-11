#!/usr/bin/env python3
"""
综合分析所有实验结果
"""

import os
import re
from datetime import datetime

def parse_results_file(file_path):
    """解析results.txt文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        results = {}
        lines = content.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key == '最佳轮次':
                    results['best_epoch'] = int(value)
                elif key == 'AUROC':
                    results['auroc'] = float(value)
                elif key == 'AUPRC':
                    results['auprc'] = float(value)
                elif key == 'F1分数':
                    results['f1'] = float(value)
                elif key == '敏感度':
                    results['sensitivity'] = float(value)
                elif key == '特异度':
                    results['specificity'] = float(value)
                elif key == '准确率':
                    results['accuracy'] = float(value)
                elif key == '精确度':
                    results['precision'] = float(value)
        
        return results
    except Exception as e:
        print(f"解析 {file_path} 失败: {e}")
        return None

def parse_train_log_for_augment_info(log_path):
    """从训练日志中解析数据增强信息"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        info = {}
        
        # 提取增强策略
        pos_match = re.search(r'正样本增强数量: 每个样本(\d+)次', content)
        neg_match = re.search(r'负样本增强数量: 每个样本(\d+)次', content)
        
        if pos_match and neg_match:
            info['pos_augment'] = int(pos_match.group(1))
            info['neg_augment'] = int(neg_match.group(1))
        
        # 提取数据集大小
        total_match = re.search(r'增强后样本总数: (\d+)', content)
        pos_ratio_match = re.search(r'正样本数: \d+ \((\d+\.\d+)%\)', content)
        
        if total_match:
            info['total_samples'] = int(total_match.group(1))
        if pos_ratio_match:
            info['pos_ratio'] = float(pos_ratio_match.group(1))
        
        # 提取训练集信息
        train_match = re.search(r'训练集: 总样本 (\d+), 正样本率 (\d+\.\d+)%', content)
        if train_match:
            info['train_samples'] = int(train_match.group(1))
            info['train_pos_ratio'] = float(train_match.group(2))
        
        return info
    except Exception as e:
        print(f"解析训练日志 {log_path} 失败: {e}")
        return {}

def analyze_all_experiments():
    """分析所有实验结果"""
    result_dir = "result"
    experiments = []
    
    # 遍历所有实验目录
    for exp_dir in os.listdir(result_dir):
        if not exp_dir.startswith("DrugBAN3D_"):
            continue
            
        exp_path = os.path.join(result_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        # 解析实验信息
        exp_info = {
            'name': exp_dir,
            'timestamp': exp_dir.split('_')[-1] if '_' in exp_dir else '',
            'strategy': exp_dir.replace('DrugBAN3D_', '').rsplit('_', 1)[0]
        }
        
        # 读取results.txt
        results_file = os.path.join(exp_path, 'results.txt')
        if os.path.exists(results_file):
            results = parse_results_file(results_file)
            if results:
                exp_info.update(results)
        
        # 读取训练日志
        log_files = [
            os.path.join(exp_path, 'logs', 'train.log'),
            os.path.join(exp_path, 'train.log')
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                augment_info = parse_train_log_for_augment_info(log_file)
                exp_info.update(augment_info)
                break
        
        # 只包含有结果的实验
        if 'auroc' in exp_info:
            experiments.append(exp_info)
    
    return experiments

def print_experiment_analysis():
    """打印实验分析结果"""
    experiments = analyze_all_experiments()
    
    # 按AUROC排序
    experiments.sort(key=lambda x: x.get('auroc', 0), reverse=True)
    
    print("=" * 100)
    print("DrugBAN3D 实验结果综合分析")
    print("=" * 100)
    
    print(f"\n📊 总实验数量: {len(experiments)}")
    
    # 显示前5名最佳结果
    print("\n🏆 TOP 5 最佳实验结果:")
    print("-" * 100)
    print(f"{'排名':<4} {'实验名称':<35} {'AUROC':<8} {'AUPRC':<8} {'F1':<8} {'轮次':<6} {'增强策略':<15}")
    print("-" * 100)
    
    for i, exp in enumerate(experiments[:5]):
        strategy = f"正{exp.get('pos_augment', '?')}负{exp.get('neg_augment', '?')}" if 'pos_augment' in exp else "未知"
        print(f"{i+1:<4} {exp['strategy']:<35} {exp.get('auroc', 0):<8.4f} {exp.get('auprc', 0):<8.4f} "
              f"{exp.get('f1', 0):<8.4f} {exp.get('best_epoch', 0):<6} {strategy:<15}")
    
    # 分析最佳实验
    if experiments:
        best_exp = experiments[0]
        print(f"\n🎯 最佳实验详细分析:")
        print(f"实验名称: {best_exp['name']}")
        print(f"策略类型: {best_exp['strategy']}")
        print(f"时间戳: {best_exp['timestamp']}")
        print(f"AUROC: {best_exp.get('auroc', 0):.4f}")
        print(f"AUPRC: {best_exp.get('auprc', 0):.4f}")
        print(f"F1分数: {best_exp.get('f1', 0):.4f}")
        print(f"最佳轮次: {best_exp.get('best_epoch', 0)}")
        print(f"敏感度: {best_exp.get('sensitivity', 0):.4f}")
        print(f"特异度: {best_exp.get('specificity', 0):.4f}")
        print(f"准确率: {best_exp.get('accuracy', 0):.4f}")
        print(f"精确度: {best_exp.get('precision', 0):.4f}")
        
        if 'pos_augment' in best_exp:
            print(f"\n📈 数据增强配置:")
            print(f"正样本增强: {best_exp['pos_augment']}次")
            print(f"负样本增强: {best_exp['neg_augment']}次")
            print(f"训练集大小: {best_exp.get('train_samples', '未知')}")
            print(f"训练集正样本比例: {best_exp.get('train_pos_ratio', '未知'):.2f}%")
    
    # 分析增强策略效果
    print(f"\n📊 增强策略效果分析:")
    print("-" * 80)
    print(f"{'策略':<15} {'实验数':<8} {'平均AUROC':<12} {'最佳AUROC':<12} {'平均轮次':<10}")
    print("-" * 80)
    
    strategy_stats = {}
    for exp in experiments:
        if 'pos_augment' in exp and 'neg_augment' in exp:
            strategy = f"正{exp['pos_augment']}负{exp['neg_augment']}"
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            strategy_stats[strategy].append(exp)
    
    for strategy, exps in strategy_stats.items():
        avg_auroc = sum(exp.get('auroc', 0) for exp in exps) / len(exps)
        max_auroc = max(exp.get('auroc', 0) for exp in exps)
        avg_epoch = sum(exp.get('best_epoch', 0) for exp in exps) / len(exps)
        print(f"{strategy:<15} {len(exps):<8} {avg_auroc:<12.4f} {max_auroc:<12.4f} {avg_epoch:<10.1f}")
    
    # 分析最差实验
    if len(experiments) > 1:
        worst_exp = experiments[-1]
        print(f"\n❌ 最差实验分析:")
        print(f"实验名称: {worst_exp['name']}")
        print(f"AUROC: {worst_exp.get('auroc', 0):.4f}")
        print(f"AUPRC: {worst_exp.get('auprc', 0):.4f}")
        print(f"最佳轮次: {worst_exp.get('best_epoch', 0)}")
        if 'pos_augment' in worst_exp:
            print(f"增强策略: 正{worst_exp['pos_augment']}负{worst_exp['neg_augment']}")
            print(f"训练集大小: {worst_exp.get('train_samples', '未知')}")
    
    # 关键发现总结
    print(f"\n🔍 关键发现:")
    if experiments:
        best_auroc = experiments[0].get('auroc', 0)
        worst_auroc = experiments[-1].get('auroc', 0) if len(experiments) > 1 else 0
        print(f"1. 最佳AUROC: {best_auroc:.4f}, 最差AUROC: {worst_auroc:.4f}, 差距: {best_auroc-worst_auroc:.4f}")
        
        # 分析早停轮次
        epochs = [exp.get('best_epoch', 0) for exp in experiments if exp.get('best_epoch', 0) > 0]
        if epochs:
            avg_epoch = sum(epochs) / len(epochs)
            print(f"2. 平均最佳轮次: {avg_epoch:.1f}, 范围: {min(epochs)}-{max(epochs)}")
        
        # 分析过拟合问题
        early_stop_exps = [exp for exp in experiments if exp.get('best_epoch', 0) < 25]
        if early_stop_exps:
            print(f"3. 过早停止实验数量: {len(early_stop_exps)}/{len(experiments)} (可能过拟合)")

if __name__ == "__main__":
    print_experiment_analysis()
