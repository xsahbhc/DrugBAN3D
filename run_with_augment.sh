#!/bin/bash

# 这个脚本使用数据增强后的图缓存进行模型训练或测试

# 设置参数
CACHE_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs"  # 原始缓存目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)  # 生成时间戳用于创建唯一目录名

# 改进的目录结构
DATA_BASE_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/data"
mkdir -p "${DATA_BASE_DIR}/original"
mkdir -p "${DATA_BASE_DIR}/augmented/graphs"
mkdir -p "${DATA_BASE_DIR}/augmented/labels"

# 增强数据保存目录
AUGMENT_DIR="${DATA_BASE_DIR}/augmented/graphs/${TIMESTAMP}"
AUGMENTED_LABELS_DIR="${DATA_BASE_DIR}/augmented/labels/${TIMESTAMP}"

# 标签文件路径
LABEL_FILE="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/labels.csv"  # 标签文件
TRAIN_LABEL="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/train_stratified.csv"  # 训练集标签
VAL_LABEL="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/val_stratified.csv"  # 验证集标签
TEST_LABEL="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/test_stratified.csv"  # 测试集标签（不增强）

CONFIG_FILE="configs/DrugBAN3D_cached.yaml"  # 配置文件

# 使用时间戳创建唯一的结果目录
BASE_RESULT_DIR="result/DrugBAN3D_balanced_augment_${TIMESTAMP}"  # 平衡增强版本

# 定义更清晰的子目录结构
MODELS_DIR="${BASE_RESULT_DIR}/models"                     # 模型文件目录
METRICS_DIR="${BASE_RESULT_DIR}/metrics"                   # 指标结果目录
CONFIG_DIR="${BASE_RESULT_DIR}/config"                     # 配置文件目录
LOGS_DIR="${BASE_RESULT_DIR}/logs"                         # 日志目录

# 创建目录结构
mkdir -p $MODELS_DIR $METRICS_DIR $CONFIG_DIR $LOGS_DIR

RESULT_DIR=$BASE_RESULT_DIR   # 保持向后兼容
LOG_FILE="${LOGS_DIR}/train.log"  # 日志文件

FORCE_REGENERATE=true  # 默认强制重新生成增强数据
TRAIN_MODEL=true  # 默认执行训练


POS_AUGMENT_COUNT=1 
NEG_AUGMENT_COUNT=0  
NUM_WORKERS=16 


# 确保基础目录存在
mkdir -p runlog

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --force|-f)
            FORCE_REGENERATE=true
            shift
            ;;
        --reuse|-r)
            # 添加新选项，允许重用最近一次的增强数据
            FORCE_REGENERATE=false
            REUSE_LATEST=true
            shift
            ;;
        --pos_augment_count)
            POS_AUGMENT_COUNT="$2"
            shift 2
            ;;
        --neg_augment_count)
            NEG_AUGMENT_COUNT="$2"
            shift 2
            ;;
        --conservative)
            # 保守策略：正1负0
            POS_AUGMENT_COUNT=1
            NEG_AUGMENT_COUNT=0
            echo "使用保守增强策略: 正1负0"
            shift
            ;;
        --aggressive)
            # 激进策略：正3负1 (不推荐，容易过拟合)
            POS_AUGMENT_COUNT=3
            NEG_AUGMENT_COUNT=1
            echo "警告: 使用激进增强策略 (正3负1)，可能导致过拟合"
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --force, -f                    强制重新生成增强数据（默认行为）"
            echo "  --reuse, -r                    重用最近一次生成的增强数据，不生成新数据"
            echo "  --pos_augment_count <数量>     指定正样本增强次数（支持小数，如0.5表示50%样本增强1次）"
            echo "  --neg_augment_count <数量>     指定负样本增强次数（支持小数，如0.5表示50%样本增强1次）"
            echo "  --conservative                 使用保守策略 (正1负0，推荐)"
            echo "  --aggressive                   使用激进策略 (正3负1，不推荐)"
            echo "  --help, -h                     显示此帮助信息"
            echo ""
            echo "默认使用精细化渐进式增强策略 (正1负0 + 渐进式)"
            exit 0
            ;;
        *)
            echo "未知选项: $key"
            echo "使用 --help 获取帮助"
            exit 1
            ;;
    esac
done

echo "======================================================"
echo "DrugBAN3D 平衡增强训练脚本"
echo "======================================================"
echo "正样本增强次数: $POS_AUGMENT_COUNT (轻度增强：50%样本增强1次)"
echo "负样本增强次数: $NEG_AUGMENT_COUNT (轻度增强：50%样本增强1次)"
echo "======================================================"
echo "轻度增强策略特点:"
echo "- 分布保持: 正负样本都轻度增强，保持32:68原始分布"
echo "- 高质量增强: 使用最保守和安全的增强方法"
echo "- 优化权重: gentle_rotate 70%, thermal_vibration 30%"
echo "- 移除激进方法: 只保留最安全的旋转和振动方法"
echo "- 超保守参数: 旋转±1度, 振动0.002Å"
echo "- 预期数据集增长: 50% (原始10436 → ~15654样本)"
echo "- 预期分布: 训练集32:68, 验证集32:68, 测试集32:68"
echo "======================================================"

# 显示当前配置文件的关键参数预览
echo "=== 当前配置文件预览 ==="
echo "配置文件: $CONFIG_FILE"
if [ -f "$CONFIG_FILE" ]; then
    echo "关键参数:"
    echo "  训练参数:"
    grep -E "BATCH_SIZE|G_LEARNING_RATE|D_LEARNING_RATE|WEIGHT_DECAY" "$CONFIG_FILE" | sed 's/^/    /'
    echo "  模型参数:"
    grep -E "HIDDEN_DIM|NUM_LAYERS|DROPOUT" "$CONFIG_FILE" | sed 's/^/    /'
    echo "  损失函数:"
    grep -E "USE_FOCAL_LOSS|FOCAL_LOSS_GAMMA|FOCAL_LOSS_ALPHA" "$CONFIG_FILE" | sed 's/^/    /'
else
    echo "  警告: 配置文件不存在!"
fi
echo "======================================================"

# 如果需要重用最近的增强数据，查找最新目录
if [ "$REUSE_LATEST" = true ] && [ "$FORCE_REGENERATE" = false ]; then
    # 查找最新的增强数据子目录
    LATEST_AUGMENT_SUB_DIR=$(find "${DATA_BASE_DIR}/augmented/graphs" -maxdepth 1 -type d -not -path "${DATA_BASE_DIR}/augmented/graphs" -printf "%T@ %f\n" 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)
    
    if [ -n "$LATEST_AUGMENT_SUB_DIR" ]; then
        # 使用最新子目录
        TIMESTAMP="$LATEST_AUGMENT_SUB_DIR"
        AUGMENT_DIR="${DATA_BASE_DIR}/augmented/graphs/${TIMESTAMP}"
        AUGMENTED_LABELS_DIR="${DATA_BASE_DIR}/augmented/labels/${TIMESTAMP}"
        
        echo "找到最新的增强数据和标签目录:"
        echo "- 增强数据目录: $AUGMENT_DIR"
        echo "- 增强标签目录: $AUGMENTED_LABELS_DIR"
        echo "将使用这些现有目录而不重新生成数据"
    else
        echo "未找到现有的增强数据目录，将重新生成数据"
        FORCE_REGENERATE=true
    fi
fi

# 仅创建和准备必要的目录
if [ "$FORCE_REGENERATE" = true ] || [ "$TRAIN_MODEL" = true ]; then
    mkdir -p "$AUGMENT_DIR"
    mkdir -p "$AUGMENTED_LABELS_DIR"
    
    echo "======================================================"
fi

# 只有在需要时才执行数据增强步骤
if [ "$FORCE_REGENERATE" = true ] && [ "$TRAIN_MODEL" = true ]; then
    # 步骤1: 使用整合脚本执行数据增强和标签生成
    echo "执行数据增强与标签生成..."
    python augment_data_with_labels.py \
        --cache_dir $CACHE_DIR \
        --output_dir $AUGMENT_DIR \
        --label_file $LABEL_FILE \
        --train_label $TRAIN_LABEL \
        --val_label $VAL_LABEL \
        --test_label $TEST_LABEL \
        --labels_output_dir $AUGMENTED_LABELS_DIR \
        --pos_augment $POS_AUGMENT_COUNT \
        --neg_augment $NEG_AUGMENT_COUNT \
        --num_workers $NUM_WORKERS | tee -a $LOG_FILE
    
    # 检查数据增强和标签生成是否成功
    if [ $? -ne 0 ] || [ ! -f "$AUGMENTED_LABELS_DIR/augmented_labels.csv" ]; then
        echo "数据增强或标签生成失败！请检查错误信息。"
        exit 1
    fi
    echo "数据增强与标签生成完成！"
fi

# 仅当需要训练时执行
if [ "$TRAIN_MODEL" = true ]; then
    # 步骤2: 修改配置文件，指向增强后的缓存目录和标签文件
    TMP_CONFIG="${CONFIG_DIR}/DrugBAN3D_augmented_${TIMESTAMP}.yaml"
    cp $CONFIG_FILE $TMP_CONFIG
    
    # 更新配置中的缓存目录和结果目录
    sed -i "s|CACHE_DIR:.*|CACHE_DIR: \"${AUGMENT_DIR}\"|g" $TMP_CONFIG
    
    # 确保结果目录设置正确 - 无论RESULT.OUTPUT_DIR还是PATH.RESULT_DIR都更新
    sed -i "s|OUTPUT_DIR:.*|OUTPUT_DIR: \"${RESULT_DIR}\"|g" $TMP_CONFIG
    sed -i "s|RESULT_DIR:.*|RESULT_DIR: \"${RESULT_DIR}\"|g" $TMP_CONFIG
    
    # 更新DATA_3D部分的配置，确保使用分割后的标签文件
    if grep -q "DATA_3D:" $TMP_CONFIG; then
        # 更新DATA_3D部分的标签文件配置
        sed -i "/DATA_3D:/,/^[a-zA-Z]/ s|LABEL_FILE:.*|LABEL_FILE: \"${AUGMENTED_LABELS_DIR}/train_augmented.csv\"|g" $TMP_CONFIG
    fi
    
    # 更新或添加数据文件路径
    if grep -q "DATA:" $TMP_CONFIG; then
        # 如果DATA部分已存在，更新里面的文件路径
        # 使用增强的训练/验证集，但保持原始测试集
        sed -i "/DATA:/,/^[a-zA-Z]/ s|TRAIN_FILE:.*|TRAIN_FILE: \"${AUGMENTED_LABELS_DIR}/train_augmented.csv\"|g" $TMP_CONFIG
        sed -i "/DATA:/,/^[a-zA-Z]/ s|VAL_FILE:.*|VAL_FILE: \"${AUGMENTED_LABELS_DIR}/valid_augmented.csv\"|g" $TMP_CONFIG
        sed -i "/DATA:/,/^[a-zA-Z]/ s|TEST_FILE:.*|TEST_FILE: \"$TEST_LABEL\"|g" $TMP_CONFIG
    else
        # 如果DATA部分不存在，添加完整的DATA配置
        echo "DATA:" >> $TMP_CONFIG
        echo "  TRAIN_FILE: \"${AUGMENTED_LABELS_DIR}/train_augmented.csv\"" >> $TMP_CONFIG
        echo "  VAL_FILE: \"${AUGMENTED_LABELS_DIR}/valid_augmented.csv\"" >> $TMP_CONFIG
        echo "  TEST_FILE: \"$TEST_LABEL\"" >> $TMP_CONFIG
    fi
    
    echo "已创建临时配置文件: $TMP_CONFIG"
    echo "已更新缓存目录为: $AUGMENT_DIR"
    echo "已更新结果目录为: $RESULT_DIR"
    echo "已更新训练/验证标签文件为增强后的标签文件"
    echo "已保持测试集使用原始标签文件: $TEST_LABEL"
    
    # 步骤3: 显示详细配置信息并运行训练
    echo "======================================================"
    echo "开始使用增强数据训练模型..."
    echo "======================================================"

    # 显示详细的配置信息
    echo "=== 详细配置信息 ===" | tee -a $LOG_FILE
    echo "配置文件: $TMP_CONFIG" | tee -a $LOG_FILE
    echo "训练数据: ${AUGMENTED_LABELS_DIR}/train_augmented.csv" | tee -a $LOG_FILE
    echo "验证数据: ${AUGMENTED_LABELS_DIR}/valid_augmented.csv" | tee -a $LOG_FILE
    echo "测试数据: $TEST_LABEL" | tee -a $LOG_FILE
    echo "缓存目录: $AUGMENT_DIR" | tee -a $LOG_FILE
    echo "结果目录: $RESULT_DIR" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

    # 显示完整配置参数详览
    echo "=== 完整配置参数详览 ===" | tee -a $LOG_FILE
    if [ -f "$TMP_CONFIG" ]; then
        echo "📋 训练参数 (TRAIN):" | tee -a $LOG_FILE
        grep -A 15 "TRAIN:" "$TMP_CONFIG" | head -12 | sed 's/^/  /' | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE

        echo "🏗️ GNN配置:" | tee -a $LOG_FILE
        grep -A 5 "GNN:" "$TMP_CONFIG" | sed 's/^/  /' | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE

        echo "🔧 解码器配置:" | tee -a $LOG_FILE
        grep -A 8 "DECODER:" "$TMP_CONFIG" | sed 's/^/  /' | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE

        echo "📊 损失函数配置:" | tee -a $LOG_FILE
        grep -E "USE_FOCAL_LOSS|FOCAL_LOSS_GAMMA|FOCAL_LOSS_ALPHA|USE_BIAS_CORRECTION" "$TMP_CONFIG" | sed 's/^/  /' | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE

        echo "🌐 3D空间特征配置:" | tee -a $LOG_FILE
        grep -A 8 "SPATIAL_3D:" "$TMP_CONFIG" | sed 's/^/  /' | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE

        echo "⚙️ 求解器配置:" | tee -a $LOG_FILE
        grep -A 10 "SOLVER:" "$TMP_CONFIG" | sed 's/^/  /' | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE

        echo "⏹️ 早停配置:" | tee -a $LOG_FILE
        grep -E "USE_EARLY_STOPPING|EARLY_STOPPING_PATIENCE" "$TMP_CONFIG" | sed 's/^/  /' | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
    else
        echo "警告: 配置文件 $TMP_CONFIG 不存在!" | tee -a $LOG_FILE
    fi
    echo "=========================" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE

    # 显示训练开始时间
    echo "训练开始时间: $(date)" | tee -a $LOG_FILE
    echo "======================================================"

    python main.py --cfg $TMP_CONFIG --data bindingdb --split stratified --use_3d --use_augmented --output-dir $RESULT_DIR | tee -a $LOG_FILE
    
    # 检查训练是否成功
    TRAIN_EXIT_CODE=$?
    echo "训练完成时间: $(date)" | tee -a $LOG_FILE

    if [ $TRAIN_EXIT_CODE -ne 0 ]; then
        echo "模型训练失败！请检查错误信息。" | tee -a $LOG_FILE
        exit 1
    fi

    # 显示训练总结
    echo "======================================================"
    echo "=== 训练总结 ===" | tee -a $LOG_FILE
    echo "训练成功完成！" | tee -a $LOG_FILE
    echo "结果保存在: $RESULT_DIR" | tee -a $LOG_FILE
    echo "配置文件: $TMP_CONFIG" | tee -a $LOG_FILE
    echo "日志文件: $LOG_FILE" | tee -a $LOG_FILE

    # 查找并显示模型文件信息
    if [ -d "$RESULT_DIR" ]; then
        MODEL_FILES=$(find "$RESULT_DIR" -name "*.pth" -o -name "*.pt" | head -5)
        if [ -n "$MODEL_FILES" ]; then
            echo "保存的模型文件:" | tee -a $LOG_FILE
            echo "$MODEL_FILES" | sed 's/^/  /' | tee -a $LOG_FILE
        fi
    fi
    echo "======================================================"

    # 清理临时文件
    rm $TMP_CONFIG
fi

# 简洁的完成信息
echo "完成时间: $(date)"

# 如果存在结果文件，显示关键指标并评估性能
if [ -f "$RESULT_DIR/results.txt" ]; then
    echo "======================================================"
    echo "性能指标："
    grep -E "AUROC|AUPRC|F1分数|最佳轮次" "$RESULT_DIR/results.txt" | sed 's/^/  /'
    echo ""

    # 检查是否达到预期性能目标
    AUROC=$(grep "AUROC:" "$RESULT_DIR/results.txt" | awk '{print $2}' 2>/dev/null)
    AUPRC=$(grep "AUPRC:" "$RESULT_DIR/results.txt" | awk '{print $2}' 2>/dev/null)

    if [ -n "$AUROC" ] && [ -n "$AUPRC" ]; then
        echo "性能评估 (目标: AUROC > 0.89, AUPRC > 0.79):"
        if (( $(echo "$AUROC > 0.89" | bc -l 2>/dev/null || echo 0) )); then
            echo "  ✓ AUROC ($AUROC) 达到预期目标"
        else
            echo "  ✗ AUROC ($AUROC) 未达到预期目标"
        fi

        if (( $(echo "$AUPRC > 0.79" | bc -l 2>/dev/null || echo 0) )); then
            echo "  ✓ AUPRC ($AUPRC) 达到预期目标"
        else
            echo "  ✗ AUPRC ($AUPRC) 未达到预期目标"
        fi
    fi
    echo "======================================================"
fi