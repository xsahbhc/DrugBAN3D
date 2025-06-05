#!/bin/bash

# 这个脚本使用数据增强后的图缓存进行模型训练或测试

# 批次大小设置 - 已从32改为64
BATCH_SIZE=64
echo "注意：批次大小已调整为 ${BATCH_SIZE}，学习率相应增加"

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
RESULT_DIR="result/DrugBAN3D_augmented_${TIMESTAMP}"  # 结果目录
LOG_FILE="runlog/run_augmented_${TIMESTAMP}.log"  # 日志文件
FORCE_REGENERATE=true  # 默认强制重新生成增强数据
TRAIN_MODEL=true  # 默认执行训练
TEST_MODEL=true  # 默认执行测试
MODEL_PATH=""  # 测试时要加载的模型路径

# 为正样本生成更多的增强样本，为负样本生成较少的增强样本
# 这有助于平衡正负样本比例
POS_AUGMENT_COUNT=3  # 每个正样本增强2次
NEG_AUGMENT_COUNT=1  # 每个负样本增强1次
NUM_WORKERS=16  # 并行处理的工作进程数

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
        --test-only)
            # 仅测试模式
            TRAIN_MODEL=false
            TEST_MODEL=true
            shift
            ;;
        --train-only)
            # 仅训练模式
            TRAIN_MODEL=true
            TEST_MODEL=false
            shift
            ;;
        --model)
            # 指定要测试的模型路径
            MODEL_PATH="$2"
            shift
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --force, -f      强制重新生成增强数据（默认行为）"
            echo "  --reuse, -r      重用最近一次生成的增强数据，不生成新数据"
            echo "  --test-only      只执行测试，不进行训练"
            echo "  --train-only     只执行训练，不进行测试"
            echo "  --model PATH     指定要测试的模型路径（仅在--test-only模式下使用）"
            echo "  --help, -h       显示此帮助信息"
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
echo "DrugBAN3D 增强数据训练/测试脚本"
echo "======================================================"
echo "开始时间: $(date)"
echo "原始缓存目录: $CACHE_DIR"

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

# 如果只执行测试模式
if [ "$TRAIN_MODEL" = false ] && [ "$TEST_MODEL" = true ]; then
    if [ -z "$MODEL_PATH" ]; then
        echo "错误: 测试模式需要指定模型路径, 使用 --model 参数"
        exit 1
    fi
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "错误: 指定的模型文件不存在: $MODEL_PATH"
        exit 1
    fi
    
    echo "测试模式: 使用模型 $MODEL_PATH"
    
    # 修改配置文件
    TEST_CONFIG="${CONFIG_FILE%.*}_test_${TIMESTAMP}.yaml"
    cp $CONFIG_FILE $TEST_CONFIG
    
    # 更新配置指向正确的测试集
    sed -i "s|TEST_FILE:.*|TEST_FILE: \"$TEST_LABEL\"|g" $TEST_CONFIG
    
    # 创建测试结果目录
    TEST_RESULT_DIR="result/DrugBAN3D_test_${TIMESTAMP}"
    mkdir -p "$TEST_RESULT_DIR"
    
    # 更新配置中的结果目录
    sed -i "s|RESULT_DIR:.*|RESULT_DIR: \"${TEST_RESULT_DIR}\"|g" $TEST_CONFIG
    
    echo "开始测试..."
    python main.py --cfg $TEST_CONFIG --data bindingdb --split stratified --use_3d --test_only --load $MODEL_PATH | tee -a "runlog/test_${TIMESTAMP}.log"
    
    echo "测试完成! 结果保存在 $TEST_RESULT_DIR"
    rm $TEST_CONFIG
    
    exit 0
fi

# 仅创建和准备必要的目录
if [ "$FORCE_REGENERATE" = true ] || [ "$TRAIN_MODEL" = true ]; then
    mkdir -p "$AUGMENT_DIR"
    mkdir -p "$AUGMENTED_LABELS_DIR"
    
    echo "增强缓存目录: $AUGMENT_DIR"
    echo "原始标签文件: $LABEL_FILE"
    echo "训练标签文件: $TRAIN_LABEL"
    echo "验证标签文件: $VAL_LABEL"
    echo "测试标签文件: $TEST_LABEL"
    echo "增强标签目录: $AUGMENTED_LABELS_DIR"
    echo "配置文件: $CONFIG_FILE"
    echo "结果目录: $RESULT_DIR"
    echo "日志文件: $LOG_FILE"
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
    TMP_CONFIG="${CONFIG_FILE%.*}_augmented_${TIMESTAMP}.yaml"
    cp $CONFIG_FILE $TMP_CONFIG
    
    # 更新配置中的缓存目录和结果目录
    sed -i "s|CACHE_DIR:.*|CACHE_DIR: \"${AUGMENT_DIR}\"|g" $TMP_CONFIG
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
    
    # 调整早停和学习率设置
    sed -i "s|EARLY_STOPPING_PATIENCE:.*|EARLY_STOPPING_PATIENCE: 20|g" $TMP_CONFIG
    sed -i "/SOLVER:/,/^[a-zA-Z]/ s|LR_SCHEDULER_TYPE:.*|LR_SCHEDULER_TYPE: \"plateau\"|g" $TMP_CONFIG
    sed -i "/SOLVER:/,/^[a-zA-Z]/ s|BATCH_SIZE:.*|BATCH_SIZE: 64|g" $TMP_CONFIG
    
    echo "已创建临时配置文件: $TMP_CONFIG"
    echo "已更新缓存目录为: $AUGMENT_DIR"
    echo "已更新结果目录为: $RESULT_DIR"
    echo "已更新训练/验证标签文件为增强后的标签文件"
    echo "已保持测试集使用原始标签文件: $TEST_LABEL"
    echo "已调整早停耐心值为20轮"
    
    # 步骤3: 运行训练
    echo "======================================================"
    echo "开始使用增强数据训练模型..."
    echo "======================================================"
    
    python main.py --cfg $TMP_CONFIG --data bindingdb --split stratified --use_3d --use_augmented | tee -a $LOG_FILE
    
    # 检查训练是否成功
    if [ $? -ne 0 ]; then
        echo "模型训练失败！请检查错误信息。"
        exit 1
    fi
    
    # 清理临时文件
    rm $TMP_CONFIG
    
    echo "======================================================"
    echo "训练完成！"
    echo "结果保存在: $RESULT_DIR"
    echo "======================================================"
    
    # 记录最佳模型路径，用于后续测试
    BEST_MODEL_PATH="${RESULT_DIR}/final_result_"*"/model_best.pt"
    BEST_MODEL_PATH=$(ls $BEST_MODEL_PATH 2>/dev/null | head -n1)
    
    if [ -f "$BEST_MODEL_PATH" ]; then
        MODEL_PATH=$BEST_MODEL_PATH
        echo "找到最佳模型: $MODEL_PATH"
    else
        echo "警告: 无法找到最佳模型文件"
        TEST_MODEL=false
    fi
fi

# 如果需要测试，并且有可用的模型路径
if [ "$TEST_MODEL" = true ] && [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH" ]; then
    echo "======================================================"
    echo "开始使用原始测试集测试模型..."
    echo "======================================================"
    
    # 创建测试配置文件
    TEST_CONFIG="${CONFIG_FILE%.*}_test_${TIMESTAMP}.yaml"
    cp $CONFIG_FILE $TEST_CONFIG
    
    # 更新测试集路径
    sed -i "s|TEST_FILE:.*|TEST_FILE: \"$TEST_LABEL\"|g" $TEST_CONFIG
    
    # 创建测试结果目录
    TEST_RESULT_DIR="${RESULT_DIR}/test_results"
    mkdir -p "$TEST_RESULT_DIR"
    
    # 更新配置中的结果目录
    sed -i "s|RESULT_DIR:.*|RESULT_DIR: \"${TEST_RESULT_DIR}\"|g" $TEST_CONFIG
    
    # 运行测试
    python main.py --cfg $TEST_CONFIG --data bindingdb --split stratified --use_3d --test_only --load $MODEL_PATH | tee -a "${LOG_FILE%.log}_test.log"
    
    # 清理临时文件
    rm $TEST_CONFIG
    
    echo "======================================================"
    echo "测试完成！"
    echo "测试结果保存在: $TEST_RESULT_DIR"
    echo "======================================================"
fi

echo "======================================================"
echo "完成时间: $(date)"
echo "======================================================"

# 创建一个记录文件，列出此次训练使用的增强数据目录
echo "记录本次操作的数据路径..."
TRAINING_RECORD="runlog/training_record.txt"
echo "执行时间: $(date)" >> $TRAINING_RECORD
echo "增强数据目录: $AUGMENT_DIR" >> $TRAINING_RECORD
echo "增强标签目录: $AUGMENTED_LABELS_DIR" >> $TRAINING_RECORD

if [ "$TRAIN_MODEL" = true ]; then
    echo "训练结果目录: $RESULT_DIR" >> $TRAINING_RECORD
    echo "训练日志文件: $LOG_FILE" >> $TRAINING_RECORD
fi

if [ "$TEST_MODEL" = true ] && [ -n "$MODEL_PATH" ]; then
    echo "测试模型: $MODEL_PATH" >> $TRAINING_RECORD
    if [ -d "$TEST_RESULT_DIR" ]; then
        echo "测试结果目录: $TEST_RESULT_DIR" >> $TRAINING_RECORD
    fi
fi

echo "---------------------------------------------------" >> $TRAINING_RECORD
echo "操作记录已保存到: $TRAINING_RECORD" 