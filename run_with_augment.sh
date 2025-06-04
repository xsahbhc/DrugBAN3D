#!/bin/bash

# 这个脚本使用数据增强后的图缓存进行模型训练

# 设置参数
CACHE_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs"  # 原始缓存目录
AUGMENT_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs_augmented"  # 增强后的缓存目录
LABEL_FILE="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/labels.csv"  # 标签文件
TRAIN_LABEL="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/train_stratified.csv"  # 训练集标签
AUGMENTED_LABELS_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/augmented_labels"  # 增强后的标签目录
CONFIG_FILE="configs/DrugBAN3D_cached.yaml"  # 配置文件
RESULT_DIR="result/DrugBAN3D_augmented_$(date +%Y%m%d_%H%M%S)"  # 结果目录
LOG_FILE="runlog/run_augmented_$(date +%Y%m%d_%H%M%S).log"  # 日志文件
FORCE_REGENERATE=false  # 是否强制重新生成增强数据

# 为正样本生成更多的增强样本，为负样本生成较少的增强样本
# 这有助于平衡正负样本比例
POS_AUGMENT_COUNT=3  # 每个正样本增强3次
NEG_AUGMENT_COUNT=1  # 每个负样本增强1次
NUM_WORKERS=16  # 并行处理的工作进程数

# 确保目录存在
mkdir -p runlog

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --force|-f)
            FORCE_REGENERATE=true
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --force, -f    强制重新生成增强数据"
            echo "  --help, -h     显示此帮助信息"
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
echo "DrugBAN3D 使用数据增强训练"
echo "======================================================"
echo "开始时间: $(date)"
echo "原始缓存目录: $CACHE_DIR"
echo "增强缓存目录: $AUGMENT_DIR"
echo "原始标签文件: $LABEL_FILE"
echo "训练标签文件: $TRAIN_LABEL"
echo "增强标签目录: $AUGMENTED_LABELS_DIR"
echo "配置文件: $CONFIG_FILE"
echo "结果目录: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo "======================================================"

# 检查是否需要强制重新生成数据
if [ "$FORCE_REGENERATE" = true ]; then
    echo "强制重新生成增强数据..."
    rm -rf "$AUGMENT_DIR" "$AUGMENTED_LABELS_DIR"
fi

# 检查增强的缓存目录和标签目录是否存在
SKIP_AUGMENTATION=false
if [ -d "$AUGMENT_DIR" ] && [ -d "$AUGMENTED_LABELS_DIR" ] && [ -f "$AUGMENTED_LABELS_DIR/augmented_labels.csv" ]; then
    echo "检测到已存在的增强数据和标签目录:"
    echo "- 增强数据目录: $AUGMENT_DIR"
    echo "- 增强标签目录: $AUGMENTED_LABELS_DIR"
    echo "跳过数据增强步骤，直接进行训练..."
    SKIP_AUGMENTATION=true
fi

# 只有在需要时才执行数据增强步骤
if [ "$SKIP_AUGMENTATION" = false ]; then
    # 步骤1: 使用整合脚本执行数据增强和标签生成
    echo "执行数据增强与标签生成..."
    python augment_data_with_labels.py \
        --cache_dir $CACHE_DIR \
        --output_dir $AUGMENT_DIR \
        --label_file $LABEL_FILE \
            --train_label $TRAIN_LABEL \
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

# 步骤2: 修改配置文件，指向增强后的缓存目录和标签文件
TMP_CONFIG="${CONFIG_FILE%.*}_augmented.yaml"
cp $CONFIG_FILE $TMP_CONFIG

# 更新配置中的缓存目录和结果目录
sed -i "s|CACHE_DIR:.*|CACHE_DIR: \"${AUGMENT_DIR}\"|g" $TMP_CONFIG
sed -i "s|RESULT_DIR:.*|RESULT_DIR: \"${RESULT_DIR}\"|g" $TMP_CONFIG

# 更新DATA_3D部分的配置，确保使用分割后的标签文件
if grep -q "DATA_3D:" $TMP_CONFIG; then
    # 更新DATA_3D部分的标签文件配置
    sed -i "/DATA_3D:/,/^[a-zA-Z]/ s|LABEL_FILE:.*|LABEL_FILE: \"${AUGMENTED_LABELS_DIR}/train_augmented.csv\"|g" $TMP_CONFIG
    # 添加验证集和测试集文件路径
    if ! grep -q "VAL_FILE:" $TMP_CONFIG; then
        sed -i "/DATA_3D:/,/^[a-zA-Z]/ s|LABEL_FILE:.*|LABEL_FILE: \"${AUGMENTED_LABELS_DIR}/train_augmented.csv\"\n  VAL_FILE: \"${AUGMENTED_LABELS_DIR}/valid_augmented.csv\"\n  TEST_FILE: \"${AUGMENTED_LABELS_DIR}/test_augmented.csv\"|g" $TMP_CONFIG
    fi
fi

# 添加增强标签文件配置
if grep -q "DATA:" $TMP_CONFIG; then
    # 如果DATA部分已存在，更新里面的文件路径
    # 使用分开的训练/验证/测试标签文件，确保正确划分数据集
    sed -i "/DATA:/,/^[a-zA-Z]/ s|TRAIN_FILE:.*|TRAIN_FILE: \"${AUGMENTED_LABELS_DIR}/train_augmented.csv\"|g" $TMP_CONFIG
    sed -i "/DATA:/,/^[a-zA-Z]/ s|VAL_FILE:.*|VAL_FILE: \"${AUGMENTED_LABELS_DIR}/valid_augmented.csv\"|g" $TMP_CONFIG
    sed -i "/DATA:/,/^[a-zA-Z]/ s|TEST_FILE:.*|TEST_FILE: \"${AUGMENTED_LABELS_DIR}/test_augmented.csv\"|g" $TMP_CONFIG
else
    # 如果DATA部分不存在，添加完整的DATA配置
    echo "DATA:" >> $TMP_CONFIG
    echo "  TRAIN_FILE: \"${AUGMENTED_LABELS_DIR}/train_augmented.csv\"" >> $TMP_CONFIG
    echo "  VAL_FILE: \"${AUGMENTED_LABELS_DIR}/valid_augmented.csv\"" >> $TMP_CONFIG
    echo "  TEST_FILE: \"${AUGMENTED_LABELS_DIR}/test_augmented.csv\"" >> $TMP_CONFIG
fi

echo "已创建临时配置文件: $TMP_CONFIG"
echo "已更新缓存目录为: $AUGMENT_DIR"
echo "已更新结果目录为: $RESULT_DIR"
echo "已更新标签文件为增强后的标签文件"

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

echo "======================================================"
echo "训练完成！"
echo "结果保存在: $RESULT_DIR"
echo "完成时间: $(date)"
echo "======================================================"

# 清理临时文件
rm $TMP_CONFIG 