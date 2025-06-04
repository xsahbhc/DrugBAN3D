#!/bin/bash

# 这个脚本使用已经增强过的数据进行模型训练，不执行数据增强步骤

# 设置参数
CACHE_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs"  # 原始缓存目录
AUGMENT_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs_augmented"  # 增强后的缓存目录
LABEL_FILE="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/labels.csv"  # 标签文件
TRAIN_LABEL="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/train_stratified.csv"  # 训练集标签
AUGMENTED_LABELS_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/augmented_labels"  # 增强后的标签目录
CONFIG_FILE="configs/DrugBAN3D_cached.yaml"  # 配置文件
RESULT_DIR="result/DrugBAN3D_train_only_$(date +%Y%m%d_%H%M%S)"  # 结果目录
LOG_FILE="runlog/run_train_only_$(date +%Y%m%d_%H%M%S).log"  # 日志文件

# 确保目录存在
mkdir -p runlog

echo "======================================================"
echo "DrugBAN3D 仅执行训练（使用已有增强数据）"
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

# 检查必要的数据是否存在
if [ ! -d "$AUGMENT_DIR" ]; then
    echo "错误: 增强数据目录 $AUGMENT_DIR 不存在!"
    exit 1
fi

if [ ! -f "$AUGMENTED_LABELS_DIR/augmented_labels.csv" ]; then
    echo "错误: 增强标签文件 $AUGMENTED_LABELS_DIR/augmented_labels.csv 不存在!"
    exit 1
fi

# 修改配置文件，指向增强后的缓存目录和标签文件
TMP_CONFIG="${CONFIG_FILE%.*}_train_only.yaml"
cp $CONFIG_FILE $TMP_CONFIG

# 更新配置中的缓存目录和结果目录
sed -i "s|CACHE_DIR:.*|CACHE_DIR: \"${AUGMENT_DIR}\"|g" $TMP_CONFIG
sed -i "s|RESULT_DIR:.*|RESULT_DIR: \"${RESULT_DIR}\"|g" $TMP_CONFIG

# 更新数据目录
ROOT_DIR="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_pdb"  # 数据根目录

# 确保配置文件中DATA_3D部分被正确设置
if grep -q "DATA_3D:" $TMP_CONFIG; then
    # 更新DATA_3D部分的配置
    sed -i "/DATA_3D:/,/^[a-zA-Z]/ s|ROOT_DIR:.*|ROOT_DIR: \"${ROOT_DIR}\"|g" $TMP_CONFIG
    sed -i "/DATA_3D:/,/^[a-zA-Z]/ s|LABEL_FILE:.*|LABEL_FILE: \"${LABEL_FILE}\"|g" $TMP_CONFIG
    sed -i "/DATA_3D:/,/^[a-zA-Z]/ s|TRAIN_FILE:.*|TRAIN_FILE: \"${TRAIN_LABEL}\"|g" $TMP_CONFIG
    sed -i "/DATA_3D:/,/^[a-zA-Z]/ s|NUM_WORKERS:.*|NUM_WORKERS: 16|g" $TMP_CONFIG
    sed -i "/DATA_3D:/,/^[a-zA-Z]/ s|DIS_THRESHOLD:.*|DIS_THRESHOLD: 5.0|g" $TMP_CONFIG
else
    # 如果DATA_3D部分不存在，添加完整的配置
    echo "DATA_3D:" >> $TMP_CONFIG
    echo "  ROOT_DIR: \"${ROOT_DIR}\"" >> $TMP_CONFIG
    echo "  LABEL_FILE: \"${LABEL_FILE}\"" >> $TMP_CONFIG
    echo "  TRAIN_FILE: \"${TRAIN_LABEL}\"" >> $TMP_CONFIG
    echo "  DIS_THRESHOLD: 5.0" >> $TMP_CONFIG
    echo "  NUM_WORKERS: 16" >> $TMP_CONFIG
fi

# 添加增强标签文件配置
if grep -q "DATA:" $TMP_CONFIG; then
    # 如果DATA部分已存在，更新里面的文件路径
    sed -i "/DATA:/,/^[a-zA-Z]/ s|TRAIN_FILE:.*|TRAIN_FILE: \"${AUGMENTED_LABELS_DIR}/augmented_labels.csv\"|g" $TMP_CONFIG
    sed -i "/DATA:/,/^[a-zA-Z]/ s|VAL_FILE:.*|VAL_FILE: \"${AUGMENTED_LABELS_DIR}/augmented_labels.csv\"|g" $TMP_CONFIG
    sed -i "/DATA:/,/^[a-zA-Z]/ s|TEST_FILE:.*|TEST_FILE: \"${AUGMENTED_LABELS_DIR}/augmented_labels.csv\"|g" $TMP_CONFIG
else
    # 如果DATA部分不存在，添加完整的DATA配置
    echo "DATA:" >> $TMP_CONFIG
    echo "  TRAIN_FILE: \"${AUGMENTED_LABELS_DIR}/augmented_labels.csv\"" >> $TMP_CONFIG
    echo "  VAL_FILE: \"${AUGMENTED_LABELS_DIR}/augmented_labels.csv\"" >> $TMP_CONFIG
    echo "  TEST_FILE: \"${AUGMENTED_LABELS_DIR}/augmented_labels.csv\"" >> $TMP_CONFIG
fi

echo "已创建临时配置文件: $TMP_CONFIG"
echo "已更新缓存目录为: $AUGMENT_DIR"
echo "已更新结果目录为: $RESULT_DIR"
echo "已更新标签文件为增强后的标签文件"

# 运行训练
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