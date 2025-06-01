#!/bin/bash

# 设置使用GPU 0
export CUDA_VISIBLE_DEVICES=0
echo "已设置使用GPU 0 (CUDA_VISIBLE_DEVICES=0)"

# 设置Comet ML API密钥
export COMET_API_KEY="OAXLoYYmTBstjpbCKpaA81PLY"
echo "已设置Comet API密钥环境变量"

# 设置配置文件路径
CONFIG_FILE="configs/DrugBAN3D_cached.yaml"

# 设置3D数据相关路径
DATA_3D_ROOT="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_pdb"
DATA_3D_LABEL="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/labels.csv"

# 设置缓存目录
CACHE_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs"

# 设置交叉验证参数
CV_FOLDS=5  # 五折交叉验证

# 创建带有时间戳的输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="result/DrugBAN3D_5fold_cv_${TIMESTAMP}"
LOG_FILE="${BASE_OUTPUT_DIR}/cv_results.log"

# 检查数据路径是否存在
if [ ! -d "$DATA_3D_ROOT" ]; then
    echo "错误: 3D数据目录 '$DATA_3D_ROOT' 不存在！"
    echo "请确认数据路径或创建目录"
    exit 1
fi

# 如果标签文件不存在，检查其他可能的位置
if [ ! -f "$DATA_3D_LABEL" ]; then
    echo "警告: 标签文件 '$DATA_3D_LABEL' 不存在"
    # 尝试查找备选的标签文件
    ALT_LABEL_FILE="${DATA_3D_ROOT}/labels.csv"
    if [ -f "$ALT_LABEL_FILE" ]; then
        echo "找到备选标签文件: $ALT_LABEL_FILE"
        DATA_3D_LABEL=$ALT_LABEL_FILE
    else
        echo "错误: 未找到任何标签文件！请确认正确的标签文件路径。"
        exit 1
    fi
fi

# 检查缓存目录是否存在
if [ ! -d "$CACHE_DIR" ]; then
    echo "警告: 缓存目录 '$CACHE_DIR' 不存在"
    echo "创建缓存目录..."
    mkdir -p "$CACHE_DIR"
fi

# 检查是否有预处理的缓存文件
CACHE_FILES_COUNT=$(find "$CACHE_DIR" -name "*.pt" | wc -l)
echo "缓存目录中找到 $CACHE_FILES_COUNT 个预处理缓存文件"

# 确保输出目录存在
mkdir -p "$BASE_OUTPUT_DIR"

# 记录训练配置
echo "=== DrugBAN3D五折交叉验证配置 ===" | tee -a "$LOG_FILE"
echo "- 配置文件: $CONFIG_FILE" | tee -a "$LOG_FILE"
echo "- 数据目录: $DATA_3D_ROOT" | tee -a "$LOG_FILE"
echo "- 标签文件: $DATA_3D_LABEL" | tee -a "$LOG_FILE"
echo "- 缓存目录: $CACHE_DIR" | tee -a "$LOG_FILE"
echo "- 交叉验证折数: $CV_FOLDS" | tee -a "$LOG_FILE"
echo "- 输出目录: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "- 预处理缓存文件数: $CACHE_FILES_COUNT" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"

# 设置环境变量以防OOM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 保存每折的AUROC和AUPRC值
declare -a AUROC_VALUES
declare -a AUPRC_VALUES
declare -a ACC_VALUES
declare -a F1_VALUES

# 运行五折交叉验证
echo "开始五折交叉验证..." | tee -a "$LOG_FILE"
echo "开始时间: $(date)" | tee -a "$LOG_FILE"

for FOLD in $(seq 1 $CV_FOLDS); do
    echo "===============================" | tee -a "$LOG_FILE"
    echo "开始第 $FOLD 折验证" | tee -a "$LOG_FILE"
    
    # 设置当前折的输出目录
    FOLD_OUTPUT_DIR="${BASE_OUTPUT_DIR}/fold_${FOLD}"
    mkdir -p "$FOLD_OUTPUT_DIR"
    
    FOLD_LOG_FILE="${FOLD_OUTPUT_DIR}/train.log"
    
    echo "- 输出目录: $FOLD_OUTPUT_DIR" | tee -a "$LOG_FILE"
    echo "- 日志文件: $FOLD_LOG_FILE" | tee -a "$LOG_FILE"
    
    # 运行当前折的训练
    echo "正在运行第 $FOLD 折训练..." | tee -a "$LOG_FILE"
    
    python main.py \
        --cfg "$CONFIG_FILE" \
        --data human \
        --split stratified \
        --seed 42 \
        --output-dir "$FOLD_OUTPUT_DIR" \
        --use_3d \
        --data_3d_root "$DATA_3D_ROOT" \
        --data_3d_label "$DATA_3D_LABEL" \
        --tag "cv_fold_${FOLD}" \
        --cv_fold $FOLD \
        --cv_total_folds $CV_FOLDS 2>&1 | tee -a "$FOLD_LOG_FILE"
    
    TRAIN_EXIT_CODE=$?
    
    # 检查训练结果
    echo "第 $FOLD 折训练完成，退出代码: $TRAIN_EXIT_CODE" | tee -a "$LOG_FILE"
    
    # 提取当前折的结果
    if [ $TRAIN_EXIT_CODE -eq 0 ] && [ -d "$FOLD_OUTPUT_DIR" ]; then
        # 从测试结果文件中提取性能指标
        TEST_RESULTS="${FOLD_OUTPUT_DIR}/test_results.txt"
        if [ -f "$TEST_RESULTS" ]; then
            echo "提取第 $FOLD 折的测试结果..." | tee -a "$LOG_FILE"
            
            # 提取AUROC
            AUROC=$(grep "AUROC" "$TEST_RESULTS" | awk '{print $2}')
            AUROC_VALUES[$FOLD]=$AUROC
            
            # 提取AUPRC
            AUPRC=$(grep "AUPRC" "$TEST_RESULTS" | awk '{print $2}')
            AUPRC_VALUES[$FOLD]=$AUPRC
            
            # 提取Accuracy
            ACC=$(grep "Accuracy" "$TEST_RESULTS" | awk '{print $2}')
            ACC_VALUES[$FOLD]=$ACC
            
            # 提取F1
            F1=$(grep "F1" "$TEST_RESULTS" | awk '{print $2}')
            F1_VALUES[$FOLD]=$F1
            
            echo "第 $FOLD 折性能指标: AUROC=$AUROC, AUPRC=$AUPRC, Accuracy=$ACC, F1=$F1" | tee -a "$LOG_FILE"
        else
            echo "警告: 未找到第 $FOLD 折的测试结果文件" | tee -a "$LOG_FILE"
            AUROC_VALUES[$FOLD]="N/A"
            AUPRC_VALUES[$FOLD]="N/A"
            ACC_VALUES[$FOLD]="N/A"
            F1_VALUES[$FOLD]="N/A"
        fi
    else
        echo "警告: 第 $FOLD 折训练失败或输出目录不存在" | tee -a "$LOG_FILE"
        AUROC_VALUES[$FOLD]="N/A"
        AUPRC_VALUES[$FOLD]="N/A"
        ACC_VALUES[$FOLD]="N/A"
        F1_VALUES[$FOLD]="N/A"
    fi
    
    echo "第 $FOLD 折验证完成" | tee -a "$LOG_FILE"
done

# 计算平均性能
echo "===============================" | tee -a "$LOG_FILE"
echo "五折交叉验证结果汇总:" | tee -a "$LOG_FILE"

# 初始化累加器
AUROC_SUM=0
AUPRC_SUM=0
ACC_SUM=0
F1_SUM=0
VALID_FOLDS=0

# 打印每折结果并累加有效值
echo "各折性能指标:" | tee -a "$LOG_FILE"
for FOLD in $(seq 1 $CV_FOLDS); do
    echo "第 $FOLD 折: AUROC=${AUROC_VALUES[$FOLD]}, AUPRC=${AUPRC_VALUES[$FOLD]}, Accuracy=${ACC_VALUES[$FOLD]}, F1=${F1_VALUES[$FOLD]}" | tee -a "$LOG_FILE"
    
    # 检查是否为有效数值，只累加有效值
    if [[ "${AUROC_VALUES[$FOLD]}" != "N/A" ]]; then
        AUROC_SUM=$(echo "$AUROC_SUM + ${AUROC_VALUES[$FOLD]}" | bc -l)
        AUPRC_SUM=$(echo "$AUPRC_SUM + ${AUPRC_VALUES[$FOLD]}" | bc -l)
        ACC_SUM=$(echo "$ACC_SUM + ${ACC_VALUES[$FOLD]}" | bc -l)
        F1_SUM=$(echo "$F1_SUM + ${F1_VALUES[$FOLD]}" | bc -l)
        VALID_FOLDS=$((VALID_FOLDS + 1))
    fi
done

# 计算平均值（仅对有效折数进行平均）
if [ $VALID_FOLDS -gt 0 ]; then
    AUROC_AVG=$(echo "scale=4; $AUROC_SUM / $VALID_FOLDS" | bc -l)
    AUPRC_AVG=$(echo "scale=4; $AUPRC_SUM / $VALID_FOLDS" | bc -l)
    ACC_AVG=$(echo "scale=4; $ACC_SUM / $VALID_FOLDS" | bc -l)
    F1_AVG=$(echo "scale=4; $F1_SUM / $VALID_FOLDS" | bc -l)
    
    echo "平均性能 (有效折数: $VALID_FOLDS):" | tee -a "$LOG_FILE"
    echo "- 平均AUROC: $AUROC_AVG" | tee -a "$LOG_FILE"
    echo "- 平均AUPRC: $AUPRC_AVG" | tee -a "$LOG_FILE"
    echo "- 平均Accuracy: $ACC_AVG" | tee -a "$LOG_FILE"
    echo "- 平均F1: $F1_AVG" | tee -a "$LOG_FILE"
    
    # 保存平均结果到总结文件
    SUMMARY_FILE="${BASE_OUTPUT_DIR}/cv_summary.txt"
    echo "平均AUROC: $AUROC_AVG" > "$SUMMARY_FILE"
    echo "平均AUPRC: $AUPRC_AVG" >> "$SUMMARY_FILE"
    echo "平均Accuracy: $ACC_AVG" >> "$SUMMARY_FILE"
    echo "平均F1: $F1_AVG" >> "$SUMMARY_FILE"
    echo "有效折数: $VALID_FOLDS / $CV_FOLDS" >> "$SUMMARY_FILE"
    
    echo "结果摘要保存到: $SUMMARY_FILE" | tee -a "$LOG_FILE"
else
    echo "警告: 没有有效的结果可供计算平均值" | tee -a "$LOG_FILE"
fi

echo "===============================" | tee -a "$LOG_FILE"
echo "五折交叉验证完成时间: $(date)" | tee -a "$LOG_FILE"
echo "结果保存在: $BASE_OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE" 