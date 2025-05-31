#!/bin/bash

# 设置Comet ML API密钥
# 请替换为您的实际密钥
export COMET_API_KEY="OAXLoYYmTBstjpbCKpaA81PLY"
echo "已设置Comet API密钥环境变量"

# 设置配置文件路径
CONFIG_FILE="configs/DrugBAN3D.yaml"

# 设置3D数据相关路径 - 更新为确认存在的路径
DATA_3D_ROOT="/home/work/workspace/shi_shaoqun/snap/3D_structure/train_pdb"
DATA_3D_LABEL="/home/work/workspace/shi_shaoqun/snap/3D_structure/labels.csv"

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

# 创建带有时间戳和优化标识的输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_OUTPUT_DIR="result/DrugBAN3D_optimized_BS64_CW_RES_CV5_${TIMESTAMP}"
mkdir -p "$MAIN_OUTPUT_DIR"

# 创建主日志文件
MAIN_LOG_FILE="${MAIN_OUTPUT_DIR}_run.log"

# 记录训练配置
echo "=== DrugBAN3D优化交叉验证训练配置 ===" | tee -a "$MAIN_LOG_FILE"
echo "- 配置文件: $CONFIG_FILE" | tee -a "$MAIN_LOG_FILE"
echo "- 数据目录: $DATA_3D_ROOT" | tee -a "$MAIN_LOG_FILE"
echo "- 标签文件: $DATA_3D_LABEL" | tee -a "$MAIN_LOG_FILE"
echo "- 输出目录: $MAIN_OUTPUT_DIR" | tee -a "$MAIN_LOG_FILE"
echo "- 批次大小: 64" | tee -a "$MAIN_LOG_FILE"
echo "- 正样本权重: 2.125" | tee -a "$MAIN_LOG_FILE"
echo "- GNN隐藏维度: 384" | tee -a "$MAIN_LOG_FILE"
echo "- GNN层数: 4" | tee -a "$MAIN_LOG_FILE"
echo "- 添加残差连接和改进Dropout" | tee -a "$MAIN_LOG_FILE"
echo "- 交叉验证折数: 5" | tee -a "$MAIN_LOG_FILE"
echo "=======================================" | tee -a "$MAIN_LOG_FILE"

# 设置交叉验证参数
CV_FOLDS=5
# 生成随机种子数组，保证每折使用不同随机种子
SEEDS=(42 66 123 789 456)

# 创建结果汇总表格
echo "折数,AUROC,AUPRC,F1,Sensitivity,Specificity,Accuracy" > "${MAIN_OUTPUT_DIR}/cv_results_summary.csv"

# 启动交叉验证
echo "开始5折交叉验证训练..." | tee -a "$MAIN_LOG_FILE"
echo "总训练时间可能较长，请耐心等待..." | tee -a "$MAIN_LOG_FILE"

for ((FOLD=1; FOLD<=CV_FOLDS; FOLD++)); do
    echo "开始第 $FOLD 折交叉验证..." | tee -a "$MAIN_LOG_FILE"
    
    # 为当前折创建输出目录
    FOLD_OUTPUT_DIR="${MAIN_OUTPUT_DIR}/fold_${FOLD}"
    FOLD_LOG_FILE="${FOLD_OUTPUT_DIR}_run.log"
    
    # 获取当前折的随机种子
    SEED=${SEEDS[$((FOLD-1))]}
    
    # 运行命令
    RUN_CMD="python main.py --cfg $CONFIG_FILE --data human --split stratified --seed $SEED --output-dir \"$FOLD_OUTPUT_DIR\" --use_3d --data_3d_root \"$DATA_3D_ROOT\" --data_3d_label \"$DATA_3D_LABEL\" --tag \"cv_fold_${FOLD}\" --cv_fold $FOLD --cv_total_folds $CV_FOLDS"
    echo "运行命令: $RUN_CMD" | tee -a "$MAIN_LOG_FILE" "$FOLD_LOG_FILE"
    
    # 执行命令并记录输出
    python main.py \
        --cfg "$CONFIG_FILE" \
        --data human \
        --split stratified \
        --seed "$SEED" \
        --output-dir "$FOLD_OUTPUT_DIR" \
        --use_3d \
        --data_3d_root "$DATA_3D_ROOT" \
        --data_3d_label "$DATA_3D_LABEL" \
        --tag "cv_fold_${FOLD}" \
        --cv_fold "$FOLD" \
        --cv_total_folds "$CV_FOLDS" 2>&1 | tee -a "$FOLD_LOG_FILE" 
    
    # 从日志中提取指标
    BEST_EPOCH=$(grep "best_model_epoch" "$FOLD_LOG_FILE" | tail -1 | awk -F'_' '{print $NF}' | awk -F'.' '{print $1}')
    AUROC=$(grep "AUROC" "$FOLD_LOG_FILE" | tail -1 | awk '{print $NF}')
    AUPRC=$(grep "AUPRC" "$FOLD_LOG_FILE" | tail -1 | awk '{print $NF}')
    F1=$(grep "F1" "$FOLD_LOG_FILE" | tail -1 | awk '{print $NF}')
    SENSITIVITY=$(grep "Sensitivity" "$FOLD_LOG_FILE" | tail -1 | awk '{print $NF}')
    SPECIFICITY=$(grep "Specificity" "$FOLD_LOG_FILE" | tail -1 | awk '{print $NF}')
    ACCURACY=$(grep "Accuracy" "$FOLD_LOG_FILE" | tail -1 | awk '{print $NF}')
    
    # 添加到汇总文件
    echo "$FOLD,$AUROC,$AUPRC,$F1,$SENSITIVITY,$SPECIFICITY,$ACCURACY" >> "${MAIN_OUTPUT_DIR}/cv_results_summary.csv"
    
    # 输出当前折的结果摘要
    echo "----- 第 $FOLD 折结果摘要 -----" | tee -a "$MAIN_LOG_FILE"
    echo "最佳轮次: ${BEST_EPOCH:-未找到}" | tee -a "$MAIN_LOG_FILE"
    echo "AUROC: ${AUROC:-未找到}" | tee -a "$MAIN_LOG_FILE"
    echo "AUPRC: ${AUPRC:-未找到}" | tee -a "$MAIN_LOG_FILE"
    echo "F1: ${F1:-未找到}" | tee -a "$MAIN_LOG_FILE"
    echo "敏感度: ${SENSITIVITY:-未找到}" | tee -a "$MAIN_LOG_FILE"
    echo "特异度: ${SPECIFICITY:-未找到}" | tee -a "$MAIN_LOG_FILE"
    echo "准确率: ${ACCURACY:-未找到}" | tee -a "$MAIN_LOG_FILE"
    echo "----------------------------" | tee -a "$MAIN_LOG_FILE"
done

# 计算平均性能指标
echo "计算5折交叉验证平均指标..." | tee -a "$MAIN_LOG_FILE"

# 使用awk计算平均值
AVG_RESULTS=$(awk -F',' 'NR>1 {auroc+=$2; auprc+=$3; f1+=$4; sens+=$5; spec+=$6; acc+=$7; count++} 
    END {printf "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", 
    auroc/count, auprc/count, f1/count, sens/count, spec/count, acc/count}' "${MAIN_OUTPUT_DIR}/cv_results_summary.csv")

# 将平均值添加到汇总文件
echo "平均,$AVG_RESULTS" >> "${MAIN_OUTPUT_DIR}/cv_results_summary.csv"

# 分解平均值以便于显示
IFS=',' read -r AVG_AUROC AVG_AUPRC AVG_F1 AVG_SENS AVG_SPEC AVG_ACC <<< "$AVG_RESULTS"

# 输出总体结果
echo "==== 5折交叉验证最终结果 ====" | tee -a "$MAIN_LOG_FILE"
echo "平均AUROC: $AVG_AUROC" | tee -a "$MAIN_LOG_FILE"
echo "平均AUPRC: $AVG_AUPRC" | tee -a "$MAIN_LOG_FILE"
echo "平均F1: $AVG_F1" | tee -a "$MAIN_LOG_FILE"
echo "平均敏感度: $AVG_SENS" | tee -a "$MAIN_LOG_FILE"
echo "平均特异度: $AVG_SPEC" | tee -a "$MAIN_LOG_FILE"
echo "平均准确率: $AVG_ACC" | tee -a "$MAIN_LOG_FILE"
echo "==============================" | tee -a "$MAIN_LOG_FILE"
echo "所有详细结果已保存到: ${MAIN_OUTPUT_DIR}" | tee -a "$MAIN_LOG_FILE"
echo "总结果日志: $MAIN_LOG_FILE" | tee -a "$MAIN_LOG_FILE"
echo "CSV汇总表: ${MAIN_OUTPUT_DIR}/cv_results_summary.csv" | tee -a "$MAIN_LOG_FILE" 