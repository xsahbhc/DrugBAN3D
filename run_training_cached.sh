#!/bin/bash

# 设置使用GPU 1
export CUDA_VISIBLE_DEVICES=1
echo "已设置使用GPU 1 (CUDA_VISIBLE_DEVICES=1)"

# 设置Comet ML API密钥
# 请替换为您的实际密钥
export COMET_API_KEY="OAXLoYYmTBstjpbCKpaA81PLY"
echo "已设置Comet API密钥环境变量"

# 设置配置文件路径
CONFIG_FILE="configs/DrugBAN3D_cached.yaml"
# 备注：该配置已使用最优参数，基于最佳实验结果 (AUROC: 0.8924)：
# 1. Batch Size: 48 (最佳稳定性)
# 2. Learning Rate: 0.0002 (生成器), 0.0004 (判别器)
# 3. Weight Decay: 0.0004 (最优正则化)
# 4. Dropout: 0.3 (最优过拟合控制)
# 5. GNN Layers: 4, Hidden Dim: 384
# 6. Focal Loss: Gamma=1.0, Alpha=0.3
# 7. 3D Spatial Fusion: 启用
# 8. Bias Correction: 启用

# 设置3D数据相关路径
DATA_3D_ROOT="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_pdb"
DATA_3D_LABEL="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_csv/labels.csv"

# 设置缓存目录
CACHE_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs"

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

# 创建带有时间戳的输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="result/DrugBAN3D_optimal_${TIMESTAMP}"
LOG_FILE="run_optimal.log"

# 确保输出目录存在
mkdir -p $(dirname "$OUTPUT_DIR")

# 记录训练配置
echo "=== DrugBAN3D最优配置训练 ===" | tee -a "$LOG_FILE"
echo "- 配置文件: $CONFIG_FILE" | tee -a "$LOG_FILE"
echo "- 数据目录: $DATA_3D_ROOT" | tee -a "$LOG_FILE"
echo "- 标签文件: $DATA_3D_LABEL" | tee -a "$LOG_FILE"
echo "- 缓存目录: $CACHE_DIR" | tee -a "$LOG_FILE"
echo "- 输出目录: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "- 批次大小: 48 (最优设置)" | tee -a "$LOG_FILE"
echo "- 最优参数: LR=0.0002/0.0004, WD=0.0004, Dropout=0.3, Focal Loss, 3D Fusion" | tee -a "$LOG_FILE"
echo "- 预期性能: AUROC > 0.89, AUPRC > 0.79" | tee -a "$LOG_FILE"
echo "- 预处理缓存文件数: $CACHE_FILES_COUNT" | tee -a "$LOG_FILE"
echo "- 日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"

# 设置环境变量以防OOM
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 运行训练，重定向输出到日志文件
echo "训练开始时间: $(date)" | tee -a "$LOG_FILE"
echo "正在运行训练..." | tee -a "$LOG_FILE"

# 运行命令，注意使用主程序的参数格式
python main.py \
    --cfg "$CONFIG_FILE" \
    --data human \
    --split stratified \
    --seed 42 \
    --output-dir "$OUTPUT_DIR" \
    --use_3d \
    --data_3d_root "$DATA_3D_ROOT" \
    --data_3d_label "$DATA_3D_LABEL" \
    --tag "optimized" 2>&1 | tee -a "$LOG_FILE"

TRAIN_EXIT_CODE=$?

# 检查训练结果
echo "训练完成时间: $(date)" | tee -a "$LOG_FILE"
echo "模型保存在: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "日志文件保存在: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===============================" | tee -a "$LOG_FILE"

# 如果训练成功完成，提取结果摘要
if [ $TRAIN_EXIT_CODE -eq 0 ] && [ -d "$OUTPUT_DIR" ]; then
    echo "=== 训练结果摘要 ===" | tee -a "$LOG_FILE"
    
    # 如果模型训练成功并保存了性能文件，从中提取信息
    METRICS_FILE="${OUTPUT_DIR}/best_metrics.txt"
    if [ -f "$METRICS_FILE" ]; then
        echo "从性能文件中提取结果..." | tee -a "$LOG_FILE"
        cat "$METRICS_FILE" | tee -a "$LOG_FILE"
    fi
    
    # 查找最佳模型文件
    BEST_MODEL_FILE=$(find "$OUTPUT_DIR" -name "best_model*.pth" | head -n 1)
    if [ -n "$BEST_MODEL_FILE" ]; then
        echo "最佳模型: $(basename "$BEST_MODEL_FILE")" | tee -a "$LOG_FILE"
        # 从文件名提取epoch (假设文件名格式为best_model_epoch_X.pth)
        EPOCH=$(basename "$BEST_MODEL_FILE" | grep -o "epoch_[0-9]*" | grep -o "[0-9]*")
        if [ -n "$EPOCH" ]; then
            echo "最佳轮次: $EPOCH" | tee -a "$LOG_FILE"
        fi
    else
        echo "未找到最佳模型文件" | tee -a "$LOG_FILE"
    fi
    
    # 如果有test_results.txt，显示测试结果
    TEST_RESULTS="${OUTPUT_DIR}/test_results.txt"
    if [ -f "$TEST_RESULTS" ]; then
        echo "测试结果:" | tee -a "$LOG_FILE"
        cat "$TEST_RESULTS" | tee -a "$LOG_FILE"
    fi
    
    echo "===============================" | tee -a "$LOG_FILE"
fi 