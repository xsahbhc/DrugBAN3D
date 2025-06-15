#!/bin/bash

# 多模态DrugBAN训练脚本
# 禁用数据增强，确保3D数据与1D/2D数据完全对应

echo "=== 多模态DrugBAN训练 ==="
echo "禁用数据增强，确保数据对应关系"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main:$PYTHONPATH"

# 激活环境
source /home/work/anaconda3/etc/profile.d/conda.sh
conda activate DrugBAN

# 检查当前目录
echo "当前工作目录: $(pwd)"
echo "Python路径: $(which python)"
echo "环境: $CONDA_DEFAULT_ENV"

# 设置输出目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="result/DrugBAN_Multimodal_Enhanced_${TIMESTAMP}"

echo "输出目录: $OUTPUT_DIR"

# 检查必要文件是否存在
echo "检查配置文件..."
if [ ! -f "configs/DrugBAN_Multimodal.yaml" ]; then
    echo "错误: 配置文件 configs/DrugBAN_Multimodal.yaml 不存在"
    exit 1
fi

echo "检查数据文件..."
DATA_3D_ROOT="/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_pdb"
DATA_1D2D_ROOT="/home/work/workspace/shi_shaoqun/snap/drugban/datasets/bindingdb_3d_sequences"
SEQID_MAPPING="/home/work/workspace/shi_shaoqun/snap/drugban/datasets/bindingdb_3d_sequences/random/seqid_mapping.csv"

if [ ! -d "$DATA_3D_ROOT" ]; then
    echo "错误: 3D数据目录不存在: $DATA_3D_ROOT"
    exit 1
fi

if [ ! -d "$DATA_1D2D_ROOT" ]; then
    echo "错误: 1D/2D数据目录不存在: $DATA_1D2D_ROOT"
    exit 1
fi

if [ ! -f "$SEQID_MAPPING" ]; then
    echo "错误: seqid映射文件不存在: $SEQID_MAPPING"
    exit 1
fi

echo "所有必要文件检查通过"

# 使用配置文件中指定的缓存目录（不创建新目录）
CACHE_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs"
echo "使用现有缓存目录: $CACHE_DIR"
echo "缓存文件数量: $(find "$CACHE_DIR" -name "*.pt" 2>/dev/null | wc -l)"

# 运行多模态训练
echo "开始多模态训练..."
python main.py \
    --cfg configs/DrugBAN_Multimodal.yaml \
    --data bindingdb \
    --split stratified \
    --output-dir "$OUTPUT_DIR" \
    --seed 42 \
    --tag "multimodal_enhanced" \
    --verbose

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "=== 多模态训练完成 ==="
    echo "结果保存在: $OUTPUT_DIR"
    
    # 显示结果摘要
    if [ -f "$OUTPUT_DIR/results.txt" ]; then
        echo "=== 训练结果摘要 ==="
        cat "$OUTPUT_DIR/results.txt"
    fi
    
    if [ -f "$OUTPUT_DIR/test_summary.csv" ]; then
        echo "=== 测试结果 ==="
        cat "$OUTPUT_DIR/test_summary.csv"
    fi
else
    echo "=== 多模态训练失败 ==="
    echo "请检查错误日志"
    exit 1
fi

echo "=== 脚本执行完成 ==="
