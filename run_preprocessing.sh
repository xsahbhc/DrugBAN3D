 #!/bin/bash

# 设置路径和参数
ROOT_DIR="/home/work/workspace/shi_shaoqun/snap/3D_structure/train_pdb"
LABEL_FILE="/home/work/workspace/shi_shaoqun/snap/3D_structure/labels.csv"
CACHE_DIR="/home/work/workspace/shi_shaoqun/snap/DrugBAN-main/cached_graphs"
NUM_WORKERS=12
DIS_THRESHOLD=5.0

# 创建缓存目录
mkdir -p $CACHE_DIR

# 执行预处理
echo "开始预处理数据..."
echo "数据目录: $ROOT_DIR"
echo "标签文件: $LABEL_FILE"
echo "缓存目录: $CACHE_DIR" 
echo "工作进程: $NUM_WORKERS"
echo "距离阈值: $DIS_THRESHOLD"

python pre_cache.py \
    --root_dir "$ROOT_DIR" \
    --label_file "$LABEL_FILE" \
    --output_dir "$CACHE_DIR" \
    --dis_threshold $DIS_THRESHOLD \
    --num_workers $NUM_WORKERS

echo "预处理完成！"