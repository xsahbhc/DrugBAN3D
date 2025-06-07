from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL_TYPE = "DrugBAN"  # 选择模型类型: "DrugBAN" 或 "DrugBAN3D"

# 通用路径配置
_C.PATH = CN()
_C.PATH.DATA_DIR = "datasets"
_C.PATH.RESULT_DIR = "result"
_C.PATH.CACHE_DIR = None  # 预处理缓存目录，如不设置则不使用缓存

# 训练配置
_C.TRAIN = CN()
_C.TRAIN.D_LEARNING_RATE = 0.0002
_C.TRAIN.G_LEARNING_RATE = 0.0001
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.D_STEPS_PER_G_STEP = 1
_C.TRAIN.RESUME_CHECKPOINT = ""
_C.TRAIN.EPOCH = 100
_C.TRAIN.GRADIENT_ACCUMULATE_STEPS = 8
_C.TRAIN.GRADIENT_CLIP_NORM = 0.0  # 添加梯度裁剪配置项
_C.TRAIN.WEIGHT_DECAY = 0.0        # L2正则化系数
_C.TRAIN.LABEL_SMOOTHING = 0.0     # 标签平滑系数
_C.TRAIN.POS_WEIGHT = 2.125        # 正样本权重参数

# 数据集配置
_C.DATA = CN()
_C.DATA.TRAIN_FILE = ""
_C.DATA.VAL_FILE = ""
_C.DATA.TEST_FILE = ""

# 求解器配置
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 50
_C.SOLVER.BATCH_SIZE = 16
_C.SOLVER.USE_MIXED_PRECISION = False  # 是否使用混合精度训练
_C.SOLVER.LR_SCHEDULER = False         # 是否使用学习率调度器
_C.SOLVER.LR_SCHEDULER_TYPE = "cosine" # 学习率调度器类型: cosine, plateau, one_cycle
_C.SOLVER.LR_WARMUP_EPOCHS = 0         # 学习率预热轮数
_C.SOLVER.GRADIENT_CLIP_NORM = 0.0     # 添加求解器梯度裁剪配置项
_C.SOLVER.SEED = 666                   # 求解器随机种子

# 结果配置
_C.RESULT = CN()
_C.RESULT.OUTPUT_DIR = "result"
_C.RESULT.SAVE_MODEL = True
_C.RESULT.SAVE_EACH_EPOCH = False  # 是否每个epoch保存一次结果
_C.RESULT.SAVE_BEST_ONLY = False   # 是否仅保存最佳模型
_C.RESULT.USE_STATE_DICT = True    # 是否使用state_dict保存模型
_C.RESULT.SAVE_TEST_PREDICTIONS = True  # 是否保存测试预测结果
_C.RESULT.SAVE_TEST_DETAILS = True      # 是否保存详细测试信息

# 领域适应配置
_C.DA = CN()
_C.DA.USE = False
_C.DA.METHOD = "None"
_C.DA.INIT_EPOCH = 0
_C.DA.LAMB_DA = 0.1
_C.DA.RANDOM_LAYER = False
_C.DA.ORIGINAL_RANDOM = False
_C.DA.USE_ENTROPY = False

# 药物分子配置
_C.DRUG = CN()
_C.DRUG.ATOM_MAX = 50
_C.DRUG.NODE_IN_FEATS = 35
_C.DRUG.NODE_IN_EMBEDDING = 128
_C.DRUG.PADDING = True
_C.DRUG.HIDDEN_LAYERS = [128, 256, 256]

# 蛋白质配置
_C.PROTEIN = CN()
_C.PROTEIN.SEQ_MAX = 1000
_C.PROTEIN.EMBEDDING_DIM = 64
_C.PROTEIN.PADDING = True
_C.PROTEIN.NUM_FILTERS = [32, 64, 96]
_C.PROTEIN.KERNEL_SIZE = [4, 8, 12]
_C.PROTEIN.NODE_IN_FEATS = 35

# 边特征配置（3D模型使用）
_C.EDGE_FEATS = CN()
_C.EDGE_FEATS.intra_l = 17  # 6(基础特征) + 11(几何特征)
_C.EDGE_FEATS.intra_p = 17  # 6(基础特征) + 11(几何特征)
_C.EDGE_FEATS.inter_l2p = 11  # 11(几何特征)
_C.EDGE_FEATS.inter_p2l = 11  # 11(几何特征)

# GNN配置（3D模型使用）
_C.GNN = CN()
_C.GNN.HIDDEN_DIM = 256
_C.GNN.NUM_LAYERS = 3

# 双线性注意力网络配置
_C.BCN = CN()
_C.BCN.HEADS = 4

# 解码器配置
_C.DECODER = CN()
_C.DECODER.IN_DIM = 256
_C.DECODER.HIDDEN_DIM = 512
_C.DECODER.OUT_DIM = 64
_C.DECODER.BINARY = 1
_C.DECODER.DROPOUT = 0.5  # 添加Dropout率参数

# 领域适应配置
_C.ADAPT = CN()
_C.ADAPT.LAMBDA = 0.1
_C.ADAPT.USE_TARGET = True
_C.ADAPT.METHOD = "CDAN"  # 可选: "None", "CDAN"
_C.ADAPT.ENTROPY_LAMBDA = 1.0
_C.ADAPT.RANDOM_DIM = 512

# 3D数据配置
_C.DATA_3D = CN()
_C.DATA_3D.ROOT_DIR = "/home/work/workspace/shi_shaoqun/snap/3D_structure/bindingdb/train_pdb"
_C.DATA_3D.LABEL_FILE = None
_C.DATA_3D.TRAIN_FILE = None
_C.DATA_3D.VAL_FILE = ''
_C.DATA_3D.TEST_FILE = ''
_C.DATA_3D.DIS_THRESHOLD = 5.0
_C.DATA_3D.NUM_WORKERS = 2  # 数据加载工作线程数

# 设置Comet
_C.COMET = CN()
_C.COMET.USE = False
_C.COMET.PROJECT = "my-project"
_C.COMET.WORKSPACE = "your_workspace"
_C.COMET.TAG = "default"  # 添加TAG字段

# 早停机制配置
_C.USE_EARLY_STOPPING = True  # 是否使用早停
_C.EARLY_STOPPING_PATIENCE = 10  # 早停耐心值，从5增加到10

# 偏置校正配置
_C.USE_BIAS_CORRECTION = True  # 是否使用偏置校正模块

# 对比损失配置
_C.USE_CONTRASTIVE_LOSS = False  # 是否使用对比损失
_C.CONTRASTIVE_LOSS_WEIGHT = 0.5  # 对比损失权重

# 焦点损失配置
_C.USE_FOCAL_LOSS = False  # 是否使用焦点损失
_C.FOCAL_LOSS_GAMMA = 2.0  # 焦点损失gamma参数
_C.FOCAL_LOSS_ALPHA = 0.25  # 焦点损失alpha参数

# 特征正则化配置
_C.USE_FEATURE_REGULARIZATION = False  # 是否使用特征正则化
_C.FEATURE_REGULARIZATION_WEIGHT = 0.01  # 特征正则化权重

# 测试相关配置
_C.TEST = CN()
_C.TEST.USE_ORIGINAL_TEST_DATA = True  # 使用原始测试数据（非增强）
_C.TEST.DEBUG_MODE = False             # 测试调试模式
_C.TEST.VERBOSE = True                 # 详细测试输出
_C.TEST.BATCH_SIZE = 64                # 测试批次大小

# 顶级测试调试模式配置
_C.DEBUG_MODE = False  # 全局调试模式开关
_C.USE_ORIGINAL_TEST_DATA = True  # 使用原始测试数据
_C.VERBOSE = True  # 详细输出模式

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()
