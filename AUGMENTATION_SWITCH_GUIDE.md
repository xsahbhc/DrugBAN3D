# 🔄 数据增强方法切换指南

## ✅ **是的，您可以随时切换！**

这个脚本设计得非常灵活，您可以轻松地在不同增强方法之间切换。

## 🎯 **三种切换方式**

### 1. **代码内切换** (推荐)

只需修改第907-911行的默认增强类型：

```python
# 当前配置 (现代方法)
augment_types = [
    'smart_conformational',  # 智能构象采样
    'binding_aware',         # 结合位点感知
    'adaptive_rotate'        # 自适应旋转
]

# 切换到原始方法
augment_types = [
    'gentle_rotate',         # 温和旋转 (原始)
    'thermal_vibration'      # 热振动 (原始)
]

# 切换到混合方法
augment_types = [
    'gentle_rotate',         # 温和旋转 (原始)
    'thermal_vibration',     # 热振动 (原始)
    'adaptive_rotate'        # 自适应旋转 (新增)
]
```

### 2. **命令行参数切换** (最灵活)

通过添加 `--augment_types` 参数：

```bash
# 使用原始方法
python augment_data_with_labels.py \
    --cache_dir cached_graphs \
    --output_dir data/augmented \
    --label_file datasets/bindingdb/train_labels.csv \
    --train_label datasets/bindingdb/train_labels.csv \
    --pos_augment 0.5 --neg_augment 0.5 \
    --augment_types gentle_rotate,thermal_vibration

# 使用现代方法
python augment_data_with_labels.py \
    --cache_dir cached_graphs \
    --output_dir data/augmented \
    --label_file datasets/bindingdb/train_labels.csv \
    --train_label datasets/bindingdb/train_labels.csv \
    --pos_augment 0.5 --neg_augment 0.5 \
    --augment_types smart_conformational,binding_aware,adaptive_rotate

# 使用单一方法测试
python augment_data_with_labels.py \
    --augment_types gentle_rotate
```

### 3. **环境变量切换**

```bash
# 设置环境变量
export DRUGBAN_AUGMENT_TYPES="gentle_rotate,thermal_vibration"
./run_with_augment.sh
```

## 📋 **所有可用的增强方法**

### 🔵 **原始方法** (已验证安全)
- `gentle_rotate`: 温和旋转 (±1度)
- `thermal_vibration`: 热振动 (0.002Å)

### 🟢 **改进方法** (新增安全)
- `adaptive_rotate`: 自适应旋转 (根据分子大小调整)

### 🟡 **高级方法** (新增高效)
- `smart_conformational`: 智能构象采样
- `binding_aware`: 结合位点感知增强
- `multi_scale`: 多尺度增强

### 🟠 **其他方法** (可选)
- `minimal_edge_dropout`: 最小边删除
- `bond_flexibility`: 键长微调
- `smart_combined`: 智能组合

## 🚀 **快速切换示例**

### 场景1: 新方法效果不好，回退到原方法

```python
# 修改第907-911行
augment_types = [
    'gentle_rotate',         # 回到原始方法
    'thermal_vibration'      # 回到原始方法
]
```

### 场景2: 只想测试单个新方法

```python
# 只测试自适应旋转
augment_types = ['adaptive_rotate']

# 只测试智能构象采样
augment_types = ['smart_conformational']
```

### 场景3: 渐进式测试

```python
# 第一步：原方法 + 1个新方法
augment_types = ['gentle_rotate', 'thermal_vibration', 'adaptive_rotate']

# 第二步：如果效果好，再添加更多
augment_types = ['gentle_rotate', 'thermal_vibration', 'adaptive_rotate', 'smart_conformational']
```

## 🔧 **实际操作步骤**

### 方法A: 修改代码 (1分钟)

1. 打开 `augment_data_with_labels.py`
2. 找到第907-911行
3. 修改 `augment_types` 列表
4. 保存文件
5. 运行 `./run_with_augment.sh`

### 方法B: 命令行参数 (无需修改代码)

```bash
# 直接指定增强方法
python augment_data_with_labels.py \
    --cache_dir cached_graphs \
    --output_dir data/augmented_original \
    --label_file datasets/bindingdb/train_labels.csv \
    --train_label datasets/bindingdb/train_labels.csv \
    --pos_augment 0.5 --neg_augment 0.5 \
    --augment_types gentle_rotate,thermal_vibration
```

## 📊 **性能对比建议**

### 🎯 **A/B测试流程**

1. **基线测试** (原方法)
   ```bash
   # 使用原方法训练
   --augment_types gentle_rotate,thermal_vibration
   # 记录 AUROC, AUPRC, F1
   ```

2. **改进测试** (新方法)
   ```bash
   # 使用新方法训练
   --augment_types smart_conformational,binding_aware,adaptive_rotate
   # 记录 AUROC, AUPRC, F1
   ```

3. **对比分析**
   - 如果新方法 AUROC > 原方法，继续使用新方法
   - 如果新方法 AUROC ≤ 原方法，回退到原方法

### 📈 **预期结果**

| 方法组合 | 预期AUROC | 训练时间 | 稳定性 |
|----------|-----------|----------|--------|
| 原方法 | 0.8917 | 快 | 高 |
| 原方法+自适应 | 0.8920+ | 中 | 高 |
| 全新方法 | 0.8925+ | 慢 | 中 |

## ⚠️ **注意事项**

### 🛡️ **安全切换**

1. **备份当前最佳模型**: 切换前保存当前最佳结果
2. **使用相同数据**: 确保对比实验使用相同的数据集
3. **固定随机种子**: 使用相同的随机种子确保可重现性
4. **记录配置**: 详细记录每次实验的配置

### 📝 **切换检查清单**

- [ ] 备份当前最佳模型和配置
- [ ] 确认新的增强方法配置
- [ ] 使用相同的数据集和随机种子
- [ ] 记录实验配置和结果
- [ ] 对比新旧方法的性能指标

## 🎯 **总结**

**是的，您完全可以随时切换！** 这个脚本的设计非常灵活：

✅ **一行代码切换**: 修改 `augment_types` 列表  
✅ **命令行切换**: 使用 `--augment_types` 参数  
✅ **无风险测试**: 可以安全地测试任何组合  
✅ **快速回退**: 随时回到原始方法  
✅ **渐进式升级**: 逐步添加新方法  

这样您就可以放心地尝试新方法，如果效果不好随时切换回原来的配置！
