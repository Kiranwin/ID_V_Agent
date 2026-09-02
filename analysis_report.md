# ACT 模型塌缩诊断报告

## 问题总结

训练后模型出现严重塌缩：
- **移动头**：100% 预测 `stop`（非停止召回率 0%）
- **意图头**：100% 预测 `travel`（decipher/kite 准确率 0%）
- 总体准确率 59.8% 仅由多数类贡献

## 根本原因：**代码逻辑问题 + 数据不平衡共同作用**

### 1. 代码问题：loss 权重设置不当

#### 问题 1.1：`move_stop_weight=0.25` 过低
```python
# vla_loss.py:42
move_class_weight[0] = weights.move_stop_weight  # 0.25
```

**影响**：
- 训练集 stop 占 64.3% (329/512)
- 下调 stop 权重到 0.25 后，**实际有效 stop 样本 ≈ 82**
- 非 stop 类共 183 个样本，但分散到 8 个方向
- 最少的类（southeast）只有 **1 个样本**，权重 1.0

**结果**：模型学会"预测 stop 永远不会太错"，因为：
- 预测 stop：64.3% 基础准确率，loss 只有 0.25 惩罚
- 预测其他方向：样本太少，学不到模式

#### 问题 1.2：没有启用 `move_direction_balance`
```python
# train_vla.py:510
parser.add_argument("--move-direction-balance", action="store_true")
# manifest 显示: "move_direction_balance": false
```

代码中有 inverse-sqrt 频率平衡逻辑（[vla_loss.py:43-49](vla_loss.py:43-49)），但**没有启用**。

### 2. 数据问题：类别极度不平衡

#### 移动类分布（训练集 128 样本）：
```
stop:      329/512 (64.3%)  ← 多数类
north:     137/512 (26.8%)  ← 次多数类
northeast:  15/512 (2.9%)
northwest:  12/512 (2.3%)
east:        7/512 (1.4%)
south:       5/512 (1.0%)
southwest:   3/512 (0.6%)
west:        3/512 (0.6%)
southeast:   1/512 (0.2%)   ← 极端少数类
```

#### 意图类分布（训练集 128 样本）：
```
travel:   55/128 (43.0%)  ← 多数类
decipher: 62/128 (48.4%)  
rescue:    7/128 (5.5%)
gate:      4/128 (3.1%)
kite:      0/128 (0%)      ← 训练集中没有
```

**验证集**中有 34 个 `kite` 样本，但训练集中 **0 个** → 模型根本没见过这个类。

### 3. 训练策略问题

#### 问题 3.1：stratified 采样没有细粒度平衡
```python
# train_vla.py:105-108
ACTION_STRATA = ("interact", "move", "other_key", "stop")
DEFAULT_STRATIFIED_RATIOS = {
    "interact": 0.35, "move": 0.35, "other_key": 0.15, "stop": 0.15,
}
```

当前的 stratified 采样只分四大类，**没有平衡 8 个移动方向** 和 **8 个意图类别**。

#### 问题 3.2：样本量可能不足
- 训练集：4096 样本（stratified 采样后）
- 500 steps，batch_size=1 → 只看了 500 个样本
- 覆盖率：500 / 4096 = **12.2%**

### 4. Teacher Forcing 衰减的影响

```python
# manifest: 
"teacher_forcing_start": 1.0,
"teacher_forcing_end": 0.0,
"teacher_forcing_decay_steps": 500
```

训练过程中 teacher forcing 从 1.0 线性衰减到 0.0，**快头需要依赖慢头预测的 condition**。

诊断报告显示：
- `fast_condition_intent_counts`: 全部是 `travel` (128/128)
- 慢头塌缩 → condition 全是 `travel` → 快头也学偏了

## 解决方案

### 立即修复（代码逻辑）

#### 方案 A：启用 move_direction_balance + 提高 stop weight
```bash
python idv_agent/scripts/train_vla.py \
  --move-direction-balance \
  --move-stop-weight 0.5 \
  --button-positive-weight 4.0 \
  --sampling stratified \
  --steps 500 \
  --max-samples 4096 \
  [其他参数不变]
```

**预期效果**：
- `move_direction_balance` 对 8 个方向做 inverse-sqrt 平衡
- `move_stop_weight 0.5` 适度降低 stop 权重（比 0.25 更温和）

#### 方案 B：增加 focal loss 或 class-balanced loss
修改 [vla_loss.py](vla_loss.py) 引入 focal loss：
```python
# 替代当前的 F.cross_entropy
move = focal_loss(fast.move_logits, batch["move_target"], gamma=2.0, alpha=move_class_weight)
```

### 中期改进（数据策略）

#### 改进 1：细粒度 stratified 采样
修改 `_stratified_subset` 在 **意图 × 移动方向** 两个维度同时平衡：
```python
# 伪代码
stratify_by = (sample["intent"], sample["dominant_move_direction"])
```

#### 改进 2：数据增强
- 对少数类（southeast, kite）做 oversample
- 或者针对性录制更多 kite/rescue/gate session

#### 改进 3：增加训练步数
- 当前 500 steps 只覆盖 12% 数据
- 建议至少 2000-4000 steps（完整 epoch）

### 长期优化（架构）

#### 优化 1：慢头单独预训练
```python
# 先固定快头，只训练慢头 200 steps
# 然后联合训练
```

#### 优化 2：调整 teacher forcing 策略
- 延长衰减周期（如 decay_steps=1000）
- 或使用 curriculum learning（先学简单类，再学复杂类）

## 验证方案

### 快速验证（30分钟）
```bash
# 1. 启用 direction balance，训练 100 steps
python idv_agent/scripts/train_vla.py \
  --move-direction-balance \
  --move-stop-weight 0.5 \
  --steps 100 \
  --max-samples 512 \
  --checkpoint checkpoints/debug/test.pt \
  [其他参数]

# 2. 运行诊断
python idv_agent/scripts/diagnose_act.py \
  --checkpoint checkpoints/debug/test.pt \
  --max-samples 128 \
  [其他参数]

# 3. 检查 move_pred_counts 是否不再全是 [512, 0, 0, ...]
```

### 完整验证（2小时）
```bash
# 启用所有改进，训练 2000 steps
python idv_agent/scripts/train_vla.py \
  --move-direction-balance \
  --move-stop-weight 0.5 \
  --steps 2000 \
  --max-samples 4096 \
  --sampling stratified \
  --checkpoint checkpoints/M3_ACT/act_balanced.pt \
  [其他参数]
```

## 结论

**主要问题是代码逻辑**：
1. `move_stop_weight=0.25` 太激进，导致模型放弃学习非 stop 类
2. `move_direction_balance` 功能存在但未启用
3. Teacher forcing 衰减 + 慢头塌缩 → 快头也塌缩

**数据问题是次要因素**：
- 类别不平衡确实存在，但可以通过代码缓解
- `kite` 训练集缺失需要补充数据

**推荐优先级**：
1. **立即**：启用 `--move-direction-balance`，调整 `move_stop_weight` 到 0.5
2. **短期**：增加训练步数到 2000+，确保覆盖完整数据集
3. **中期**：改进 stratified 采样逻辑，补充 kite/rescue 数据
