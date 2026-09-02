# ACT 模型塌缩诊断（修正版）

## 核心事实

### 实际训练配置（来自 manifest）
- **训练集**: 4096 样本（stratified 采样后）
- **验证集**: 2579 样本
- **训练量**: batch_size=1 × 500 steps = **500 样本次**
- **数据覆盖率**: 500 / 4096 = **12.2%**
- **关键超参**:
  - `move_stop_weight: 0.25`
  - `move_direction_balance: False` ← 未启用
  - `teacher_forcing: 1.0 → 0.0` over 500 steps

### 全量数据真实分布（21个session，7264样本）
```
意图分布:
  decipher: 3679 (50.6%)
  travel:   3007 (41.4%)
  rescue:    365 (5.0%)
  gate:      213 (2.9%)

移动方向分布（每样本4步，共29,056步）:
  stop:      18031 (62.1%)
  north:      8621 (29.7%)
  northeast:   708 (2.4%)
  northwest:   656 (2.3%)
  east:        324 (1.1%)
  south:       188 (0.6%)
  west:        216 (0.7%)
  southeast:   172 (0.6%)
  southwest:   140 (0.5%)
```

### 模型表现（128样本抽样诊断）
- **移动头**: 100% 预测 stop（512/512），非停止召回率 0%
- **意图头**: 100% 预测 travel（128/128）
- **验证集问题**: kite 有 34 个样本，但训练集中 0 个（未见过的类）

---

## 问题诊断

### 主要问题 1：训练覆盖率过低（12.2%）

**最严重的问题**：batch_size=1, steps=500 只训练了 500 次，但数据集有 4096 个样本。

**影响**：
- stratified 采样后，少数类（rescue 5%, gate 3%）本来就稀疏
- 只看 12.2% 的数据，**少数类几乎看不到**
- southeast (0.6%), southwest (0.5%) 这些方向，在 500 步中可能只出现 **3-10 次**

**这是为什么验证集准确率为 0% 的根本原因**。

### 主要问题 2：move_direction_balance 未启用

代码中有 inverse-sqrt 频率平衡逻辑（[vla_loss.py:43-49](../idv_agent/training/vla_loss.py:43-49)），但配置中 `"move_direction_balance": false`。

**影响**：
- stop (62%) 被 `move_stop_weight=0.25` 降权到 15.5% 等效权重
- north (30%) 保持 1.0 权重
- 其他 7 个方向（8%）共享 1.0 权重，每个 ~1%
- **模型发现**: 预测 stop 虽然降权，但基础准确率 62% 仍然最稳

### 主要问题 3：move_stop_weight=0.25 过于激进

原始分布 62% stop，降权到 0.25 后：
- 有效 stop 样本 ≈ 62% × 0.25 = 15.5% 等效权重
- north 30% 权重 1.0
- **理论上** north 应该被学习得更好

**但实际结果**：移动头全预测 stop，说明：
1. 训练步数太少（500步），模型还在"试探阶段"
2. 0.25 太低导致梯度不稳定，模型索性放弃 stop 类 → 但又因为没见过足够的其他类 → 最终退化到预测多数类

### 次要问题 4：Teacher Forcing 衰减导致级联塌缩

慢头全预测 travel → fast_condition 全是 travel → 快头学习的都是"travel 模式下的动作"。

但这不是根因，因为如果慢头准确率高，这个机制应该工作正常。

### 次要问题 5：验证集中有训练集未见类别

诊断显示验证集有 34 个 kite 样本，但训练集中 0 个 → 这个类别必然准确率 0%。

但全量数据没有 kite（只有 decipher/travel/rescue/gate），说明**验证集本身可能来自不同分布**。

---

## 根本原因排序

1. **训练步数严重不足** (12.2% 覆盖率) ← **最关键**
2. **move_direction_balance 未启用** ← 代码有但没用
3. **move_stop_weight=0.25 配合低覆盖率导致不稳定**
4. 数据不平衡（次要，因为全量数据 62% stop 不算极端）

---

## 解决方案（按优先级）

### 优先级 1：增加训练步数

```bash
# 至少训练 2-4 个完整 epoch
--steps 8000  # 2 epoch: 4096 × 2 / 1 = 8192
--batch-size 2  # 如果显存允许，提高 batch size
```

**预期效果**：
- 少数类至少被看到 20-40 次（当前可能只有 3-5 次）
- 模型有足够样本学习 8 个移动方向的模式

### 优先级 2：启用 move_direction_balance

```bash
--move-direction-balance
```

**预期效果**：
- 自动用 inverse-sqrt 平衡 8 个方向
- southwest (0.5%) 会得到 ~4.5× 权重，southeast 同理

### 优先级 3：调整 move_stop_weight

```bash
--move-stop-weight 0.5  # 比 0.25 更温和
```

**原因**：
- 0.25 在低覆盖率（12%）时太激进
- 0.5 降低 stop 权重但不至于完全忽略
- 配合 `move_direction_balance`，stop 仍会适当降权

### 优先级 4：调整 teacher forcing 策略

```bash
--teacher-forcing-decay-steps 2000  # 延长衰减周期
```

**原因**：
- 当前 500 步就衰减完，慢头还没学好就开始自回归
- 延长到 2000 步，让慢头先稳定

### 优先级 5：检查验证集

验证集中有 kite，但训练集没有 → 检查验证集是否混入了其他模式的数据。

```bash
# 检查验证集文件
cat data/vla_raw_sessions/20260902_155023_887599/vla_chunks_v4.jsonl | \
  python -c "import json, sys; [print(json.loads(l)['slow_label']['intent']) for l in sys.stdin if l.strip()]" | \
  sort | uniq -c
```

---

## 推荐训练命令

```bash
python idv_agent/scripts/train_vla.py \
  --data <your_data_path> \
  --val-data <your_val_path> \
  --model-path <model_path> \
  --init-checkpoint <M2_VG_checkpoint> \
  --steps 8000 \
  --batch-size 2 \
  --max-samples 4096 \
  --sampling stratified \
  --move-direction-balance \
  --move-stop-weight 0.5 \
  --button-positive-weight 4.0 \
  --teacher-forcing-start 1.0 \
  --teacher-forcing-end 0.0 \
  --teacher-forcing-decay-steps 2000 \
  --checkpoint checkpoints/M3_ACT/act_fixed.pt
```

**关键改动**：
- `steps 8000` → 2 个完整 epoch（覆盖率 200%）
- `batch-size 2` → 如果显存允许，提高批量
- `move-direction-balance` → 启用自动平衡
- `move-stop-weight 0.5` → 更温和的降权
- `teacher-forcing-decay-steps 2000` → 延长衰减

---

## 快速验证（推荐先跑）

```bash
# 1000 步快速验证
python idv_agent/scripts/train_vla.py \
  --steps 1000 \
  --max-samples 2048 \
  --move-direction-balance \
  --move-stop-weight 0.5 \
  --checkpoint checkpoints/debug/test_1k.pt \
  <其他必需参数>

# 诊断
python idv_agent/scripts/diagnose_act.py \
  --checkpoint checkpoints/debug/test_1k.pt \
  --max-samples 256 \
  <其他必需参数>
```

**验证指标**：
- `move_pred_counts` 不再是 `[512, 0, 0, ...]`
- `move_nonstop_prediction_rate_on_nonstop_targets` > 0.3
- `slow_intent_pred_names` 出现 decipher/rescue/gate

---

## 总结

**诊断结论修正**：
- ❌ 之前：主要是 `move_stop_weight=0.25` 问题
- ✅ 现在：**主要是训练步数不足（12.2% 覆盖率）**，配合未启用 `move_direction_balance`

**本质问题**：
- 少数类在 500 步中只出现 3-10 次，根本学不到
- 即使代码逻辑完美，12% 覆盖率也无法学习 8 个移动方向 + 4-8 个意图类别

**立即行动**：
1. 增加训练步数到 8000（2 epoch）
2. 启用 `--move-direction-balance`
3. 调整 `move_stop_weight` 到 0.5
