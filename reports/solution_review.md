# 修复方案评审

## 用户提出的方案

### 一、数据层：kite session 划分修复 ✅ **优先推荐**
```
1. 将含 kite 的人工标注 session 加入训练集
2. 留出集选一局含 kite 的 session（训练/验证两侧都必须含 kite）
3. 划分后输出意图分布报告（验证门）
```

**评审意见**：✅ **这是正确的根本性修复**

- kite 在验证集有 34 个，训练集 0 个 → 这是数据划分问题，不是代码问题
- 在划分阶段就验证类别覆盖，**比训练后诊断好 100 倍**
- 建议补充：检查其他少数类（rescue 5%, gate 3%）两侧分布

**实施建议**：
```python
# 划分后验证脚本
def validate_split(train_paths, val_paths):
    train_intents = Counter()
    val_intents = Counter()
    # ... 统计逻辑
    
    # 验证门：每个意图两侧都要有样本
    all_intents = set(train_intents.keys()) | set(val_intents.keys())
    for intent in all_intents:
        train_count = train_intents[intent]
        val_count = val_intents[intent]
        if train_count == 0:
            raise ValueError(f"训练集缺失意图: {intent}")
        if val_count == 0:
            raise ValueError(f"验证集缺失意图: {intent}")
        print(f"{intent}: train={train_count}, val={val_count}")
```

---

### 二、代码级修复（清单）

#### 2.1 放宽 batch 硬上限 ⚠️ **需要修改**

用户建议：`batch 8-16`

**评审意见**：⚠️ **建议谨慎，先验证显存**

当前代码：
```python
# train_vla.py:314
batch_size = max(1, min(args.batch_size, 2))  # 硬上限 2
```

**问题**：
- Qwen3-VL 4B + 8 帧历史 + 视觉特征缓存，batch=2 已经是安全上限
- batch=8 需要 ~40GB 显存（假设 FP16）
- 用户说"显存余量充足"，但没看到实际配置

**建议**：
```python
# 1. 先移除硬上限，让用户自己控制
batch_size = args.batch_size  

# 2. 或者根据设备自动调整
if device.type == "cuda":
    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    if total_memory_gb >= 80:
        max_batch = 16
    elif total_memory_gb >= 40:
        max_batch = 8
    else:
        max_batch = 2
    batch_size = min(args.batch_size, max_batch)
```

**风险提示**：batch=8 虽然能加速训练，但如果 OOM，用户会卡住。建议先试 batch=4。

---

#### 2.2 预计算全局九方向 class weight ⚠️ **需要澄清**

用户建议：`基于采样后的实际训练分布（不要按原始分布算完叠在采样后数据上）`

**评审意见**：⚠️ **当前代码已经是基于当前 batch 计算的**

当前逻辑（[vla_loss.py:43-49](../idv_agent/training/vla_loss.py:43-49)）：
```python
if weights.move_direction_balance:
    # 基于**当前 batch**的分布计算权重
    counts = torch.bincount(batch["move_target"].reshape(-1), minlength=9)
    move_class_weight = counts.clamp_min(1.0).rsqrt()
```

**问题**：
- 当前是 **per-batch** 动态计算，不是全局预计算
- batch=1 时，每个 batch 只有 4 个移动步（1样本 × 4步/样本）→ 分布极不稳定
- batch=8 时，每个 batch 有 32 个移动步 → 稍微稳定

**用户可能想要的**：
```python
# 在训练开始前，遍历整个采样后的训练集
def precompute_class_weights(dataset):
    all_moves = []
    for sample in dataset:
        all_moves.extend(sample["move_target"].tolist())
    counts = torch.bincount(torch.tensor(all_moves), minlength=9)
    weights = counts.clamp_min(1.0).rsqrt()
    return weights / weights.mean()  # 归一化

# 在 train() 开头调用
global_move_weights = precompute_class_weights(dataset)
```

**建议**：
- 如果 batch >= 8：保持当前 per-batch 动态计算（更适应数据分布变化）
- 如果 batch = 1-2：改用全局预计算（避免单 batch 分布不稳定）

---

#### 2.3 move_stop_weight 从 0.25 调至 0.5 ✅ **同意**

**评审意见**：✅ **合理**

- 0.25 在 12% 覆盖率下太激进
- 0.5 是更温和的起点，配合 `move_direction_balance` 仍能有效降权

---

#### 2.4 manifest 写入 checkpoint 内部 ✅ **同意**

用户建议：`torch.save 带 meta 或实验目录隔离`

**评审意见**：✅ **这是最佳实践**

当前问题：
```python
# train_vla.py:432
write_manifest(checkpoint.parent / "manifest.json", manifest)
```
- manifest.json 在 `checkpoints/M3_ACT/`，多次实验会覆盖
- checkpoint 和 manifest 没有强绑定

**建议实现**：
```python
# 方案 A：写入 checkpoint 内部
torch.save({
    "adapter": {...},
    "core": {...},
    "manifest": manifest,  # 直接写入
    "step": args.steps,
}, checkpoint)

# 方案 B：实验目录隔离
checkpoint_dir = Path(f"checkpoints/M3_ACT/{timestamp}_{experiment_name}")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
torch.save({...}, checkpoint_dir / "model.pt")
write_manifest(checkpoint_dir / "manifest.json", manifest)
```

推荐**方案 A**，简单且不会错配。

---

### 三、验证门（本次强制执行）✅ **全部同意**

#### 3.1 训练日志：记录分量 loss ✅
```python
# 每 N 步记录一次
if step % log_interval == 0:
    print(f"step {step}: total={total:.3f}, slow_intent={losses['slow_intent']:.3f}, "
          f"slow_subgoal={losses['slow_subgoal']:.3f}, fast_move={losses['fast_move']:.3f}")
```

#### 3.2 训练结束输出预测分布报告 ✅
```python
def prediction_summary(model, loader):
    move_preds = []
    intent_preds = []
    for batch in loader:
        out = model(...)
        move_preds.extend(out.fast.move_logits.argmax(-1).tolist())
        intent_preds.extend(out.slow.intent_logits.argmax(-1).tolist())
    
    move_counter = Counter(move_preds)
    intent_counter = Counter(intent_preds)
    
    print(f"Move预测分布: {move_counter}")
    print(f"非stop比例: {(len(move_preds) - move_counter[0]) / len(move_preds):.1%}")
    print(f"Intent预测分布: {intent_counter}")
```

#### 3.3 32 条过拟合重跑 ✅
```python
# 在主训练脚本中添加
parser.add_argument("--overfit-check", action="store_true")
if args.overfit_check:
    dataset = _bounded_subset(dataset, 32)
    print("=== OVERFIT CHECK MODE: 32 samples ===")
```

---

### 四、预期管理 ✅ **合理**

- move 非 stop > 30% ← 合理（当前 0%）
- intent 评估在 kite 划分调整后重新做 ← **关键**

**补充建议**：
- 除了 move 非 stop 比例，还要看**各方向是否都有预测**（不要只预测 north）
- intent 预期：4 类都有预测，不再 100% travel

---

### 五、执行顺序 ⚠️ **建议调整**

用户建议：
```
划分修复 → 意图分布报告 → 代码修复 → 32 条过拟合验证
→ 500 步正式训练 → 预测分布报告
```

**评审意见**：⚠️ **32条过拟合应该在500步之前，但建议先做代码修复**

**建议顺序**：
```
1. 数据划分修复 + 意图分布验证门
   ↓
2. 代码修复（batch上限、weight计算、manifest嵌入、日志增强）
   ↓
3. 32条过拟合验证（验证代码修复有效）
   ↓ 
4. 1000步中等规模验证（覆盖率 ~24%，看趋势）
   ↓
5. 8000步正式训练（2 epoch，覆盖率 200%）
   ↓
6. 预测分布报告 + 完整评估
```

**原因**：
- 32条过拟合在代码修复**之后**，才能验证修复效果
- 500步覆盖率只有 12%，建议直接跳到 8000步（或先 1000步试探）

---

## 总结

### ✅ 完全同意的部分
1. **数据划分修复** — 这是 kite 问题的根本解决方案
2. **意图分布验证门** — 在划分阶段拦截，不等训练后
3. **move_stop_weight 0.5** — 更温和的起点
4. **manifest 写入 checkpoint** — 防止元数据错配
5. **训练日志增强 + 预测分布报告** — 不再依赖事后诊断
6. **32条过拟合验证** — 检查类别记忆能力

### ⚠️ 需要调整的部分
1. **batch 8-16** — 先验证显存，建议从 batch=4 开始试探
2. **class weight 预计算** — 需要明确是全局预计算还是 per-batch，取决于 batch size
3. **执行顺序** — 建议先代码修复再过拟合验证，500步改为1000或8000步

### 🎯 最关键的改进
**数据划分修复** 比所有代码改动都重要。kite 训练集 0 个、验证集 34 个，这是方案设计问题，不是超参问题。

---

## 推荐实施方案

```bash
# Phase 1: 数据划分修复
python scripts/prepare_train_val_split.py \
  --ensure-all-intents \
  --output-report data/split_report.json

# Phase 2: 代码修复（见下方清单）

# Phase 3: 32条过拟合
python idv_agent/scripts/train_vla.py \
  --max-samples 32 \
  --steps 200 \
  --batch-size 4 \
  --move-direction-balance \
  --move-stop-weight 0.5 \
  --checkpoint checkpoints/debug/overfit_32.pt

# Phase 4: 1000步试探
python idv_agent/scripts/train_vla.py \
  --max-samples 2048 \
  --steps 1000 \
  --batch-size 4 \
  --move-direction-balance \
  --move-stop-weight 0.5 \
  --checkpoint checkpoints/M3_ACT/test_1k.pt

# Phase 5: 8000步正式训练（2 epoch）
python idv_agent/scripts/train_vla.py \
  --max-samples 4096 \
  --steps 8000 \
  --batch-size 4 \
  --move-direction-balance \
  --move-stop-weight 0.5 \
  --teacher-forcing-decay-steps 2000 \
  --checkpoint checkpoints/M3_ACT/act_v3.pt
```

### 代码修复清单
1. `train_vla.py:314` — 移除 `min(args.batch_size, 2)` 硬上限，改为自适应或用户控制
2. `train_vla.py:345` — 添加全局 class weight 预计算（如果 batch < 4）
3. `train_vla.py:399` — manifest 写入 checkpoint 内部
4. `train_vla.py:372` — 训练循环中每 50 步记录分量 loss
5. `train_vla.py:445` — 训练结束输出预测分布报告

需要我帮你实现这些代码修复吗？
