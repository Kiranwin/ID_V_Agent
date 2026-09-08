# 15 · m28 状态—决策接口与时序缓存协议

> 本文冻结快慢 VLA 的实现边界。模型、数据集和部署调度器必须遵守本协议；如果实现需要改变其中任一项，先修改本文和 schema，再改代码。

## 1. 总体决定（m28 当前协议）

采用**单一共享 M0 + 空间 State Trunk + 状态条件 Joint Planner**，不是多个并列动作头：

```text
8 帧图像 + 任务条件
  → M0 单帧 backbone（每帧最多一次）
  → FrameFeatureCache（滚动缓存）
  → ordered spatial feature + last-first delta
  → State Trunk z
      ├─ state heads：intent/subgoal/grounding/phase/target/steering/path
      └─ Joint Planner(z + predicted soft beliefs)：move/camera/Q/duration
  → h0 执行 6 帧 → 重新观察
```

m28 的 slow/fast temporal 分支仍保留为兼容与诊断路径，但不再是部署动作的第二套决策器；
真正执行的动作只来自 `StateDecisionExpert` 的 Joint Planner。不能复制一套完整视觉语言
backbone。架构变更记录见 [docs/18-架构变更历史.md](18-架构变更历史.md)。

## 2. 时序建模位置（冻结）

选择：**backbone 逐帧编码，时序编码器在 backbone 之后聚合**。

不采用把 3～8 帧拼成一次多图输入的方案，原因是：

- 每个新时间步都会重新计算整段视觉语言输入；
- 帧数增加时延迟和显存近似线性增加；
- 运行时无法复用已经计算过的旧帧特征。

### 2.1 具体接口

任务指令和 `mode` 在一个 session/episode 内是常量。文本侧只编码一次，结果放入
`TaskConditionCache`；逐帧的 `encode_frame` 只执行图像侧 backbone 和已有条件的投影，
不能每帧重新 tokenize 或计算 instruction/mode embedding。

```python
class TaskConditionCache:
    instruction_embedding: Tensor  # episode 内固定
    mode_embedding: Tensor         # episode 内固定
    task_id: str
    mode: str

task_cache = backbone.encode_task_once(instruction, mode)
frame_feature = backbone.encode_frame(
    image,
    instruction_embedding=task_cache.instruction_embedding,
    mode_embedding=task_cache.mode_embedding,
)  # 每帧只跑图像侧
```

```python
frame_feature = backbone.encode_frame(image, task_cache)  # [D]

temporal_feature = temporal_encoder(
    frame_feature_window,          # [L, D], L <= 8
    frame_time_deltas              # [L]
)  # [D_t]
```

`L=8` 是运行时缓存和训练的上限；短序列（最少 3 帧）允许在录制丢帧或会话开头使用，
通过 `valid_mask` 标记，不补造图像。第一版训练默认使用 8 帧滚动窗口，仍可从旧的
3 帧样本迁移为带 mask 的短窗口。

诊断时序编码器使用轻量 causal GRU，不能再次调用 M0；它只处理已经缓存的
`frame_feature`。但 m28 的 deployed planner 使用 ordered spatial State Trunk，
不把 slow/fast 两个 GRU 的输出当作第二套动作决策。

## 3. m28 状态—决策融合机制（当前冻结）

选择：**共享视觉状态 + 模型预测 belief 的 soft embedding + Joint Planner**。

不采用裸向量相加，也不采用 cross-attention 或离散扩散；当前 MVP 需要确定性、低维、
可审计的滚动动作。

```python
state = StateTrunk(ordered_spatial_window)
belief = predict_state_heads(state)
planner_input = concat(state, soft_embedding(belief))
action_h0 = JointPlanner(planner_input)[..., 0]
```

`belief` 必须是模型预测的 logits/probabilities；训练真值只进入 masked loss，不能替换
planner 输入。`intent` 表示任务阶段，`phase/steering/path` 表示当前镜头/路径策略，
grounding 表示画面事实；它们共同决定 move/camera/Q，而不是各自拥有互不协调的动作头。

旧 m27 的 FiLM/slow-condition 梯度耦合和并列动作头保留在历史代码路径中，只供兼容/诊断；
m28 当前验收不以它证明模块协同，而以 Joint Planner 的共享状态梯度和状态 ablation 证明。

接口定义：

```python
class SlowVLAOutput:
    intent_id: int
    subgoal_id: int
    context_embedding: Tensor  # [256]
    refresh_policy: int

class StateDecisionInput:
    spatial_feature_window: Tensor  # [L, cells, D] 的有序视觉特征
    task_condition: Tensor          # episode 内固定
    history_actions: Tensor         # 仅作诊断/协议输入，不改写 m28 camera prior
```

## 4. FrameFeatureCache 运行时协议（冻结）

缓存是 m28 单模型状态—动作闭环的必需组件，不是可选优化。

```python
class FrameFeatureCache:
    max_length = 8
    # 每项：frame_index, timestamp_ns, feature[D], valid
    append(item) -> None
    window(length=8) -> (features[L,D], timestamps[L], valid_mask[L])
    latest() -> item
```

捕获频率和 planner 推理频率**解耦**。捕获线程只负责把最新帧编码后写入缓存；planner
循环按自己的固定周期读取缓存中的最新窗口，不等待也不逐帧绑定捕获节奏：

```text
Capture loop（例如 30 FPS）：
  grab → M0.encode_frame 一次 → append cache

Planner loop（例如 10/15 Hz）：
  按固定 deadline 读取 cache.latest_window(8)
  → State Trunk → state heads → Joint Planner → 执行 h0
```

如果捕获频率高于快头，期间的特征只进入缓存，不额外触发快头；如果快头快于捕获，
它读取同一最新窗口并依据时间戳/`valid_mask` 工作，不能重复调用 backbone 伪造新帧。
每个采集帧的 `M0.encode_frame` 最多调用一次。

快循环的固定周期由 `next_deadline += 1 / fast_hz` 维护，不能使用“处理完上一帧后再
sleep”的相对延时写法。慢循环独立调度，不能阻塞快循环。

每个 h0 动作执行 6 帧后必须重新读取视觉窗口并重算状态；不把同一旧窗口下的后续
动作提前发送。`stop_or_replan` 只能作为 planner 的风险/重规划信号，不能替代重新观察。

缓存要求：

- 读写由同一个推理线程或读写锁保护；
- 图像帧、特征和时间戳必须一一对应；
- 丢帧时保留时间戳间隔并设置 `valid_mask=0`，禁止静默复制上一帧伪造时序；
- 慢头刷新不能阻塞快头，慢头计算期间快头继续使用上一份有效条件；
- 若缓存为空或慢层条件过期，快头输出 `stop_or_replan`，由安全策略接管。

目标频率暂为 15 Hz。频率验收必须测完整的 `encode_frame + State Trunk + state heads +
Joint Planner + executor handoff` 路径；不能只测旧 fast head 而漏掉状态预测开销。

## 5. 慢标签采样粒度（冻结）

### 5.1 标注粒度

慢标签以**帧为索引、片段为语义来源**：人工标注的是时间片段，推荐使用
`scripts/init_intent_segments.py` 生成候选边界，再由人工只确认顶层 intent；构建器把片段标签
前向扩展到片段内的每一帧。每帧都必须有：

```json
"slow_label": {
  "valid": true,
  "segment_id": "seg_003",
  "intent": "search",
  "subgoal": "find_cipher",
  "subgoal_source": "rule",
  "subgoal_rule_version": "subgoal.v1"
}
```

片段边界之外或无法判断的帧使用 `valid=false`，不能用相邻片段跨边界填充。
`context_embedding` 不人工标注，由慢头根据帧和上述离散标签学习生成。

### 5.2 训练采样

`vla_chunks.jsonl` 的每条样本仍以 `anchor_frame` 为中心，同时保存 anchor 的慢标签，
并在 `observations.frames[]` 中保存每个历史帧对应的 `slow_label`。例如：

```json
{
  "observations": {
    "frames": [
      {"frame_index": 120, "slow_label": {"valid": true, "intent": "search", "subgoal": "find_cipher", "subgoal_source": "rule"}},
      {"frame_index": 123, "slow_label": {"valid": true, "intent": "search", "subgoal": "find_cipher", "subgoal_source": "rule"}},
      {"frame_index": 126, "slow_label": {"valid": true, "intent": "decipher", "subgoal": "face_cipher", "subgoal_source": "rule"}}
    ]
  },
  "slow_label": {"valid": true, "intent": "decipher", "subgoal": "face_cipher", "subgoal_source": "rule"},
  "loss_mask": {"slow": 1, "fast": 1}
}
```

`loss_mask.slow` 由真实时间戳决定，而不是简单用全局帧号取模：

```text
slow_loss_mask = 1，当 anchor_timestamp
                  到上一次慢层训练采样时间已达到 slow_period
                  或该帧是事件触发点；否则为 0。
```

默认 `slow_period = 1.0s`，允许配置为 0.5～2.0s。`loss_mask.fast` 对所有有效动作块
样本为 1。这样每帧都有慢标签，但慢头不会被按快层频率重复计权。

subgoal 的固定词汇、intent 映射和半自动派生规则见
[docs/16-Subgoal词汇表v1.md](16-Subgoal词汇表v1.md)。人工标注只要求顶层 `intent`；
`subgoal` 必须记录 `subgoal_source` 和规则版本。

### 5.3 与现有 v3 的关系

v4 不保留顶层 `intent` 字段，避免它与 `slow_label.intent` 产生双重真源；训练器只读取
anchor/历史帧中的 `slow_label.intent`。现有 v3 记录中的 `intent` 空字符串仅用于兼容，新的 VLA 训练集必须包含
`slow_label` 和 `loss_mask`。新增字段后将使用 `vla.action_chunk.v4`；旧 v3 只作为
兼容/迁移输入，不能直接送入快慢 VLA 训练。`build_vla_chunks.py` 后续改造时必须从
帧级/片段级标注读取这些字段；不能在训练 collator 中临时猜测慢标签。

## 6. 训练与部署边界

- WK/VG 训练更新 M0 的共享语义能力；
- ACT 训练同时更新 `slow_temporal`、`fast_temporal`、Slow Head 和 Fast Head；
  两个 GRU 参数独立，但都下游于同一个 FiLM 条件融合（见第 3 节耦合说明）；
- 快慢头使用联合损失，但先冻结 backbone，再逐步解冻顶部 LoRA；
- 推理只有一个 M0 特征流，慢头和快头不各自重复跑视觉 backbone（各自的时序 GRU
  仍要分别跑一次，但都消费同一份 `frame_feature`，不重复调用 M0）；
- 如果未来因显存或延迟必须拆成两个推理进程，仍从同一个 M0 checkpoint 导出，并视为
  部署复制，不得进行两套互不约束的独立微调。

### 6.1 m28 状态预测的训练/推理闭环（冻结）

m28 训练不能把真实 intent/grounding/path 标签喂给 planner，否则会产生 state exposure
bias。第一阶段所有 planner 输入都必须是 state heads 的预测 logits，经 soft embedding
进入 Joint Planner；真实 sidecar 只进入 masked auxiliary loss。

```text
视觉窗口
        → shared State Trunk z
        → state heads → predicted belief
        → soft embedding(predicted belief) + z
        → Joint Planner → move/camera/Q
```

当前不采用 teacher forcing 直接替换 planner 状态。若未来增加跨 tick belief state，
必须先增加连续决策序列数据，并单独验收 hidden-state reset、延迟和状态漂移。
