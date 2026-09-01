# 15 · 快慢 VLA 接口与时序缓存协议

> 本文冻结快慢 VLA 的实现边界。模型、数据集和部署调度器必须遵守本协议；如果实现需要改变其中任一项，先修改本文和 schema，再改代码。

## 1. 总体决定

采用**单一共享 M0 + 两个任务头**，不是两个独立微调模型：

```text
每帧图像
  → M0 单帧 backbone（每帧最多一次）
  → FrameFeatureCache（滚动缓存）
  → 共享时序编码器
      ├─ Slow Head：低频 intent/subgoal/context
      └─ Fast Head：高频 action chunk
```

慢头和快头可以有各自的小型投影层或 LoRA target，但不能复制一套完整视觉语言
backbone。模式适配器仍按 `M0 + mode LoRA` 管理。

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

时序编码器使用轻量 causal Transformer/GRU，不能再次调用 M0。它只处理已经缓存的
`frame_feature`，因此增加历史帧不会重复运行 backbone。

## 3. context_embedding 融合机制（冻结）

选择：**离散条件嵌入 + 连续 context 经过 MLP 后对每个视觉帧特征做 FiLM 调制**。

不采用裸向量相加（尺度和语义不稳定），第一版也不采用 cross-attention（会增加每步
的 token 交互和 KV 开销）。

```python
condition = MLP(concat(
    Embedding(intent_id),       # 64 维
    Embedding(subgoal_id),      # 64 维
    LayerNorm(context_embedding), # 256 维
    mode_embedding,              # 32 维
))                              # → gamma[D], beta[D]

conditioned_frame = LayerNorm(
    (1 + tanh(gamma)) * frame_feature + beta
)
```

同一个 `gamma/beta` 应用于当前窗口中的每个帧特征，然后再进入共享时序编码器；快头
从时序特征读取慢层条件。这里使用的是**上一份有效的慢层条件**（`condition_{t-1}`），
避免慢头和当前时刻的时序特征形成循环依赖；慢头在消费完当前窗口后生成
`condition_t`，从下一采集帧开始生效。慢头输出的 `context_embedding` 维度固定为 256。
慢头 warm-up 阶段，快头损失对慢头输出使用 stop-gradient；联合训练稳定后再解除该限制。

接口定义：

```python
class SlowVLAOutput:
    intent_id: int
    subgoal_id: int
    context_embedding: Tensor  # [256]
    refresh_policy: int

class FastVLAInput:
    temporal_feature: Tensor    # [D_t]
    slow_condition: Tensor      # [C], 由上述 MLP 生成
    history_actions: Tensor
```

## 4. FrameFeatureCache 运行时协议（冻结）

缓存是单模型多头部署的必需组件，不是可选优化。

```python
class FrameFeatureCache:
    max_length = 8
    # 每项：frame_index, timestamp_ns, feature[D], valid
    append(item) -> None
    window(length=8) -> (features[L,D], timestamps[L], valid_mask[L])
    latest() -> item
```

捕获频率和快头推理频率**解耦**。捕获线程只负责把最新帧编码后写入缓存；快循环按
自己的固定周期读取缓存中的最新窗口，不等待也不逐帧绑定捕获节奏：

```text
Capture loop（例如 30 FPS）：
  grab → M0.encode_frame 一次 → append cache

Fast loop（例如 10/15 Hz）：
  按固定 deadline 读取 cache.latest_window(8)
  → temporal_encoder → Fast Head → 执行动作块

Slow loop（例如 1 Hz 或事件触发）：
  读取同一 cache.latest_window(8)
  → Slow Head → 更新 condition_t
```

如果捕获频率高于快头，期间的特征只进入缓存，不额外触发快头；如果快头快于捕获，
它读取同一最新窗口并依据时间戳/`valid_mask` 工作，不能重复调用 backbone 伪造新帧。
每个采集帧的 `M0.encode_frame` 最多调用一次。

快循环的固定周期由 `next_deadline += 1 / fast_hz` 维护，不能使用“处理完上一帧后再
sleep”的相对延时写法。慢循环独立调度，不能阻塞快循环。

动作块执行期间，快头仍按固定周期滚动重预测；是否提前打断当前 chunk 由
`stop_or_replan` 决定。

慢头刷新发生在第 6 步之后；刷新完成前，快头继续使用 `condition_{t-1}`。因此慢头
即使超时或报错，也不会暂停快头主循环。

缓存要求：

- 读写由同一个推理线程或读写锁保护；
- 图像帧、特征和时间戳必须一一对应；
- 丢帧时保留时间戳间隔并设置 `valid_mask=0`，禁止静默复制上一帧伪造时序；
- 慢头刷新不能阻塞快头，慢头计算期间快头继续使用上一份有效条件；
- 若缓存为空或慢层条件过期，快头输出 `stop_or_replan`，由安全策略接管。

目标频率为 15 Hz。15 Hz 是否达标只测 `encode_frame + temporal_encoder + fast_head` 的
路径；慢头耗时单独统计，不计入快头每步预算。

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

现有记录中的 `intent` 空字符串仍可被旧样本校验通过，但新的 VLA 训练集必须包含
`slow_label` 和 `loss_mask`。新增字段后将使用 `vla.action_chunk.v4`；旧 v3 只作为
兼容/迁移输入，不能直接送入快慢 VLA 训练。`build_vla_chunks.py` 后续改造时必须从
帧级/片段级标注读取这些字段；不能在训练 collator 中临时猜测慢标签。

## 6. 训练与部署边界

- WK/VG 训练更新 M0 的共享语义能力；
- ACT 训练共享时序编码器、Slow Head 和 Fast Head；
- 快慢头使用联合损失，但先冻结 backbone，再逐步解冻顶部 LoRA；
- 推理只有一个 M0 特征流，慢头和快头不各自重复跑视觉 backbone；
- 如果未来因显存或延迟必须拆成两个推理进程，仍从同一个 M0 checkpoint 导出，并视为
  部署复制，不得进行两套互不约束的独立微调。
