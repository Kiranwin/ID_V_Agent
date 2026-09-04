# ACT 视觉动作专家设计

## 目标与证据

目标不是让中间特征对图像有微小变化，而是使部署时输出的 move、camera、intent 必须依赖画面，并通过 image-zero、image-shuffle 两个严格反事实门禁。

冻结 Qwen 的空间网格线性 probe 已在 79 个训练 session、20 个保留 session 上证明存在可学视觉信号：只用最新画面的 move 准确率为 70.55%，超过 62.50% 的多数类基线；只用画面的 decipher/travel macro recall 为 92.27%。因此问题不在采集、回放、v5 标签或视觉塔。

现行 m6 堆栈将画面、时间和 history 先融合再由同一 fast head 分类。即使 cell projector、GRU input residual、post-GRU visual residual、history dropout 已加入，最终 GRU different-scene cosine 仍为 0.999938，且 normal/image-zero/image-shuffle 指标完全一致。历史和慢层先验可在共享 logits 上覆盖视觉梯度；继续增加 residual 没有可证伪性。

## 架构

保留已实现的冻结 Qwen 8x8 token、SpatialCellProjector、8 帧 v5 输入和独立 fast/slow GRU。新增 `VisualActionExpert`，直接从最后有效帧的保胞投影特征和“末帧减首有效帧”变化特征产生：

```text
f_last, Δf = f_last - f_first               # 每项 2560，视觉专属
  -> visual trunk (Linear -> GELU -> Linear -> GELU)
  -> visual fast heads: move/dx/dy/buttons/duration/intent-context
  -> visual slow heads: intent/subgoal/context/refresh

temporal + history -> existing fast/slow prior heads
final categorical logits = visual_logits + 0.5 * prior_logits
final duration = clamp(visual_duration + 0.5*(prior_duration - 6), 1, 30)
```

视觉 expert 的 logits 永远是最终 logits 的加数；history 只能以固定上界 0.5 作残差修正，不能通过可学习 gate 把视觉分支置零。视觉 expert 的输入不含 `history_actions`。同一视觉 expert 会单独承担所有有真值的 action/intent/subgoal loss（与主输出同一类别权重及 mask），使画面预测能力是一个训练目标而不是希望发生的副作用。部署只使用模型预测，绝不输入真值。

`VisualActionExpert` 复用 `FastVLAHead(history_action_dim=0)` 和 `SlowVLAHead` 的输出契约，新增 `VisualExpertOutput` 暴露 `fast`、`slow`；`FastSlowVLAOutput` 增加 `visual`。最终输出仍在既有 `.fast` / `.slow` 字段，故 ActionChunkPolicy、ActionDecoder 与执行器无需改变动作协议。

## 训练、缓存与 checkpoint

`compute_vla_loss` 接收可选 visual fast/slow 输出，新增逐头 `visual_*` 损失并以 `visual_aux=1.0` 加入 total。类别权重继续每头独立、bounded inverse-sqrt `[0.35, 3.0]`，训练图像增强与 history dropout=0.5 继续仅用于训练。

新 checkpoint schema 为 `m7_act.visual_expert.v1`。旧 m6 与 M2 均 fail-closed。checkpoint core state 自然包含 visual expert；adapter state 仍严格为 raster projector 和 condition projection。M2 不加载。

raw feature cache 仍保存冻结 `[64,1024]`，训练时重新运行可训练 SpatialCellProjector，因此不缓存 expert 或其随机增强输出。增强与 raw feature cache 仍互斥。当前 cache 与 decorrelated session 的覆盖不完整，此次严格训练继续原图实时视觉路径，不能把不完整 cache 当作训练数据。

## 验收

1. 单元：改变 `f_last` 或 `Δf` 必须改变 visual logits；history 改变不得改变 visual logits；主 logits 必须按精确公式融合 visual + 0.5 prior。
2. 训练：30--60 step FP16+GradScaler 训练有限、loss 下降、无非法梯度。
3. 深层：same-frame cosine >= 0.999，different-scene final fast feature < 0.99；左右遮挡 feature L2 和 output L1 均 > 1e-4。
4. 反事实：image-zero 与 image-shuffle 对 move/camera/intent 均下降至少 `max(0.05, normal * 0.15)`。
5. 分类：camera 0 桶误转 <15%，稀有桶和 decipher recall 提升不以多数类崩坏为代价。没有比较 baseline 时报告为不可证明，不能声明“类别验收通过”。
6. 未通过任一视觉或分类硬门禁的 checkpoint 不得用于 MVP、dry-run 或部署。

## 不变约束

- ACT 从 base Qwen 初始化；M2 继续禁用。
- 2080 Ti 使用 FP16 + GradScaler。
- v5 协议不变：8 帧、stride=3、8 条宏动作 history、每步 6 帧、30 FPS、`time_deltas=None`。
- 文档仅写入 `docs/`。
