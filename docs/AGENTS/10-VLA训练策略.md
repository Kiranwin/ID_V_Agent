# 10 · VLA 多模式与 ActVLP 训练策略

> 2026-09-08：新 MVP 标签/时间设计以 [03 §0](03-数据格式.md) 为准。以下为旧 v5
> 训练实现。v6 导出不能喂给当前 train_vla；固定 decipher 不训练单类别 intent CE，
> 用户最新要求 Q=当前 interact_prompt，允许短时连续 Q。Q 独立 mask/BCE，不读
> replay Q、过去 Q 或 decoding 结果来抑制；损失组件已实现，旧训练器尚未迁移。

> M29 SGAN 已实现：预测 facts/phase/steering/path 经过 soft beliefs 才能进入
> navigation；原始视觉到导航的旁路被禁止。请以 [03 §0.13](03-数据格式.md) 的
> checkpoint、介入和独立 Q/状态验收为准。M29 训练入口是 `train_m29`，不是旧 v5
> `train_vla`；没有完整 decoding 和导航审核覆盖时只允许运行 `--task q`。

## 1. 多模式 LoRA 路线

标准模式数据训练共享底座 `M0`。M0 固定：

- 视觉输入与时序编码方式；
- VLA v2 动作空间（九方向移动、五档相机、六个按键状态）；
- 键鼠执行器的底层交互语义；
- 因果帧-动作对齐和动作块长度。

新模式只训练参数高效适配器：

```text
M0 + lora_joint_hunt
M0 + lora_blackjack
```

录制命令的 `--mode` 写入 raw session 的 `meta.json`；构建 v3 `vla_chunks.jsonl` 时复制
为样本顶层 `mode`，并生成语言条件 token：

```text
<mode:standard>
<mode:joint_hunt>
<mode:blackjack>
```

模式 token 只作为条件，不改变共享动作头的维度。若某模式没有某个动作，使用数据
分布和可选动作 mask 约束，不新建第二套 schema。模式切换时加载对应 LoRA；未提供
适配器时回退 M0。LoRA 合并/量化必须以 2080 Ti 的显存和实测延迟为准，不预设多适配器
同时驻留。

## 2. ActVLP 的选择性借鉴

JARVIS-VLA 的 ActVLP 提供的核心启发是：不要只用轨迹动作数据训练 VLA，先用与任务
相关的非轨迹视觉语言数据改善模型的知识、识别和空间 grounding，再做轨迹动作学习。
本项目只采用这个训练顺序，不照搬其 Minecraft 数据、手柄动作 token、模型规模或 PPO。

### 阶段 WK · 游戏知识后训练（文本为主）

训练入口为 `idv_agent/scripts/train_wk_lora.py`，输出 `M1_WK`；只接受已审核样本，
并将 `<mode:standard|joint_hunt|blackjack>` 写入每条监督 prompt。WK 完成后必须按
[17-三阶段训练继承协议.md](17-三阶段训练继承协议.md) 传递给 VG，再传递给 ACT。

冻结视觉编码器，只用 LoRA/QLoRA 训练语言部分，数据来自人工审核的第五人格知识库：

- PC 键位及交互前置条件（Q 交互、Space 翻越/校准、E 治疗等）；
- 标准、联合狩猎、黑杰克的规则差异；
- 密码机、狂欢之椅、板窗、出口等对象的功能和状态转移；
- “看到什么状态时可以做什么”的短问答和反事实问答。

目标是让模型掌握游戏语义，不让它直接学习按键轨迹。知识答案必须来自版本化、人工
审核的资料；模型生成的问答只能作为候选，不能未经审核进入训练集。

### 阶段 VG · 视觉语言与空间 grounding

VG 必须加载 WK 阶段输出的 `M1_WK`，保留并冻结 `lora_wk`，再训练视觉 grounding 适配器/投影，输出 `M2_VG`；不得从原始 Qwen 重新开始。完整继承链见 [17-三阶段训练继承协议.md](17-三阶段训练继承协议.md)。

用新录制帧和少量标注构造图像问答、对象定位、遮挡/高亮识别样本，例如：

- “画面中是否存在密码机/交互提示？”
- “扳手图标位于画面哪一侧？”
- “目标被墙遮挡时，黄色高亮代表什么？”

YOLO 框可作为标注工具和辅助字段，但模型推理不能强依赖 YOLO。增强只使用保持游戏
几何语义的裁剪、亮度、色彩和轻微平移，不使用会改变方向或键位含义的水平翻转。

### 阶段 ACT · VLA 动作块监督

长期协议可评估加载 VG 阶段输出的 `M2_VG`；但当前 M2/VG 有已知质量问题，ACT 一律从
基础 Qwen 初始化，禁止传 `--init-checkpoint`，并在 manifest 记录
`base_without_m2`。ACT 优化视觉 action expert、控制辅助头和 fast/slow 头；部署不读取
任何标注 JSONL。完整数据契约见 [03-数据格式.md](03-数据格式.md) §7。

在继承后的模型上训练因果动作块：

```text
最多 8 帧滚动窗口（最少 3 帧，valid_mask 标记短窗口）+ <mode:...> + 任务指令 + 历史动作
→ 4 步 move_dir / camera buckets / buttons / duration
```

动作训练只使用通过对齐校验的新 `vla_raw` session。当前协议训练与执行只消费第 0 个
未来宏动作（6 帧约 200ms），然后重新观察；模型保留四步输出张量仅为 shape 兼容。
标准模式先训练 M0，其他模式在固定 M0 上训练 LoRA。第一版不做 PPO 或在线探索。

#### 当前 ACT 训练输入（MVP）

训练前必须运行 `audit_mvp_training_data`；它验证 v5 schema、时间对齐、整 session split、
control replay 与 canonical h0 同源、以及权重统计来源。当前 canonical 数据为：

```text
data/mvp_vla_v5_event_centered_decision_label_v3_rebucket675
  train: 732 chunks / 79 sessions
  val:   182 chunks / 20 sessions
```

训练标签分三层，不能混写：

1. `slow_label + action_chunk[h0]` 是不可变的 Raw Input 行为复刻；相机 bucket 与执行器
   像素查表同源。
2. `act_grounding_annotations.jsonl` 是训练期画面事实辅助真值；缺标行使用 mask，不视为
   no-cipher 负例。
3. `camera_control_{train,val}.*.annotated.jsonl` 是训练期镜头意图辅助真值；train 与 val
   文件必须显式分开，模型只消费预测 logits，runtime 不读取 sidecar。

m28 的唯一 deployed 动作路径是 `StateDecisionExpert`：完整有序视觉窗口先生成共享
state `z`，从 `z` 预测 intent、grounding 和 navigation control；Joint Planner 再读取
`z + soft(predicted state)` 联合生成 move/camera/Q。它不是使用标注真值的规则状态机，
也不是把 `desired_turn` 作为独立、不会执行的旁路头。

主类损失采用每头独立、训练集 h0 统计的 inverse-sqrt 权重：`clip(w, 0.35, 3.0)`，
每头均值归一化为 1。控制六头同样独立平衡，六个 CE 先取均值，再由单一
`camera_control_loss_weight` 调节相对强度；不得对所有头做全局归一化。

图像增强只在训练加载期间启用，验证/部署/feature cache 路径恒为原图。一个 8 帧窗口
共享同一组 hue、saturation、brightness、contrast、translate 参数，避免伪造时序变化；
启用 `--image-augmentation` 时禁止 `--raw-feature-cache`。

当前完整训练调用的核心形状如下（按 checkpoint 路径替换）：

```powershell
python -m idv_agent.scripts.train_vla `
  --data data/mvp_vla_v5_event_centered_decision_label_v3_rebucket675/train `
  --val-data data/mvp_vla_v5_event_centered_decision_label_v3_rebucket675/val `
  --grounding-annotations data/mvp_act_grounding_pool_v1/act_grounding_annotations.jsonl `
  --val-grounding-annotations data/mvp_act_grounding_pool_v1/act_grounding_annotations.jsonl `
  --camera-control-annotations data/mvp_act_grounding_pool_v1/camera_control_train.v2.rebucket675.annotated.jsonl `
  --val-camera-control-annotations data/mvp_act_grounding_pool_v1/camera_control_val.v2.rebucket675.annotated.jsonl `
  --grounding-annotated-fraction 0.50 `
  --camera-control-annotated-fraction 0.35 `
  --execution-horizon 1 --camera-prior-scale 0 `
  --class-balance --image-augmentation
```

control 行是 grounding 行的严格子集。上述联合采样把 control / other-grounding / ordinary
ACT 的目标 draw mass 固定为 `0.35 / 0.15 / 0.50`，而不是让 48 条 control 行被 128 条
grounding 行稀释成约 18.75%。这提高标签曝光，不代表 48 条已足够泛化。

### m28 的时序与耦合边界

两份实时控制架构审查带来的可执行约束是：实时控制先是观测—状态—动作的闭环，而不是
语言生成或随机采样问题。当前不引入离散扩散、CFG、Top-k、熵正则、VQ-VAE 或 IRL；
这些方法不能修复错误的观测—动作对齐或稀缺的路径语义标签。

- h0 是唯一可以执行的动作，固定 6 帧后必须重观测；训练、离线指标和 executor 都以此为准。
- 需要记录并验收实际 `capture_ts → inference_start/end → command_send → first_post_action_capture`
  时延；`action_delay_frames=1` 是数据构建约定，不是部署延迟已被证明相等。
- move/camera 当前由同一 planner state 生成，但最后仍是多个 factorized output head；
  因此 smoke 必须增加联合动作验收（前进+左右转、path_follow+移动、Q prompt 时停转+Q），
  不能只看各头独立 accuracy。
- 当前 m28 不把 action history 作为 deployed planner 的主输入，避免历史复读捷径；若要
  加入上一已执行宏动作作为本体状态，必须单独做 history-zero/image-zero 消融，证明它没有
  重新替代图像。

## 3. 资源与部署约束

- RTX 2080 Ti 无 BF16：训练使用 FP16 + GradScaler；必要时用梯度累积和梯度检查点。
- 世界知识/视觉 grounding 阶段优先小规模 LoRA，不追求 JARVIS-VLA 论文中的数据规模。
- 快速动作推理必须保持滚动 chunk；知识问答不进入每帧控制链路。
- 评估拆开记录：知识准确率、空间 grounding、动作 chunk 准确率、Q 触发 F1、完成率、
  p95 推理延迟。不能用知识问答得分替代实机动作成功率。

## 4. 当前执行顺序

```text
新模式字段与 mode token
→ 新 raw session 录制
→ WK/VG 小数据集与校验集
→ M0 标准模式动作训练
→ joint_hunt / blackjack LoRA
→ 逐模式实机评估
```
