# 10 · VLA 多模式与 ActVLP 训练策略

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

ACT 必须加载 VG 阶段输出的 `M2_VG`，继承 `lora_wk` 与 VG 语义能力；第一轮冻结 Qwen 主干和两阶段 LoRA，只训练 `slow_temporal`/`fast_temporal`（分离时序编码器，见 [18-架构变更历史.md](18-架构变更历史.md)）、FiLM、Slow Head 和 Fast Head。完整继承链见 [17-三阶段训练继承协议.md](17-三阶段训练继承协议.md)。

在继承后的模型上训练因果动作块：

```text
最多 8 帧滚动窗口（最少 3 帧，valid_mask 标记短窗口）+ <mode:...> + 任务指令 + 历史动作
→ 4 步 move_dir / camera buckets / buttons / duration
```

动作训练只使用通过对齐校验的新 `vla_raw` session。标准模式先训练 M0，其他模式
在固定 M0 上训练 LoRA。第一版不做 PPO 或在线探索。

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
