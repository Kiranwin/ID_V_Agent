# AGENTS.md

本文件给 AI 助手（Claude Code / 其他）在该仓库工作时提供主线指引。**文档以 `docs/` 为唯一入口**，不要再新增分散的 `.md` 于仓库根目录；唯一例外是用户明确要求的根目录 `GOAL_STATUS.md`，用于 MVP 断点恢复。

## 断点恢复（每次启动必做）

如果仓库根目录存在 `GOAL_STATUS.md`，在检查代码、数据、训练或部署之前必须先完整阅读它。
它记录当前 MVP 证据、禁止放行条件、下一步及备份位置；每次有实质训练/数据/架构/门禁状态变化后必须更新。训练命令使用阻塞式前台运行，等命令退出再读取结果，不得启动轮询子进程或重复训练。

## 项目一句话

以《第五人格》为环境构建 VLA agent：**连续画面+指令+模式 → 动作块序列**。当前阶段求生者·律师，PC 互通版，官方自定义剧本沙盒。VLA 原生 action chunk 是训练主线，规则策略仅作安全兜底。

## 硬性约束（勿违反）

- **合规红线**：只跑官方自定义剧本 / 训练营，不碰真人排位；不用内存读取、DLL 注入、截帧 hook。
- **采集必须管理员权限**（NeAC 内核级反作弊会阻断用户态 pynput hook，管理员提升 UIPI 才放行）。默认注入只发 dry-run，`--send-input` 仅在自定义剧本/训练模式启用。
- **文档唯一入口是 `docs/`**。根目录只有 `README.md`（导航）、`AGENTS.md`（本文件）、`requirements.txt`。
- 硬件 Turing 2080 Ti：**无 BF16**，训练用 FP16 + GradScaler。
- 已确认决策见 README「已确认决策」，勿再摇摆（平台/角色/硬件/最终 VLA/沙盒）。

## 可快速修改的关键位置

| 想改什么 | 位置 |
|---|---|
| 动作类别 / 连续通道约定 | `idv_agent/configs/schema.py` |
| 键位映射（换了角色/改了键） | `idv_agent/configs/keymap_survivor.py` |
| 意图类别与 16 维原型 | `idv_agent/configs/intent.py` |
| VLA 训练逻辑与超参 | `idv_agent/training/` |
| 采集参数（FPS、区域、灵敏度尺度） | `idv_agent/capture/screen_capture.py` |
| 原始事件→逐帧动作预处理 | `idv_agent/labels/extract.py`, `idv_agent/scripts/extract.py` |
| 破译状态机 | `idv_agent/labels/state_machine.py` |
| VLA 主干适配器 | `idv_agent/model/qwen_backbone_adapter.py`, `siglip_vision_adapter.py` |
| 策略抽象(可插拔规则/学习) | `idv_agent/model/policy.py` |
| 部署调度 | `idv_agent/agent/realtime_agent.py` |
| 规则 agent（律师版兜底 P1） | `idv_agent/agent/rule_agent.py` |
| 对局记忆（P11） | `idv_agent/agent/memory.py` |

## 关键概念速览

- **动作空间**：VLA v4 九方向移动 + 五档相机桶 + 六个按键状态，见 `vla/action_chunk.py`。
- **时序输入**：`FrameFeatureCache` 滚动窗口最多 8 帧（最少 3 帧，`valid_mask` 标记短窗口），滚动预测 4 步宏动作块。
- **时序编码器已分离**：`SharedFastSlowVLA` 内 `slow_temporal`/`fast_temporal` 是两个独立 GRU，慢头（intent/subgoal）和快头（action chunk）梯度互不干扰，详见 `docs/18-架构变更历史.md` 与 `idv_agent/model/fast_slow_vla.py`。
- **三阶段训练继承**：WK（游戏知识 LoRA）→ VG（视觉 grounding，冻结 lora_wk）→ ACT（动作块，冻结 Qwen 主干+两阶段 LoRA），后一阶段必须加载前一阶段 checkpoint，见 `docs/17-三阶段训练继承协议.md`。
- **VG 标注**：YOLO 只保留 `cipher_visible/cipher_highlight/interact_prompt` 三类；帧状态由 `frame_states.jsonl` 标注 `decoding/idle/walking/chased`，转换后写入 VG JSON 的顶层 `state`。
- **破译状态机**：第五人格按一次 Q 即自动破译（无后续按键事件），需识别该隐式状态并把解码段语义重写为 `INTERACT_HOLD`（`labels/state_machine.py`）。
- **数据文件**：每 raw session 一个 `frames/`+`events.csv`+`mouse_deltas.csv`（Raw Input 相对位移）+`frame_timestamps.csv`+`meta.json`，构建脚本产出 `vla_chunks_v4.jsonl`。相机动作只来自 `mouse_deltas.csv`，不使用绝对坐标。
- **训练精度**：RTX 2080 Ti 使用 FP16 + GradScaler；VLA 训练不再依赖旧 `samples.jsonl`/BC 管线。

## 变更工作流

1. 先想清楚改哪一层，动 `configs` 前先看是否会影响下游（schema 是全局约定）。
2. 每次小步改完跑相关测试（无 GPU/游戏也能跑)：`python -m idv_agent.scripts.smoke_test`。
3. 改模型 / 训练逻辑时跑 `tests/` 或 `scripts/smoke_test` 里对应的 forward 检查。
4. 涉及文档：所有内容写进 `docs/`，按编号命名（`03-数据格式`、`15-快慢VLA接口`…）；架构/性能类变更追加到 `docs/18-架构变更历史.md`，不新建编号文档。
5. 提交前 `git status` 自查，不残留测试产物/临时文件。

## 反作弊操作知识（重要）

- NeAC 存在时用户态键盘/鼠标 hook 被吞：管理员运行时才收到进游戏内的 W/A/S/D 事件。非管理员只收到系统级事件。
- 真机注入 `SendInput` 带 `LLMHF_INJECTED` 标志，排位/匹配有检测压力。默认 dry-run。

## 依赖

见 `requirements.txt`（torch 2.6 + cu124、transformers、peft、accelerate、modelscope、qwen-vl-utils、dxcam、pynput、pygetwindow、opencv、pandas、numpy、sentencepiece）。模型权重缓存在 `~/.cache/modelscope/hub/models/`（Qwen3-VL-4B-Instruct ~8GB、siglip2-base-patch16-224 ~1.4GB）。
