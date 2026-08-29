# AGENTS.md

本文件给 AI 助手（Claude Code / 其他）在该仓库工作时提供主线指引。**文档以 `docs/` 为唯一入口**，不要再新增分散的 `.md` 于仓库根目录。

## 项目一句话

以《第五人格》为环境构建 VLA agent：**画面+状态 → 动作序列**。当前阶段求生者·律师，PC 互通版，官方自定义剧本沙盒。架构为快慢双系统分层，快层按 M1 规则 → M2 小策略网络(BC) → M3 3B VLA 演进。

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
| 训练超参与设备 | `idv_agent/training/config.py` |
| 采集参数（FPS、区域、灵敏度尺度） | `idv_agent/capture/screen_capture.py` |
| 破译状态机 / 动作提取参数 | `idv_agent/labels/state_machine.py`, `extract.py` |
| 快层网络 | `idv_agent/model/fast_controller.py` |
| 慢层三头网络 | `idv_agent/model/game_actor_critic.py` |
| 策略抽象(可插拔规则/学习) | `idv_agent/model/policy.py` |
| 部署调度 | `idv_agent/agent/realtime_agent.py` |
| 规则 agent（律师版兜底 P1） | `idv_agent/agent/rule_agent.py` |
| 对局记忆（P11） | `idv_agent/agent/memory.py` |

## 关键概念速览

- **动作空间**：20 离散类别 + 4 连续（move_x/y, cam_dx/dy），见 `configs/schema.py`。
- **快慢连接**：慢层输出 → `intent_vector[16]` + `skeleton_idx` → 注入快层条件输入。
- **破译状态机**：第五人格按一次 Q 即自动破译（无后续按键事件），需识别该隐式状态并把解码段语义重写为 `INTERACT_HOLD`（`labels/state_machine.py`）。
- **数据文件**：每 session 一个 `frames/`+`events.csv`+`mouse_positions.csv`+`meta.json`，构建脚本产出 `samples.jsonl`。
- **BC 训练**：`bc_fast`（快层）+ `bc_slow`（慢层），均 FP16 + GradScaler。

## 变更工作流

1. 先想清楚改哪一层，动 `configs` 前先看是否会影响下游（schema 是全局约定）。
2. 每次小步改完跑相关测试（无 GPU/游戏也能跑)：`python -m idv_agent.scripts.smoke_test`。
3. 改模型 / 训练逻辑时跑 `tests/` 或 `scripts/smoke_test` 里对应的 forward 检查。
4. 涉及文档：所有内容写进 `docs/`，按编号命名（`01-审查报告`、`02-架构设计`…）。
5. 提交前 `git status` 自查，不残留测试产物/临时文件。

## 反作弊操作知识（重要）

- NeAC 存在时用户态键盘/鼠标 hook 被吞：管理员运行时才收到进游戏内的 W/A/S/D 事件。非管理员只收到系统级事件。
- 真机注入 `SendInput` 带 `LLMHF_INJECTED` 标志，排位/匹配有检测压力。默认 dry-run。

## 依赖

见 `requirements.txt`（torch 2.6 + cu124、transformers、peft、accelerate、modelscope、qwen-vl-utils、dxcam、pynput、pygetwindow、opencv、pandas、numpy、sentencepiece）。模型权重缓存在 `~/.cache/modelscope/hub/models/`（Qwen3-VL-4B-Instruct ~8GB、siglip2-base-patch16-224 ~1.4GB）。
