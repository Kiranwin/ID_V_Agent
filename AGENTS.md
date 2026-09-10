# AGENTS.md

For long-running asynchronous work:
- Empty `write_stdin` polls MUST use `yield_time_ms >= 180000`; prefer `300000` when intermediate output is not needed.
- Training and evaluation tasks must use `yield_time_ms = 300000`.
- `functions.wait` MUST use `yield_time_ms >= 180000`.
- `functions.exec` MUST set its outer `@exec yield_time_ms` at least 30000 ms longer than the longest nested tool wait, so the outer code cell does not yield first.
- Do not apply the long wait to non-empty `write_stdin` calls that send interactive input. These tools return early when the process or cell completes.
- Do not wake the model merely to report that work is still running

本文件给 AI 助手（Claude Code / 其他）在该仓库工作时提供主线指引。**文档以 `docs/` 为唯一入口**，不要再新增分散的 `.md` 于仓库根目录；唯一例外是用户明确要求的根目录 `GOAL_STATUS.md`，用于 MVP 断点恢复。

文档分区（新增文档按此归位）：

| 位置 | 内容 |
|---|---|
| `docs/user-guide.md` | 当前操作入口：录制 → 回放 → 标注 → 数据准备 → 训练 → 运行 |
| `docs/run-agent.md` | 当前运行入口：字段、依赖脚本、trace 结构、故障排查 |
| `docs/data-collection-tool.md` | 采集工具 |
| `docs/replay-tool.md` | 数据回放 |
| `docs/data-workbench-v6.md` | V6 标注工作台 |
| `docs/AGENTS/` | 历史设计与实验记录（00 状态、03 数据格式、04 环境、05 操作手册、10 训练策略、12–17 WK/VG/快慢 VLA、18 架构变更历史、19 ACT 协议） |

## 断点恢复（每次启动必做）

如果仓库根目录存在 `GOAL_STATUS.md`，在检查代码、数据、训练或部署之前必须先完整阅读它。
它记录当前 MVP 证据、禁止放行条件、下一步及备份位置；每次有实质训练/数据/架构/门禁状态变化后必须更新。训练命令使用阻塞式前台运行，等命令退出再读取结果，不得启动轮询子进程或重复训练。

## 项目一句话

以《第五人格》为环境构建 VLA agent：**连续画面 + 状态 → 动作序列**。当前阶段求生者·律师，PC 互通版，官方自定义剧本沙盒。唯一运行架构是 **SGAN**（State-Guided Action Network，checkpoint 内部名仍为 m29），规则/ACT 旧栈已删除。

## 硬性约束

- **文档唯一入口是 `docs/`**。根目录只有 `README.md`（导航）、`AGENTS.md`（本文件）、`requirements.txt`。
- 硬件 Turing 2080 Ti：，训练用 FP16 + GradScaler。
- 环境 conda idv312

## 可快速修改的关键位置

| 想改什么 | 位置 |
|---|---|
| 动作类别 / 连续通道约定 | `idv_agent/configs/schema.py` |
| 键位映射（换了角色/改了键） | `idv_agent/configs/keymap_survivor.py` |
| 意图类别与 16 维原型 | `idv_agent/configs/intent.py` |
| SGAN 训练逻辑与超参 | `idv_agent/scripts/train_m29.py`, `idv_agent/training/` |
| 采集参数（FPS、区域、灵敏度尺度） | `idv_agent/capture/screen_capture.py` |
| 冻结 Qwen 主干加载与适配 | `idv_agent/model/qwen_backbone_adapter.py` |
| SGAN checkpoint 契约 | `idv_agent/model/m29_checkpoint.py` |
| V6 数据契约与导出 | `idv_agent/scripts/prepare_mvp_v6.py` |
| SGAN 数据集 / 损失 | `idv_agent/training/m29_dataset.py`, `idv_agent/training/m29_loss.py`, `idv_agent/training/prompt_q_loss.py` |
| 实时部署入口（唯一 SGAN 运行入口） | `idv_agent/scripts/run_agent.py` |
| SGAN 运行时调度（20 FPS 采集 / ≤5 Hz 推理 / 50 Hz 执行） | `idv_agent/scripts/run_agent.py`（`run()`） |
| SGAN 离线评估与视觉依赖门禁 | `idv_agent/scripts/evaluate_model_ability.py` |
| SGAN 推理与命令合并 | `idv_agent/agent/m29_policy.py` |
| SGAN 网络定义（状态→决策→导航） | `idv_agent/model/state_guided_action.py` |

## 架构（当前唯一链路）

```text
采集线程 20 FPS ─► 单槽覆盖（旧帧丢弃并计数）
                       │
推理线程 ≤5 Hz ────────┘
   冻结 Qwen3-VL 视觉编码 → 8 帧 visual history + 2 维执行反馈
       │
       ├─ 视觉事实：visibility / bbox / prompt_bbox / decoding / prompt
       ├─ 顶层决策：phase / steering / path
       └─ Q = prompt head：当前帧直接输出，不被 phase 或 decoding 否决
       │
导航：9 档移动 + 两轴 5 档镜头（吃语义 belief + 连续预测状态）
       │
命令线程 50 Hz：200 ms 宏动作、400 ms 陈旧结果门禁、按键持有合并
       │
SendInput（默认 dry-run；未过门禁的 checkpoint 需显式授权）
```

实现落点：网络 `idv_agent/model/state_guided_action.py`，checkpoint 契约
`idv_agent/model/m29_checkpoint.py`，冻结主干 `idv_agent/model/qwen_backbone_adapter.py`
+ `idv_agent/training/m29_features.py`，数据集与损失 `idv_agent/training/m29_*.py`，
训练 `idv_agent/scripts/train_m29.py`，运行 `idv_agent/scripts/run_agent.py`，
评估 `idv_agent/scripts/evaluate_model_ability.py`。设计演进与历史决策见
`docs/AGENTS/18-架构变更历史.md`。

## 关键概念速览

- **动作空间**：9 档移动 + 两轴 5 档镜头（-110/-25/0/25/110 px），每 200 ms 一个宏动作；Q 是独立的当前帧反应输出。
- **时序输入**：SGAN 消费最多 8 帧视觉 history（`valid_mask` 标记短窗口、`relative_times_s` 必须严格递增且最新帧为 0）与 2 维执行反馈；推理耗时是 runtime 门禁，不作为网络输入。
- **层级**：视觉事实 → 顶层决策（phase/steering/path）→ 导航；导航只吃语义 belief + 连续预测状态，没有从原始视觉特征直达动作的捷径。
- **Q 契约**：`interact_prompt` 为真 → 输出 Q token，为假不输出；允许短时间连续触发，不要求落地按下的精确边沿，见 `idv_agent/vla/prompt_q.py`。
- **训练继承链**：`train_m29 --task q` → `--task full`（`--init-checkpoint` 继承 Q 头）→ 可选 `--task q_interact`；后一阶段必须加载前一阶段 `m29.pt`。
- **数据文件**：每 raw session 一个 `frames/`+`events.csv`+`mouse_deltas.csv`（Raw Input 相对位移）+`frame_timestamps.csv`+`meta.json`；V6 标注与导出走工作台 + `prepare_mvp_v6`，产出 `data/derived/m29/*.jsonl`。相机动作只来自 `mouse_deltas.csv`。
- **训练精度**：RTX 2080 Ti 使用 FP16 + GradScaler，checkpoint 默认 `deployable=false`。

## 变更工作流

1. 先想清楚改哪一层，动 `configs` 前先看是否会影响下游（schema 是全局约定）。
2. 每次小步改完跑相关测试（无 GPU/游戏也能跑)：`python -m idv_agent.scripts.smoke_test` 与 `pytest tests -q`。
3. 改模型 / 训练逻辑时跑 `tests/` 里对应的 forward 检查；改运行入口时至少跑一次 `run_agent --mode SGAN` 的 dry-run。
4. 涉及文档：操作类改动同步 `docs/user-guide.md` 与 `docs/agent.md`；工具类改动同步 `docs/tools/`；架构/性能类变更追加到 `docs/AGENTS/18-架构变更历史.md`，不新建编号文档。根目录只保留 `README.md`、`AGENTS.md`、`GOAL_STATUS.md`。
5. 提交前 `git status` 自查，不残留测试产物/临时文件。

## 依赖

见 `requirements.txt`（torch 2.6 + cu124、transformers、peft、accelerate、modelscope、qwen-vl-utils、dxcam、pynput、pygetwindow、opencv、pandas、numpy、sentencepiece）。模型权重缓存在 `~/.cache/modelscope/hub/models/`（Qwen3-VL-4B-Instruct ~8GB）。
