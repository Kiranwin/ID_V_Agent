# SGan 运行入口（run_agent）

更新：2026-09-10。`run_agent` 现在只有一种运行架构：**SGan**（State-Guided
Action Network）。它是原 M29 的对外名称，checkpoint 内部 schema、文件命名和
模块名仍是 `m29`（例如 `m29.pt`、`m29_policy.py`、`train_m29.py`、
`evaluate_model_ability.py`）。

旧的 `--mode rule`（规则安全兜底）与 `--mode act`（旧 ACT 同步 runtime）已从本
入口移除，传这两个值会被 argparse 直接拒绝。

相关文档：[user-guide.md](user-guide.md)（六步操作）、[data-workbench-v6.md](data-workbench-v6.md)（采集/回放/
标注工具）、[AGENTS/](AGENTS/)（历史设计与实验记录）。

## 1. 它做什么

输入是连续画面，输出只有两件事：

1. **导航动作**：9 档移动 + 两轴 5 档镜头，每 200 ms 一个宏动作；
2. **Q**：由当前帧的 `interact_prompt` 视觉证据直接决定的快速反应层输出。

三个循环互相解耦，互不阻塞：

```text
capture loop   20 FPS   抓屏 → 单槽覆盖（旧帧被丢弃并计数）→ 推理线程取走
inference loop ≤5 Hz    取最新帧 → Qwen 视觉编码 → SGan 前向 → 发布一个 decision
executor loop  50 Hz    消费最新 decision，合并按键/鼠标动作，维护按键持有状态
```

推理线程是模型与 history 的唯一 owner；执行线程是输入设备的唯一 owner。Q 只要
`q_logits.sigmoid() >= 0.5` 就发送，不被 phase、decoding 状态或“之前按过没有”
否决；导航动作则受 400 ms 陈旧结果门禁和 200 ms 宏动作时钟约束。

默认 **dry-run**：只跑完整时序并把命令写进 trace，不发送键鼠。

## 2. 依赖的脚本与子文件

| 文件 | 角色 |
|---|---|
| `idv_agent/scripts/run_agent.py` | 本入口：唯一 `--mode sgan`，负责参数校验与转发 |
| `idv_agent/scripts/run_agent.py`（`run()`） | 运行时引擎：20 FPS 采集线程、推理线程、50 Hz 命令线程、trace 落盘 |
| `idv_agent/agent/m29_policy.py` | `M29Policy`（模型推理 + 8 帧 history）与 `M29CommandMerger`（动作合并、Q 发送、按键释放） |
| `idv_agent/agent/observation_clock.py` | `CapturedObservation`、`LatestObservationSlot`（单槽覆盖）、`ObservationActionClock`（200 ms 动作 / 400 ms 陈旧门禁） |
| `idv_agent/model/m29_checkpoint.py` | `load_m29_checkpoint`：严格校验 checkpoint schema 与控制契约，不兼容 m28/v5 |
| `idv_agent/model/state_guided_action.py` | SGan 网络定义：视觉事实 → 顶层决策 → 导航，`PHASES`、`CAMERA_COMMANDS` 常量 |
| `idv_agent/training/m29_features.py` | `load_m29_encoder`：冻结的 Qwen 视觉编码器 + 特征投影 |
| `idv_agent/capture/screen_capture.py` | `ScreenCapture` / `CaptureConfig`：按窗口标题抓屏，宽度上限 1334 |
| `idv_agent/agent/action_decoder.py` | `Command`：`press` / `release` / `mouse_move` 指令载体 |
| `idv_agent/scripts/train_m29.py` | 训练入口，产出 `m29.pt` + `manifest.json` + `metrics.json` |
| `idv_agent/scripts/evaluate_model_ability.py` | 离线模型能力评估：标签指标（`--mode metrics`）、视觉依赖门禁（`--mode gate`）或两者（默认 `all`） |
| `tests/test_m29_hierarchy.py`、`tests/test_m29_pipeline.py`、`tests/test_configs.py` | 层级契约、checkpoint 往返、CLI 参数回归 |

## 3. 运行前准备

| 需要准备 | 说明 |
|---|---|
| 环境 | `conda activate idv312`（CUDA 版 PyTorch；`--device cuda` 时没有 CUDA 会直接报错） |
| checkpoint | `m29.pt` **文件**路径，不是目录。例如 `checkpoints/m29_q_interact_fix300_20260910/m29.pt` |
| Qwen 基础模型 | Qwen3-VL checkpoint 目录；省略时读取 checkpoint manifest 的 `base_model` |
| trace 路径 | 必须是一个**尚不存在**的路径，运行不会覆盖旧 trace |
| 游戏窗口 | 官方自定义剧本/训练模式，窗口标题与 `--title` 一致 |

checkpoint 必须满足 `m29_checkpoint.py` 的契约，否则加载即失败：

| 契约字段 | 要求 |
|---|---|
| `schema` | `m29.state_guided_action.v3` |
| `manifest.camera_commands` | `[-110, -25, 0, 25, 110]` |
| `manifest.action_ms` | `200` |
| `manifest.q_contract` | `interact_prompt_to_q.v1` |
| `manifest.hierarchy` | `predicted_beliefs_only` |
| `manifest.state_bottleneck` | `discrete_straight_through_v1` 或 `soft_continuous_geometry_v2` |

## 4. 使用说明

最小 dry-run（只记录命令，不发送键鼠）：

```powershell
conda activate idv312
python -m idv_agent.scripts.run_agent `
  --mode sgan `
  --checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt `
  --model-path C:\Users\kiran\.cache\modelscope\models\Qwen--Qwen3-VL-4B-Instruct\snapshots\master `
  --device cuda `
  --duration 10 `
  --trace reports/sgan_dryrun_20260910.jsonl
```

让 `task=q_interact` 的 checkpoint 同时输出导航动作（仍是 dry-run，不解除门禁）：

```powershell
python -m idv_agent.scripts.run_agent --mode sgan `
  --checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt `
  --model-path <Qwen目录> --device cuda --duration 8 --enable-navigation `
  --trace reports/sgan_nav_dryrun_20260910.jsonl
```

真实发送键鼠（仅官方自定义剧本/训练模式，且需要显式授权未过门禁的 checkpoint）：

```powershell
python -m idv_agent.scripts.run_agent --mode sgan `
  --checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt `
  --model-path <Qwen目录> --device cuda --duration 10 --enable-navigation `
  --send-input --allow-undeployed-send-input `
  --trace reports/sgan_sandbox_20260910.jsonl
```

运行结束会打印 `captures=` / `encoded=`，并提示 trace 位置。`captures` 是实际抓到
的帧数，`encoded` 是真正送进模型推理的帧数；两者差距反映推理吞吐上限（当前环境
约 5 FPS），`empty_grabs`、`overwritten_captures`、`superseded_results` 说明采集是
否跑赢推理。

## 5. 字段总览

| 字段 | 取值 / 默认 | 必填 | 说明 |
|---|---|---|---|
| `--mode` | `sgan`，默认 `sgan` | 否 | 只有这一个取值；`rule` / `act` / `m29` 会被拒绝 |
| `--checkpoint` | 路径 | **是** | `m29.pt` 文件；不存在直接报错 |
| `--model-path` | 路径，默认 `None` | 否 | Qwen3-VL 目录；省略时用 manifest 的 `base_model` |
| `--title` | 字符串，默认 `第五人格` | 否 | 被捕获窗口标题 |
| `--device` | 默认 `cuda` | 否 | 推理设备；`cuda` 不可用时报错退出 |
| `--duration` | 浮点秒，默认 `30.0` | 否 | 运行时长，必须为正数 |
| `--trace` | 路径 | **是** | trace JSONL 输出；路径必须不存在 |
| `--send-input` | 开关，默认关 | 否 | 真实发送键鼠 |
| `--allow-undeployed-send-input` | 开关，默认关 | 否 | 允许 `deployable=false` 的 checkpoint 真发送，必须与 `--send-input` 同时出现 |
| `--enable-navigation` | 开关，默认关 | 否 | `task=q_interact` checkpoint 也输出导航动作 |

### 各字段使用样例

`--mode`：固定 `sgan`；写错会立即失败，不会静默回退到别的策略。

```powershell
--mode sgan     # 唯一合法值
--mode rule     # 报错：invalid choice
```

`--checkpoint`：指向 `.pt` 文件。带目录会报 `checkpoint 不存在`。

```powershell
--checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt
```

`--model-path`：推荐显式给出，避免 manifest 里的绝对路径换机器后失效。

```powershell
--model-path C:\Users\kiran\.cache\modelscope\models\Qwen--Qwen3-VL-4B-Instruct\snapshots\master
# 省略时：--checkpoint ...（读 manifest.base_model）
```

`--title`：窗口标题必须与游戏窗口一致，否则抓到空帧。

```powershell
--title "第五人格"
```

`--device`：正常用 `cuda`；显式 `--device cpu` 只用于不追求帧数的接口自检。

```powershell
--device cuda
--device cpu
```

`--duration`：短诊断用 5~10 秒，沙盒验证用 10~30 秒。

```powershell
--duration 5      # 冒烟
--duration 30     # 常规
--duration 0      # 报错：--duration 必须为正数
```

`--trace`：每次运行换新文件名；同一路径第二次运行会被拒绝。

```powershell
--trace reports/sgan_dryrun_20260910.jsonl
```

`--send-input`：默认不发送；只有确认在官方自定义剧本/训练模式时才加。

```powershell
--send-input
```

`--allow-undeployed-send-input`：当前训练出的 checkpoint 都是
`deployable=false`，真实发送需要它同时出现；单独使用会报错。

```powershell
--send-input --allow-undeployed-send-input     # 合法
--allow-undeployed-send-input                  # 报错：必须与 --send-input 同时使用
```

`--enable-navigation`：`manifest.task` 为 `full` 时导航本来就开启；只有
`task=q_interact` 的 Q 专项 checkpoint 需要这个开关才会输出移动/镜头。

```powershell
--enable-navigation
```

## 6. trace 文件结构

第一行是 header：

```json
{"schema": "m29.runtime_trace.v1", "source": "dry_run_simulation",
 "counts": {"captures": 95, "encoded": 22, "empty_grabs": 0, "superseded_results": 0},
 "overwritten_captures": 73, "elapsed_s": 5.02, "inference_worker_alive": false, "errors": []}
```

| header 字段 | 含义 |
|---|---|
| `schema` | 固定 `m29.runtime_trace.v1` |
| `source` | `dry_run_simulation`（未发送输入）或 `live_input`（已真发送） |
| `counts.captures` / `counts.encoded` | 采集帧数 / 推理帧数 |
| `counts.empty_grabs` | 抓屏返回空帧的次数 |
| `counts.superseded_results` | 推理结果还没被执行器消费就被新结果覆盖的次数 |
| `overwritten_captures` | 采集帧在单槽里被覆盖丢弃的次数 |
| `inference_worker_alive` | 退出时推理线程是否仍然存活（应为 `false`） |
| `errors` | 线程内异常字符串列表；非空时进程会重新抛出 |

其余行是事件。每个被接受的 decision 记录一行：

```json
{"event": "decision", "sequence": 21, "captured_ns": 1234567890, "ready_ns": 1234999999,
 "submitted_ns": 1235000000, "phase": "approach", "decoding_probability": 0.03,
 "prompt_probability": 0.91, "q": true, "navigation_applied": true,
 "move": 1, "camera": [0, 0], "feedback": "Q submitted is an attempt, not decoding success"}
```

| 事件字段 | 含义 |
|---|---|
| `event` | `decision` / `action_end` / `stale_result` |
| `sequence` | 采集帧序号，对应 `captured_ns` |
| `captured_ns` / `ready_ns` / `submitted_ns` | 抓帧、模型输出、命令提交时刻（单调 ns） |
| `phase` | 顶层阶段：`search` / `approach` / `align` / `interact` / `maintain_decode` |
| `prompt_probability` | 当前帧 `interact_prompt` 概率，就是 Q 的唯一判据 |
| `decoding_probability` | 破译状态概率（只观察，不否决 Q） |
| `q` | 本次是否提交了 Q 按键 |
| `navigation_applied` | 本次导航动作是否被动作时钟接受 |
| `move` | 9 档移动序号：`0`不动 `1`前 `2`右前 `3`右 `4`右后 `5`后 `6`左后 `7`左 `8`左前 |
| `camera` | `[dx, dy]` 相对鼠标像素，取值属于 `-110 / -25 / 0 / 25 / 110` |

`action_end` 表示 200 ms 宏动作结束并释放按键；`stale_result` 表示该 decision 超过
400 ms 或已经过期，被丢弃而未执行。Q 的提交只代表“按下了”，不代表游戏内真的进入
破译，验收时要结合录像与后续帧的 `decoding_probability`。

## 7. 门禁与安全

1. **默认 fail-closed**：不加 `--send-input` 就永远不发送键鼠，只写 trace。
2. **checkpoint 门禁**：`manifest.deployable=false` 时，`--send-input` 单独使用会被
   `run_agent.py` 拒绝；只有显式加 `--allow-undeployed-send-input` 才放行，用于已经
   授权的标准模式沙盒验证。
3. **模式门禁**：仅在官方自定义剧本/训练模式运行；不要对正常对局使用。
4. **退出释放**：`shutdown` 无条件释放 `w/a/s/d/q`，避免残留按住状态。
5. **trace 不可覆盖**：防止把上一次证据冲掉；要复跑就换文件名。

## 8. 常见故障

| 现象 | 原因与处理 |
|---|---|
| `trace 已存在，请换一个新路径` | 换文件名或删除旧 trace |
| `checkpoint 不存在` | `--checkpoint` 必须指到 `m29.pt` 文件本身 |
| `checkpoint is not M29` | 传了 m28/v5 的权重；SGan 只接受 m29 schema |
| `M29 checkpoint control contract mismatch` | checkpoint 的 `action_ms` / `camera_commands` / `q_contract` / `hierarchy` 与 runtime 不一致 |
| `M29 checkpoint/Qwen feature dimension mismatch` | `--model-path` 与训练时用的 base model 不是同一个 |
| `SGan 需要 CUDA` | 在 `idv312` 环境运行，或显式 `--device cpu` 做接口自检 |
| `captures` 很大但 `encoded` 很小 | 正常现象：Qwen 视觉约 5 FPS，采集 20 FPS，单槽会丢弃中间帧（记入 `overwritten_captures`） |

## 9. 与训练、评估的关系

```text
train_m29.py  →  checkpoints/<run>/m29.pt + manifest.json + metrics.json
                        │
                        ├── evaluate_model_ability.py   离线指标 + 视觉依赖门禁
                        └── run_agent --mode sgan  实时运行（本文件）
```

`train_m29 --task q` 先训 Q 专项 checkpoint；`train_m29 --task full` 用
`--init-checkpoint <Q阶段m29.pt>` 继承视觉提示头，再训练导航与顶层决策。两个阶段
产出的 `m29.pt` 都能被 `run_agent --mode sgan` 加载，区别只在
`manifest.task`（`q_interact` 需要 `--enable-navigation` 才有导航输出）。
