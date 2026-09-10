# ID_V_Agent

以《第五人格》为环境，构建「模型根据游戏画面自主决策动作」的 VLA（Vision-Language-Action）agent。

> 一句话定位：**画面 + 状态进，动作序列出**。当前阶段为求生者·律师（PC 互通版，官方自定义剧本沙盒）。

## 快速导航

| 你想做什么 | 看这里 |
|---|---|
| **第一次上手：六步操作** | [docs/user-guide.md](docs/user-guide.md) |
| 运行入口字段、依赖脚本、trace 结构 | [docs/agent.md](docs/agent.md) |
| 标注工作台 / 录制 / 回放工具 | [docs/tools/](docs/tools/) |
| 架构设计与实验历史 | [docs/AGENTS/18-架构变更历史.md](docs/AGENTS/18-架构变更历史.md) |
| 数据格式与动作空间（历史权威版） | [docs/AGENTS/03-数据格式.md](docs/AGENTS/03-数据格式.md) |
| 环境、硬件与反作弊须知 | [docs/AGENTS/04-环境与硬件.md](docs/AGENTS/04-环境与硬件.md) |
| 当前进度与执行顺序（历史快照） | [docs/AGENTS/00-当前状态.md](docs/AGENTS/00-当前状态.md) |
| 给 AI 助手（Claude/其他）的协作指南 | [AGENTS.md](AGENTS.md) |

## 架构

当前唯一的运行架构是 **SGAN**（State-Guided Action Network，原 M29）。它先把画面压缩
成「视觉事实」，再由顶层决策指导导航动作；Q 完全独立于这套层级，由当前帧的交互提示
直接决定：

```text
采集线程 20 FPS ─► 单槽覆盖（旧帧丢弃并计数）
                       │
推理线程 ≤5 Hz ────────┘
   冻结 Qwen3-VL 视觉编码 → 8 帧 visual history + 2 维执行反馈
       │
       ├─ 视觉事实：visibility / bbox / prompt_bbox / decoding / prompt
       ├─ 顶层决策：phase / steering / path（只吃事实概率与时间状态）
       └─ Q = prompt head：当前帧直接输出，不被 phase 或 decoding 否决
       │
导航：9 档移动 + 两轴 5 档镜头（吃事实与决策的语义 belief + 连续预测状态）
       │
命令线程 50 Hz：200 ms 宏动作、400 ms 陈旧结果门禁、按键持有合并
       │
SendInput（默认 dry-run；未过门禁的 checkpoint 需显式授权）
```

关键约束：标签永远不作为前向输入；导航没有从原始视觉特征直达动作的捷径；推理耗时只作为
runtime 门禁与 trace 指标，不进入网络；顶层状态由模型自己预测后再喂给导航。逐帧跑大模型
不满足实时预算，所以采集、推理、执行是三个解耦线程。

代码地图：

| 层 | 位置 |
|---|---|
| 网络定义（事实 → 决策 → 导航） | `idv_agent/model/state_guided_action.py` |
| checkpoint 契约与加载 | `idv_agent/model/m29_checkpoint.py` |
| 冻结 Qwen 主干与特征投影 | `idv_agent/model/qwen_backbone_adapter.py`, `idv_agent/training/m29_features.py` |
| 数据集与损失 | `idv_agent/training/m29_dataset.py`, `m29_loss.py`, `prompt_q_loss.py` |
| 训练入口 | `idv_agent/scripts/train_m29.py` |
| 实时运行（引擎 + CLI） | `idv_agent/scripts/run_agent.py` |
| 推理与命令合并 / 时序时钟 | `idv_agent/agent/m29_policy.py`, `idv_agent/agent/observation_clock.py` |
| 离线评估与视觉依赖门禁 | `idv_agent/scripts/evaluate_model_ability.py` |

逐字段说明与 trace 结构见 [docs/agent.md](docs/agent.md)，设计演进见
[docs/AGENTS/18-架构变更历史.md](docs/AGENTS/18-架构变更历史.md)。

## 已确认决策

1. **平台**：仅 PC 互通版（DDA 捕获 + SendInput 注入），不用模拟器。
2. **首发角色**：求生者·律师（自带地图 = 慢层感知捷径，Tab 看密码机/队友位置）。
3. **硬件**：本地 RTX 2080 Ti 22GB（Turing， FP16 + GradScaler）。
4. **最终走 VLA**：SGAN 原生动作输出是唯一训练主线；不做传统“感知-规划-控制”规则栈。
5. **战场**：官方自定义剧本沙盒（3 Bot 队友 + 1 Bot 监管者）。

## 全管线

录制 → 回放测试 → 数据标注 → 数据准备 → 模型训练 → 模型推理运行。每一步只保留一个入口。
逐步操作看 [docs/user-guide.md](docs/user-guide.md)，本节只列命令与产物。

### 1. 录制

```powershell
# 原始会话（默认 20 FPS；管理员终端，游戏窗口保持前台）
python -m idv_agent.scripts.record_vla --window-title "第五人格" --mode standard --fps 20 `
  --output data/raw_sessions

# 完整性校验：文件齐全、CSV 字段、时间戳单调、事件落在 session 时间范围内
python -m idv_agent.scripts.validate_vla_raw data/raw_sessions/<session_id>

# 可选诊断：捕获延迟基准 / Windows Raw Input 相对位移
python -m idv_agent.scripts.benchmark_capture --title "第五人格" --seconds 60
python -m idv_agent.scripts.diagnose_raw_mouse
```

产出 `data/raw_sessions/<session_id>/`：

```text
meta.json                录制模式、帧率、对齐延迟
frame_timestamps.csv     帧号 + perf_counter_ns 时间戳
events.csv               键鼠事件流
mouse_deltas.csv         Raw Input 相对位移（相机动作唯一来源，不用绝对坐标）
frames/                  连续 JPEG 帧
```

### 2. 回放测试

```powershell
# 回放录制的鼠标轨迹（默认 dry-run，只打印）
python -m idv_agent.scripts.replay_mouse data/raw_sessions/<session_id>
python -m idv_agent.scripts.replay_mouse data/raw_sessions/<session_id> --send-input

# 网页回放：工作台里按原始时间间隔预览画面与标注框
python -m idv_agent.scripts.data_workbench --root data/raw_sessions --port 8765
```

### 3. 数据标注

先在 X-AnyLabeling 里完成引擎标注，再用工作台完成 V6 标注（阶段/目标/路径/动作/Q）。
**不要手改 CSV**，工作台会写入每个 session 自己的 `navigation_review_with_state.csv`。

```powershell
# 标注工作台（浏览器打开 http://127.0.0.1:8765/）
python -m idv_agent.scripts.data_workbench --root data/raw_sessions --port 8765

# 引擎标注导入 V6 工程
python -m idv_agent.scripts.import_mvp_anylabeling import <workspace> --output <out_dir> `
  --annotator <name> --completion-note "已完成 MVP 标注" --split train --scenario-group <group>

# 工作台导航审核结果回写工程
python -m idv_agent.scripts.import_mvp_anylabeling review <workspace> `
  --csv <session>/navigation_review_with_state.csv --output <out_dir> --reviewer <name>
```

详细字段与简化流程见 [docs/tools/data-workbench-v6.md](docs/tools/data-workbench-v6.md)。

### 4. 数据准备

```powershell
# 几何窗口审计（相机动作是否覆盖到足够的窗口）
python -m idv_agent.scripts.audit_mvp_camera_windows data/raw_sessions/<session_id> `
  --output reports/camera_windows.json

# 工程自检 / 导出（export-q 只出 Q 监督，export 出完整 V6 行）
python -m idv_agent.scripts.prepare_mvp_v6 validate <workspace> --require-reviewed
python -m idv_agent.scripts.prepare_mvp_v6 export   <workspace> --output <train.jsonl>
python -m idv_agent.scripts.prepare_mvp_v6 export-q <workspace> --output <q.jsonl>

# 合并成一个训练文件
python -m idv_agent.scripts.assemble_m29_dataset `
  --inputs <a.jsonl> <b.jsonl> --output data/derived/m29/<name>.jsonl

# 正式数据门禁（train/val 不允许 session 或 scenario-group 重叠）
python -m idv_agent.scripts.audit_m29_v6_data `
  --workspaces <ws...> --q-data <q.jsonl> `
  --full-train <train.jsonl> --full-val <val.jsonl> --output reports/m29_data_audit.json
```

### 5. 模型训练

`train_m29` 的继承链是 `q → full → q_interact`，后一阶段必须用 `--init-checkpoint` 继承前一
阶段的 `m29.pt`（Q 头的视觉提示监督不能在 full 阶段从零开始）：

```powershell
# Q 专项：当前帧 interact_prompt → Q token
python -m idv_agent.scripts.train_m29 --task q `
  --data data/derived/prompt_q/<q_train>.jsonl `
  --val-data data/derived/prompt_q/<q_val>.jsonl `
  --model-path <Qwen3-VL-4B 权重目录> --steps 300 `
  --output checkpoints/m29_q

# full：继承 Q 头，训练状态/决策/导航
python -m idv_agent.scripts.train_m29 --task full --init-checkpoint checkpoints/m29_q/m29.pt `
  --data data/derived/m29/<full_train>.jsonl --val-data data/derived/m29/<full_val>.jsonl `
  --model-path <Qwen3-VL-4B 权重目录> --steps 1000 `
  --output checkpoints/m29_full
```

训练产物固定为 `m29.pt` + `manifest.json` + `metrics.json`；checkpoint 默认
`deployable=false`，要进入真实输入必须过离线视觉依赖门禁与独立验证。

### 6. 模型推理运行

```powershell
# 离线模型能力评估：标签指标 + 视觉依赖门禁（--mode metrics|gate|all，默认 all）
python -m idv_agent.scripts.evaluate_model_ability --checkpoint checkpoints/m29_full/m29.pt `
  --data data/derived/m29/<val>.jsonl --model-path <Qwen3-VL-4B 权重目录> `
  --device cuda --output reports/m29_eval.json

# 实时运行（默认 dry-run，只写 trace）
python -m idv_agent.scripts.run_agent --mode sgan `
  --checkpoint checkpoints/m29_full/m29.pt --model-path <Qwen3-VL-4B 权重目录> `
  --device cuda --duration 10 --trace reports/sgan_dryrun.jsonl

# 真实发送（仅官方自定义剧本/训练模式；未过门禁的 checkpoint 需显式授权）
python -m idv_agent.scripts.run_agent --mode sgan `
  --checkpoint checkpoints/m29_full/m29.pt --model-path <Qwen3-VL-4B 权重目录> `
  --device cuda --duration 10 --enable-navigation `
  --send-input --allow-undeployed-send-input --trace reports/sgan_sandbox.jsonl
```

## 目录结构

```text
ID_V_Agent/
├── docs/                 # 全部文档（文档唯一入口）
│   ├── user-guide.md     # 六步操作手册（当前）
│   ├── agent.md          # 运行入口字段 / 依赖 / trace（当前）
│   ├── tools/            # 采集 / 回放 / 标注工作台说明（当前）
│   └── AGENTS/           # 00–19 历史设计与实验记录
├── idv_agent/
│   ├── configs/          # 动作空间 schema / 键位 / 模式 / 意图与 subgoal 词表
│   ├── capture/          # 屏幕捕获(dxcam) + 输入日志(pynput) + Raw Input 位移
│   ├── model/            # 冻结 Qwen3-VL 适配器 / 时序缓存 / SGan 网络与 checkpoint
│   ├── training/         # SGan 数据集 / 特征 / 损失
│   ├── data_tools/       # 标注工作台后端 + 审核表读写 + 静态资源
│   ├── agent/            # 部署层：SGan 推理与命令合并 / 动作解码 / 注入 / 时序时钟
│   ├── vla/              # 录制与 Q 契约（action_chunk / prompt_q）
│   └── scripts/          # CLI 入口（录制/回放/标注/数据准备/训练/推理运行）
├── tests/                # 单元测试（无 GPU / 无游戏可跑）
├── checkpoints/          # 训练产物（gitignored）
└── data/                 # 录制与派生数据（gitignored）
```

## 快速开始

```bash
# 1. 装依赖（详见 docs/AGENTS/04-环境与硬件.md）
pip install -r requirements.txt

# 2. 无 GPU / 无游戏也能验证管线
python -m idv_agent.scripts.smoke_test

# 3. 环境只读自检（不启动游戏、不发送输入）
python -m idv_agent.scripts.doctor

# 4. 录制原始会话
python -m idv_agent.scripts.record_vla --window-title "第五人格" --mode standard --fps 20
python -m idv_agent.scripts.validate_vla_raw data/raw_sessions/<session_id>

# 5. 标注 → 数据准备 → 训练 → 运行
# 见 docs/user-guide.md 或上面「全管线」各节
```

## 文档地图

| 文档 | 定位 |
|---|---|
| [docs/user-guide.md](docs/user-guide.md) | 当前操作入口：六步全管线 |
| [docs/agent.md](docs/agent.md) | 当前运行入口：字段、依赖脚本、trace 结构、故障排查 |
| [docs/tools/](docs/tools/) | 当前工具说明：采集、回放、V6 标注工作台 |
| [docs/AGENTS/](docs/AGENTS/) | 历史记录：00 状态、03 数据格式、04 环境、05 操作手册、10 训练策略、12–17 WK/VG/快慢 VLA、18 架构变更历史、19 ACT 协议 |

## 合规边界

网易对「辅助类脚本」零容忍，处罚可升级到封实名+封设备。本项目**只**运行于官方自定义剧本（Bot 对战），
不连真人匹配/排位；不读取内存、不做 DLL 注入、不截帧 hook。任何自动化在条款上仍属灰色，
风险自担，项目定位为技术研究。
