# ID_V_Agent

以《第五人格》为环境，构建「模型根据游戏场景自主决策动作」的 VLA（Vision-Language-Action）agent。

> 一句话定位：**画面 + 状态进，动作序列出**。当前阶段为求生者·律师（PC 互通版，官方自定义剧本沙盒）。

---

## 快速导航

| 你想做什么 | 看这里 |
|---|---|
| 查看当前真实进度与唯一执行顺序（先看） | [docs/00-当前状态.md](docs/00-当前状态.md) |
| 了解数据格式与动作空间 | [docs/03-数据格式.md](docs/03-数据格式.md) |
| 了解 YOLO → VG 标注与训练消费路径 | [docs/14-VG标注与消费路径.md](docs/14-VG标注与消费路径.md) |
| 快慢 VLA 时序、条件融合与特征缓存接口 | [docs/15-快慢VLA接口与时序缓存协议.md](docs/15-快慢VLA接口与时序缓存协议.md) |
| Subgoal 词汇表与半自动派生规则 | [docs/16-Subgoal词汇表v1.md](docs/16-Subgoal词汇表v1.md) |
| WK→VG→ACT 三阶段训练继承协议 | [docs/17-三阶段训练继承协议.md](docs/17-三阶段训练继承协议.md) |
| 已完成的架构/性能变更历史 | [docs/18-架构变更历史.md](docs/18-架构变更历史.md) |
| ACT 实时推理接入协议 | [docs/19-ACT实时推理接入.md](docs/19-ACT实时推理接入.md) |
| 搭环境 / 硬件 / 反作弊须知 | [docs/04-环境与硬件.md](docs/04-环境与硬件.md) |
| 录制 → 构建 → 训练 → 部署 实操 | [docs/05-操作手册.md](docs/05-操作手册.md) |
| 多模式 LoRA 与 ActVLP 训练策略 | [docs/10-VLA训练策略.md](docs/10-VLA训练策略.md) |
| 给 AI 助手（Claude/其他）的协作指南 | [AGENTS.md](AGENTS.md) |

## 架构一图流

```
┌─────────────────────────────────────────────┐
│ VLA 主链路 (滚动 action chunk)                 │
│   连续多帧 + 指令 + mode → 并行动作块          │
├─────────────────────────────────────────────┤
│ 安全兜底                                       │
│   规则策略 + YOLO/状态机 → dry-run/短时控制    │
├─────────────────────────────────────────────┤
│ 感知层 (轻量结构化, 不逐帧跑大模型)              │
│   检测 / HUD 提取 / 事件检测(受击·挂椅·爆点)     │
├─────────────────────────────────────────────┤
│ 环境层                                       │
│   DDA 捕获 + SendInput 注入 + 官方自定义剧本沙盒 │
└─────────────────────────────────────────────┘
```

快慢双系统共享同一 VLA 基座：捕获线程持续写入单帧特征缓存，快头按固定频率消费窗口，慢头低频更新意图与上下文。慢头和快头各自使用独立的时序 GRU（`slow_temporal`/`fast_temporal`），消除了两个任务对同一份循环权重的竞争；纯大模型逐帧闭环不满足本项目的实时预算。详见 [docs/00-当前状态.md](docs/00-当前状态.md) 和 [docs/15-快慢VLA接口与时序缓存协议.md](docs/15-快慢VLA接口与时序缓存协议.md)。

## 已确认决策（勿再摇摆）

1. **平台**：仅 PC 互通版（DDA 捕获 + SendInput 注入），不用模拟器。
2. **首发角色**：求生者·律师（自带地图 = 慢层感知捷径，Tab 看密码机/队友位置）。
3. **硬件**：本地 RTX 2080 Ti 22GB（Turing，无 BF16 → FP16 + GradScaler）+ 可租云 GPU（ACT 训练用）。
4. **最终走 VLA**：VLA 原生动作块是唯一训练主线；规则策略和 YOLO 仅作教师、标注辅助与安全兜底。
5. **战场**：官方自定义剧本沙盒（3 Bot 队友 + 1 Bot 监管者）。合规红线：不碰真人排位、不用内存/hook/截帧工具。

## 数据与训练完整流程

从录制到实机部署的单向数据流。每一步的详细协议、字段定义和边界条件见对应的 `docs/*.md`；本节只给出流程骨架和命令入口。

### 1. 数据采集

用 `record_vla` 在官方自定义剧本/训练营录制原始会话（管理员权限，游戏窗口保持前台）：

```bash
python -m idv_agent.scripts.record_vla --window-title "第五人格" --mode standard --fps 30
```

产出 `data/new_vla_raw_sessions/<session_id>/`（脚本默认输出目录；校验通过、构建完 chunks 后再归档进 `data/vla_raw_sessions/` 作为正式训练语料）：

```text
meta.json                录制模式、帧率、对齐延迟
frame_timestamps.csv     帧号 + perf_counter_ns 时间戳
events.csv               键鼠事件流
mouse_deltas.csv         Raw Input 相对位移（相机动作唯一来源，不用绝对坐标）
frames/                  连续 JPEG 帧
```

### 2. 数据处理：校验 → 提取 → 意图标注 → 构建 action chunk

```bash
# 完整性校验
python -m idv_agent.scripts.validate_vla_raw data/new_vla_raw_sessions/<session_id>

# 生成意图片段候选（只给候选边界，不自动决定意图）
python -m idv_agent.scripts.init_intent_segments data/new_vla_raw_sessions/<session_id>
# 人工只需确认/调整顶层 intent（decipher/kite/rescue/rotate/travel/search/gate/idle）

# 校验片段覆盖完整
python -m idv_agent.scripts.validate_intent_segments data/new_vla_raw_sessions/<session_id>/intent_segments.csv --require-full-coverage

# 构建 v4 action chunks（自动从 mouse_deltas.csv 重跑逐帧动作提取）
python -m idv_agent.scripts.build_vla_chunks data/new_vla_raw_sessions/<session_id> --schema-version vla.action_chunk.v4 --history 8
```

产出 `vla_chunks_v4.jsonl`：每条样本包含 3~8 帧历史观测、4 步未来动作块（九方向移动 + 五档相机桶 × 2 轴 + 6 按键状态 + duration）、每帧的 `slow_label`（intent/subgoal）和按真实时间戳生成的 `loss_mask`。subgoal 由 `subgoal.v1` 规则从状态/事件自动派生，不需人工逐帧标注。详见 [docs/03-数据格式.md](docs/03-数据格式.md)。

### 3. 视觉标注（供 WK/VG 训练使用）

```bash
# 按局抽帧，划分 train/val
python -m idv_agent.scripts.prepare_yolo_dataset --sessions <session_ids> --output data/yolo_cipher --stride 5

# X-AnyLabeling 标注后转换为 YOLO txt（三类：cipher_visible/cipher_highlight/interact_prompt）
python -m idv_agent.scripts.convert_anylabeling_to_yolo data/vg_cipher_<session_id> --split train

# 从事件流自动推断帧状态（decoding/idle/walking），再生成 grounding/QA
python -m idv_agent.scripts.infer_frame_states data/vg_cipher_<session_id>
python -m idv_agent.scripts.generate_frame_qa data/vg_cipher_<session_id> --annotations vg_annotations.jsonl --states frame_states.jsonl --intent search
```

产出 `vg_annotations.jsonl` / `vg_grounding.jsonl` / `frame_qa.jsonl`，详见 [docs/14-VG标注与消费路径.md](docs/14-VG标注与消费路径.md)。

### 4. 三阶段训练：WK → VG → ACT

三阶段严格继承，后一阶段必须加载前一阶段 checkpoint，不从原始 Qwen 权重重新开始（详见 [docs/17-三阶段训练继承协议.md](docs/17-三阶段训练继承协议.md)）：

```text
Qwen3-VL base B0
  └─ WK LoRA → M1_WK   （游戏知识后训练，冻结视觉塔，只训语言 LoRA）
       └─ VG（冻结 lora_wk）→ M2_VG   （视觉 grounding/空间定位）
            └─ ACT（加载 M2_VG）→ M3_ACT   （动作块监督）
```

```bash
# WK：游戏知识后训练（只接受人工审核样本）
python -m idv_agent.scripts.train_wk_lora --data idv_agent/data/wk/wk_train.jsonl --output-dir checkpoints/M1_WK

# VG：视觉 grounding（必须从 M1_WK 加载）
python -m idv_agent.scripts.train_vg_grounding --data data/vg_cipher_<session_id>/vg_grounding.jsonl \
  --init-checkpoint checkpoints/M1_WK --output-dir checkpoints/M2_VG

# ACT：动作块监督（必须从 M2_VG 加载；分离时序编码器 slow_temporal/fast_temporal 只在此阶段训练）
# --data 只接受一个逗号/分号分隔的字符串，不会自动展开 glob；先拼接好路径再传入
train_data=$(ls data/mvp_vla/train/*.jsonl | paste -sd, -)
val_data=$(ls data/mvp_vla/val/*.jsonl | paste -sd, -)
python -m idv_agent.scripts.train_vla --init-checkpoint checkpoints/M2_VG --model-path <Qwen3-VL-4B 权重目录> \
  --data "$train_data" --val-data "$val_data" \
  --sampling stratified --steps 500 --checkpoint checkpoints/M3_ACT/act_v3.pt
```

ACT 阶段的模型是**单模型共享 backbone + 分离时序编码器 + 两个任务头**：

```text
FrameFeatureCache(≤8 帧) → FiLM(condition) → ┬─ slow_temporal → Slow Head（intent/subgoal/context）
                                             └─ fast_temporal → Fast Head（action chunk）
```

RTX 2080 Ti 无 BF16，训练全程 FP16 + GradScaler。当前所有 500-step 规模的候选 checkpoint 均为诊断性实验，尚未达到进入 dry-run 的标准，详见 [docs/00-当前状态.md](docs/00-当前状态.md)。

### 5. 实时推理部署

`run_agent` 现在只保留 **SGan**（原 M29）一种运行架构，旧的 `--mode rule` 与
`--mode act` 已移除；字段、依赖脚本与 trace 说明见 [docs/agent.md](docs/agent.md)。

```text
Capture loop（20 FPS）：抓屏 → 单槽覆盖 → 推理线程取走最新帧
Inference loop（≤5 Hz）：冻结 Qwen 视觉编码 → SGan 前向 → 发布最新 decision
Executor loop（50 Hz）：合并导航按键/鼠标动作与 Q，维护 200 ms 宏动作和 400 ms 陈旧门禁
```

```bash
python -m idv_agent.scripts.run_agent --mode sgan --title "第五人格" \
  --checkpoint checkpoints/<run>/m29.pt \
  --model-path <Qwen3-VL-4B 权重目录> \
  --device cuda --duration 10 --trace reports/sgan_dryrun.jsonl
```

默认 `dry-run`，只记录命令不发送；真实发送仅限官方自定义剧本/训练模式，且未过
门禁的 checkpoint 需要同时传 `--send-input --allow-undeployed-send-input`。

### 6. 评估

```bash
python -m idv_agent.scripts.evaluate_vla --checkpoint checkpoints/M3_ACT/<candidate>.pt \
  --init-checkpoint checkpoints/M2_VG --model-path <Qwen3-VL-4B 权重目录> \
  --data data/vla_raw_sessions/<holdout_session>/vla_chunks_v4.jsonl --output reports/vla_eval.json
```

离线指标：移动/相机/按键准确率、duration MAE、intent/subgoal 准确率。实机指标（暂未开始）：存活率、单局修机进度贡献、被击倒次数/牵制时长、救援成功率、逃生率，与同剧本人类基线对比。

## 目录结构

```
ID_V_Agent/
├── docs/                 # 全部文档（本项目的文档唯一入口）
├── idv_agent/            # 主包
│   ├── configs/          # 动作空间 / 键位 / 意图类别 / 训练配置
│   ├── capture/          # 屏幕捕获(dxcam) + 输入日志(pynput) + 窗口检测
│   ├── labels/           # 破译状态机等标注辅助工具
│   ├── model/            # VLA 主干适配器 / 快慢头 / 策略抽象
│   ├── training/         # VLA 训练实现
│   ├── agent/            # 部署：调度器 / 差分解码 / 注入 / 规则 agent / 记忆 / 感知
│   └── scripts/          # CLI 入口（录制/构建/检查/统计/训练/部署/smoke）
├── tests/                # 单元测试（无 GPU / 无游戏可跑）
├── checkpoints/          # 训练产物（gitignored）
└── data/                 # 录制的会话数据（gitignored）
```

## 快速开始

```bash
# 1. 装依赖（详见 docs/04）
pip install -r requirements.txt

# 2. 无 GPU / 无游戏也能验证管线
python -m idv_agent.scripts.smoke_test

# 3. M0 环境只读自检（不启动游戏、不发送输入）
python -m idv_agent.scripts.doctor

# 4. VLA 原始录制（管理员权限，详见 docs/05）
python -m idv_agent.scripts.record_vla --window-title "第五人格" --mode standard

# VLA 原始会话完整性检查
python -m idv_agent.scripts.validate_vla_raw data/new_vla_raw_sessions/<session_id>

# M0 捕获延迟基准（管理员终端、游戏窗口已打开；只读不注入）
python -m idv_agent.scripts.benchmark_capture --title "第五人格" --seconds 60

# 5. 构建 VLA action chunks
python -m idv_agent.scripts.build_vla_chunks data/new_vla_raw_sessions/<session_id> --schema-version vla.action_chunk.v4
```

## 合规边界

网易对「辅助类脚本」零容忍，处罚可升级到封实名+封设备。本项目**只**运行于官方自定义剧本（Bot 对战），不连真人匹配/排位；不读取内存、不做 DLL 注入、不截帧 hook。任何自动化在条款上仍属灰色，风险自担，项目定位为技术研究。
