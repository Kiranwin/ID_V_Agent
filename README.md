# ID_V_Agent

以《第五人格》为环境，构建「模型根据游戏场景自主决策动作」的 VLA（Vision-Language-Action）agent。

> 一句话定位：**画面 + 状态进，动作序列出**。当前阶段为求生者·律师（PC 互通版，官方自定义剧本沙盒）。

---

## 快速导航

| 你想做什么 | 看这里 |
|---|---|
| 查看当前真实进度与唯一执行顺序（先看） | [docs/00-当前状态.md](docs/00-当前状态.md) |
| 了解项目全貌与已确认决策 | [docs/02-架构设计.md](docs/02-架构设计.md) |
| 了解历史代码审查与重构取舍 | [docs/01-审查报告.md](docs/01-审查报告.md) |
| 了解数据格式与动作空间 | [docs/03-数据格式.md](docs/03-数据格式.md) |
| 了解 YOLO → VG 标注与训练消费路径 | [docs/14-VG标注与消费路径.md](docs/14-VG标注与消费路径.md) |
| 快慢 VLA 时序、条件融合与特征缓存接口 | [docs/15-快慢VLA接口与时序缓存协议.md](docs/15-快慢VLA接口与时序缓存协议.md) |
| Subgoal 词汇表与半自动派生规则 | [docs/16-Subgoal词汇表v1.md](docs/16-Subgoal词汇表v1.md) |
| 搭环境 / 硬件 / 反作弊须知 | [docs/04-环境与硬件.md](docs/04-环境与硬件.md) |
| 录制 → 构建 → 训练 → 部署 实操 | [docs/05-操作手册.md](docs/05-操作手册.md) |
| 里程碑与当前进度 | [docs/06-里程碑.md](docs/06-里程碑.md) |
| 项目时间规划（周级执行表） | [docs/08-项目时间规划.md](docs/08-项目时间规划.md) |
| 竞品与同类项目调研（VLA/游戏 Agent 可借鉴优点） | [docs/07-竞品调研.md](docs/07-竞品调研.md) |
| 可行性评估与启动检查 | [docs/09-可行性与启动检查.md](docs/09-可行性与启动检查.md) |
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

快慢双系统共享同一 VLA 基座：捕获线程持续写入单帧特征缓存，快头按固定频率消费窗口，慢头低频更新意图与上下文。纯大模型逐帧闭环不满足本项目的实时预算。

## 已确认决策（勿再摇摆）

1. **平台**：仅 PC 互通版（DDA 捕获 + SendInput 注入），不用模拟器。
2. **首发角色**：求生者·律师（自带地图 = 慢层感知捷径，Tab 看密码机/队友位置）。
3. **硬件**：本地 RTX 2080 Ti 22GB（Turing，无 BF16 → FP16 + GradScaler）+ 可租云 GPU（M3 微调用）。
4. **最终走 VLA**：VLA 原生动作块是唯一训练主线；规则策略和 YOLO 仅作教师、标注辅助与安全兜底。
5. **战场**：官方自定义剧本沙盒（3 Bot 队友 + 1 Bot 监管者）。合规红线：不碰真人排位、不用内存/hook/截帧工具。

## 目录结构

```
ID_V_Agent/
├── docs/                 # 全部文档（本项目的文档唯一入口）
├── idv_agent/            # 主包
│   ├── configs/          # 动作空间 / 键位 / 意图类别 / 训练配置
│   ├── capture/          # 屏幕捕获(dxcam) + 输入日志(pynput) + 窗口检测
│   ├── labels/           # 破译状态机等标注辅助工具
│   ├── model/            # VLA 主干适配器 / 策略抽象
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
python -m idv_agent.scripts.validate_vla_raw data/vla_raw_sessions/<session_id>

# M0 捕获延迟基准（管理员终端、游戏窗口已打开；只读不注入）
python -m idv_agent.scripts.benchmark_capture --title "第五人格" --seconds 60

# 5. 构建 VLA action chunks
python -m idv_agent.scripts.build_vla_chunks data/vla_raw_sessions/<session_id>
```

## 合规边界

网易对「辅助类脚本」零容忍，处罚可升级到封实名+封设备。本项目**只**运行于官方自定义剧本（Bot 对战），不连真人匹配/排位；不读取内存、不做 DLL 注入、不截帧 hook。任何自动化在条款上仍属灰色，风险自担，项目定位为技术研究。
