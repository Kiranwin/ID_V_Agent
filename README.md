# ID_V_Agent

以《第五人格》为环境，构建「模型根据游戏场景自主决策动作」的 VLA（Vision-Language-Action）agent。

> 一句话定位：**画面 + 状态进，动作序列出**。当前阶段为求生者·律师（PC 互通版，官方自定义剧本沙盒）。

---

## 快速导航

| 你想做什么 | 看这里 |
|---|---|
| 了解项目全貌与已确认决策 | [docs/02-架构设计.md](docs/02-架构设计.md) |
| 了解历史代码审查与重构取舍 | [docs/01-审查报告.md](docs/01-审查报告.md) |
| 了解数据格式与动作空间 | [docs/03-数据格式.md](docs/03-数据格式.md) |
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
│ 慢决策层 (秒级, 异步)                          │
│   VLM 全局规划 + 对局记忆 → intent + skeleton │
├─────────────────────────────────────────────┤
│ 快控制层 (8-30Hz)                             │
│   M1 规则 agent → M2 小策略网络(BC) → M3 VLA  │
│   帧 + intent + skeleton → 动作类别 + 连续值    │
├─────────────────────────────────────────────┤
│ 感知层 (轻量结构化, 不逐帧跑大模型)              │
│   检测 / HUD 提取 / 事件检测(受击·挂椅·爆点)     │
├─────────────────────────────────────────────┤
│ 环境层                                       │
│   DDA 捕获 + SendInput 注入 + 官方自定义剧本沙盒 │
└─────────────────────────────────────────────┘
```

快慢双系统分层是 Cradle / SIMA 2 / Gemini Robotics 的共同范式，也是对抗场景唯一可行的实时结构（纯 LLM 闭环 300ms-20s/动作，无法应对实时对抗）。

## 已确认决策（勿再摇摆）

1. **平台**：仅 PC 互通版（DDA 捕获 + SendInput 注入），不用模拟器。
2. **首发角色**：求生者·律师（自带地图 = 慢层感知捷径，Tab 看密码机/队友位置）。
3. **硬件**：本地 RTX 2080 Ti 22GB（Turing，无 BF16 → FP16 + GradScaler）+ 可租云 GPU（M3 微调用）。
4. **最终走 VLA**：快层按「M2 SigLIP2 小策略网络 → M3 VLA」演进，策略接口可插拔（见 `idv_agent/model/policy.py`）。
5. **战场**：官方自定义剧本沙盒（3 Bot 队友 + 1 Bot 监管者）。合规红线：不碰真人排位、不用内存/hook/截帧工具。

## 目录结构

```
ID_V_Agent/
├── docs/                 # 全部文档（本项目的文档唯一入口）
├── idv_agent/            # 主包
│   ├── configs/          # 动作空间 / 键位 / 意图类别 / 训练配置
│   ├── capture/          # 屏幕捕获(dxcam) + 输入日志(pynput) + 窗口检测
│   ├── labels/           # 按键状态机 / 动作提取 / 意图推断 / 数据集 / JSONL
│   ├── model/            # 快层 FastController / 慢层 GameActorCritic / 策略抽象
│   ├── training/         # bc_fast / bc_slow（FP16 + GradScaler）
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

# 3. 真实录制（管理员权限，详见 docs/05）
python -m idv_agent.scripts.record --list-windows

# 4. 数据统计（确认类别分布，P4 验证）
python -m idv_agent.scripts.stats --session-dir data/sessions/<session_id>
```

## 合规边界

网易对「辅助类脚本」零容忍，处罚可升级到封实名+封设备。本项目**只**运行于官方自定义剧本（Bot 对战），不连真人匹配/排位；不读取内存、不做 DLL 注入、不截帧 hook。任何自动化在条款上仍属灰色，风险自担，项目定位为技术研究。
