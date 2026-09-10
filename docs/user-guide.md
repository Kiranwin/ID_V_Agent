# User Guide · 从录制到运行

本文是上手这份代码的唯一操作入口：按顺序做完六步，就能从零得到一次可复现的
SGan（原 M29）运行。运行字段、trace 结构与故障排查见 [agent.md](agent.md)；
工具细节见 [tools/](tools/)；历史设计与实验记录见 [AGENTS/](AGENTS/)。

## 0. 准备

| 需要准备 | 说明 |
|---|---|
| 环境 | `conda activate idv312`（CUDA 版 PyTorch，RTX 2080 Ti 用 FP16） |
| 权限 | 管理员终端；游戏必须跑在官方自定义剧本/训练营 |
| 游戏窗口 | 窗口标题与命令里的 `--window-title`/`--title` 一致，保持前台 |
| Qwen 权重 | 本地 Qwen3-VL-4B 目录（约 8GB），路径后面反复用到 |

先做一次只读自检，确认依赖和硬件都在位：

```powershell
python -m idv_agent.scripts.smoke_test   # 无 GPU / 无游戏也能跑
python -m idv_agent.scripts.doctor       # 采集与模型依赖、CUDA、窗口
```

整条链路是单向的，任何一步的产物都不会被下一步改写：

```text
录制 → 回放测试 → 数据标注 → 数据准备 → 模型训练 → 模型推理运行
raw session       V6 审核表      JSONL       m29.pt      trace / 输入
```

## 1. 录制

```powershell
python -m idv_agent.scripts.record_vla --window-title "第五人格" `
  --mode standard --fps 20 --max-seconds 60 --output data/raw_sessions

# 完整性校验：文件齐全、CSV 字段、时间戳单调、事件落在 session 时间范围内
python -m idv_agent.scripts.validate_vla_raw data/raw_sessions/<session_id>
```

每个 session 只录一个明确片段，MVP 阶段优先覆盖「开局转向找机 → 靠近 → 出现交互
提示 → Q → 破译」。采集固定 20 FPS；相机动作只来自 `mouse_deltas.csv`（Raw Input
相对位移），不使用绝对坐标。

录制有问题时（帧率不稳、视角不转）用这两个只读诊断：

```powershell
python -m idv_agent.scripts.benchmark_capture --title "第五人格" --seconds 60
python -m idv_agent.scripts.diagnose_raw_mouse
```

## 2. 回放测试

```powershell
# 回放鼠标位移（默认 dry-run，只打印；确认无误才加 --send-input）
python -m idv_agent.scripts.replay_mouse data/raw_sessions/<session_id>

# 网页回放：按原始时间间隔看画面、标注框与决策端点
python -m idv_agent.scripts.data_workbench --root data/raw_sessions --port 8765
```

回放要确认三件事：画面与时间戳对齐、鼠标位移与视角转动一致、端点切在你真正想监督
的那一帧。详见 [tools/replay-tool.md](tools/replay-tool.md)。

## 3. 数据标注

先在 X-AnyLabeling 完成引擎标注（`interact_prompt` 等框），再用工作台完成 V6 语义
标注：主要阶段、实际破译状态、目标密码机、镜头方式、路径、动作标签来源。
**不要手改 CSV**，工作台会写进每个 session 自己的
`navigation_review_with_state.csv`。

```powershell
python -m idv_agent.scripts.data_workbench --root data/raw_sessions --port 8765

# 引擎标注导入 V6 工程
python -m idv_agent.scripts.import_mvp_anylabeling import <workspace> --output <out_dir> `
  --annotator <name> --completion-note "已完成 MVP 标注" `
  --split train --scenario-group <group>

# 导航审核结果回写工程
python -m idv_agent.scripts.import_mvp_anylabeling review <workspace> `
  --csv <session>/navigation_review_with_state.csv --output <out_dir> --reviewer <name>
```

标注口径、默认值与简化流程见 [tools/data-workbench-v6.md](tools/data-workbench-v6.md)。

## 4. 数据准备

```powershell
# 几何窗口审计：相机动作是否覆盖训练需要的窗口
python -m idv_agent.scripts.audit_mvp_camera_windows data/raw_sessions/<session_id> `
  --output reports/camera_windows.json

# 工程自检与导出
python -m idv_agent.scripts.prepare_mvp_v6 validate <workspace> --require-reviewed
python -m idv_agent.scripts.prepare_mvp_v6 export   <workspace> --output <train.jsonl>
python -m idv_agent.scripts.prepare_mvp_v6 export-q <workspace> --output <q.jsonl>

# 合并成训练文件
python -m idv_agent.scripts.assemble_m29_dataset `
  --inputs <a.jsonl> <b.jsonl> --output data/derived/m29/<name>.jsonl

# 正式数据门禁：train/val 不允许 session 或 scenario-group 重叠
python -m idv_agent.scripts.audit_m29_v6_data `
  --workspaces <ws...> --q-data <q.jsonl> `
  --full-train <train.jsonl> --full-val <val.jsonl> `
  --output reports/m29_data_audit.json
```

门禁不通过的数据只能做诊断性训练，不能作为正式 checkpoint 的放行依据。

## 5. 模型训练

继承链是 `q → full → q_interact`，后一阶段必须用 `--init-checkpoint` 继承前一阶段
的 `m29.pt`：

```powershell
# Q 专项：当前帧 interact_prompt → Q token
python -m idv_agent.scripts.train_m29 --task q `
  --data data/derived/prompt_q/<q_train>.jsonl `
  --val-data data/derived/prompt_q/<q_val>.jsonl `
  --model-path <Qwen3-VL-4B 权重目录> --steps 300 `
  --output checkpoints/m29_q

# full：继承 Q 头，训练视觉事实 / 顶层决策 / 导航
python -m idv_agent.scripts.train_m29 --task full --init-checkpoint checkpoints/m29_q/m29.pt `
  --data data/derived/m29/<full_train>.jsonl --val-data data/derived/m29/<full_val>.jsonl `
  --model-path <Qwen3-VL-4B 权重目录> --steps 1000 `
  --output checkpoints/m29_full
```

产物固定为 `m29.pt` + `manifest.json` + `metrics.json`。checkpoint 默认
`deployable=false`：训练 loss 下降不等于可以发送输入。

## 6. 模型推理运行

```powershell
# 离线模型能力评估：标签指标 + 视觉依赖门禁（--mode metrics|gate|all）
python -m idv_agent.scripts.evaluate_model_ability `
  --checkpoint checkpoints/m29_full/m29.pt `
  --data data/derived/m29/<val>.jsonl --model-path <Qwen3-VL-4B 权重目录> `
  --device cuda --mode all --output reports/m29_eval.json

# 实时 dry-run：只写 trace，不发送键鼠
python -m idv_agent.scripts.run_agent --mode sgan `
  --checkpoint checkpoints/m29_full/m29.pt --model-path <Qwen3-VL-4B 权重目录> `
  --device cuda --duration 10 --trace reports/sgan_dryrun.jsonl

# 真实发送：仅官方自定义剧本/训练模式，未过门禁的 checkpoint 需显式授权
python -m idv_agent.scripts.run_agent --mode sgan `
  --checkpoint checkpoints/m29_full/m29.pt --model-path <Qwen3-VL-4B 权重目录> `
  --device cuda --duration 10 --enable-navigation `
  --send-input --allow-undeployed-send-input --trace reports/sgan_sandbox.jsonl
```

评估报告里的 `gate_pass=true` 表示图像置空、图像打乱、history 清零都会改变输出，
bbox 平移会改变导航，且 Q 没有假触发；`false` 时进程以退出码 2 结束。

## 7. 常见问题

| 现象 | 处理 |
|---|---|
| `trace 已存在` | trace 不覆盖，换新文件名 |
| `checkpoint 不存在` | `--checkpoint` 要指到 `m29.pt` 文件本身 |
| `checkpoint is not M29` | 传了旧 m28/v5 权重；当前只接受 `m29.state_guided_action.v3` |
| `SGan 需要 CUDA` | 在 `idv312` 环境运行；CPU 只适合接口自检 |
| `captures` 远大于 `encoded` | 正常：采集 20 FPS、Qwen 推理约 5 FPS，中间帧被单槽丢弃 |
| `--send-input` 被拒绝 | checkpoint 未过门禁；确认沙盒后再加 `--allow-undeployed-send-input` |

## 8. 文档地图

| 文档 | 内容 |
|---|---|
| [user-guide.md](user-guide.md) | 本文：六步操作 |
| [agent.md](agent.md) | 运行入口字段、依赖脚本、trace 结构、故障排查 |
| [tools/](tools/) | 采集、回放、标注工作台的使用说明 |
| [AGENTS/](AGENTS/) | 历史设计与实验记录（00 状态、03 数据格式、18 架构历史、19 ACT 协议等） |
