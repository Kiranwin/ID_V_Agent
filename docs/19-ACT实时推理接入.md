# ACT 实时推理接入协议

`idv_agent.scripts.run_agent --mode act` 接入 M3_ACT 动作块 checkpoint。运行器只使用官方自定义剧本/训练营，默认 `dry-run`，`--send-input` 仍要求管理员权限。

## 时序协议

- 冷启动条件固定为 `intent=travel`、`subgoal=observe`、72 维 `history_actions` 全零；积累至少 22 个采集帧后才开始推理。
- 文本侧 instruction/mode embedding 在 episode 初始化时计算一次；每帧只调用视觉编码器并写入 `FrameFeatureCache`。
- v5 视觉输入固定为最近 8 帧、相邻采样间隔 3 帧（缓存保留 22 帧）；这与构建器的 `history=8/history_stride=3` 完全一致。
- 慢层默认 1 Hz，消费上述 8 帧和执行器已经应用的真实 6 帧宏动作历史；其输出作为后续快层条件。
- 快层默认 15 Hz，独立于捕获频率从缓存读取最新窗口。当前动作块执行期间不打断；推理结果只保留一个 latest pending 块并覆盖旧结果，当前块结束后立即使用最新块。
- 特征年龄超过 `--max-feature-age-s` 时清空 pending、停止执行并释放所有按键。
- v5 `duration_frames` 固定为 6；部署不使用未按 variable-duration 协议训练的 duration head。

## 示例

```powershell
python -m idv_agent.scripts.run_agent `
  --mode act `
  --act-checkpoint checkpoints/M3_ACT/act_mvp_b8_mb8_300.pt `
  --act-init-checkpoint checkpoints/M2_VG `
  --device cuda `
  --duration 30
```

## 训练/离线对齐协议

离线 `evaluate_vla.py` 使用 8 帧视觉窗口、数据集中真实的 8 个 6 帧宏动作
`history_actions`，且不注入时间差（等价全零 `time_deltas`）。ACT v5 实机必须使用同一协议：
8 帧、stride=3、8 个 6 帧宏动作 history、`time_deltas=None`。

- `--history-frames` 固定为 8；其它值会被拒绝。
- `--history-stride` 固定为 3；其它值会被拒绝。
- `--use-time-deltas` 已禁止。当前 checkpoint 未用真实时间差训练，不能通过参数注入负时间差。
- `--zero-history`：诊断模式，每次推理都用全零 history，不读取执行器历史（可单独观察自回归 history 的影响）。

对齐复现：直接使用默认参数，不传 `--use-time-deltas`、不传 `--zero-history`；观察完整 4-step `chunk=` 序列与 logits 差异。

实机发送只在确认处于官方自定义剧本/训练模式后显式加入 `--send-input`。建议先用默认 dry-run 检查动作日志，再进行发送测试。
