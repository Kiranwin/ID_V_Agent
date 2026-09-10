"""CLI 入口。

全管线：录制 `record_vla` → 回放 `replay_mouse` / `data_workbench` →
标注 `data_workbench` / `import_mvp_anylabeling` → 数据准备 `prepare_mvp_v6` /
`assemble_m29_dataset` / `audit_*` → 训练 `train_m29` → 推理运行 `run_agent` →
离线评估 `evaluate_model_ability`（指标 + 视觉依赖门禁）。
"""
