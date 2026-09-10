"""SGan（State-Guided Action Network，原 M29）实时运行入口。

用法（默认 dry-run，只记录命令、不发送键鼠）：

    python -m idv_agent.scripts.run_agent --mode sgan ^
      --checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt ^
      --model-path <Qwen3-VL-4B 权重目录> ^
      --trace reports/sgan_dryrun.jsonl --duration 30

真发送（仅官方自定义剧本/训练模式，且必须显式授权）：

    python -m idv_agent.scripts.run_agent --mode sgan ... ^
      --send-input --allow-undeployed-send-input

本入口只保留 SGan：旧的 rule 安全兜底与旧 ACT 同步 runtime 已移除。
字段、依赖脚本、trace 结构与使用样例见 docs/agent.md。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m idv_agent.scripts.run_agent",
        description="SGan（原 M29 状态引导策略）实时运行入口；默认 dry-run。",
    )
    p.add_argument("--mode", choices=["sgan"], default="sgan",
                   help="运行架构；只保留 sgan（原 m29），旧的 rule/act 已移除")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="SGan checkpoint 文件（例如 checkpoints/<run>/m29.pt）")
    p.add_argument("--model-path", default=None,
                   help="Qwen3-VL 基础模型目录；省略时读取 checkpoint manifest 的 base_model")
    p.add_argument("--title", default="第五人格",
                   help="被捕获的窗口标题")
    p.add_argument("--device", default="cuda",
                   help="推理设备，例如 cuda")
    p.add_argument("--duration", type=float, default=30.0,
                   help="运行时长（秒）")
    p.add_argument("--trace", type=Path, required=True,
                   help="运行 trace JSONL 输出路径；必须是不存在的路径")
    p.add_argument("--send-input", action="store_true",
                   help="真实发送键鼠；仅官方自定义剧本/训练模式，默认关闭")
    p.add_argument("--allow-undeployed-send-input", action="store_true",
                   help="显式允许 deployable=false 的 checkpoint 真发送；必须与 --send-input 同时使用")
    p.add_argument("--enable-navigation", action="store_true",
                   help="task=q_interact 的 checkpoint 也输出导航动作（dry-run/诊断用）")
    args = p.parse_args(argv)

    if args.duration <= 0:
        p.error("--duration 必须为正数")
    if args.allow_undeployed_send_input and not args.send_input:
        p.error("--allow-undeployed-send-input 必须与 --send-input 同时使用")
    if not args.checkpoint.is_file():
        p.error(f"checkpoint 不存在: {args.checkpoint}")
    if args.trace.exists():
        p.error(f"trace 已存在，请换一个新路径: {args.trace}")

    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "SGan 需要 CUDA；当前 torch.cuda.is_available()=False。"
            "请在装有 CUDA 版 PyTorch 的 idv312 环境运行。"
        )

    from types import SimpleNamespace

    from idv_agent.scripts.run_m29 import run as run_sgan

    print(f"[run_agent] mode=sgan, dry_run={not args.send_input}, duration={args.duration}s")
    result = run_sgan(SimpleNamespace(
        checkpoint=args.checkpoint,
        model_path=args.model_path,
        title=args.title,
        device=args.device,
        duration=args.duration,
        trace=args.trace,
        send_input=args.send_input,
        enable_navigation=args.enable_navigation,
        allow_undeployed_send_input=args.allow_undeployed_send_input,
    ))
    counts = result["counts"]
    print(f"[run_agent] 结束，captures={counts['captures']} encoded={counts['encoded']}")
    print(f"[sgan:trace] {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
