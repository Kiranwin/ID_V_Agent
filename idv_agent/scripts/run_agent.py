"""RealtimeAgent CLI 入口（M1 规则 / M2 学习快层）。

用法（dry-run 默认安全）：
    python -m idv_agent.scripts.run_agent --mode rule --duration 30
    python -m idv_agent.scripts.run_agent --mode fast --fast-ckpt checkpoints/fast_controller/final.pt --duration 30
    python -m idv_agent.scripts.run_agent --mode rule --send-input   # 真发送（仅自定义剧本/训练模式！）

注意：
- 默认 dry-run（只打印命令不发键鼠）。
- --send-input 需要 pydirectinput，且**仅限官方自定义剧本/训练模式**（合规红线）。
- F12 紧急退出，释放所有按住的键。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from idv_agent.agent.action_executor import ActionExecutor
from idv_agent.agent.memory import MatchMemory
from idv_agent.agent.realtime_agent import RealtimeAgent
from idv_agent.agent.rule_agent import RuleAgent
from idv_agent.capture.screen_capture import CaptureConfig


def parse_region(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--region 需要 4 个整数：left,top,w,h")
    return tuple(parts)


def build_policy(args, device):
    if args.mode == "rule":
        from idv_agent.model.policy import RulePolicy
        rp = RulePolicy(RuleAgent())
        # 感知占位：给规则 agent 一个空 spatial（M1 未接视觉时先走 MOVE_LOOK 扫视）
        return rp, None
    if args.mode == "fast":
        import torch
        from idv_agent.configs.intent import INTENT_VECTOR_DIM
        from idv_agent.configs.schema import NUM_CATEGORIES, NUM_CONTINUOUS
        from idv_agent.model.dummy_vision import DummyVisionEncoder
        from idv_agent.model.fast_controller import FastController
        from idv_agent.model.policy import LearnedPolicy
        from idv_agent.training.bc_fast import FastControllerWithSkeleton

        # 若给定了 ckpt，优先从 ckpt 的 extra 元数据恢复维度，避免与训练默认不一致导致静默部分加载
        vision_hidden = args.vision_hidden
        skeleton_dim = args.skeleton_dim
        sk_vocab_size = args.skeleton_vocab_size
        use_siglip = args.use_siglip
        ckpt = None
        if args.fast_ckpt is not None and Path(args.fast_ckpt).exists():
            import torch as _torch
            ckpt = _torch.load(args.fast_ckpt, map_location="cpu", weights_only=False)
            if "model_state" not in ckpt:
                raise ValueError(f"{args.fast_ckpt} 不是有效的 FastController checkpoint（缺 model_state）")
            meta = ckpt.get("extra", {}) or ckpt
            vision_hidden = int(meta.get("vision_hidden_size", vision_hidden))
            skeleton_dim = int(meta.get("skeleton_dim", skeleton_dim))
            sk_vocab_size = int(meta.get("skeleton_vocab_size", sk_vocab_size))
            # 关键：视觉主干类型必须与训练一致。siglip 训练的 ckpt 若用 dummy 构造则视觉权重静默不匹配。
            use_siglip = bool(meta.get("use_siglip", use_siglip))
            if use_siglip:
                vision_hidden = int(meta.get("vision_hidden_size", 768))
            print(f"[run_agent] ckpt 维度: vision_hidden={vision_hidden}, "
                  f"skeleton_dim={skeleton_dim}, skeleton_vocab_size={sk_vocab_size}, "
                  f"use_siglip={use_siglip}")

        if use_siglip:
            from idv_agent.model.siglip_vision_adapter import load_siglip_vision_encoder
            try:
                from modelscope import snapshot_download
                siglip_path = snapshot_download("google/siglip2-base-patch16-224")
            except Exception:
                siglip_path = "google/siglip2-base-patch16-224"
            vision, _, vision_hidden, _ = load_siglip_vision_encoder(
                siglip_path, dtype=torch.float32, device=device, freeze=True,
            )
            print(f"[run_agent] 使用 SigLIP2 视觉主干 (hidden={vision_hidden})，"
                  "注意：SigLIP 推理需真实模型权重可用")
        else:
            from idv_agent.model.dummy_vision import DummyVisionEncoder
            vision = DummyVisionEncoder(hidden_size=vision_hidden)
        _fast = FastController(
            vision_encoder=vision, vision_hidden_size=vision_hidden,
            intent_dim=INTENT_VECTOR_DIM, skeleton_dim=skeleton_dim,
            num_categories=NUM_CATEGORIES, num_continuous=NUM_CONTINUOUS,
        )
        model = FastControllerWithSkeleton(
            fast=_fast, skeleton_vocab_size=sk_vocab_size,
            skeleton_dim=skeleton_dim,
        ).to(device)
        if ckpt is not None:
            missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
            if missing:
                print(f"[run_agent] 警告：加载后缺失 {len(missing)} 个 key：{missing}")
            if unexpected:
                print(f"[run_agent] 警告：{len(unexpected)} 个 key 未被使用")
            step = ckpt.get("step", "?")
            print(f"[run_agent] 加载 fast ckpt step={step}")
        else:
            print("[run_agent] 无 ckpt → 随机初始化（仅供调试，行为不可控）")
        policy = LearnedPolicy(model, device, image_size=args.image_size)
        return policy, model
    raise ValueError(f"未知 mode: {args.mode}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rule", "fast"], default="rule",
                   help="rule=M1规则闭环，fast=M2学习快层")
    p.add_argument("--title", default="Identity V")
    p.add_argument("--region", type=parse_region, default=None)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--vision-hidden", type=int, default=192, help="需与训练一致（bc_fast 默认 192）")
    p.add_argument("--skeleton-dim", type=int, default=64, help="需与训练一致（bc_fast 默认 64）")
    p.add_argument("--skeleton-vocab-size", type=int, default=16)
    p.add_argument("--fast-ckpt", type=Path, default=None)
    p.add_argument("--use-siglip", action="store_true",
                   help="fast 模式：用 SigLIP2 vision（需真实权重）。有 ckpt 时以 ckpt 元数据为准")
    p.add_argument("--send-input", action="store_true", help="真发送键鼠（仅沙盒！）")
    p.add_argument("--dry-run", action="store_true", help="显式声明仅记录命令（默认行为）")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.send_input and args.dry_run:
        p.error("--send-input 与 --dry-run 不能同时使用")

    import torch
    device = torch.device(args.device)
    policy, _ = build_policy(args, device)

    capture_cfg = CaptureConfig(
        fps=args.fps,
        region=args.region,
        window_title=args.title,
    )
    executor = ActionExecutor(dry_run=not args.send_input)
    agent = RealtimeAgent(
        policy=policy,
        executor=executor,
        capture_config=capture_cfg,
        target_fps=args.fps,
        memory=MatchMemory(),
        slow_planner=None,   # M1 暂不接 VLM；M2 接入 slow_planner 需手动构造
        device=device,
    )
    print(f"[run_agent] mode={args.mode}, dry_run={not args.send_input}, duration={args.duration}s")
    max_frames = int(args.fps * args.duration)
    agent.run(max_frames=max_frames)
    print(f"[run_agent] 结束，帧数={agent.frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
