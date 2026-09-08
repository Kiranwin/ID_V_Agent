"""RealtimeAgent CLI 入口（规则安全兜底）。

用法（dry-run 默认安全）：
    python -m idv_agent.scripts.run_agent --mode rule --duration 30
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
from idv_agent.agent.act_policy import ACTPolicy
from idv_agent.agent.memory import MatchMemory
from idv_agent.agent.realtime_agent import RealtimeAgent
from idv_agent.agent.rule_agent import CipherVisualServo, RuleAgent
from idv_agent.capture.screen_capture import CaptureConfig
from idv_agent.agent.action_decoder import ActionDecoder
from idv_agent.agent.perception import CipherMachineDetector
from idv_agent.configs.keymap import DEFAULT_SURVIVOR_KEYMAP
from idv_agent.configs.schema import ActionCategory


def parse_region(s: str):
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--region 需要 4 个整数：left,top,w,h")
    return tuple(parts)


def build_policy(args, device):
    if args.mode == "rule":
        from idv_agent.model.policy import RulePolicy
        # M1 默认使用视觉伺服：屏幕中心走廊对齐后前进，左右移仅留给
        # blocked 脱困分支；无 bbox 几何时仍回退到旧的粗粒度规则。
        rp = RulePolicy(RuleAgent(visual_servo=CipherVisualServo()))
        return rp, None
    if args.mode == "act":
        if not args.act_checkpoint:
            raise ValueError("ACT 模式必须提供 --act-checkpoint")
        closed_loop_observer = None
        if args.closed_loop_trace is not None:
            # This is an external observation of the next captured frame. It
            # is intentionally separate from ACT logits; decoding state still
            # requires a supplied frame-state detector/annotation.
            from idv_agent.agent.perception import CipherMachineDetector
            detector = CipherMachineDetector(
                template_path=str(args.cipher_template) if args.cipher_template else None,
                yolo_model_path=str(args.yolo_model) if args.yolo_model else None,
                debug=args.debug_yolo,
            )
            closed_loop_observer = lambda frame: detector.detect(frame).as_spatial()
        policy = ACTPolicy.from_checkpoint(
            args.act_checkpoint, model_path=args.model_path,
            instruction=args.instruction, mode=args.game_mode, device=device,
            capture_fps=args.fps, fast_hz=args.fast_hz, slow_hz=args.slow_hz,
            history_frames=args.history_frames,
            history_stride=args.history_stride,
            use_time_deltas=args.use_time_deltas,
            zero_history=args.zero_history,
            max_feature_age_s=args.max_feature_age_s,
            closed_loop_trace=args.closed_loop_trace,
            closed_loop_episode_id=args.closed_loop_episode_id,
            closed_loop_observer=closed_loop_observer,
        )
        return policy, policy
    raise ValueError(f"未知 mode: {args.mode}")


def run_image_test(args, policy) -> int:
    """Offline perception/policy replay; never opens DXGI or sends input."""
    import cv2
    if args.test_image:
        paths = [Path(p) for p in args.test_image]
    else:
        root = Path(args.test_dir)
        paths = sorted(p for p in root.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[::args.test_stride]
    if not paths:
        raise ValueError("测试输入为空：请提供有效图片或非空目录")
    detector = CipherMachineDetector(
        template_path=str(args.cipher_template) if args.cipher_template else None,
        yolo_model_path=str(args.yolo_model) if args.yolo_model else None,
        debug=args.debug_yolo,
    )
    decoder = ActionDecoder(DEFAULT_SURVIVOR_KEYMAP)
    memory = MatchMemory()
    print(f"[test] offline images={len(paths)} mode={args.mode} send_input=False")
    for i, path in enumerate(paths, 1):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[test] skip unreadable path={path}")
            continue
        detection = detector.detect(frame)
        spatial = detection.as_spatial()
        state = {
            "intent_vector": [0.0] * 16,
            "skeleton_idx": 0,
            "spatial": spatial,
            "memory": memory.summary(),
        }
        out = policy.decide(frame, state)
        cmds = decoder.decode(out.category_id, out.continuous)
        try:
            action_name = ActionCategory(int(out.category_id)).name
        except ValueError:
            action_name = str(out.category_id)
        print(
            f"[test:{i}] file={path.name} "
            f"detect={spatial} action={action_name} "
            f"continuous={tuple(round(float(v), 3) for v in out.continuous)} "
            f"commands={cmds}"
        )
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rule", "act"], default="rule",
                   help="rule=规则安全兜底；act=ACT 动作块策略")
    p.add_argument("--title", default="第五人格")
    p.add_argument("--cipher-template", type=Path, default=None,
                   help="密码机模板图片（建议同分辨率、同地图；不传则不触发自动找机）")
    p.add_argument("--yolo-model", type=Path, default=None,
                   help="YOLO 密码机检测权重（best.pt）")
    p.add_argument("--debug-yolo", action="store_true", help="打印 YOLO 原始检测框和结果")
    p.add_argument("--region", type=parse_region, default=None)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cipher-detect-interval", type=float, default=4.0,
                   help="密码机慢层检测间隔（秒，建议 3~5）")
    p.add_argument("--cam-pixel-scale", type=float, default=120.0,
                   help="legacy rule 模式相机归一化尺度；ACT v5 不使用此参数")
    p.add_argument("--trajectory-log", type=Path, default=None,
                   help="可选：将每帧感知状态、宏控制动作和执行命令写入 JSONL")
    p.add_argument("--trajectory-frames", type=Path, default=None,
                   help="可选：配合 --trajectory-log 保存对应帧 JPEG（用于示范轨迹训练）")
    p.add_argument("--send-input", action="store_true", help="真发送键鼠（仅沙盒！）")
    p.add_argument("--dry-run", action="store_true", help="显式声明仅记录命令（默认行为）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--act-checkpoint", type=Path, default=None,
                   help="M3_ACT 动作块 checkpoint")
    p.add_argument("--act-init-checkpoint", type=Path, default=None,
                   help="已废弃：ACT 当前不加载 M2_VG")
    p.add_argument("--model-path", default=None,
                   help="Qwen 基础模型目录（ACT 当前必填，不加载 M2 manifest）")
    p.add_argument("--instruction", default="找到密码机，靠近并进入破译")
    p.add_argument("--game-mode", choices=["standard", "joint_hunt", "blackjack"], default="standard")
    p.add_argument("--fast-hz", type=float, default=15.0)
    p.add_argument("--slow-hz", type=float, default=1.0)
    p.add_argument("--history-frames", type=int, default=8)
    p.add_argument("--history-stride", type=int, default=3,
                   help="ACT 视觉窗口采样间隔；v5 固定为 3 帧")
    p.add_argument("--use-time-deltas", action="store_true",
                   help="将真实帧时间差送入 temporal GRU；默认关闭以对齐训练/离线评估")
    p.add_argument("--zero-history", action="store_true",
                   help="诊断模式：每次推理都使用全零 history，不读取执行器历史")
    p.add_argument("--max-feature-age-s", type=float, default=0.5)
    p.add_argument("--closed-loop-trace", type=Path, default=None,
                   help="ACT 闭环证据 JSONL；需要由外部观察器接入阶段字段")
    p.add_argument("--closed-loop-episode-id", default=None)
    test_group = p.add_mutually_exclusive_group()
    test_group.add_argument("--test-image", action="append", default=None,
                            help="离线测试单张图片；可重复传入多张")
    test_group.add_argument("--test-dir", type=Path, default=None,
                            help="离线测试图片目录（按文件名排序）")
    p.add_argument("--test-stride", type=int, default=1,
                   help="--test-dir 每隔多少张图片测试一张")
    args = p.parse_args(argv)

    if args.send_input and args.dry_run:
        p.error("--send-input 与 --dry-run 不能同时使用")
    if args.test_stride <= 0:
        p.error("--test-stride 必须为正数")
    if args.trajectory_frames is not None and args.trajectory_log is None:
        p.error("--trajectory-frames 需要同时指定 --trajectory-log")
    if (args.test_image or args.test_dir) and args.send_input:
        p.error("离线测试模式禁止 --send-input")
    if args.test_dir is not None and not args.test_dir.is_dir():
        p.error(f"测试目录不存在: {args.test_dir}")

    import torch
    device = torch.device(args.device)
    if args.mode == "act" and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "ACT 需要 CUDA；当前 torch.cuda.is_available()=False。"
            "请在安装了 CUDA 版 PyTorch 的 idv312 环境中运行，"
            "或先用 benchmark/CPU 小模型诊断，不能把 CPU 的低帧数当作 ACT 结果。"
        )
    if args.send_input:
        # 提前阻止“看似启动成功、实际运行在非管理员会话”的危险误用。
        import ctypes
        if not (hasattr(ctypes, "windll") and ctypes.windll.shell32.IsUserAnAdmin()):
            raise RuntimeError(
                "--send-input 必须从管理员权限终端运行；当前 Python 进程未提升。"
            )
    policy, act_policy = build_policy(args, device)

    if args.test_image or args.test_dir:
        if act_policy is not None:
            p.error("ACT 模式需要连续帧缓存，离线单图测试请使用 benchmark_act_realtime 或规则模式")
        return run_image_test(args, policy)

    capture_cfg = CaptureConfig(
        fps=args.fps,
        region=args.region,
        window_title=args.title,
    )
    if act_policy is not None:
        if args.test_image or args.test_dir:
            return run_image_test(args, policy)
        # ACTActionChunkExecutor emits through the same safety-aware executor
        # as the legacy path; dry-run remains the default.
        act_executor = ActionExecutor(dry_run=not args.send_input)
        act_policy.executor.set_send(act_executor.execute)
        print(f"[run_agent] mode=act, dry_run={not args.send_input}, duration={args.duration}s")
        count = act_policy.run_capture(capture_cfg, duration_s=args.duration,
                                       max_frames=int(args.fps * args.duration))
        print(f"[run_agent] 结束，帧数={count}")
        if act_policy.last_run_stats:
            print(f"[act:stats] {act_policy.last_run_stats}")
            if count < 22:
                print("[act:warning] 少于 22 个有效观察帧，v5 的 8 帧/stride=3 窗口未预热；本次没有形成有效 ACT 决策证据。")
        act_executor.shutdown()
        return 0
    executor = ActionExecutor(dry_run=not args.send_input)
    agent = RealtimeAgent(
        policy=policy,
        executor=executor,
        capture_config=capture_cfg,
        target_fps=args.fps,
        memory=MatchMemory(),
        slow_planner=None,   # M1 暂不接 VLM；M2 接入 slow_planner 需手动构造
        cipher_template=str(args.cipher_template) if args.cipher_template else None,
        yolo_model_path=str(args.yolo_model) if args.yolo_model else None,
        yolo_debug=args.debug_yolo,
        cipher_detection_interval_s=args.cipher_detect_interval,
        cam_pixel_scale=args.cam_pixel_scale,
        trajectory_log=args.trajectory_log,
        trajectory_frames_dir=args.trajectory_frames,
        device=device,
    )
    print(f"[run_agent] mode={args.mode}, dry_run={not args.send_input}, duration={args.duration}s")
    max_frames = int(args.fps * args.duration)
    agent.run(max_frames=max_frames)
    print(f"[run_agent] 结束，帧数={agent.frame_count}")
    mem = agent.memory.summary()
    print(f"[cipher] detections={mem.get('cipher_detection_count', 0)} "
          f"last={mem.get('cipher_last_detection', {})}")
    for stage, stats in agent.latency.summary().items():
        print(f"[latency] {stage}: mean={stats['mean']:.2f}ms "
              f"p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms "
              f"p99={stats['p99']:.2f}ms max={stats['max']:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
