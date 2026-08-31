"""M0 环境自检：不启动游戏、不捕获屏幕、不发送输入。"""

from __future__ import annotations

import ctypes
import importlib.util
import platform
import sys


REQUIRED_RUNTIME = ("torch", "numpy", "pandas", "PIL", "cv2")
CAPTURE_INPUT = ("dxcam", "pynput", "pygetwindow", "pydirectinput")
MODEL_RUNTIME = ("torchvision", "transformers", "peft", "accelerate", "modelscope")


def _available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _line(label: str, ok: bool, detail: str = "") -> None:
    mark = "OK" if ok else "MISSING"
    suffix = f" — {detail}" if detail else ""
    print(f"[{mark:7}] {label}{suffix}")


def main() -> int:
    print("=== ID_V_Agent M0 doctor (read-only) ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    failed = False
    py_ok = sys.version_info[:2] in {(3, 10), (3, 11), (3, 12), (3, 13)}
    _line("Python 3.10–3.13", py_ok, "项目依赖尚未针对 3.14 验证" if not py_ok else "")
    failed |= not py_ok

    for name in REQUIRED_RUNTIME:
        ok = _available(name)
        _line(f"runtime:{name}", ok)
        failed |= not ok

    for name in CAPTURE_INPUT:
        _line(f"capture:{name}", _available(name), "管理员+Windows 游戏环境需要" if not _available(name) else "")

    for name in MODEL_RUNTIME:
        _line(f"model:{name}", _available(name), "训练/真实模型路径需要" if not _available(name) else "")

    is_admin = bool(ctypes.windll.shell32.IsUserAnAdmin()) if sys.platform == "win32" else False
    _line("管理员权限", is_admin, "请用‘以管理员身份运行’启动终端" if not is_admin else "")

    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        detail = f"{torch.__version__}, CUDA={torch.version.cuda}, device={torch.cuda.get_device_name(0) if cuda else 'none'}"
        _line("Torch CUDA", cuda, detail)
    except Exception as exc:
        _line("Torch CUDA", False, str(exc))
        failed = True

    try:
        from idv_agent.capture.input_logger import list_windows
        wins = list_windows()
        _line("Identity V 窗口", any("identity" in w.lower() or "第五人格" in w for w in wins),
              f"可见窗口数={len(wins)}；游戏启动后重试" if not any("identity" in w.lower() or "第五人格" in w for w in wins) else "")
    except Exception as exc:
        _line("Identity V 窗口", False, str(exc))

    runtime_ready = py_ok and all(_available(n) for n in REQUIRED_RUNTIME)
    live_ready = runtime_ready and all(_available(n) for n in CAPTURE_INPUT) and is_admin
    if failed:
        conclusion = "当前仅可运行离线测试；先修复上述基础缺项。"
    elif live_ready:
        conclusion = "运行时与管理员权限满足；启动自定义剧本后继续做短时 dry-run 与延迟验收。"
    else:
        conclusion = "基础运行时满足；真实 M0 仍需管理员权限、采集依赖和游戏窗口。"
    print("\n结论：" + conclusion)
    # 缺少可选捕获/模型依赖时仍返回 0，方便把 doctor 用作信息查询；
    # 仅基础 runtime 或 Python 版本异常才返回非零。
    core_failed = (not py_ok) or any(not _available(n) for n in REQUIRED_RUNTIME)
    return 1 if core_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
