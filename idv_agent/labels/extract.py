"""从 session 目录（events.csv / mouse_positions.csv / frame 时间网格）推导每帧动作。

输出 per_frame_actions.csv：
    frame_id, timestamp_ns, category_id, category_name,
    move_x, move_y, cam_dx, cam_dy, held_keys, text_action

结合 StateReplay（按键重放）+ DecodeStateTracker（破译隐式状态）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from idv_agent.configs.keymap import DEFAULT_SURVIVOR_KEYMAP, SurvivorKeymap
from idv_agent.configs.schema import (
    ActionCategory,
    CONT_CAM_DX,
    CONT_CAM_DY,
    CONT_MOVE_X,
    CONT_MOVE_Y,
    ExtractParams,
    NUM_CONTINUOUS,
)
from idv_agent.labels.state_machine import DecodeStateTracker, StateReplay


def display_key(code: str) -> str:
    """将输入日志代码转为稳定的人类可读名称。"""
    if code.startswith("key:"):
        return code[4:].upper()
    if code.startswith("btn:"):
        return code[4:].upper()
    return code.upper()

# 破译隐式状态：这些帧的原语义被重写为 INTERACT_HOLD
_DECODE_REWRITE_CATEGORIES = frozenset({
    ActionCategory.NOOP, ActionCategory.LOOK,
    ActionCategory.VAULT, ActionCategory.DROP_BOARD,
})


def move_axis_vector(move_x: float, move_y: float, normalize_diagonal: bool = True) -> tuple[float, float]:
    """把 raw 键位移归一化。normalize_diagonal=True 时对角线保持单位向量。"""
    mag = (move_x * move_x + move_y * move_y) ** 0.5
    if mag < 1e-6:
        return (0.0, 0.0)
    if normalize_diagonal:
        return (move_x / mag, move_y / mag)
    return (float(move_x), float(move_y))


def infer_category(
    frame: FrameState,
    decode_status,  # DecodeStatus
    keymap: SurvivorKeymap,
    params: ExtractParams,
) -> tuple[int, tuple[float, float], str]:
    """从 FrameState + 破译状态推导 (category_id, (move_x, move_y raw), text_action)。"""
    held = frame.held_durations_ms
    released = frame.just_released_durations_ms

    # ---- 破译态重写 ----
    if decode_status.active and not decode_status.calibration:
        # 任何不冲突的类别重写为 INTERACT_HOLD；校准帧保留 IDLE_观察但下面仍归 INTERACT_HOLD
        return int(ActionCategory.INTERACT_HOLD), (0.0, 0.0), "INTERACT_HOLD(decode)"

    # ---- WASD 移动（连续值） ----
    mx = 0.0
    my = 0.0
    if keymap.move_left in held:
        mx -= 1.0
    if keymap.move_right in held:
        mx += 1.0
    if keymap.move_back in held:
        my -= 1.0
    if keymap.move_forward in held:
        my += 1.0
    nx, ny = move_axis_vector(mx, my, params.normalize_diagonal)

    # ---- 交互 / 校准 ----
    if decode_status.active and decode_status.calibration:
        # 校准帧：语义仍是破译中，但标记 calibration 供后续视觉标注
        return int(ActionCategory.INTERACT_HOLD), (nx, ny), "INTERACT_HOLD(decode+calibrate)"

    # 交互键 tap vs hold
    if keymap.interact in released:
        dur = released[keymap.interact]
        cat = ActionCategory.INTERACT_TAP if dur < params.tap_threshold_ms else ActionCategory.INTERACT_RELEASE
        return int(cat), (nx, ny), display_key(keymap.interact)
    if keymap.interact in held:
        dur = held[keymap.interact]
        cat = ActionCategory.INTERACT_HOLD if dur >= params.tap_threshold_ms else ActionCategory.INTERACT_TAP
        return int(cat), (nx, ny), display_key(keymap.interact)

    # 翻窗/翻板/校准 Space（非破译态 = 翻越）
    if keymap.vault in held:
        return int(ActionCategory.VAULT), (nx, ny), display_key(keymap.vault)
    if keymap.vault in released:
        return int(ActionCategory.VAULT), (nx, ny), display_key(keymap.vault)

    # 技能 / 道具 / 地图 / 表情
    if keymap.skill_1 in held:
        dur = held[keymap.skill_1]
        return int(ActionCategory.SKILL_1_HOLD if dur >= params.tap_threshold_ms else ActionCategory.SKILL_1_TAP), (nx, ny), display_key(keymap.skill_1)
    if keymap.skill_2 in held:
        dur = held[keymap.skill_2]
        return int(ActionCategory.SKILL_2_HOLD if dur >= params.tap_threshold_ms else ActionCategory.SKILL_2_TAP), (nx, ny), display_key(keymap.skill_2)
    for i, k in enumerate(keymap.item_keys):
        if k in held:
            return int(ActionCategory.ITEM_1 + i), (nx, ny), display_key(k)
    if keymap.map_view in held:
        return int(ActionCategory.MAP), (nx, ny), display_key(keymap.map_view)
    if keymap.emote in held:
        return int(ActionCategory.EMOTE), (nx, ny), display_key(keymap.emote)

    # ---- 视角（LOOK） vs 静止 ----
    cam_motion = abs(frame.mouse_dx) + abs(frame.mouse_dy)
    has_movement = (nx * nx + ny * ny) > 1e-6
    if cam_motion >= params.cam_motion_dead_zone_px:
        cat = ActionCategory.MOVE_LOOK if has_movement else ActionCategory.LOOK
    elif has_movement:
        cat = ActionCategory.MOVE
    else:
        cat = ActionCategory.NOOP
    return int(cat), (nx, ny), ""


def build_frame_grid(
    events: pd.DataFrame,
    positions: pd.DataFrame,
    frame_ts: list[int],
    keymap: SurvivorKeymap,
    params: ExtractParams,
) -> list[dict]:
    """核心提取逻辑：返回每帧动作 dict 列表。"""
    replay = StateReplay(events, positions, end_ts_ns=frame_ts[-1] if frame_ts else None)
    decode_tracker = DecodeStateTracker(keymap.interact, keymap.vault, params)

    rows: list[dict] = []
    prev_ts = None
    for idx, ts in enumerate(frame_ts):
        frame = replay.frame_state(ts, prev_ts)
        status = decode_tracker.on_frame(ts, frame)

        cat_id, (nx, ny), text_action = infer_category(frame, status, keymap, params)

        cam_dx = np.clip(frame.mouse_dx / params.cam_pixel_scale if frame.mouse_dx else 0.0, -1, 1)
        cam_dy = np.clip(frame.mouse_dy / params.cam_pixel_scale if frame.mouse_dy else 0.0, -1, 1)
        cont = [0.0] * NUM_CONTINUOUS
        cont[CONT_MOVE_X] = nx
        cont[CONT_MOVE_Y] = ny
        cont[CONT_CAM_DX] = cam_dx
        cont[CONT_CAM_DY] = cam_dy

        held_keys = "+".join(sorted(frame.held_durations_ms.keys(), key=str)) or ""

        rows.append({
            "frame_id": idx,
            "timestamp_ns": ts,
            "category_id": cat_id,
            "category_name": ActionCategory(cat_id).name,
            "move_x": nx, "move_y": ny, "cam_dx": cam_dx, "cam_dy": cam_dy,
            "held_keys": held_keys,
            "text_action": text_action,
            # P2：校准帧标记（后续视觉辅助标注用）
            "decode_calibration": int(status.calibration),
        })
        prev_ts = ts
    return rows


def extract_session(session_dir: Path,
                    keymap: Optional[SurvivorKeymap] = None,
                    params: Optional[ExtractParams] = None) -> pd.DataFrame:
    """从 session 目录提取 per_frame_actions，写回 per_frame_actions.csv 并返回 DataFrame。"""
    keymap = keymap or DEFAULT_SURVIVOR_KEYMAP
    params = params or ExtractParams()

    events = pd.read_csv(session_dir / "events.csv")
    positions = pd.read_csv(session_dir / "mouse_positions.csv")

    meta = {}
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fps = float(meta.get("effective_fps") or meta.get("target_fps") or 30.0)
    n_frames = int(meta.get("num_frames") or 0)
    start_ts = int(meta.get("start_ts_ns") or (events["timestamp_ns"].min() if len(events) else 0))
    end_ts = int(meta.get("end_ts_ns") or (events["timestamp_ns"].max() if len(events) else start_ts + int(1e9)))

    # 新录制优先使用真实抓帧时间戳；旧 session 无该文件时回退到线性网格。
    frame_ts_path = session_dir / "frame_timestamps.csv"
    if frame_ts_path.exists():
        ts_df = pd.read_csv(frame_ts_path)
        frame_ts = [int(x) for x in ts_df["timestamp_ns"].tolist()]
        if n_frames and len(frame_ts) > n_frames:
            frame_ts = frame_ts[:n_frames]
    else:
        frame_ts = [int(start_ts + int(i / fps * 1e9)) for i in range(n_frames)] if n_frames else []

    rows = build_frame_grid(events, positions, frame_ts, keymap, params)
    df = pd.DataFrame(rows)
    out = session_dir / "per_frame_actions.csv"
    df.to_csv(out, index=False)
    return df
