from types import SimpleNamespace

import pytest
import torch

from idv_agent.scripts.benchmark_act_realtime import (
    ActionChunkStep,
    LatestFrameSlot,
    FixedRateGate,
    RealtimeWindow,
    action_step_to_text,
    parse_output_size,
)


def test_latest_frame_slot_replaces_stale_capture_and_reports_age():
    slot = LatestFrameSlot()
    slot.put("old", captured_at=1.0)
    slot.put("new", captured_at=2.0)
    item = slot.take_latest()
    assert item is not None
    assert item.payload == "new"
    assert item.captured_at == 2.0
    assert slot.take_latest() is None


def test_fixed_rate_gate_is_independent_of_producer_ticks():
    gate = FixedRateGate(15.0, start=0.0)
    assert gate.due(0.0)
    assert not gate.due(0.01)
    assert gate.due(0.067)
    assert gate.next_deadline > 0.067


def test_parse_output_size_requires_width_height():
    assert parse_output_size("640,384") == (640, 384)


def test_action_step_to_text_maps_v4_semantics_without_legacy_category():
    step = ActionChunkStep(
        move_dir=1, camera_dx=2, camera_dy=-1,
        buttons=(1, 0, 0, 0, 0, 0), duration_frames=6,
    )
    text = action_step_to_text(step)
    assert text == "move=north camera=(2,-1) buttons=interact duration=6"


def test_realtime_window_keeps_last_three_features_and_pads_before_warmup():
    window = RealtimeWindow(max_length=3)
    window.append(torch.tensor([1.0]), frame_index=0, timestamp_ns=100)
    features, valid, indices = window.snapshot()
    assert features.shape == (1, 1, 1)
    assert valid.tolist() == [[True]]
    assert indices == [0]
    for i in range(1, 5):
        window.append(torch.tensor([float(i)]), frame_index=i, timestamp_ns=100 + i)
    features, valid, indices = window.snapshot()
    assert features.flatten().tolist() == [2.0, 3.0, 4.0]
    assert valid.tolist() == [[True, True, True]]
    assert indices == [2, 3, 4]


def test_realtime_window_returns_exact_v5_eight_frame_stride_three_window():
    from idv_agent.scripts.benchmark_act_realtime import RealtimeWindow

    window = RealtimeWindow(max_length=22)
    for index in range(22):
        window.append(torch.tensor([float(index)]), frame_index=index, timestamp_ns=index + 1)

    assert window.ready(history_frames=8, history_stride=3)
    features, valid, indices = window.snapshot_window(history_frames=8, history_stride=3)
    assert features.flatten().tolist() == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
    assert valid.tolist() == [[True] * 8]
    assert indices == [0, 3, 6, 9, 12, 15, 18, 21]


def test_realtime_window_protocol_requires_full_span():
    from idv_agent.scripts.benchmark_act_realtime import RealtimeWindow

    window = RealtimeWindow(max_length=22)
    for index in range(21):
        window.append(torch.tensor([float(index)]), frame_index=index, timestamp_ns=index + 1)
    assert not window.ready(history_frames=8, history_stride=3)
    with pytest.raises(RuntimeError, match="未预热"):
        window.snapshot_window(history_frames=8, history_stride=3)
