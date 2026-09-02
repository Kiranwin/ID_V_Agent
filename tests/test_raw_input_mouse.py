import pytest

from idv_agent.capture.raw_input_mouse import MouseDelta, _parse_raw_mouse_delta, summarize_deltas


def test_summarize_deltas_reports_nonzero_rate_and_totals():
    deltas = [
        MouseDelta(100, 0, 0),
        MouseDelta(200, 3, -2),
        MouseDelta(300, -1, 4),
    ]

    summary = summarize_deltas(deltas)

    assert summary["samples"] == 3
    assert summary["nonzero_samples"] == 2
    assert summary["nonzero_rate"] == pytest.approx(2 / 3)
    assert summary["total_dx"] == 2
    assert summary["total_dy"] == 2
    assert summary["max_abs_dx"] == 3
    assert summary["max_abs_dy"] == 4


def test_summarize_empty_deltas_is_well_formed():
    summary = summarize_deltas([])
    assert summary == {
        "samples": 0,
        "nonzero_samples": 0,
        "nonzero_rate": 0.0,
        "total_dx": 0,
        "total_dy": 0,
        "max_abs_dx": 0,
        "max_abs_dy": 0,
    }


def test_parse_raw_mouse_delta_decodes_relative_report():
    # 8-byte RAWINPUTHEADER + RAWMOUSE fields through lLastX/lLastY.
    raw = bytearray(8 + 20)
    raw[0:4] = (0).to_bytes(4, "little")
    raw[8 + 12:8 + 16] = int(-7).to_bytes(4, "little", signed=True)
    raw[8 + 16:8 + 20] = int(5).to_bytes(4, "little", signed=True)
    assert _parse_raw_mouse_delta(bytes(raw), 8) == (-7, 5)


def test_parse_raw_mouse_delta_ignores_absolute_report():
    raw = bytearray(8 + 20)
    raw[8:10] = (1).to_bytes(2, "little")
    assert _parse_raw_mouse_delta(bytes(raw), 8) is None
