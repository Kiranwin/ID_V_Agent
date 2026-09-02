import pandas as pd
import pytest

from idv_agent.scripts.replay_mouse import ReplayEvent, load_replay_events


def test_load_replay_events_merges_keyboard_and_raw_mouse_on_relative_timeline(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    (session / "events.csv").write_text(
        "timestamp_ns,kind,code,value\n"
        "1000,key_down,key:w,1\n"
        "1100,key_down,key:w,1\n"
        "1300,key_up,key:w,1\n",
        encoding="utf-8",
    )
    mouse = tmp_path / "mouse_deltas.csv"
    mouse.write_text(
        "timestamp_ns,dx,dy\n"
        "9000,2,0\n"
        "9200,-1,3\n",
        encoding="utf-8",
    )

    events = load_replay_events(session, mouse_deltas=mouse)

    assert events == [
        ReplayEvent(0.0, "press", "key:w"),
        ReplayEvent(0.0, "mouse_move", None, 2.0, 0.0),
        ReplayEvent(0.0000002, "mouse_move", None, -1.0, 3.0),
        ReplayEvent(0.0000001, "release", "key:w"),
    ]
