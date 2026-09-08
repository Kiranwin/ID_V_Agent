import json
from idv_agent.agent.mvp_closed_loop import ClosedLoopTraceWriter, ClosedLoopTransition, verify_transitions
from idv_agent.scripts.evaluate_mvp_closed_loop import main

def _rows(source="live_sandbox", causal=True):
    return [
        ClosedLoopTransition("ep", 1, {"visible": "no"}, {"camera_dx": 1}, {"visible": "yes", "distance": "far"}, source, causal),
        ClosedLoopTransition("ep", 2, {"visible": "yes", "distance": "far"}, {"move_dir": 1}, {"visible": "yes", "distance": "near"}, source, causal),
        ClosedLoopTransition("ep", 3, {"visible": "yes", "distance": "near"}, {"interact": 0}, {"visible": "yes", "interact_prompt": 1}, source, causal),
        ClosedLoopTransition("ep", 4, {"visible": "yes", "interact_prompt": 1}, {"interact": 1}, {"frame_state": "decoding"}, source, causal),
    ]

def test_live_sandbox_trace_passes_all_stages():
    report = verify_transitions(_rows())
    assert report["mvp_pass"] is True
    assert report["stage_counts"] == {"find_cipher": 1, "approach_cipher": 1, "q_prompt": 1, "decode_entry": 1}

def test_human_replay_cannot_pass_causal_gate():
    report = verify_transitions(_rows("human_replay", False))
    assert report["stage_counts"]["decode_entry"] == 1
    assert report["mvp_pass"] is False

def test_trace_writer_requires_post_action_observation_to_append():
    from pathlib import Path
    path = Path(".mvp_closed_loop_writer_test.jsonl")
    try:
        with ClosedLoopTraceWriter(path, episode_id="ep") as writer:
            writer.append(before={"visible": "no"}, action={"camera_dx": 1}, after={"visible": "yes"})
        row = ClosedLoopTransition.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert row.source == "live_sandbox" and row.causal is True
    finally:
        path.unlink(missing_ok=True)

def test_cli_returns_nonzero_for_noncausal_trace():
    from pathlib import Path
    trace = Path(".mvp_closed_loop_test_trace.jsonl")
    report = Path(".mvp_closed_loop_test_report.json")
    try:
        trace.write_text("\n".join(json.dumps({"episode_id": r.episode_id, "step": r.step, "before": r.before, "action": r.action, "after": r.after, "source": "human_replay", "causal": False}) for r in _rows()) + "\n", encoding="utf-8")
        assert main(["--trace", str(trace), "--output", str(report)]) == 2
    finally:
        trace.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
