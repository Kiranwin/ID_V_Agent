from idv_agent.agent.act_action_executor import ACTActionChunkExecutor
from idv_agent.agent.act_scheduler import ACTChunkScheduler
from idv_agent.scripts.benchmark_act_realtime import ActionChunkStep


def _step(move=0, dx=0, dy=0, buttons=(0, 0, 0, 0, 0, 0), duration=2):
    return ActionChunkStep(move, dx, dy, buttons, duration)


def test_chunk_executor_repeats_step_for_duration_and_releases_on_end():
    calls = []
    executor = ACTActionChunkExecutor(send=lambda cmds: calls.extend(cmds), dry_run=True)
    executor.submit([_step(move=1, duration=2), _step(duration=1)])
    executor.tick()
    executor.tick()
    executor.tick()
    executor.shutdown()
    assert any(getattr(c, "kind", None) == "press" and c.code == "key:w" for c in calls)
    assert sum(getattr(c, "kind", None) == "release" for c in calls) >= 1


def test_chunk_executor_rejects_invalid_duration():
    executor = ACTActionChunkExecutor(send=lambda _: None)
    try:
        executor.submit([_step(duration=0)])
    except ValueError:
        return
    raise AssertionError("duration=0 must be rejected")


def test_chunk_executor_holds_step_across_fast_ticks():
    calls = []
    executor = ACTActionChunkExecutor(send=lambda cmds: calls.extend(cmds), capture_fps=30, tick_hz=15)
    executor.submit([_step(move=1, duration=6)])
    for _ in range(2):
        executor.tick()
    presses = [c for c in calls if getattr(c, "kind", None) == "press"]
    assert len(presses) == 1
    assert executor.active


def test_scheduler_requests_chunks_at_fixed_rate_and_stops_stale_features():
    calls = []
    executor = ACTActionChunkExecutor(send=lambda cmds: calls.extend(cmds), capture_fps=30, tick_hz=15)
    requested = []
    scheduler = ACTChunkScheduler(
        predict=lambda payload: requested.append(payload) or [_step(move=1, duration=6)],
        executor=executor, fast_hz=15, max_feature_age_s=0.2,
    )
    scheduler.update_feature("f", timestamp=0.0)
    scheduler.tick(now=0.0)
    scheduler.tick(now=0.01)
    assert requested == ["f"]
    scheduler.tick(now=0.30)
    assert not executor.active
    assert any(getattr(c, "kind", None) == "release" for c in calls)


def test_scheduler_keeps_latest_prediction_until_active_chunk_finishes():
    calls = []
    executor = ACTActionChunkExecutor(send=lambda cmds: calls.extend(cmds), capture_fps=30, tick_hz=15)
    requested = []
    scheduler = ACTChunkScheduler(
        predict=lambda payload: requested.append(payload) or [_step(move=len(requested), duration=1)],
        executor=executor, fast_hz=15, max_feature_age_s=1.0,
    )
    scheduler.update_feature("f", timestamp=0.0)
    scheduler.tick(now=0.0)  # first block is submitted
    scheduler.update_feature("g", timestamp=0.05)
    scheduler.tick(now=0.067)  # prediction is refreshed, but block is active
    assert requested == ["f", "g"]
    assert executor.active
    scheduler.tick(now=0.134)  # active block completes; latest block is used
    assert executor.active
    assert len(requested) == 3


def test_scheduler_consumes_chunk_at_wall_clock_rate_not_call_rate():
    calls = []
    executor = ACTActionChunkExecutor(send=lambda cmds: calls.extend(cmds), capture_fps=30, tick_hz=15)
    scheduler = ACTChunkScheduler(
        predict=lambda _payload: [_step(move=1, duration=6)],
        executor=executor, fast_hz=15, max_feature_age_s=1.0,
    )
    scheduler.update_feature("f", timestamp=0.0)
    # Calls arrive at capture cadence (30 Hz), while execution must consume
    # six frames over 200 ms, not six frames over 100 ms.
    for i in range(4):
        scheduler.tick(now=i / 30.0)
    assert executor.active
    scheduler.tick(now=0.24)
    # The first six-frame step has completed after ~200 ms and the pending
    # prediction starts immediately; it was not consumed twice as fast.
    assert executor.active
    assert any(getattr(command, "kind", None) == "release" for command in calls)
