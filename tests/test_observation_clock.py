from idv_agent.agent.observation_clock import (
    CapturedObservation, LatestObservationSlot, ObservationActionClock,
)


def test_slow_worker_takes_latest_not_capture_backlog():
    slot = LatestObservationSlot()
    for i in range(20):
        slot.put(CapturedObservation(i, i * 50_000_000, i))
    assert slot.take().sequence == 19
    assert slot.take() is None
    assert slot.overwritten == 19


def test_repeated_q_or_mouse_result_cannot_trigger_twice():
    clock = ObservationActionClock()
    observation = CapturedObservation(1, 100_000_000, None)
    assert clock.accept(observation, ready_ns=266_000_000, now_ns=270_000_000)
    assert not clock.accept(observation, ready_ns=266_000_000, now_ns=480_000_000)
    assert clock.release_due(470_000_000)


def test_new_sequence_from_during_action_does_not_count_as_after():
    clock = ObservationActionClock()
    assert clock.accept(CapturedObservation(1, 0, None), ready_ns=160_000_000, now_ns=170_000_000)
    assert not clock.accept(CapturedObservation(2, 200_000_000, None),
                            ready_ns=366_000_000, now_ns=380_000_000)
    assert clock.accept(CapturedObservation(3, 400_000_000, None),
                        ready_ns=566_000_000, now_ns=570_000_000)


def test_old_feature_is_rejected_even_if_just_finished_encoding():
    clock = ObservationActionClock()
    assert not clock.accept(CapturedObservation(1, 0, None),
                            ready_ns=500_000_000, now_ns=500_000_000)
