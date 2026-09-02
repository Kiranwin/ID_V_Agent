from idv_agent.scripts.record_vla import RecordingControls


def test_f9_toggles_recording_state():
    controls = RecordingControls()

    controls.on_key("f9")
    assert controls.consume_toggle() is True
    assert controls.consume_toggle() is False

    controls.on_key("f9")
    assert controls.consume_toggle() is True


def test_f10_requests_exit_and_stops_active_recording():
    controls = RecordingControls()
    controls.on_key("f9")
    assert controls.consume_toggle() is True

    controls.on_key("f10")
    assert controls.exit_requested is True
    assert controls.consume_toggle() is True


def test_unrelated_keys_are_ignored():
    controls = RecordingControls()
    controls.on_key("f8")
    assert controls.consume_toggle() is False
    assert controls.exit_requested is False
