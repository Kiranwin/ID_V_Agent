from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(("pixels", "expected"), [
    (-49, -2), (-48, -1), (-13, -1), (-12, 0), (0, 0), (12, 0),
    (13, 1), (48, 1), (49, 2),
])
def test_v5_camera_bucket_uses_shared_pixel_edges(pixels, expected):
    from idv_agent.vla.action_chunk import camera_bucket_from_pixels

    assert camera_bucket_from_pixels(pixels) == expected


def test_act_executor_uses_shared_bucket_representative_pixels():
    from idv_agent.agent.act_action_executor import ACTActionChunkExecutor

    commands = []
    executor = ACTActionChunkExecutor(send=commands.extend, capture_fps=30, tick_hz=15)
    executor.submit([SimpleNamespace(move_dir=0, camera_dx=1, camera_dy=-2,
                                     buttons=(0, 0, 0, 0, 0, 0), duration_frames=6)])
    executor.tick()

    mouse = next(command for command in commands if command.kind == "mouse_move")
    assert (mouse.dx_px, mouse.dy_px) == (25.0, -110.0)
