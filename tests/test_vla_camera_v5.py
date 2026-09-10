import pytest


@pytest.mark.parametrize(("pixels", "expected"), [
    (-68, -2), (-67.5, -1), (-13, -1), (-12, 0), (0, 0), (12, 0),
    (13, 1), (67.5, 1), (68, 2),
])
def test_v5_camera_bucket_uses_shared_pixel_edges(pixels, expected):
    from idv_agent.vla.action_chunk import camera_bucket_from_pixels

    assert camera_bucket_from_pixels(pixels) == expected
