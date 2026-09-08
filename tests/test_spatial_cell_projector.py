"""CPU contracts for ACT's fixed-raster spatial cell projection."""

from __future__ import annotations

import torch


def test_raster_cell_projector_preserves_cell_location_and_gradients():
    from idv_agent.model.spatial_cell_projector import SpatialCellProjector

    projector = SpatialCellProjector(dim=4, k=2, cell_dim=3, output_dim=5)
    left = torch.zeros(1, 4, 4, requires_grad=True)
    left.data[0, 0, 0] = 2.0
    right = left.detach().clone().requires_grad_(True)
    right.data[:, [0, 3]] = right.data[:, [3, 0]]

    left_output = projector(left)
    right_output = projector(right)

    assert left_output.shape == (1, 5)
    assert torch.isfinite(left_output).all()
    assert not torch.allclose(left_output, right_output, atol=1e-7, rtol=1e-7)
    left_output.square().mean().backward()
    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert projector.cell_projection.weight.grad is not None
    assert torch.isfinite(projector.cell_projection.weight.grad).all()


def test_raster_cell_projector_rejects_non_grid_inputs():
    from idv_agent.model.spatial_cell_projector import SpatialCellProjector

    projector = SpatialCellProjector(dim=4, k=2, cell_dim=3, output_dim=5)
    for bad in (torch.zeros(1, 3, 4), torch.zeros(1, 4, 3), torch.zeros(4, 4)):
        try:
            projector(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {tuple(bad.shape)}")


def test_raster_cell_projector_preserves_per_cell_magnitude_signal():
    """A visual-token scale change must not be erased before raster fusion."""
    from idv_agent.model.spatial_cell_projector import SpatialCellProjector

    projector = SpatialCellProjector(dim=4, k=2, cell_dim=3, output_dim=5)
    dark = torch.ones(1, 4, 4)
    bright = dark * 2.0

    dark_output = projector(dark)
    bright_output = projector(bright)

    assert not torch.allclose(dark_output, bright_output, atol=1e-6, rtol=1e-6)
