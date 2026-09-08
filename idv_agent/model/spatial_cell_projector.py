"""Fixed-raster spatial projection for ACT visual grounding.

Unlike attention pooling, this module never reduces the 8×8 visual grid to a
single weighted average.  It applies one shared transform per cell, retains
the cell's raster position, and lets the final fusion layer assign different
weights to every spatial location.
"""

from __future__ import annotations

import torch
from torch import nn


class SpatialCellProjector(nn.Module):
    """Compress ordered ``[N,k*k,D]`` tokens into the ACT frame vector."""

    def __init__(self, *, dim: int, k: int = 8, cell_dim: int = 32,
                 output_dim: int = 2560):
        super().__init__()
        if dim < 1 or k < 1 or cell_dim < 1 or output_dim < 1:
            raise ValueError("dim/k/cell_dim/output_dim 必须为正数")
        self.dim = int(dim)
        self.k = int(k)
        self.n_cells = self.k * self.k
        self.cell_dim = int(cell_dim)
        self.output_dim = int(output_dim)
        # Do not LayerNorm each visual cell.  Per-cell normalization erases
        # image-dependent token magnitude, which is a real visual signal (a
        # real frame and a black frame became indistinguishable before fusion).
        # A fixed scale keeps the initial linear activation numerically tame
        # without making the representation invariant to that signal.
        self.register_buffer("input_scale", torch.tensor(self.dim ** -0.5), persistent=True)
        self.cell_projection = nn.Linear(self.dim, self.cell_dim)
        self.position = nn.Parameter(torch.empty(1, self.n_cells, self.cell_dim))
        self.fusion_projection = nn.Linear(self.n_cells * self.cell_dim, self.output_dim,
                                           bias=False)
        nn.init.normal_(self.position, std=self.cell_dim ** -0.5)

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        if cells.ndim != 3 or cells.shape[1:] != (self.n_cells, self.dim):
            raise ValueError(
                f"cells 必须是 [N,{self.n_cells},{self.dim}]，收到 {tuple(cells.shape)}")
        scaled = cells.to(self.cell_projection.weight.dtype) * self.input_scale.to(cells)
        per_cell = torch.nn.functional.gelu(self.cell_projection(scaled))
        positioned = per_cell + self.position.to(per_cell)
        return self.fusion_projection(positioned.flatten(start_dim=1))
