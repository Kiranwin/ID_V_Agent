"""k×k 空间 token 编码器（C 档阶段一：独立验证模块，不改现役契约）。

背景
----
现役视觉链路把视觉塔 per-token 输出 ``mean-pool`` 成单向量再投影，画面空间
(在哪个格子)在进入主干前就被抹掉（probe 实测 raw cos≈0.9999，画面差异仅占
该层约 2.5% 能量）。

本模块是 C 级方案的第一阶段验收件：不改 ``encode_frame``/``_raw_visual_forward_batch``
/``visual_projection`` 等现役路径或 checkpoint 契约，独立在「冻结视觉塔原始
per-patch token」上做 **固定 k×k grid spatial average-pool**，产出保留位置的
``[N, k*k, D]`` 空间 token 序列（未投影）。用它离线对照：画面可分性在“保留空间”
vs “全局 mean” 间的差异，能否显著回升——据此决定是否值得进入阶段二（接投影+GRU，
会真正破坏并重建视觉投影契约）。

几何约定（Qwen3-VL 视觉塔）
---------------------------
视觉塔对一张图输出该图全部 patch 的 per-token 行（可能经过 2×2 spatial-merge 下
采样一次；图片含 t 个 tile 时 tile 沿宽并排）。因此有效网格几何由实际行数 + 
``grids=(t,h,w)`` 反推，绝不静默 mean。拍不到就报错。

只读模型权重、不训练、无学习参数；可 CPU 跑。它与现役 ``encode_raw_frames``
唯一的差别是保留胞位置、不做全局平均。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn


def per_image_rows_and_geo(
    grids: Sequence[Sequence[int]],
    merge_size: int,
    out_rows: int,
) -> tuple[list[int], list[tuple[int, int]]]:
    """从每图 grid ``(t,h,w)`` 与视觉塔实际总行数反推：每块图像素行数 + (gh,gw)。

    采用 Qwen3 的可能两形态：unmerged 行数 = t·h·w，或 spatial-merge 后的
    t·(h/M)·(w/M)。遍历 batch 里每张、按与总行匹配者归对应渲染序；每图的 geo
    选择能使其逐行计数和=已知行数的那个形态。
    """
    ms = int(merge_size)
    # 先逐图估计两形态：unmerged(t·h·w, 网格 h×(w·t)) 与 merged(t·(h/M)·(w/M))。
    n_un_list, n_me_list = [], []
    geo_un_list, geo_me_list = [], []
    for t, h, w in grids:
        t, h, w = int(t), int(h), int(w)
        n_un_list.append(t * h * w)
        n_me_list.append(max(1, t * (h // ms) * (w // ms)))
        geo_un_list.append((h, w * t))                # t tile 横向并排 -> 宽×t
        geo_me_list.append((h // ms, (w // ms) * t))
    tot_un = sum(n_un_list)
    tot_me = sum(n_me_list)
    if out_rows == tot_un:
        return n_un_list, geo_un_list
    if out_rows == tot_me:
        return n_me_list, geo_me_list
    raise ValueError(
        f"无法唯一反推：out_rows={out_rows} 不等于 unmerged 总行 {tot_un} "
        f"也不等于 merged 总行 {tot_me}（grids={[list(g) for g in grids]}, ms={ms})")


def grid_spatial_pool(tokens: torch.Tensor, rows: int, gh: int, gw: int,
                      k: int) -> torch.Tensor:
    """单图全部 per-patch token -> k×k 胞平均 token ``[k*k, D]``。

    Args:
        tokens: ``[rows, D]`` 该图全部视觉 token。
        rows:   必须 == gh*gw。
        gh,gw:  该图有效网格高宽（tile 已并入 gw）。
        k:      k×k 目标胞。
    """
    if tokens.ndim != 2:
        raise ValueError(f"tokens 需 [rows,D]，收到 {tuple(tokens.shape)}")
    if rows != gh * gw:
        raise ValueError(f"rows={rows} ≠ gh×gw={gh}×{gw}")
    D = tokens.shape[-1]
    grid = tokens.reshape(gh, gw, D)
    # 两轴切成 k 段（允许不均分，段内取 mean）
    hs = _spans(gh, k)
    ws = _spans(gw, k)
    cells = []
    for (h0, h1) in hs:
        for (w0, w1) in ws:
            cells.append(grid[h0:h1, w0:w1].mean(dim=(0, 1)))
    return torch.stack(cells, dim=0)          # [k*k, D]


def grid_spatial_pool_batch(grids: torch.Tensor, k: int) -> torch.Tensor:
    """Pool a same-geometry image batch ``[N,H,W,D]`` into ``[N,k*k,D]``."""
    if grids.ndim != 4:
        raise ValueError("batch grids 必须是 [N,H,W,D]")
    _, gh, gw, _ = grids.shape
    cells = []
    for (h0, h1) in _spans(gh, k):
        for (w0, w1) in _spans(gw, k):
            cells.append(grids[:, h0:h1, w0:w1].mean(dim=(1, 2)))
    return torch.stack(cells, dim=1)


def _spans(length: int, k: int) -> list[tuple[int, int]]:
    """把 [0,length) 切成 k 段（整除边界，余量给前段）。"""
    if k <= 0:
        raise ValueError("k 必须为正")
    if length < k:
        # 网格比 k 小：退化为重复可取整网格。
        return [(i, i + 1) if i < length else (length - 1, length) for i in range(k)]
    out = []
    base = length // k
    rem = length % k
    lo = 0
    for i in range(k):
        take = base + (1 if i < rem else 0)
        out.append((lo, lo + take))
        lo += take
    # 防御：可能因取整溢出
    out[-1] = (out[-1][0], length)
    return out


class SpatialGridEncoder(nn.Module):
    """冻结图塔 per-patch 特征 + 固定 k×k 胞 tokeniser（阶段一独立件）。

    Encode PIL/预处理图 -> ``[N, k*k, D]``（无投影、保胞位置）。用 adapter 的
    ``get_image_features`` 取 per-token 输出；不调用现役 mean 分支。

    Args:
        adapter: ``Qwen3VLBackboneAdapter``（或同类带 model.get_image_features 的包装）。
        k:  每轴胞数，默认 8 => k×k=64 super-token。
        merge_size: 视觉塔空间合并因子（与 qwen3 一致 2）。
    """

    def __init__(self, adapter, k: int = 8, *, merge_size: int = 2):
        super().__init__()
        self.adapter = adapter
        self.k = int(k)
        self.merge_size = int(merge_size)

    def forward(self, images: Sequence[Any]) -> torch.Tensor:
        return self._image_tokens(images)

    def _image_tokens(self, images: Sequence[Any]):
        prepared = [self.adapter._prepare_image(im) for im in images]
        pixel = torch.cat([p["pixel_values"] for p in prepared], dim=0)
        merged = {"pixel_values": pixel}
        if all("image_grid_thw" in p for p in prepared):
            merged["image_grid_thw"] = torch.cat([p["image_grid_thw"] for p in prepared], dim=0)
        else:
            raise ValueError("spatial 编码必须提供 image_grid_thw")
        # 取冻结视觉塔的公共 per-token 辅助：在 Qwen3VL(ForConditionalGeneration)
        # 上为 get_image_features；必要时沿 model.model 再试一层。
        image_features = getattr(self.adapter.model, "get_image_features", None)
        if image_features is None and hasattr(self.adapter.model, "model"):
            image_features = getattr(self.adapter.model.model, "get_image_features", None)
        if image_features is None:
            visual = getattr(self.adapter.model, "visual", None)
            if visual is None and hasattr(self.adapter.model, "model"):
                visual = getattr(self.adapter.model.model, "visual", None)
            if visual is None:
                raise RuntimeError("adapter 无 per-token visual 输出")
            parts = []
            for item in prepared:
                kwargs = {"hidden_states": item["pixel_values"],
                          "grid_thw": item["image_grid_thw"]}
                try:
                    part = visual(**kwargs)
                except TypeError:
                    part = visual(pixel_values=item["pixel_values"],
                                  image_grid_thw=item["image_grid_thw"])
                if hasattr(part, "last_hidden_state"):
                    part = part.last_hidden_state
                elif isinstance(part, (tuple, list)):
                    part = next(x for x in part if isinstance(x, torch.Tensor))
                if part.ndim == 2:
                    part = part.unsqueeze(0)
                if part.ndim != 3:
                    raise ValueError("visual 输出必须为 [B,L,D]")
                parts.append(part)
            if len({part.shape[1] for part in parts}) == 1:
                out = torch.cat(parts, dim=0)
            else:
                out = torch.cat([part.reshape(-1, part.shape[-1]) for part in parts], dim=0)
        else:
            out = image_features(pixel_values=merged["pixel_values"],
                                 image_grid_thw=merged["image_grid_thw"])
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            out = next(x for x in out if isinstance(x, torch.Tensor))
        grids = merged["image_grid_thw"].tolist()
        dev = out.device
        if out.ndim == 3 and out.shape[0] == len(images):
            # [B,L,D]
            return self._from_batched(out.float(), grids, dev)
        if out.ndim == 2:
            return self._from_flattened(out.float(), grids, dev)
        raise ValueError(f"spatial 要求 [B,L,D] 或 [总token,D]，收到 {tuple(out.shape)}")

    def _from_batched(self, out, grids, dev):
        # out: [B, Lx, D].  Resolve each image independently because a batch
        # may contain different image_grid_thw resolutions.
        outs = []
        for i in range(out.shape[0]):
            vec = out[i]
            row_list, geo_list = per_image_rows_and_geo(
                [grids[i]], self.merge_size, vec.shape[0]
            )
            outs.append(grid_spatial_pool(vec, row_list[0], *geo_list[0], self.k))
        return torch.stack(outs, 0).to(dev)

    def _from_flattened(self, tok, grids, dev):
        rows_per, geo = per_image_rows_and_geo(grids, self.merge_size, tok.shape[0])
        if len(set(rows_per)) == 1 and len(set(geo)) == 1:
            gh, gw = geo[0]
            return grid_spatial_pool_batch(
                tok.reshape(len(rows_per), gh, gw, tok.shape[-1]), self.k
            ).to(dev)
        outs = []
        s = 0
        for rows, (gh, gw) in zip(rows_per, geo):
            chunk = tok[s:s + rows]
            s += rows
            outs.append(grid_spatial_pool(chunk, rows, gh, gw, self.k))
        return torch.stack(outs, 0).to(dev)


def build_spatial_grid_encoder(adapter, k: int = 8) -> SpatialGridEncoder:
    """阶段一旁路封装：给任意现役 adapter，输出 [N,k*k,D]（不动 adapter 权重）。"""
    return SpatialGridEncoder(adapter, k=k)
