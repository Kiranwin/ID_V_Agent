"""对照：k×k 空间 token 画面可分性 vs 现役 mean-pool（阶段一验收对照）。

加载同一 adapter（冻结 Qwen 视觉塔），对同一批差异大截图，分别获得：
  - diagnostic pooled: encode_raw_frames(...).mean(dim=1) → [N, 1024]
  - spatial k×k:  build_spatial_grid_encoder(...)   → [N, k*k, 1024]（保胞位）
比较两者画面两两余弦/L2 与“画面归一信号能量”，给出空间保留是否显著带回
画面可分性的量化结论，供决定阶段二(接投影/时序)是否值得。

用法（idv312 python）:
  python -m idv_agent.scripts.probe_spatial_vs_pooled \
      --checkpoint checkpoints/M3_ACT/act_mvp_b8_mb8_300.pt \
      --frames tmp/feature_activity/00000003.jpg tmp/feature_activity/00002888.jpg tmp/feature_activity/00003715.jpg \
      --device cuda --k 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from idv_agent.model.spatial_grid_feature import build_spatial_grid_encoder
from idv_agent.scripts.train_vla import _load_act_backbone


def _signal(v: torch.Tensor) -> dict:
    # v: [N,*];返回两两 cos(归一) 与 norm-L2(原始幅) 均低→画面相似。
    v = v.double()
    top = v.reshape(v.shape[0], -1)
    mag = top.norm(dim=1)
    cos, l2 = [], []
    n = top.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            cos.append(float(torch.nn.functional.cosine_similarity(top[i], top[j], dim=0)))
            l2.append(float((top[i] - top[j]).norm() / max(float(mag[i]), 1e-9)))
    return {"n": n, "cos_mean": sum(cos) / len(cos),
            "residual_mean": sum(1.0 - c for c in cos) / len(cos),
            "l2_norm_mean": sum(l2) / len(l2),
            "cos": cos, "l2": l2}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--frames", nargs="*", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--k", type=str, default="8", help="逗号分隔 k 值列表，如 1,4,8")
    args = p.parse_args(argv)

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp = dev.type == "cuda"
    dtype = torch.float16 if amp else torch.float32
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    ckpt = Path(args.checkpoint)
    manifest = json.loads((ckpt.parent / "manifest.json").read_text(encoding="utf-8"))
    base_model = manifest["base_model"]
    init_ckpt = manifest["parent"]
    probe_images = [Image.open(path).convert("RGB") for path in args.frames]
    names = [Path(x).name for x in args.frames]

    adapter, _, _ = _load_act_backbone(base_model, init_ckpt, dtype=dtype, device=dev)
    adapter.eval()

    raw_cells = adapter.encode_raw_frames(probe_images, micro_batch_size=len(args.frames))  # [N,k*k,1024]
    pooled = raw_cells.mean(dim=1)  # diagnostic only; production never consumes this path
    s_p = _signal(pooled)

    spatial_by_k = {}
    print(f"\n[probe] baseline pooled {tuple(pooled.shape)} 残差信息 {s_p['residual_mean']:.3e} (cos {s_p['cos_mean']:.6f})", flush=True)
    print("\n   k    spatial.shape         cos均值      1-cos残差(spatial)   残差/池    形态")

    for k in ks:
        enc = build_spatial_grid_encoder(adapter, k=k)
        spatial = enc(probe_images)                      # [N, k*k, 1024]
        # 度量 A：展平保胞序（[N, k*k*D]）——胞位置并入向量，位置差异驱动可分性。
        flat = spatial.reshape(spatial.shape[0], -1)
        s_flat = _signal(flat)
        # 度量 B：跨胞 平均回到 [N, 1024]（同 pooled 维，纯校验胞均值等价均 mean+sanity）
        pooled_dim = spatial.mean(dim=1)
        s_pd = _signal(pooled_dim)
        ratio = s_flat["residual_mean"] / s_p["residual_mean"] if s_p["residual_mean"] > 0 else float("inf")
        spatial_by_k[k] = {
            "shape": list(spatial.shape),
            "signal_flat": s_flat, "signal_cellmean": s_pd,
            "ratio_vs_pooled_flat": ratio,
        }
        print(f"  {k:>2}    {str(tuple(spatial.shape)):<16} "
              f"{s_flat['cos_mean']:>9.6f}   {s_flat['residual_mean']:>11.3e}   {ratio:>7.1f}x   (cellmean cos={s_pd['cos_mean']:.6f})")

    # 跨 k 趋势 + 给阶段二结论（含维度 caveat：展平到 k*k*D 会把“多胞并列”放大成
    # 更高维，天然比 single mean 更能拉开两图；故结论关注“保胞序位置信号”，非单纯维度）。
    print("\n== 空间保留趋势(k=1 应≈mean sanity): ==")
    trend = []
    for k in ks:
        r = spatial_by_k[k]["ratio_vs_pooled_flat"]
        cm = spatial_by_k[k]["signal_cellmean"]["cos_mean"]
        trend.append((k, r, spatial_by_k[k]["signal_flat"]["cos_mean"], cm))
        print(f"  k={k}: 展平残差提升 {r:.1f}x vs pooled（flat cos="
              f"{spatial_by_k[k]['signal_flat']['cos_mean']:.6f}, cellmean cos={cm:.6f}）")
    base_cm = (trend[0][3] + (trend[-1][3] if len(trend) > 1 else trend[0][3])) / 2
    best = max(trend, key=lambda t: t[1])
    verdict = (
        [f"最强 k={best[0]}：画面可分展平残差约 {best[1]:.0f}x 于 pooled，细胞均值 cos={best[3]:.6f}。"
         "按胞保位展平显著拉开画面 → 佐证现役 mean-pool 是画面信息瓶颈，具备进入阶段二(接投影+时序)依据。",
         "caveat：展平维度 k²×D 本身抬高可比性；最终是否增益须以阶段二接投影后 GRU 输入端的活性(原 probe)复测为准。"]
        if best[1] >= 2.0 and best[2] < 0.99999 else
        [f"最强 k={best[0]} 展平残差提升仅 {best[1]:.1f}x（cellmean cos={best[3]:.6f}），"
         "空间保留增益有限 → 先复核几何/采样/聚合再进阶段二。"]
    )
    for line in verdict:
        print(" -", line)

    result = {
        "checkpoint": str(ckpt.resolve()),
        "frames": args.frames, "ks": ks, "device": str(dev),
        "pooled_baseline": {"shape": list(pooled.shape), **s_p},
        "spatial_by_k": spatial_by_k,
        "best": {"k": best[0], "lift_x": best[1], "cos": best[2]},
        "verdict": verdict,
    }
    out = Path("reports/probe_spatial_vs_pooled.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[probe] → {out}")


if __name__ == "__main__":
    main()
