"""分离式 M3_ACT_spatial 的画面可分活性（对照旧 mean 基线的 L3/L4）。

旧 M3_ACT(mean)实测：FiLM conditioned(GRU 输入段) signal≈3.8e-3 → fast GRU末态
≈5.0e-6(“踩塌”)。此处验证新分离+spatial ACT 是否保住画面可分：同一 3 张图，
在工作链 FiLM 输入段与 fast_temporal 末态(res. 静态 4 帧窗口)取 signal_margin。
用法同 probe_feature_activity：加载 checkpoint 后 前向，输出 json。
复用 spatial 通路、不进训练。scheme: sep schema = slow/fast temporal 分离故无需
LegacyCore，但 core loss-free forward 直接 SharedFastSlowVLA 即可。
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch

from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.scripts.train_vla import _load_act_base_backbone
from PIL import Image
from PIL import ImageDraw


def _softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    flat = logits.reshape(logits.shape[0], -1).float()
    return torch.softmax(flat, dim=-1)


def _layer_metrics(features_a: torch.Tensor, features_b: torch.Tensor, names):
    a = features_a.float(); b = features_b.float()
    n = len(names)
    same_cos, same_l2, mag, inter_cos, inter_ns = [], [], [], [], []
    for i in range(n):
        same_l2.append(float((a[i] - b[i]).norm()))
        same_cos.append(float(torch.nn.functional.cosine_similarity(a[i], b[i], dim=0)))
        mag.append(float(a[i].norm()))
    for i in range(n):
        for j in range(i + 1, n):
            inter_cos.append(float(torch.nn.functional.cosine_similarity(a[i], a[j], dim=0)))
            inter_ns.append(float((a[i] - a[j]).norm() / max(1e-9, mag[i])))
    mean_same = sum(same_l2) / max(1, n)
    mean_cos = sum(inter_cos) / max(1, len(inter_cos))
    mean_ns = sum(inter_ns) / max(1, len(inter_ns))
    signal = (sum(1 - c for c in inter_cos) /
              max(1, len(inter_cos)) + mean_ns)
    return {"shape": list(a.shape), "inter": {"cos_mean": mean_cos, "l2_over_mag_mean": mean_ns},
            "same": {"l2_mean": mean_same, "cos_mean": sum(same_cos) / max(1, n)},
            "signal_margin": float(signal)}


def _pair_metrics(reference: dict[str, torch.Tensor], altered: dict[str, torch.Tensor]) -> dict:
    out = {}
    for key in reference:
        a, b = reference[key].float(), altered[key].float()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        out[key] = {"cos_mean": float(cos.mean()), "l2_mean": float((a - b).norm(dim=-1).mean())}
    return out


def _occlude(image: Image.Image, side: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    if side == "left":
        draw.rectangle((0, 0, width // 2, height), fill=(0, 0, 0))
    else:
        draw.rectangle((width // 2, 0, width, height), fill=(0, 0, 0))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/M3_ACT_spatial/act_spatial_b1_030.pt")
    p.add_argument("--model-path", default="", help="基础 Qwen 模型；默认从 ACT manifest 读取")
    p.add_argument("--init-checkpoint", default="", help="已废弃：不会加载 M2")
    p.add_argument("--device", default="cuda")
    p.add_argument("--frames", nargs="*", default=["tmp/feature_activity/00000003.jpg",
                                                       "tmp/feature_activity/00002888.jpg",
                                                       "tmp/feature_activity/00003715.jpg"])
    args = p.parse_args(argv)
    ckpt = Path(args.checkpoint)
    act_manifest = json.loads((ckpt.parent / "manifest.json").read_text(encoding="utf-8"))
    base = args.model_path or act_manifest["base_model"]
    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    amp = dev.type == "cuda"
    dtype = torch.float16 if amp else torch.float32
    adapter, _ = _load_act_base_backbone(base, dtype=torch.float16 if amp else torch.float32, device=dev)
    adapter.eval()
    # Diagnostic spatial_agg is loaded explicitly from the ACT checkpoint.
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    adapter.visual_projection.load_state_dict(saved["adapter"]["visual_projection"])
    adapter.condition_projection.load_state_dict(saved["adapter"]["condition_projection"])
    if "spatial_agg" in saved["adapter"]:
        agg_state = saved["adapter"]["spatial_agg"]
        adapter._ensure_spatial_agg(int(agg_state["position"].shape[-1])).load_state_dict(agg_state)
    manifest = json.load(open(Path(ckpt).parent / "manifest.json", encoding="utf-8"))
    td = int(manifest["training"]["temporal_dim"])
    core = SharedFastSlowVLA(adapter.act_feature_dim, temporal_dim=td,
                             history_action_dim=72).to(dev)
    core.load_state_dict(saved["core"])
    core.eval()

    paths = args.frames
    images = [Image.open(p).convert("RGB") for p in paths]
    names = [Path(p).name for p in paths]

    # 任务 condition
    with torch.no_grad():
        task = adapter.encode_task_once("解码密码机，转点牵制监管者，必要时救援并开启大门。", "standard", task_id="probe")
    ins = task.instruction_embedding.detach(); mode = task.mode_embedding.detach()

    def collect(frame_images):
        with torch.no_grad():
            enc = adapter.encode_frames(frame_images, task, micro_batch_size=len(frame_images)).float()  # [N,2560]
            # 拆分 condition。encode_frames 输出 = project(+cond from project_raw_features内 condition加)
            # encode_frames 输出 = agg→proj→(+const 任务 cond)；把它当 per-frame spatial
            # 特征做静态 4 帧窗口进时序，与旧 mean ACT 的 L3/L4 对照。
            cond = core.initial_condition(enc.shape[0], device=dev, mode_id=0)
            valid = torch.ones(enc.shape[0],4,dtype=torch.bool,device=dev)
            z = torch.zeros(enc.shape[0],4,dtype=torch.float32,device=dev)
            feats = enc.unsqueeze(1).expand(-1,4,-1).clone()
            film = core.conditioner(feats, cond.intent_id, cond.subgoal_id,
                                    cond.context_embedding.to(enc), cond.mode_id)
            fast = core.fast_temporal(film, valid_mask=valid, time_deltas=z)
            slow = core.slow_temporal(film, valid_mask=valid, time_deltas=z)
            slow_out = core.slow_head(slow)
            h = torch.zeros(fast.shape[0],72,dtype=fast.dtype,device=dev)
            fast_out=core.fast_head(fast,history_actions=h)
            return {
              "proj_after_cond": enc.cpu(),
              "film_gru_input_mean": film.mean(dim=1).cpu(),
              "fast_gru": fast.cpu(),
              "slow_ctx": slow_out.context_embedding.cpu(),
              "fast_probs": _softmax_logits(fast_out.move_logits).cpu(),
            }
    normal = collect(images)
    repeat = collect(images)
    left = collect([_occlude(image, "left") for image in images])
    right = collect([_occlude(image, "right") for image in images])
    order=[("proj_after_cond","space_agg→proj(+任务cond) 每帧"),
           ("film_gru_input_mean","FiLM conditioned(GRU 输入段)"),
           ("fast_gru","fast temporal GRU 末态(4帧静态)"),
           ("slow_ctx","slow context_embedding"),
           ("fast_probs","fast move  分布")]
    layers={}
    print(f"{'层':<34} signal_margin   1-cos  cos")
    for k,label in order:
        m=_layer_metrics(normal[k],repeat[k],names)
        layers[k]={"label":label,**m}
        print(f"{label:<34} {m['signal_margin']:.3e}  {1-m['inter']['cos_mean']:.3e}  {m['inter']['cos_mean']:.6f}")
    perturbations = {"left_occlusion": _pair_metrics(normal, left),
                     "right_occlusion": _pair_metrics(normal, right)}
    report = {"ckpt": str(ckpt), "layers": layers, "perturbations": perturbations,
              "determinism": {key: layers[key]["same"] for key, _ in order}}
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/probe_act_spatial_film.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n空间扰动 cosine（越低表示决策特征越敏感）：")
    for side, values in perturbations.items():
        print(f"  {side}: fast_gru={values['fast_gru']['cos_mean']:.6f}, "
              f"fast_probs={values['fast_probs']['cos_mean']:.6f}")


if __name__ == "__main__":
    main()
