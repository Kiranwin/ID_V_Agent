# VLA Raw Feature Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache deterministic frozen Qwen vision-tower pooled features incrementally, then train the projection and VLA heads from cache without stale-feature or gradient errors.

**Architecture:** The cache stores only the frozen tower output after the canonical `mean(dim=1)` pooling and before `visual_projection`/task conditioning. A manifest-backed directory contains one tensor shard per session plus an index keyed by normalized image path, so new sessions append only missing frames. Training looks up raw features, applies current trainable projections and instruction/mode conditioning, and preserves the existing path for adapters without raw-cache support.

**Tech Stack:** Python 3.12, PyTorch 2.6.0+cu124, PIL, JSON/torch serialization, pytest.

**Spec:** User-approved chat design: frozen pooled output, exact realtime/cache equality, incremental session updates.

## Global Constraints

- Canonical pooling is `vision_output.mean(dim=1)` for `[N,L,D]` outputs; `[N,D]` outputs are already pooled.
- Precompute uses `model.eval()` and `torch.inference_mode()`; cache lookup must be bitwise equal to direct frozen-tower output for the same prepared image.
- Cache keys are normalized absolute image paths; instruction/mode are never part of raw feature identity.
- Cache writes are append-only by session/shard and must not rewrite existing feature tensors.
- Trainable `visual_projection` and `condition_projection` always run after lookup.

---

### Task 1: Raw frozen feature API and exact projection split

**Files:**
- Modify: `idv_agent/model/qwen_backbone_adapter.py`
- Test: `tests/test_frame_cache_behavior.py`

**Interfaces:**
- Add `encode_raw_frames(images, *, micro_batch_size=None) -> Tensor[N,D_raw]`.
- Add `project_raw_features(raw_features, task_cache) -> Tensor[N,hidden_size]`.
- Keep `encode_frame`/`encode_frames` behavior by composing raw encoding and projection.

- [ ] **Step 1: Write failing tests** for raw output shape, exact equality between `encode_frame` and `project_raw_features(encode_raw_frames(...))`, and deterministic eval mode.
- [ ] **Step 2: Run the focused tests** and verify they fail because the new methods do not exist.
- [ ] **Step 3: Implement the split** using the same `_visual_forward_batch` pooling rule and `torch.inference_mode()` only in callers that request frozen precompute; do not add rounding or dtype conversion beyond existing model output handling.
- [ ] **Step 4: Run focused tests** and verify exact `torch.equal` output equality.
- [ ] **Step 5: Commit** with `feat: expose frozen raw vision features`.

### Task 2: Incremental manifest-backed precompute utility

**Files:**
- Create: `idv_agent/scripts/precompute_vla_features.py`
- Test: `tests/test_frame_feature_cache.py`

**Interfaces:**
- `collect_unique_frame_paths(data_paths) -> list[Path]`.
- `update_feature_cache(data_paths, model_path, output_dir, device, micro_batch_size) -> dict`.
- CLI options: `--data`, `--model-path`, `--output`, `--device`, `--micro-batch-size`.

- [ ] **Step 1: Write failing tests** for deduplication, manifest metadata, and second-run incremental behavior (zero new encodes).
- [ ] **Step 2: Run tests** and verify the utility is missing.
- [ ] **Step 3: Implement** a manifest containing model/processor identity, pooling version, raw dimension, dtype, per-path shard location, and counts; write each new session to its own `.pt` shard and update the index atomically.
- [ ] **Step 4: Run tests** including a changed/new session and verify only new paths are encoded.
- [ ] **Step 5: Commit** with `feat: add incremental VLA raw feature precompute`.

### Task 3: Training lookup integration and zero-difference regression

**Files:**
- Modify: `idv_agent/scripts/train_vla.py`
- Modify: `idv_agent/scripts/evaluate_vla.py`
- Test: `tests/test_frame_cache_behavior.py`, `tests/test_frame_feature_cache.py`

**Interfaces:**
- `encode_batch(..., raw_feature_cache=None)` accepts a manifest-backed lookup object or dictionary of raw features.
- Training CLI adds `--raw-feature-cache`; evaluation accepts the same cache format.

- [ ] **Step 1: Write failing tests** showing `forward_loss` uses lookup, current projections still receive gradients, and direct-vs-cache feature/loss tensors are exactly equal under eval/no-grad.
- [ ] **Step 2: Run tests** and verify cache lookup is ignored or unsupported.
- [ ] **Step 3: Implement** lookup before image loading; project looked-up raw tensors with the current task cache, retain existing persistent final-feature cache only for eval compatibility, and report `cache_hits`, `cache_misses`, `unique_frames_encoded`, and `new_frames_encoded`.
- [ ] **Step 4: Run focused tests, smoke test, compileall, and a small CPU fake-adapter integration check.**
- [ ] **Step 5: Commit** with `feat: train VLA from incremental raw feature cache`.

### Task 4: Documentation and verification

**Files:**
- Modify: `docs/04-环境与硬件.md`
- Modify: `docs/18-帧特征缓存修复报告.md`

- [ ] **Step 1:** Document canonical pooling, exact-equality guarantee, cache invalidation fields, incremental workflow, and storage/statistics output.
- [ ] **Step 2:** Run the relevant pytest files, `python -m idv_agent.scripts.smoke_test`, `python -m compileall -q idv_agent`, and `git diff --check`.
- [ ] **Step 3:** Commit with `docs: document incremental raw feature cache`.
