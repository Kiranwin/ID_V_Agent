# ACT Visual Expert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every deployed ACT action head include a separately supervised, history-free visual prediction path.

**Architecture:** `VisualActionExpert` receives last-frame and first-to-last visual change vectors, predicts fast/slow outputs independently, and is added to bounded temporal/history prior logits. The loss supervises both visual outputs and fused outputs with the same masks and per-head weights.

**Tech Stack:** PyTorch 2.6, CUDA FP16 + GradScaler on RTX 2080 Ti, frozen Qwen3-VL, pytest.

**Spec:** `docs/superpowers/specs/2026-09-04-act-visual-grounding-design.md`

## Global Constraints

- Do not load M2; all new ACT checkpoints use `m7_act.visual_expert.v1` and reject previous schemas.
- Keep v5: 8 frames, stride 3, 8 action history steps, 6-frame action duration, 30 FPS, `time_deltas=None`.
- Preserve cache/augmentation separation; augmented image features are never retained in raw or inference caches.
- A checkpoint is deployable only after the specified feature, visual-dependency, and classification gates pass.

---

### Task 1: Visual expert and bounded output fusion

**Files:**
- Modify: `idv_agent/model/vla_heads.py`
- Modify: `idv_agent/model/fast_slow_vla.py`
- Test: `tests/test_visual_action_expert.py`

**Interfaces:** `VisualActionExpert(frame_feature_dim, temporal_dim, horizon)` returns `VisualExpertOutput(fast: FastVLAOutput, slow: SlowVLAOutput, feature: Tensor)`. `fuse_fast_outputs(visual: FastVLAOutput, prior: FastVLAOutput, prior_scale: float = 0.5) -> FastVLAOutput` and `fuse_slow_outputs(visual: SlowVLAOutput, prior: SlowVLAOutput, prior_scale: float = 0.5) -> SlowVLAOutput` retain existing output shapes.

- [ ] **Step 1: Write focused failing tests**

```python
def test_visual_expert_is_history_independent_and_fusion_is_bounded():
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=72)
    frames = torch.randn(2, 3, 8)
    condition = core.initial_condition(2)
    visual_a = core(frames, condition, history_actions=torch.zeros(2, 72), run_slow=True)
    visual_b = core(frames, condition, history_actions=torch.ones(2, 72), run_slow=True)
    assert torch.equal(visual_a.visual.fast.move_logits, visual_b.visual.fast.move_logits)
    assert torch.allclose(visual_a.fast.move_logits,
                          visual_a.visual.fast.move_logits + 0.5 * visual_a.prior_fast.move_logits)
```

- [ ] **Step 2: Run the focused test and confirm it fails because `visual` and `prior_fast` do not exist.**

Run: `python -m pytest tests/test_visual_action_expert.py -q`

- [ ] **Step 3: Implement `VisualActionExpert` and pure fusion functions.**

Use last valid and first valid indices from `valid_mask`; use `[last, last-first]` as the expert input; no `history_actions` argument is accepted. Reuse action dimensions from `FastVLAHead` / `SlowVLAHead`. Fuse categorical logits and button logits as `visual + 0.5 * prior`; fuse durations with the spec formula.

- [ ] **Step 4: Wire the expert into `SharedFastSlowVLA`.**

Keep `.fast` and `.slow` as fused deployment output, add `.visual`, `.prior_fast`, and `.prior_slow` only for loss/diagnostics.

- [ ] **Step 5: Run focused model tests.**

Run: `python -m pytest tests/test_visual_action_expert.py tests/test_separated_temporal.py tests/test_act_policy.py -q`

- [ ] **Step 6: Commit.**

```powershell
git add idv_agent/model/vla_heads.py idv_agent/model/fast_slow_vla.py tests/test_visual_action_expert.py
git commit -m "feat: add bounded visual action expert"
```

### Task 2: Explicit visual supervision and m7 checkpoint contract

**Files:**
- Modify: `idv_agent/training/vla_loss.py`
- Modify: `idv_agent/scripts/train_vla.py`
- Modify: `idv_agent/model/act_checkpoint.py`
- Test: `tests/test_visual_action_expert.py`
- Test: `tests/test_act_visual_checkpoint.py`

**Interfaces:** `compute_vla_loss(fast, slow, batch, weights, visual_fast=None, visual_slow=None)` emits `visual_fast_move`, `visual_fast_camera`, `visual_fast_buttons`, `visual_fast_duration`, `visual_slow_intent`, and `visual_slow_subgoal`; `VLALossWeights.visual_aux` defaults to 1.0.

- [ ] **Step 1: Write failing loss and schema tests.**

```python
assert losses["total"] == fused_total + weights.visual_aux * visual_total
with pytest.raises(ValueError, match="m7_act"):
    load_visual_grounded_act_checkpoint(adapter, core, m6_checkpoint)
```

- [ ] **Step 2: Run tests and confirm failure.**

Run: `python -m pytest tests/test_visual_action_expert.py tests/test_act_visual_checkpoint.py -q`

- [ ] **Step 3: Factor masked per-output loss computation and supervise fused plus visual output.**

Apply exactly the existing fast/slow masks and class weights to visual heads. Keep the consistency loss fused-only. Send `output.visual.fast` and `output.visual.slow` from both slow/fast train passes into loss.

- [ ] **Step 4: Bump schema to m7 and preserve fail-closed restore behavior.**

`core.load_state_dict` must load the visual expert state strictly; adapter state contract remains raster-only.

- [ ] **Step 5: Run focused tests and commit.**

Run: `python -m pytest tests/test_visual_action_expert.py tests/test_act_visual_checkpoint.py tests/test_train_vla_inheritance.py -q`

```powershell
git add idv_agent/training/vla_loss.py idv_agent/scripts/train_vla.py idv_agent/model/act_checkpoint.py tests/test_visual_action_expert.py tests/test_act_visual_checkpoint.py
git commit -m "feat: supervise ACT visual expert"
```

### Task 3: End-to-end diagnostics and measured validation

**Files:**
- Modify: `docs/00-当前状态.md`
- Modify: `docs/18-架构变更历史.md`

**Interfaces:** existing `evaluate_feature_activity.py`, `evaluate_visual_dependency.py`, and `evaluate_vla.py` continue consuming `.fast` / `.slow` fused outputs, so their hard gates remain unchanged.

- [ ] **Step 1: Run full unit and integration suite before training.**

Run: `python -m pytest -q --basetemp .pytest-tmp-m7-pretrain`

- [ ] **Step 2: Train 60 steps on all 480 decorrelated train chunks and all 118 held-out chunks.**

Use `--class-balance --image-augmentation --sampling stratified --history-dropout-p 0.5 --require-training-behavior`; do not provide a raw cache while augmentation is enabled.

- [ ] **Step 3: Run held-out classifier, feature-activity and four-condition visual-dependency evaluators.**

Use their fixed stratified 32-sample cohort for the visual gates and record nonzero return from a failed gate as an expected test result, not as a tool failure.

- [ ] **Step 4: Run full regression suite and diff check.**

Run: `python -m pytest -q --basetemp .pytest-tmp-m7-final` and `git diff --check`.

- [ ] **Step 5: Record measured outcomes without promoting failed checkpoints. Commit docs.**

```powershell
git add docs/00-当前状态.md docs/18-架构变更历史.md
git commit -m "docs: record m7 visual expert validation"
```
