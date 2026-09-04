# Visual Dependency Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed offline acceptance tool that proves each ACT checkpoint uses current visual frames instead of only history or priors.

**Architecture:** Add a standalone evaluator that loads one or more M3_ACT checkpoints, evaluates the exact same dataset under four conditions (`normal`, `image_zero`, `image_shuffle`, `history_zero`), and writes one JSON report. Image conditions use fresh feature caches; history_zero only clears the model input history. A pure gate function compares move, camera, and intent accuracy against normal and makes the process exit nonzero unless both image-degraded conditions pass all three drop requirements.

**Tech Stack:** Python, PyTorch, PIL, existing `VLASequenceDataset`, `VLASequenceCollator`, `encode_batch`, and ACT checkpoint loading helpers.

**Spec:** User-approved in-chat visual-dependency acceptance design, 2026-09-04.

## Global Constraints

- Evaluation is offline and read-only with respect to game input; it never sends keyboard or mouse events.
- The four conditions use identical samples, labels, ordering, model weights, and deployment-style slow-to-fast inference.
- The accepted image degradation threshold is `max(0.05, normal_metric * 0.15)` for each of move, camera, and intent metrics.
- Both `image_zero` and `image_shuffle` must pass all three metric drops; `history_zero` is reported diagnostically and is not sufficient for visual acceptance.
- Reports and documentation live under `docs/` or the requested output path; no root-level documentation is added.

---

### Task 1: Pure perturbation and gate contracts

**Files:**
- Create: `idv_agent/scripts/evaluate_visual_dependency.py`
- Test: `tests/test_visual_dependency_gate.py`

**Interfaces:**
- Produces `_dependency_gate(normal, degraded, required_metrics) -> dict` with per-metric deltas, thresholds, failures, and `pass`.
- Produces `_shuffle_frame_paths(frame_paths_by_sample) -> list[list[str | None]]` with a deterministic cyclic permutation.
- Produces `_image_transform(mode)` for normal, zero, and blur image conditions.

- [ ] **Step 1: Write failing tests** for zero/blur transforms, non-self shuffle, and fail-closed gate behavior.
- [ ] **Step 2: Run only the new tests** and verify they fail because the new module does not yet exist.
- [ ] **Step 3: Implement the pure helpers** with finite-value checks and explicit failure reasons.
- [ ] **Step 4: Run the new tests** and verify they pass.

### Task 2: Four-condition checkpoint evaluator

**Files:**
- Modify: `idv_agent/scripts/evaluate_visual_dependency.py`
- Test: `tests/test_visual_dependency_gate.py`

**Interfaces:**
- `evaluate_checkpoint(checkpoint, args) -> dict` evaluates all four conditions with separate feature caches.
- CLI accepts repeated `--checkpoint` values and/or `--checkpoint-dir`, plus the existing model/data/init/device arguments.
- CLI writes a JSON report and exits `0` only when every checkpoint has `visual_dependency_gate_pass=true`; otherwise exits `2`.

- [ ] **Step 1: Add tests** for condition naming, history zeroing, and all-checkpoints aggregation using lightweight fake model hooks.
- [ ] **Step 2: Run those tests** and verify the missing evaluator behavior fails.
- [ ] **Step 3: Implement evaluation** by reusing deployment-style slow prediction followed by fast prediction, then aggregate move/camera/intent accuracy for every condition.
- [ ] **Step 4: Add deterministic cyclic image shuffling** over the complete sample set while preserving each sample's history/actions and targets.
- [ ] **Step 5: Add CLI checkpoint discovery, JSON output, and nonzero gate exit behavior.
- [ ] **Step 6: Run focused tests** and verify all pass.

### Task 3: Documentation and final verification

**Files:**
- Modify: `docs/00-当前状态.md`
- Modify: `docs/18-架构变更历史.md`

- [ ] **Step 1: Document** the four conditions, exact threshold, report path convention, and fail-closed checkpoint rule.
- [ ] **Step 2: Run the full test suite, compileall, and diff checks.
- [ ] **Step 3: Run the tool's `--help` and a pure gate smoke invocation to verify CLI wiring.
- [ ] **Step 4: Commit only the visual-dependency tool, tests, and documentation; preserve unrelated worktree changes.
