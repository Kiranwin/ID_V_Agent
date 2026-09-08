# MVP Goal Status

> This is the repository's durable recovery record for the active MVP goal.
> It is the one explicit root-level Markdown exception to the documentation
> policy in `AGENTS.md`. Read it before starting VLA work and update it after
> every material state change, experiment, blocker, or deployment decision.

## Goal

In the official custom scenario/training mode only, complete the learned MVP
behavior without external rule control: **at match start, turn to find a cipher
machine, approach it, and press Q to enter decoding when the interaction UI is
visible**.

Deployment remains fail-closed until the offline visual-dependency gate and
all relevant data/protocol checks pass. Do not use `--send-input` or classify a
checkpoint as deployable merely because its training loss decreases.

## Startup and Update Protocol

1. Read this file before inspecting or modifying VLA data, training, model, or
   deployment code.
2. Verify the paths, git worktree/branch, checkpoint files, and reports named
   below before treating any status as current.
3. Use `data/new_vla_raw_sessions` as the raw MVP source. Do not modify it;
   derived data and annotation-pool products are separate, recoverable files.
4. For every training command, run the command as a blocking foreground task.
   Do not start a second process or repeatedly poll it. Read its artifacts only
   after the command exits.
5. After a material result, update this file with evidence paths, pass/fail
   gates, the next concrete action, and any newly created backups.

## Root-Cause Review Addendum (2026-09-08)

Read-only diagnosis requested by the user was completed in the primary
`vla-train` checkout at `ecae81f` (including m28 commit `7bd3106`). No model,
dataset, training run, or deployment permission changed. Detailed evidence
and the proposed repair order are recorded in `docs/18-架构变更历史.md`, section
“MVP 根因复核：监督冲突、评估口径与执行时间”.

New confirmed issues to resolve before interpreting another full run:

- m28 replay-camera CE and desired-turn CE both supervise the same executed
  h0 logits without arbitration. Canonical train labels disagree on dx in
  15/48 rows, dy in 12/48, either axis in 23/48 (val: 13/46, 17/46, 25/46).
  A CPU gradient probe confirms opposing updates on conflicting classes.
  Preserve raw replay evidence while defining one explicit execution target.
- The standard `evaluate_vla.py` still scores all four action steps and uses
  the default four-step loss. This differs from h0 training/deployment; it
  does not invalidate the separate h0 training/visual-dependency reports.
- A CPU scheduler/executor probe with 250-ms tick intervals emits W press
  and release in the same tick: elapsed time before a queued step starts is
  incorrectly charged to that new 200-ms step. `run_capture` also passes a
  pre-encode timestamp to tick, excluding encode latency from freshness.
  Actual GPU/capture latency remains unmeasured; no game inputs were sent.

Effective full-train supervision is 128 grounding endpoints and 48 control
endpoints (31 sessions), including only 6 search and 2 left-detour control
labels. The latest m28 128-sample smoke actually consumes 9 control rows;
its validation move predictions are forward on all 64 samples. This smoke
is not a full-task architecture result. Historical m26 h0 train/val camera
zero false-turn rates are 0.6146/0.6218, and held-out Q event recall is 7/20;
visual dependence alone did not establish reliable actions.

Keep m28 architecture fixed while resolving these independently testable
contracts, then perform the existing per-loss/per-module gradient trace and
an explicit small-set h0 fit check. Preserve all current offline gates and
deployment restrictions. A no-input dry-run cannot by itself demonstrate
agent-caused decoding entry; eventual sandbox success requires actual
action-to-next-observation evidence after prerequisites and authorization.

## Closed-Loop Evidence Contract (2026-09-08)

Implemented `idv_agent.agent.mvp_closed_loop` and
`idv_agent.scripts.evaluate_mvp_closed_loop`. A trace row now requires
`before`, the actually submitted h0 `action`, and the first post-action
`after` observation. The verifier reports `find_cipher`, `approach_cipher`,
`q_prompt`, and `decode_entry` separately. Only rows marked
`source=live_sandbox` and `causal=true` can pass `mvp_pass`; human replay is
diagnostic evidence only. This adds an acceptance contract, not a live game
result; no sandbox trace has been recorded yet.

ACT runtime integration is now wired: `ACTPolicy` accepts
`closed_loop_trace`, `closed_loop_episode_id`, and an external
`closed_loop_observer`; the executor callback records the submitted h0 and the
trace writer flushes it only after at least 6 newly observed frames. `run_agent`
exposes `--closed-loop-trace` and `--closed-loop-episode-id`, and the usage and
evaluation commands are documented in `docs/19-ACT实时推理接入.md`. The current
CLI observer supplies detector fields when a template/YOLO model is given;
`decode_entry` still requires an external frame-state observer to provide
`frame_state=decoding`. No live sandbox run has been performed.

Runtime diagnostic (2026-09-08): a user run ended with `帧数=4`. The CLI had
defaulted ACT to `--device cpu`; this machine reports
`torch.cuda.is_available()=False`. Qwen3-VL-4B CPU inference therefore consumed
the short run before the required 22-frame warmup, so that invocation produced
no valid ACT decision or closed-loop evidence. The CLI now defaults to CUDA,
fails early when ACT is requested without CUDA, and prints capture statistics
(`processed_frames`, `predictions`, `empty_grabs`, `reused_frames`, `device`).

Model repair (2026-09-08): m28 training now exposes and records explicit
`slow_intent_weight`, `slow_subgoal_weight`, and `consistency_weight` values.
Defaults are `0.25`, `0.125`, and `0.1`; action/camera/Q weights are unchanged.
This targets the observed m28 smoke peak dominated by `slow_intent`/`slow_subgoal`
without using gradient clipping as a substitute for a model fix. No new smoke
has run yet because this session has no CUDA device.

## Current Working Context

| Item | Current value |
|---|---|
| Worktree | `C:\Codespace\ID_V_Agent\.worktrees\training-augmentation-balance` |
| Branch | `codex/training-augmentation-balance` |
| Python | `C:\Users\kiran\miniconda3\envs\idv312\python.exe` |
| Hardware | Local RTX 2080 Ti; FP16 + GradScaler; no BF16 |
| ACT initialization | Base Qwen only; M2/VG is disabled for ACT |
| Raw MVP source | `C:\Codespace\ID_V_Agent\data\new_vla_raw_sessions` |
| Canonical ACT dataset | `C:\Codespace\ID_V_Agent\data\mvp_vla_v5_event_centered_decision_label_v3_rebucket675` |
| ACT split | train=732 chunks / 79 sessions; val=182 chunks / 20 sessions; no session overlap |

## Data and Annotation State

- Canonical event-centered data has 99 Q events: train=79, val=20, across 98
  sessions. All Q events are labeled `decipher`.
- Same-session visual supervision is complete at
  `C:\Codespace\ID_V_Agent\data\mvp_act_grounding_pool_v1`.
- The annotation pool has 1,380 images: train=1,076 and val=304. Current
  reannotation result: `cipher_visible=1,304`, `cipher_highlight=72`,
  `interact_prompt=1,186`; 17 explicitly no-cipher images remain.
- Annotation contract: `interact_prompt` requires a visible/highlight cipher
  box; `target_side` comes from the prompt box center; prompt implies
  `cipher_reachable=1`. The pool validator passes.
- Current ACT decision-frame lookup coverage is train=128/732 and val=64/182.
  Unmatched chunks have `grounding_mask=0`; they are never converted to false
  negatives.
- Data backups are retained inside the annotation-pool directory, including
  `labels.before_reannotation_20260906_221910` and timestamped prior auxiliary
  JSONL files. Do not delete them without an explicit request.
- Camera-control semantic labels are complete: train=48/48 and val=46/46.
  The canonical rebucketed replay-evidence copies are
  `camera_control_{train,val}.v2.rebucket675.annotated.jsonl`; they preserve
  all human `desired_turn_*`/path fields and update only replay evidence.
- On 2026-09-08 all 99 raw-session `vla_chunks_v5.jsonl` files were rebuilt
  with shared edges `[12.0, 67.5]`: `±13..±67 → ±1`, `|px|≥68 → ±2`.
  Every v5 audit passed (983 chunks).  The old v2 MVP directory is retained;
  new canonical v3 has the same session split and 732/182 chunks.  H0 class
  weights and provenance are frozen in
  `class_balance_h0_rebucket675.json` under the v3 dataset root.

## Current Architecture

- Camera labels are v5 Raw Input replay labels using the shared pixel bucket
  convention and executor lookup table. Training labels and deployed pixel
  commands remain same-source.
- The deployed ACT path is m28 `StateDecisionExpert`: ordered 8-frame spatial
  visual features enter one shared State Trunk `z`; intent/subgoal, grounding,
  phase/target/steering/path are predicted from `z`; one Joint Planner consumes
  `z` plus its own soft predicted beliefs and emits move/camera/Q/duration.
  Runtime never reads annotation JSONL or teacher-forced labels.
- m28 camera prior is fixed to zero.  `desired_turn_dx/dy` aliases the planner
  h0 camera logits, so human control labels supervise the action actually
  eligible for execution.  Replay camera remains immutable behavior evidence.
- Checkpoint schema for the next ACT run is
  `m28_act.state_conditioned_joint_planner.v1`; every m27 and earlier
  checkpoint fails closed and is historical evidence only.

## Latest Evidence: m23 Grounding-Balanced Training

| Item | Evidence |
|---|---|
| Checkpoint | `C:\Codespace\ID_V_Agent\checkpoints\m23_grounding50_2epoch_local_idv312\act.pt` |
| Training setup | 366 steps, batch=4, lr=1e-4, image augmentation, history dropout=0.5, class balance, grounding loss=1.0 |
| Grounding sampling | 50% annotated target fraction; 128 annotated / 604 unannotated source chunks |
| Optimization | loss `11.2358 -> 2.0292`; minimum `1.1806`; max gradient norm `95.8847`; checkpoint loss delta `0.0` |
| Held-out grounding subset | 14 annotated chunks: presence accuracy `0.4286`, side accuracy `0.2143`, bbox L1 `0.1405`, prompt recall `1.0` but precision `0.4` |
| Visual gate report | `C:\Codespace\ID_V_Agent\reports\m23_visual_dependency_full.json` |

### m23 Deployment Decision: **REJECT / FAIL-CLOSED**

The full 182-sample, four-condition gate failed:

- `image_zero`: move drop `0.1358` passes; intent drop `0.3381` passes;
  camera drop `0.0251 < 0.05` fails.
- `image_shuffle`: intent drop `0.3478` passes; move drop `0.0263 < 0.05`
  fails; camera drop `0.0439 < 0.05` fails.

`m23` has learned useful visual intent behavior, but camera action is still
not sufficiently grounded and must not be run in dry-run or send-input mode.

### m23 Grounding-to-Camera Alignment Audit

Reports:

- `C:\Codespace\ID_V_Agent\reports\m23_grounding_camera_alignment_train.json`
- `C:\Codespace\ID_V_Agent\reports\m23_grounding_camera_alignment_val.json`

The same-frame target label is informative for future camera replay but is not
currently connected to the ACT camera-logit path. On aligned, non-prompt
windows, side and dx have the expected sign relationship: e.g. in validation,
`side=left` has negative dx `18/28` nonzero-or-zero comparable decisions while
`side=right` has positive dx `12/16`; center is mostly dx=0. In contrast,
every held-out `prompt=1/reachable=1` macro step is dx=0 and dy=0 (72/72),
which correctly expresses “already aligned; stop turning”. Therefore the next
architecture change should condition the learned camera head on the model's
own visual side/prompt representation, not add a deployment rule or redefine
the replay labels.

## Next Action

1. Reduce camera zero-bucket false turns while preserving the passed visual
   dependency and spatial-feature gates. Do not deploy while camera-dx/dy
   zero-target false-turn exceeds `0.15`.
2. **Human annotation required before another camera retrain.** The prompt=0
   camera replay target is high entropy even conditional on current bbox/side:
   define and annotate the intended camera-control target/phase for those
   frames. Do not conceal this with a deployment threshold or more epochs.
3. After all offline gates and class/over-action checks pass, perform the
   sandbox dry-run sequence and verify find-machine, approach, Q interaction,
   and decoding entry. Keep `--send-input` disabled until that evidence exists.

The previous full m26 checkpoint is not reusable: its class-balance tables were
computed over all four action horizons while the causal loss trained only step
0. The corrected implementation has passed a fresh smoke; the next full run
must be trained from the base Qwen with the corrected step-0 tables.

## m27 Camera-Control Semantic Smoke (2026-09-08)

The completed human labels are now consumed only during ACT training by
visual-only control heads (`phase`, `target_id`, `steering`, `path`, and
`desired_turn_dx/dy`).  Their *predicted* logits are fused into camera logits;
runtime never reads label JSONL or applies a path rule.  The checkpoint schema
is therefore `m27_act.visual_camera_control.v1`; older m26 checkpoints fail
closed against the expanded head state.

The first 60-step m27 smoke exposed an auxiliary-scale defect: summing six
control cross-entropies yielded a peak control loss `9.0361` and gradient norm
`120.30` (step 5), so the training-behavior gate correctly rejected it.  The
repair averages the six head losses, retaining independent class weights and
the single `camera_control_loss_weight` lambda.

The repaired local run completed at
`C:\Codespace\ID_V_Agent\checkpoints\m27_control_rebucket675_normalized_smoke60_local_idv312`:

- base Qwen only; M2/VG disabled; m27 schema; v3 rebucket675 data;
  train=128 / val=64 sampled chunks; 48/46 completed train/val semantic labels
  available in their full datasets;
- loss `8.1845 -> 3.5596` (minimum `2.1173`); max gradient `70.1498`; all
  gradients/components finite; checkpoint reload delta `0`; behavior gate **passes**;
- this is only a numerical/integration smoke, not a deployment candidate:
  validation h0 camera zero false-turn remains dx=`0.6750`, dy=`0.6538`, both
  far above `0.15`; visual dependency and full held-out control metrics must
  be rerun after full training.

The training evaluator now records held-out metrics for every semantic control
head and masks all unannotated rows.  Next: train all 732 chunks / validate all
182 chunks with the new schema and then run control, visual-dependency, class,
and sandbox gates before any dry-run deployment.

## m27 Control-Sampling Audit and Full-Run Decision (2026-09-08)

The first completed full m27 run is retained at
`C:\Codespace\ID_V_Agent\checkpoints\m27_control_rebucket675_full366_local_idv312\act.pt`.
It trained 366 steps, but its four-condition visual-dependency report at
`C:\Codespace\ID_V_Agent\reports\m27_control_rebucket675_full366_visual_dependency.json`
fails camera: image-zero drop `0.0310 < 0.05` and cross-session image-shuffle
does not lower camera accuracy.  It remains **rejected / fail-closed**.

Its held-out semantic-control report is
`C:\Codespace\ID_V_Agent\reports\m27_control_rebucket675_full366_control_eval.json`.
On its 46 labeled validation frames, `desired_turn_dx` accuracy is `0.2609`
and steering accuracy is `0.3261`; this is insufficient evidence of useful
route-aware visual control.

The original full command made a real sampling mistake: control rows were a
subset of the 128 grounding rows, so ordinary 50% grounding oversampling gave
the 48 control rows only about `18.75%` of draws.  The trainer now implements
joint sampling with explicit mass for control / other-grounding / ordinary ACT
rows.  With `control=0.35`, `grounding=0.50`, the full train population is
`48 / 80 / 604`, with target draw masses `0.35 / 0.15 / 0.50`; class weights
remain independently derived from the complete 48 training control labels.

The numerical smoke using the new sampler completed at
`C:\Codespace\ID_V_Agent\checkpoints\m27_control_oversample35_smoke60_local_idv312`:
loss `7.4780 -> 4.5355` (min `2.0378`), max gradient `77.1086`, finite
components, checkpoint delta `0`, and training-behavior gate passed.  It is
not a semantic result: `--max-samples=128` left only **9** control rows in the
bounded source subset, so repeated 35% sampling only replayed those nine.
The 46-row held-out result (`...\reports\m27_control_oversample35_smoke60_control_eval.json`)
is `desired_turn_dx=0.2174`, below the previous full result, but cannot test
the intended full-48-label training regime.  Do not draw an architectural
conclusion from that truncated smoke and do not deploy it.

To make data isolation auditable rather than relying on exact frame keys, the
trainer now has explicit `--val-grounding-annotations` and
`--val-camera-control-annotations`.  A train run with `--val-data` and camera
control supervision fails unless the latter is supplied.  The next full run
will use train-only control labels (48) and val-only control labels (46), with
the 732/182 session split unchanged.

### Historical m27 Result — Superseded

The m27 checkpoint/results above remain useful evidence about sparse control
coverage and replay ambiguity, but must not be retrained, evaluated as a
candidate, or deployed.  m28 supersedes its parallel-head architecture.

## Data-to-Training Contract Consolidation (2026-09-08)

The data pipeline and labels were consolidated after the camera-supervision
diagnosis.  The current single source of truth is `docs/03-数据格式.md` §7;
`docs/00-当前状态.md` is the concise execution entrypoint and `docs/10`/`14`
now describe ACT consumption and human annotation without carrying old v2 or
48-pixel bucket guidance.

The chain is deliberately split into three non-overwritable label layers:

1. **Raw replay behavior**: `events.csv` plus `mouse_deltas.csv` share the
   frame clock; `extract_session` produces `per_frame_actions.csv`; v5 sums
   each future six-frame Raw Input window before bucketing.  `action_chunk`
   and h0 `replay_camera_dx/dy` remain an immutable record of the human
   demonstration.
2. **Visual facts**: `act_grounding_annotations.jsonl` contains cipher box,
   side, Q prompt and reachability.  The pool intentionally includes Q-neighbor
   frames; only exact decision endpoints attach to an ACT row.  Missing labels
   are masked, never silently converted to no-cipher negatives.
3. **Visual control intent**: `camera_control_{train,val}.v2.rebucket675`
   contains phase/target/steering/path/desired turn.  It is exact-endpoint
   auxiliary supervision only.  It may differ from replay for detours or
   recording-latency decisions but it never rewrites replay.  Deployment
   consumes only the model's predicted logits, never sidecar JSONL.

New command `idv_agent.scripts.audit_mvp_training_data` is the required
fail-closed dataset readiness check.  It validates the complete v5 record
contract, session-held-out split, grounding split membership, exact control
endpoint and replay-h0 provenance, and h0 class-balance artifact convention.
The real canonical audit passed and is saved at:

`C:\Codespace\ID_V_Agent\reports\mvp_v5_rebucket675_training_data_audit.json`

Evidence: train=732 chunks/79 sessions, val=182/20, zero session overlap;
grounding=1,076/304 frames; control=48/46 complete rows; control replay
matches canonical h0; h0 dx counts `[-2,-1,0,+1,+2]=[53,84,467,63,65]`; no
audit errors.  Focused pipeline/label regression tests: **63 passed**.

This proves the current artifacts are internally consistent, not that the
48/46 control labels are sufficient to learn generalized obstacle navigation.
The next data expansion must add independently held-out, fully labeled route
examples (search, acquisition, left/right detours, approach, Q prompt) before
claiming camera-control generalization.  No deployment authorization changes.

## Architecture Coordination Audit: Required m28 Direction (2026-09-08)

The current m27 graph is not yet a true `image -> state -> decision -> action`
policy.  It has a structural coordination defect:

- `VisualActionExpert.fast` (move/Q/duration and the direct camera paths) and
  `VisualActionExpert.slow` (intent/subgoal) are independent readouts from the
  same frozen visual pair feature.  There is no shared trainable state trunk
  and the action heads do not consume the visual slow intent/subgoal logits.
- Runtime's one-Hz `SlowCondition` is passed through FiLM into the diagnostic
  temporal branch, but deployed m27 move/Q are visual-only and camera prior is
  fixed to zero.  Thus `travel/decipher` is displayed in logs but is not an
  upstream causal decision condition for the deployed camera action.
- Grounding/control predictions are fused into final camera logits through a
  zero-initialized linear residual, but move and Q do not consume these states;
  `desired_turn_*` is evaluated as an auxiliary head while the executor still
  uses the replay-camera argmax.  Consequently path-follow can be predicted
  without controlling the turn actually sent to the game.

Do not solve this only by adding data or loss weights.  The required next
architecture is an m28 **shared State-Decision Core**:

```text
8-frame spatial visual feature
  -> shared state trunk z (no mean-pool / preserves spatial cells)
  -> predicted belief logits: intent, nav phase, target, side/reachable/prompt,
     steering mode, path strategy, desired move/turn
  -> differentiable soft state embedding + z
  -> one joint h0 planner: move + camera + Q
  -> executor lookup table / 6-frame rolling re-observation
```

All planner inputs must be model predictions, never annotation sidecars or
ground-truth teacher-forced state.  Replay actions remain supervision across
all rows; annotated `desired_turn` is direct planner supervision where present
and control-state losses teach why a replay action occurred.  The deployed
camera must eventually come from this state-conditioned planner, not an
unconditioned replay argmax.  A later learned recurrent belief filter may add
state persistence only after contiguous decision sequences are prepared;
the current decorrelated chunks cannot truthfully train such a transition.

### m28 implementation status

The m28 core is now implemented in `idv_agent/model/vla_heads.py` as
`StateDecisionExpert` and is the deployed `core.visual_expert`.  It constructs
shared state `z` from the ordered eight-frame spatial feature sequence; all
state logits and the joint action planner share it.  The planner takes only
soft embeddings of its own predicted beliefs.  `desired_turn_dx/dy` aliases
the planner's executed h0 camera logits.  Checkpoint schema is now
`m28_act.state_conditioned_joint_planner.v1`; all m27 checkpoints are
fail-closed.  This is an architecture implementation, not a trained result.

Focused regression: **66 passed** (state/action gradient path, desired-turn
wire identity, all-predicted-belief planner input, data/label contract,
camera buckets and checkpoint rejection).  A new m28 numerical smoke must run
from base Qwen before any full train; the interrupted m27 oversampling full
run, even if it finishes, is superseded and must not be evaluated as a
deployment candidate.

Required acceptance for m28 before a full run: (1) state loss and action loss
share the state trunk with verified nonzero gradients, (2) inference uses the
same all-predicted soft state path as training, (3) state-ablation/shuffle with
images held constant measurably degrades annotated desired-turn/move/Q metrics,
(4) image zero/shuffle and current visual gates still pass, and (5) h0 camera
is sourced from the planner actually executed.  The architecture is implemented;
GPU smoke, full training and deployment evidence are not.  All existing
checkpoints remain fail-closed.

### m28 post-implementation architecture review

The shared State Trunk fixes the m27 parallel-head disconnection: action loss
now reaches state/intent/steering and `desired_turn` is the actual h0 camera
logit.  It does **not** yet prove full control closure:

1. Joint Planner has one shared hidden/state input, but its final move/camera/Q
   outputs are still factorized linear heads.  Add joint-action acceptance for
   `move+turn`, `path_follow+move`, and `prompt -> camera=0 + Q`; independent
   head accuracy is insufficient.
2. The action-history vector is intentionally excluded from the m28 deployed
   planner to prevent action-autocorrelation shortcuts.  No current learned
   body-velocity/belief memory exists; do not add it until continuous decision
   sequence data and image/history ablations can prove it improves control
   rather than recreating a prior shortcut.
3. Raw v5 timestamps prove offline capture/action-label ordering, but runtime
   does not yet record `capture -> inference -> command send -> next capture`.
   The first m28 smoke must add a dry-run timing trace before claiming that
   `action_delay_frames=1` represents actual deployment latency.
4. Reports proposing discrete diffusion, CFG, Top-k randomness, VQ-VAE action
   primitives, filtering raw replay, physical-engine constraints, or IRL are
   explicitly deferred.  We lack verified game physics/reward/continuous
   state, and stochasticity would worsen safe h0/Q control.  Their applicable
   insight is state-conditioned joint action plus measured rolling timing.

### m28 alignment repair after full architecture review

The second review found and repaired a hidden m27 compatibility path that
would otherwise invalidate the m28 claim.  `StateDecisionExpert` originally
received only pure visual features while task/mode flowed through the old
FiLM/SlowCondition diagnostic branch.  `ACTPolicy` also ran that old slow loop
every second and logged its intent, despite the m28 planner not consuming it.

The deployed path is now one pass everywhere: adapter task/mode projection is
explicitly computed as `conditioned_visual - pure_visual` and projected into
the m28 State Trunk; runtime no longer runs or logs a SlowCondition loop;
it logs predicted state intent/phase/steering from the same planner pass that
produces h0.  Training, standard evaluation, visual-dependency evaluation,
camera-control evaluation, feature-activity evaluation, zero-camera diagnosis,
ACT diagnosis, and realtime benchmark were all changed from two-pass legacy
SlowCondition calls to the same one-pass m28 call.  The realtime benchmark
now reports `planner_ticks`, not a fictitious independent slow tick.

Regression evidence: **109 passed** across m28 planner, deployment, temporal
compatibility, benchmark, executor, data and checkpoint contracts.  This
repairs input/decision-path consistency; it does not replace the still-required
dry-run timestamp trace, GPU smoke, full train, or MVP gates.

## m28 GPU Smoke Series: Numerical Gate Not Yet Passed (2026-09-08)

Before the first m28 smoke, the complete CPU suite passed **311** tests and
the canonical data readiness audit passed.  After correcting the historical
checkpoint error-message test and adding m28 loss-scale guards, the complete
suite passes **313** tests.  All runs used base Qwen only, the audited v3
rebucket675 128/64 smoke subset, h0-only execution, image augmentation,
train/val sidecar separation, joint control sampling 0.35 / grounding 0.50,
and `lr=1e-4`.

| Smoke checkpoint | Targeted change | Loss initial → final | Max unscaled grad | Result |
|---|---|---:|---:|---|
| `m28_state_joint_smoke60_local_idv312` | initial shared state planner | 8.3708 → 2.3236 | 246.49 (step 10) | reject |
| `m28_state_joint_groundingmean_smoke60_local_idv312` | grounding 5-head mean instead of sum | 5.7619 → 1.6586 | 176.85 (step 4) | reject |
| `m28_state_joint_tasknorm_smoke60_local_idv312` | normalized/bounded task injection | 5.7928 → 2.1314 | 178.89 (step 4) | reject |
| `m28_state_joint_fusionnorm_smoke60_local_idv312` | post-fusion shared-state LayerNorm | 5.7885 → 2.2246 | 137.20 (step 4) | reject |

Every run had finite loss/components, no invalid gradient tensors, and
checkpoint reload delta `0`; the training-behavior gate rejects them solely
because its max gradient norm is `100`.  The corrections materially reduce
the peak (`246.49 → 137.20`) but do not meet the fail-closed threshold.
The persistent peak batch is not control-supervised; after grounding is
averaged it is dominated by state loss (`slow_intent≈5.73`,
`slow_subgoal≈1.71`) rather than a non-finite numerical failure.

**Decision:** do not relax the 100 threshold, lower learning rate as a
cosmetic workaround, or start full training.  The next code task is a
per-module / per-loss gradient attribution trace at the peak batch, separating
the m28 State Trunk, task-state injection, intent/subgoal heads, soft belief
embeddings, and Joint Planner.  Use this trace to change one explicit loss
lambda or gradient path, then run one new 60-step smoke.  All m28 smoke
checkpoints remain non-deployment diagnostics.

## Latest Evidence: m25 Loss-Scale Smoke

| Item | Evidence |
|---|---|
| Checkpoint | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m25_rolling_smoke60_scaled_local_idv312\\act.pt` |
| Metrics | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m25_rolling_smoke60_scaled_local_idv312\\metrics.json` |
| Training | 60 steps, 128 train samples, 64 validation samples, base Qwen only, M2/VG disabled |
| Optimization | loss `3.8789 -> 3.5390`; finite throughout; training behavior gate passed; command exit code `0` |
| Validation signal | camera-dx zero false-turn rate `0.5500`; dx recall `0.1429, 0.0000, 0.4500, 0.0000, 0.2500`; intent recall `0.8000` / `0.5254` for observed classes |

This smoke validates the corrected causal loss scale and numerical stability,
but it is not a deployment result. The full visual-dependency gate was not run
for this checkpoint, camera remains weak, and the class-balance acceptance is
`not_comparable` because no acceptance baseline was supplied.

## Latest Evidence: m25 Full Training and Visual Gate

| Item | Evidence |
|---|---|
| Checkpoint | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m25_rolling_full366_local_idv312\\act.pt` |
| Metrics / manifest | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m25_rolling_full366_local_idv312\\metrics.json`, `manifest.json` |
| Training | 366 steps, 732 train chunks, 182 validation chunks, batch=4, `execution_horizon=1`, image augmentation, class balance, 50% grounding sampling, base Qwen only |
| Optimization | loss `8.2220 -> 0.8976`; checkpoint loss delta `0`; command exit code `0`; all recorded gradients finite |
| Validation | intent accuracy `0.8462`; interaction precision/recall `0.6364/0.3500`; camera-dx zero false-turn rate `0.5630`; grounding side accuracy `0.4531` |
| Four-condition gate | `C:\\Codespace\\ID_V_Agent\\reports\\m25_rolling_full366_visual_dependency.json`; exit code `1`; `visual_dependency_gate_pass=false` |

The model is measurably image-sensitive: camera-dx argmax flip rate is
`0.6964` for image-zero and `0.4368` for image-shuffle. However, this does
not constitute useful visual grounding. Balanced normal/zero/shuffle scores are
move `0.264/0.167/0.364`, camera `0.256/0.225/0.229`, and intent
`0.860/0.500/0.513`. Thus image-zero camera drops only `0.0310` (required
`0.05`), while image-shuffle camera drops `0.0272` and move improves by
`0.1002`; both image conditions fail. The checkpoint is rejected and remains
fail-closed.

## m26 Protocol Repair (Implemented)

The m25 failure diagnosis separated two defects from the model itself:

- The default `camera_prior_scale=0.5` improved ordinary camera accuracy using
  temporal priors (`0.7527` prior branch accuracy) but obscured useful visual
  camera decisions. With a true prior-free diagnostic, the visual model passes
  all four required image-drop checks.
- `image_shuffle` previously mapped every row to the next ordered row, often a
  near-identical chunk from the same session. The gate now deterministically
  maps each row to a different session, and fails if fewer than two sessions
  are available.

The new `m26_act.visual_camera_no_prior.v1` schema permanently fixes camera
prior to zero in core construction, training, checkpoint loading, offline
evaluation, and realtime benchmarking; m25 and older checkpoints now fail
closed. On the 182-sample m25 diagnostic with the m26 output protocol and
cross-session shuffle, image-zero drops move/camera/intent by
`0.0973/0.0879/0.3604`, and image-shuffle drops by
`0.0999/0.0596/0.3318`; all satisfy the gate. This is diagnostic evidence
only: m25 was trained with camera prior `0.5`, so m26 must be retrained before
any deployment decision.

## Latest Evidence: m26 Visual-Camera Smoke

| Item | Evidence |
|---|---|
| Checkpoint | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m26_visual_camera_smoke60_local_idv312\\act.pt` |
| Metrics / manifest | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m26_visual_camera_smoke60_local_idv312\\metrics.json`, `manifest.json` |
| Training | 60 steps, 128 train samples, 64 validation samples, `camera_prior_scale=0`, image augmentation, per-head class balance, 50% grounding sampling, base Qwen only |
| Optimization | loss `3.8622 -> 3.6044`; max gradient norm `76.9136`; all component losses finite; checkpoint loss delta `0`; behavior gate passed; command exit code `0` |
| Protocol | manifest/checkpoint schema `m26_act.visual_camera_no_prior.v1`; M2/VG disabled |

This smoke validates numerical stability and the visual-only camera protocol,
but does not establish generalization or deployment readiness. Its 64-sample
validation camera-dx zero false-turn rate is `0.6750`; full training and the
complete 182-sample cross-session visual gate are still required.

## Latest Evidence: m26 Causal-Balance Smoke

| Item | Evidence |
|---|---|
| Checkpoint | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m26_causal_balance_smoke60_local_idv312\\act.pt` |
| Metrics / manifest | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m26_causal_balance_smoke60_local_idv312\\metrics.json`, `manifest.json` |
| Optimization | 60 steps, loss `3.9061 -> 3.6445`; max gradient norm `76.8631`; checkpoint loss delta `0`; behavior gate passed; command exit code `0` |
| Causal class tables | `execution_horizon=1`; camera-dx counts `[11,8,85,9,15]`, camera-dy counts `[0,0,110,17,1]`, move counts `[37,75,11,2,0,0,0,0,3]` |

This smoke validates the corrected causal class-balance source and remains a
numerical smoke only. The full 732/182 training and visual-dependency gate are
still pending.

## Latest Evidence: m26 Causal-Balance Full Training and Gates

| Item | Evidence |
|---|---|
| Checkpoint | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m26_causal_balance_full366_local_idv312\\act.pt` |
| Metrics / manifest | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m26_causal_balance_full366_local_idv312\\metrics.json`, `manifest.json` |
| Training | 366 steps, 732 train chunks, 182 validation chunks, `execution_horizon=1`, `camera_prior_scale=0`, corrected h0 class-balance tables, image augmentation, 50% grounding sampling, base Qwen only |
| Optimization | loss `8.1873 -> 0.8854`; checkpoint loss delta `0`; command exit code `0`; class-balance manifest horizon `1` |
| Visual dependency gate | `C:\\Codespace\\ID_V_Agent\\reports\\m26_causal_balance_full366_visual_dependency_crosssession_grounded.json`; exit code `0`; all image-zero/image-shuffle move/camera/intent checks passed |
| Feature activity gate | `C:\\Codespace\\ID_V_Agent\\reports\\m26_causal_balance_full366_feature_activity.json`; exit code `0`; gate passed |

The full checkpoint now has verified visual dependency and non-collapsed
spatial/deep features. The normal validation camera-dx zero false-turn rate is
still high (`0.6218`) and the rare-class acceptance has not yet been compared
against a declared baseline; this is not yet deployment authorization.

The live realtime benchmark also exposed the machine-side limitation: model
loading completed, but DXGI returned no display device in the current
non-interactive session. This is an environment blocker for live capture only;
it does not replace the offline checkpoint gates.

## Latest Evidence: Camera Zero-Bucket Calibration Diagnostic

| Item | Evidence |
|---|---|
| Report | `C:\\Codespace\\ID_V_Agent\\reports\\m26_camera_zero_calibration_diagnostic.json` |
| Method | Fit a zero-class logit bias on train only; evaluate unchanged on the 182-sample held-out validation set |
| Train result | bias `+1.5249`; zero-target false-turn `0.1478`; nonzero macro recall `0.3546` |
| Held-out result | zero-target false-turn `0.2269`; recalls for buckets `[-2,-1,0,+1,+2]` = `[0.1000,0.0909,0.7731,0.0833,0.0500]` |

This rejects “just add a zero-bucket threshold” as the fix. The model's
visual camera logits are image-sensitive, but prompt=0 held-out decisions do
not generalize: the annotation subset has only 46 prompt=0 samples, while its
camera targets span all five buckets; side supervision is also sparse
(`left=7`, `right=4`). The current evidence supports a data/target ambiguity
or observation-to-action mismatch as the primary blocker, not insufficient
epochs or a missing image pathway.

## Latest Evidence: Prompt=0 Camera Target Ambiguity

| Item | Evidence |
|---|---|
| Entropy reports | `C:\\Codespace\\ID_V_Agent\\reports\\m26_grounding_camera_causal_entropy_train.json`, `C:\\Codespace\\ID_V_Agent\\reports\\m26_grounding_camera_causal_entropy_val.json` |
| Scope | Same-frame human grounding, causal action horizon 0 only; read-only audit |
| Train prompt=0 dx | 48 samples; entropy `2.1511` bits; dominant bucket only `35.4%`; counts `{-2:9,-1:8,0:17,+1:3,+2:11}` |
| Val prompt=0 dx | 46 samples; entropy `2.1894` bits; dominant bucket only `32.6%`; counts `{-2:9,-1:5,0:15,+1:5,+2:12}` |
| Prompt=1 contrast | train `77/80` dx=0; validation `18/18` dx=0 |

Bounding-box horizontal position partly explains edge turns, but does not
resolve central targets: validation central/small (`x2|small`) contains all
five dx buckets. The smallest useful human label for each selected prompt=0
camera decision frame is therefore not another bbox; it is:

1. `camera_control_phase`: `search`, `target_acquire`, `target_align`, or
   `hold`.
2. `camera_target_id`: whether the visible cipher is the navigation goal
   (`target_cipher` / `other_visible` / `none`).
3. `camera_steering_mode`: `target_center`, `path_follow`, `search_sweep`,
   or `hold`—the camera can follow a walkable detour while the cipher remains
   the navigation target.
4. `path_strategy`: `direct`, `detour_left`, `detour_right`, or `unknown`;
   required only when steering mode is `path_follow`.
5. `desired_turn_dx` and `desired_turn_dy`: intended one-step correction bucket
   (`-2..2`) for the camera's current steering mode.

Label the existing grounding-pool prompt=0 frames first (train=48, val=46),
with priority on central/small visible cipher frames and every nonzero replay
bucket. Preserve the existing raw replay label unchanged; these are new
training-only semantic targets. Resume only after the labels pass a coverage
and enum validator, then use their model predictions—not sidecar values—in the
camera head.

### Annotation Workspace Ready (v2)

The pending templates have been generated and validated without modifying raw
sessions, canonical chunks, or the existing grounding JSONL:

- `C:\\Codespace\\ID_V_Agent\\data\\mvp_act_grounding_pool_v1\\camera_control_train.v2.pending.jsonl` — 48 rows
- `C:\\Codespace\\ID_V_Agent\\data\\mvp_act_grounding_pool_v1\\camera_control_val.v2.pending.jsonl` — 46 rows

For each row, replace `status: pending` with `complete` and fill exactly:
`camera_control_phase`, `camera_target_id`, `camera_steering_mode`,
`path_strategy`, `desired_turn_dx`, and `desired_turn_dy`. The validator
command is:

```powershell
python -m idv_agent.scripts.prepare_camera_control_annotations validate --input <completed.jsonl>
```

It rejects incomplete rows, duplicate ids, invalid enums, `hold` with a
nonzero desired turn, and `target_align` without a visible control target.

### Current External Blocker (2026-09-07)

Validation from the m26 worktree confirms the annotation workspace is intact
but not yet filled: `camera_control_train.v2.pending.jsonl` is `0/48 complete`
and `camera_control_val.v2.pending.jsonl` is `0/46 complete`. There is no
completed camera-control sidecar to train against. Do not run another camera
retrain, add a runtime threshold, or enable deployment while this remains the
case; doing so would repeat the proven prompt=0 ambiguity failure.

Resume sequence after the user provides completed labels:

1. Run the strict validator on both JSONL files (without `--allow-pending`).
2. Audit split, field coverage, enums, and desired-turn distributions.
3. Add these targets to the training-only camera supervision path, with model
   predictions—not sidecar values—used at deployment.
4. Run a 60-step smoke, full 366-step run, visual dependency/feature gates,
   class safety gate, then sandbox dry-run on an interactive DXGI desktop.

## Latest Evidence: m26 Full Training and Cross-Session Gate

| Item | Evidence |
|---|---|
| Checkpoint | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m26_visual_camera_full366_local_idv312\\act.pt` |
| Metrics / manifest | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m26_visual_camera_full366_local_idv312\\metrics.json`, `manifest.json` |
| Optimization | 366 steps, loss `8.1786 -> 0.8788`, checkpoint loss delta `0`, command exit code `0`, schema `m26_act.visual_camera_no_prior.v1` |
| Validation | intent accuracy `0.8462`; camera-dx zero false-turn rate `0.6218`; camera-dy zero false-turn rate `0.2323`; grounding side accuracy `0.4531` |
| Gate | `C:\\Codespace\\ID_V_Agent\\reports\\m26_visual_camera_full366_visual_dependency_crosssession_grounded.json`; exit code `1`; `visual_dependency_gate_pass=false` |

The full m26 checkpoint is rejected. Move and intent satisfy both image
degradation checks, but camera does not: image-zero balanced camera drop is
`0.0458 < 0.05`, and cross-session image-shuffle camera drop is `-0.0039`.
The grounded subset shows the failure is concentrated before interaction:
`prompt=1/reachable=1` camera accuracy is `0.9167` on 18 samples, while
`prompt=0` is `0.3913` on 46 samples; `side=right` has only 4 samples.

The report also exposed that m26's causal loss consumed only action step 0,
while class-balance histograms still counted all four horizon steps. This is
now corrected in code and covered by regression tests; the existing m26
checkpoint must not be reused because its weights were trained with the old
histograms.

## Latest Evidence: m24 Full Grounding-Balanced Training

| Item | Evidence |
|---|---|
| Checkpoint | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m24_groundingbalanced_2epoch_local_idv312\\act.pt` |
| Metrics / manifest | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m24_groundingbalanced_2epoch_local_idv312\\metrics.json`, `manifest.json` |
| Training | 366 steps, all 732 train chunks, batch=4, image augmentation, history dropout=0.5, 50% annotated sampler, base Qwen only |
| Optimization | loss `12.0795 -> 1.5974`; finite throughout; all-sample diagnostics completed |
| Grounding validation | 64 annotated chunks: presence `0.6094` accuracy / `0.6545` recall; prompt `0.8594` accuracy / `0.9444` recall; reachable `0.8281` accuracy / `0.9444` recall; side `0.4531`; bbox L1 `0.1128` |
| Full gate report | `C:\\Codespace\\ID_V_Agent\\reports\\m24_groundingbalanced_visual_dependency_full.json` |

### m24 Grounding-Balanced Deployment Decision: **REJECT / FAIL-CLOSED**

This is the strongest normal-condition model so far: on all 182 held-out
chunks, normal accuracy is move `0.5357`, camera `0.5591`, intent `0.8516`;
balanced accuracy is move `0.2829`, camera `0.2500`, intent `0.8636`.
Prompt/reachability calibration improved materially over the 30-step smoke.
The history-zero condition is identical to normal, as required by the
history-free visual move/intent design.

It still fails the visual-dependency gate and cannot be deployed:

- `image_zero`: move drop `0.1163` passes and intent drop `0.3636` passes,
  but camera drop is only `0.0206 < 0.05`.
- `image_shuffle`: intent drop `0.3478` passes, but move changes by `-0.0016`
  and camera drop is only `0.0318 < 0.05`.

The evaluator exits nonzero when a gate fails; that is expected fail-closed
behavior after it has written the complete report. The training command also
returned nonzero only because `--require-training-behavior` is deliberately a
30--60-step smoke gate while this run had 366 steps. The checkpoint, manifest,
metrics, and full diagnostics had already been written and are valid; do not
interpret that option-contract failure as an optimization failure.

### Next Diagnostic Decision

Do not merely add more epochs. The calibrated manifest shows that of the 128
annotated training decision frames, target side is `none=8`, `left=5`,
`center=105`, `right=10`. The sparse left/right supervision and the
observation-to-future Raw Input replay offset are likely limiting the learned
side-to-five-bucket camera mapping. Audit this relationship conditionally by
side, prompt/reachability, and action horizon before choosing between a
causal camera-target reformulation and targeted human labels. No deployment
rule may substitute for that learned mapping.

## m25 Causal Rolling-Step Protocol (Implemented, Not Yet Trained)

The per-horizon audit reports are at
`C:\\Codespace\\ID_V_Agent\\reports\\m24_grounding_camera_alignment_train.json`
and `C:\\Codespace\\ID_V_Agent\\reports\\m24_grounding_camera_alignment_val.json`.
They show that the observation-frame side label predicts the first action's
dx sign (held-out: left is negative `5/7`, right is positive `4/4`), while
steps 2--3 become mixed after the game view has changed. Runtime previously
executed all four six-frame actions (~0.8 s) before a new block could start,
which made those stale-image targets part of both training loss and deployment.

`m25_act.rolling_causal_step.v1` fixes that protocol end-to-end without an
external rule: the model still predicts a four-step tensor, but the trainer,
offline evaluator, and ACT executor share `execution_horizon=1`. They
supervise/score/submit only action 0 (six frames = 200 ms), then observe the
next real frame before issuing the next model decision. `m24` and all older
checkpoints fail closed because their four-step loss/execution semantics are
not compatible. The implementation and causal-loss regression are covered by
75 focused tests. The first m25 smoke wrote
`C:\\Codespace\\ID_V_Agent\\checkpoints\\m25_rolling_smoke60_local_idv312\\act.pt`
and `metrics.json`, but was rejected by the numerical behavior gate: maximum
unscaled gradient norm was `103.3916` at step 4 versus the `100.0` limit.
Loss still decreased `7.4353 -> 6.7463`, all component losses were finite
(maximum `13.9462`), and no NaN/Inf was reported. Its 64-sample validation
also had camera-dx zero false-turn rate `0.6000`; it is not a deployment
candidate. Re-run m25 smoke with a conservative learning rate before a full
training and complete four-condition gate.

The lower-`lr` retry also failed (`max_grad=117.1901`, final loss `9.0805`),
which ruled out learning rate as the root cause. The actual issue was loss
scale: restricting `_masked_mean` from 4 future actions to 1 causal action
amplified each fast-head sample gradient by 4x. The fix is implemented and
covered by 77 focused tests: m25 fast move/camera/button/event/duration loss
is normalized by `execution_horizon / 4`, preserving the new causal target
mask, class weights, and head lambdas. Re-run the smoke at the original
`lr=1e-4` before a full m25 run.

## Latest Evidence: m24 Learned Camera-Grounding Fusion Smoke

| Item | Evidence |
|---|---|
| Checkpoint | `C:\\Codespace\\ID_V_Agent\\checkpoints\\m24_camera_grounding_smoke30_local_idv312\\act.pt` |
| Schema / initialization | `m24_act.camera_grounding_fusion.v1`; base Qwen only, without M2 |
| Training setup | 30 steps, batch=4, max source samples=64, grounding annotated fraction=0.5 |
| Optimization | loss `11.3464 -> 6.2800`; minimum `4.7771`; max gradient norm `78.683`; behavior smoke passed (finite loss/gradients and deterministic feature checks) |
| Full gate report | `C:\\Codespace\\ID_V_Agent\\reports\\m24_visual_dependency_full.json` |

### m24 Deployment Decision: **REJECT / FAIL-CLOSED**

The evaluator completed over all 182 held-out chunks. The checkpoint fails
both required image conditions when assessed with balanced accuracy:

- `image_zero`: move `0.1441 -> 0.1667` (drop `-0.0226`), camera
  `0.2368 -> 0.2329` (drop `0.0040`), intent `0.4019 -> 0.5000` (drop
  `-0.0981`); all fail their required drops.
- `image_shuffle`: move `0.1441 -> 0.1473` (drop `-0.0031`) and camera
  `0.2368 -> 0.2300` (drop `0.0068`) fail; intent does pass (`0.4019 ->
  0.2483`, drop `0.1536`, threshold `0.0603`).

The image tensors do affect raw outputs (for example, image-zero camera-dx
argmax flips on `83.4%` of actions), but that sensitivity does not translate
to the required *correct* visual decisions. `history_zero` is effectively
unchanged, confirming the intended history-free move/intent path, but it does
not satisfy the visual-dependency requirement. Do not deploy or enable
`--send-input` for this checkpoint.

## Important Historical Evidence

- m21, without same-session visual grounding, failed the full visual gate.
  The root cause was not merely epochs: Raw Input mouse replay lacks a target
  position/reachability label.
- m22 (30 steps) proved the new auxiliary path has finite gradients and learns
  loss, but also failed the visual gate. m23 improved visual intent and
  image-zero move dependence, but not camera enough.
- `data/wk` and `data/vg_cipher_20260901_new` are optional and previously
  unreliable; do not use them to replace same-session evidence without a
  separate audit.
