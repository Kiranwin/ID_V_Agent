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

## Current Working Context

| Item | Current value |
|---|---|
| Worktree | `C:\Codespace\ID_V_Agent\.worktrees\training-augmentation-balance` |
| Branch | `codex/training-augmentation-balance` |
| Python | `C:\Users\kiran\miniconda3\envs\idv312\python.exe` |
| Hardware | Local RTX 2080 Ti; FP16 + GradScaler; no BF16 |
| ACT initialization | Base Qwen only; M2/VG is disabled for ACT |
| Raw MVP source | `C:\Codespace\ID_V_Agent\data\new_vla_raw_sessions` |
| Canonical ACT dataset | `C:\Codespace\ID_V_Agent\data\mvp_vla_v5_event_centered_decision_label_v2` |
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

## Current Architecture

- Camera labels are v5 Raw Input replay labels using the shared pixel bucket
  convention and executor lookup table. Training labels and deployed pixel
  commands remain same-source.
- The deployed ACT path is the history-free `VisualActionExpert`; temporal
  history is diagnostic/prior-only and cannot replace visual move or intent
  logits. Camera retains a bounded temporal prior contribution.
- m22 added **training-only** heads on the direct visual-pair feature for
  cipher presence, normalized box, target side, interaction prompt, and
  reachability. These heads never consume YOLO/runtime rules and deployment
  never reads annotation JSONL.
- Checkpoint schema for the next ACT run is
  `m26_act.visual_camera_no_prior.v1`. m19/m22/m23/m24/m25 checkpoints fail
  closed for the current protocol and are not deployment candidates.

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

1. Run a 60-step m26 numerical smoke with `camera_prior_scale=0`, then inspect
   its embedded manifest and behavior acceptance.
2. If the smoke passes, run the full 366-step m26 training and the revised
   cross-session four-condition gate. The checkpoint is deployable only when
   all image-zero/image-shuffle move/camera/intent requirements pass.
3. After an offline pass, perform the sandbox dry-run sequence and verify the
   learned policy completes find-machine, approach, Q interaction, and decoding
   entry. Keep `--send-input` disabled until that evidence exists.

The previous full m26 checkpoint is not reusable: its class-balance tables were
computed over all four action horizons while the causal loss trained only step
0. The corrected implementation has passed a fresh smoke; the next full run
must be trained from the base Qwen with the corrected step-0 tables.

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
