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
- Checkpoint schema is `m24_act.camera_grounding_fusion.v1`. Older m19/m22/m23
  checkpoints fail closed and are not deployment candidates.

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

1. The m24 learned camera-side/prompt fusion path is implemented: it consumes
   only predicted visual grounding logits, affects only camera dx/dy, and has
   a zero-initialized fusion layer. Schema is now
   `m24_act.camera_grounding_fusion.v1`; m22/m23 fail closed.
2. Run a short 30--60 step behavior smoke as a blocking command. If it is
   numerically sound, run a bounded training experiment; then run the complete
   four-condition gate as a blocking command.
3. Update this file with the result. A checkpoint is deployable only when all
   image-zero and image-shuffle move/camera/intent requirements pass.

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
