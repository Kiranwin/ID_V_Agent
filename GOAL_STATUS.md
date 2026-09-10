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
3. Use `data/raw_sessions` as the raw MVP source.
   derived data and annotation-pool products are separate, recoverable files.
4. After a material result, update this file with evidence paths, pass/fail
   gates, the next concrete action, and any newly created backups.

## Root-Cause Review Addendum (2026-09-08)

**2026-09-10 run_agent 入口收敛为 SGan：** 按用户要求，`run_agent` 删除
`--mode rule`（规则安全兜底）与 `--mode act`（旧 ACT 同步 runtime）两条分支及其
参数，只保留原 M29 运行链路并对外重命名为 **SGan**（State-Guided Action
Network；checkpoint schema 与模块名仍为 m29，`m29.pt` 可直接加载）。新字段为
`--mode sgan`、`--checkpoint`（必填，`.pt` 文件）、`--model-path`（可省略，回退
`manifest.base_model`）、`--title`、`--device`、`--duration`、`--trace`（必填，
路径必须不存在）、`--send-input`、`--allow-undeployed-send-input`、
`--enable-navigation`。行为不变：默认 dry-run；`deployable=false` 的 checkpoint
单独用 `--send-input` 仍被 `run_m29.py` 拒绝。旧 rule/act 分支上的“--send-input
必须管理员终端”预检随分支一并移除；SGan 路径沿用 run_m29 的 deployable 门禁。
完整字段与 trace 说明写入
`docs/agent.md`；受影响文档 `README.md`、`docs/05-操作手册.md`、
`docs/19-ACT实时推理接入.md` 已同步，行为回归测试
`tests/test_configs.py -k run_agent` 2 passed。当前标准模式沙盒命令为：

`conda run -n idv312 python -m idv_agent.scripts.run_agent --mode sgan --checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt --model-path <Qwen目录> --device cuda --duration 10 --enable-navigation --send-input --allow-undeployed-send-input --trace reports/sgan_sandbox_<date>.jsonl`

**2026-09-10 新标注批次合并与分布检查：** 新增的 19 个 raw session
`20260909_204705_204448` 至 `20260909_205722_049628` 全部通过
`validate_vla_raw`。已将当前 25 个已完成 V6 标注文件按原有 split 合并为：
`data/derived/m29/20260910_all_new_train_v1.jsonl`（1008 行、21 个
session）和 `data/derived/m29/20260910_all_new_val_v1.jsonl`（213 行、4
个 session）。合并后 train/val 无重复 endpoint ID。

新的 full train 分布为：phase search/approach/align/interact/maintain_decode
=72/565/34/110/226；navigation move=0/1/2/3/4/7/8
=102/434/31/17/3/3/63；camera_dx 桶
`-110/-25/0/25/110=14/46/468/63/62`；camera_dy
`-25/0/25=1/624/28`。验证集 phase=21/116/8/22/46，navigation move
只覆盖 0/1（18/116），camera_dx=`-25/0/25/110=7/104/8/15`，camera_dy
`0/25=124/10`。Q 正负、破译正负、五阶段和横向镜头多桶均已有覆盖，纵向
镜头首次出现非零桶，但仍极不均衡。

当前正式数据门禁仍失败：train 与 val 都属于 scenario group
`decliper`，`audit_m29_v6_data` 报告
`reports/mvp_v6_design/m29_v6_data_audit_20260910_new.json` 的唯一错误是
`full train/val session or scenario-group overlap`。因此这些数据可以用于
训练分布和动作头再训练，不能直接作为最终独立验证集；下一步应把至少一
个完整 scenario group 留作 val，或重新录制独立场景组后再训练正式 checkpoint。

**2026-09-10 新数据 full 训练验证：** 训练入口原先错误地要求 val 的 path
也必须包含多个类别；现已改为训练集强制类别多样、验证集只要求字段可评估，
19 个 M29 回归测试通过。随后以 v7 几何动作架构在
`20260910_all_new_train_v1.jsonl`（1008 行）上训练 1000 步，并在
`20260910_all_new_val_v1.jsonl`（213 行）评估。输出位于
`checkpoints/m29_full_newdata_diag1000_20260910`。训练稳定，峰值裁剪前
梯度 24.10，无梯度故障，checkpoint reload 一致。

训练/验证结果：move 92.3%/85.1%（balanced 95.9%/75.0%），camera-dx
68.8%/53.7%（balanced 86.3%/54.1%），camera-dy 94.9%/88.8%
（balanced 98.2%/57.2%），joint navigation 61.9%/48.5%。验证集中
camera-dy=+25 recall 仅 20%。状态方面 decoding 97.4%/97.2%，但 Q 正类
recall 22.7%/9.5%，interact phase recall 14.5%/18.2%；这两项直接阻塞
MVP 的“看见提示→Q→进入破译”。bbox/prompt_bbox 验证 mean L1 为
0.211/0.283，也仍偏高。

本次 checkpoint 是共享 `decliper` 场景组的诊断模型，manifest 保持
`deployable=false`。下一步必须优先修复 Q/interact 的采样或两阶段初始化，
并增加非零 camera-dy 与独立 scenario-group 验证；不能直接进入发送输入。

**2026-09-09 M29 full training diagnosis:** a 120-step FP16+GradScaler run on
`checkpoints/m29_full_random80_20_diag120_20260909` completed without a
gradient failure (peak pre-clip norm 14.30; checkpoint reload delta 0). The
random 80/20 split is intentionally diagnostic and has session/group overlap.
Q accuracy is 93.5% train / 90.6% val; phase is 91.6% / 90.6%; decoding is
98.1% / 98.1%. Navigation remains collapsed: move balanced accuracy is
25.0% train and 25.0% val, with every class except move=1 at zero recall;
camera-dx balanced accuracy is 20.0% on both splits, predicting only bucket
0 (the 0-pixel command); camera-dy has only bucket 0 in train and is therefore
not a meaningful learned result. The action loss stays near its initial
cross-entropy (move 2.30 -> 2.02, camera-dx 1.64 -> 1.24) while state losses
fall sharply. This rules out “only too few steps” and confirms action-head
collapse under the current labels/conditioning. The data are strongly
imbalanced (move 1=103/169 navigation rows; camera-dx 0=125/169), and the
same coarse phase/visibility state maps to multiple actions. Next diagnosis
must address action-label coverage/conditioning and per-class sampling or
loss before any deployment claim; the checkpoint remains `deployable=false`.

**2026-09-09 same-data overfit diagnosis:** train and validation were made
content-identical (214 rows; validation copy only changes the protocol split
field) and trained for 300 steps from the Q checkpoint at
`checkpoints/m29_full_overfit_same_train_20260909`. State heads overfit as
expected (phase 96.3%, Q 94.4%), and total loss fell from 3.64 to 0.11, but
navigation did not overfit at all: move stayed at 60.9% accuracy / 25.0%
balanced accuracy with only class 1 recalled; camera-dx stayed at 74.0% /
20.0% with only bucket 0 recalled. This is decisive evidence that the current
M29 navigation conditioning/label representation cannot memorize the existing
action rows, even when train and validation are identical. The immediate
blocker is therefore inside the navigation target/conditioning contract, not
generalization or insufficient training duration. The checkpoint remains
`deployable=false`.

**2026-09-09 decisive true-belief navigation experiment:** using the same 169
navigation rows, the navigation trunk and action heads were trained directly
on ground-truth facts/decision beliefs for 1000 CPU steps. The resulting
training fit reached move 99.4%, camera-dx 94.1%, camera-dy 100%, and joint
action 94.1%. The constructed true belief had 151 unique inputs; only three
exact belief values carried conflicting action labels. Therefore the action
labels are mostly learnable and the action head implementation can memorize
them. The failure in the end-to-end M29 overfit is specifically caused by the
predicted-state discrete belief bottleneck: predicted visibility/phase/etc.
are not stable or informative enough for navigation, and their hard argmax
values erase target-position/action distinctions. Do not solve this by adding
more epochs. The next architecture change must preserve continuous visual
target geometry and/or history features into navigation while retaining the
top-level predicted-state influence; re-run the same-data overfit gate after
that change. Checkpoint remains `deployable=false`.

**2026-09-09 anti-collapse verification:** the true-belief navigation model
was retrained and evaluated with per-class output counts plus counterfactual
input ablations. On the original beliefs it reached move 98.8%, camera-dx
95.3%, and 94.7% joint accuracy; the majority-class baselines are move 60.9%
and camera-dx 74.0%. Outputs used all observed move classes (35/103/3/28 for
classes 0/1/2/8) and all camera-dx buckets (4/8/128/16/13), so this is not a
majority-class-only solution. Zeroing bbox reduced joint accuracy to 51.5%,
shuffling bbox to 61.5%, and mirroring bbox to 24.9%; camera-dx fell to 31.4%
under the mirror. A constant bbox still retained 75.7% joint accuracy because
phase/steering also carry signal, so bbox is not the only feature, but the
large counterfactual drops confirm that the fitted action depends on target
geometry. Camera-dy cannot be validated for diversity because every current
navigation label is bucket 0. This verifies true-belief fitting, while the
end-to-end predicted-belief model remains collapsed and non-deployable.

**2026-09-09 predicted-belief loss localization:** code and checkpoint review
locate the information loss at three concrete operations in
`state_guided_action.py`. `PredictedFacts.probabilities()` hardens visibility
and prompt with argmax/threshold, then multiplies the predicted target bbox by
`visibility[:,:2]`; one visibility error therefore erases all target geometry.
`PredictedDecision.probabilities()` hardens phase, steering and path again, so
navigation receives only 13 one-hot semantic bits plus four predicted bbox
coordinates, rather than the visual state that produced them. The learned
predicted bbox is also weakly supervised: its measured mean L1 is about 0.21
while the facts block contributes only 0.25 times the average of four losses,
so geometry receives an effective low weight. The navigation trunk has no raw
visual/history or confidence input by design. This explains why true beliefs
fit but predicted beliefs collapse: state classification can be accurate while
the continuous geometry and action-relevant uncertainty are wrong. The next
fix should retain continuous fact probabilities/geometry (with confidence)
and add an action-relevant temporal geometry signal to navigation, while
keeping predicted states as a required conditioning path.

**2026-09-09 navigation/action-head repair v1:** the M29 state-to-navigation
boundary now preserves probability-valued categorical/binary beliefs and no
longer hard-gates the continuous target/prompt bbox by visibility argmax. This
keeps predicted state as the only navigation condition while retaining
uncertainty and target geometry for the move, camera-dx and camera-dy heads.
Focused M29 hierarchy/pipeline regression passes: 19 tests. Existing
checkpoints are incompatible with this behavior contract and remain
non-deployable; the required next gate is a fresh same-data end-to-end
overfit run followed by anti-collapse counterfactual checks.

**2026-09-09 navigation/action-head repair v2:** M29 schema is now v3. The
navigation trunk receives both the soft predicted semantic belief and the
continuous temporal state that generated it; move and camera heads use
separate trunks so move gradients cannot dominate camera learning. The sampler
stratifies full-navigation rows by `(move, camera_dx, camera_dy)` while keeping
state-only rows. Focused regression remains green (19 tests). A fresh 300-step
same-data end-to-end overfit from the new architecture reached move 86.4%
accuracy / 92.4% balanced accuracy, with recalls 91.4/81.6/100/96.4% for the
observed move classes 0/1/2/8. Camera-dx reached 34.3% raw accuracy but 78.3%
balanced accuracy, with recall 100/100/15.2/76.2/100% across all five buckets;
the model no longer outputs only bucket 0. Camera-dy remains uninformative
because all 169 current labels are bucket 0. The repaired model now passes the
anti-collapse criterion for move and camera-dx, but needs a longer/cleaner
camera-dx fit and new camera-dy labels before deployment gates can be
considered. Checkpoint remains `deployable=false`.

**2026-09-09 v3 overfit acceptance:** the repaired architecture was trained
for 1000 steps on the identical train/validation content with stratified
navigation sampling. The new checkpoint
`checkpoints/m29_full_overfit_v3_balanced_1000_20260909` reached move 98.2%
accuracy / 99.3% balanced accuracy and camera-dx 89.9% / 96.5% balanced
accuracy. Move recall for observed classes 0/1/2/8 is 100/97.1/100/100%;
camera-dx recall for buckets 0..4 is 100/100/87.2/95.2/100%. Camera-dy is
still a single-class label (169/169 bucket 0), so no diversity claim is made
for that head. Train and validation metrics are identical by design, proving
the repaired predicted-state path can memorize the available action set and
no longer collapses to the majority move/camera outputs. A separate
counterfactual geometry check remains required before calling the checkpoint
deployable; current manifest remains `deployable=false`.

**2026-09-09 v4 geometry/action-head repair:** M29 v3 now has separate move
and camera trunks plus an explicit bbox/prompt geometry trunk whose logits are
added to move, camera-dx and camera-dy. The continuous temporal state is
attenuated during training so it cannot be the sole memorization shortcut.
Regression: 19 focused tests pass. A fresh 1000-step same-data run at
`checkpoints/m29_full_overfit_v4_geometry_regularized_1000_20260909` reaches
move 98.2% accuracy / 98.3% balanced accuracy and camera-dx 90.5% / 95.4%,
with all observed movement and camera-dx classes recalled. Camera-dy remains
single-class in the source data. This satisfies the overfit and anti-collapse
action-head gate, but final geometry-dependence counterfactual output must be
recorded before deployment; checkpoint remains `deployable=false`.

**2026-09-09 v5 geometry-dependence repair:** Added an explicit geometry trunk
from predicted target/prompt bbox to all three action heads, with scale 2.0,
and deterministic attenuation of the continuous temporal state during
training. M29 metadata identifies the boundary as `soft_continuous_geometry_v2`;
focused M29 tests remain 19/19. The 1000-step same-data run reaches move
98.2% / 98.3% balanced accuracy and camera-dx 90.5% / 95.4% balanced accuracy,
with all observed classes recalled. Bbox counterfactuals change action logits
(mean move/camera-dx logit changes: zero 0.39/0.39, shuffle 0.07/0.07, mirror
0.16/0.16); zeroing the continuous state instead reduces joint action accuracy
from 91.1% to 42.6%. This proves the repaired heads use predicted geometry and
predicted state rather than only a majority action. Camera-dy remains
single-class in current labels, so its diversity is unverified. Checkpoint
remains `deployable=false`.

**2026-09-09 objective completion audit:** The final v7 checkpoint
`checkpoints/m29_full_overfit_v7_direct_geometry10_1000_20260909` satisfies the
requested M29 repair gate. The predicted-state path overfits the identical
train/validation set with move 96.4% accuracy / 98.1% balanced accuracy and
camera-dx 94.1% / 97.6%, while recalling every observed class. The final
counterfactual test changes bbox to zero, shuffled, and mirrored values and
measures mean absolute logit changes for move/camera-dx/camera-dy of
`1.859/1.661/2.426`, `0.639/0.649/0.837`, and `1.214/1.234/1.731`
respectively. All three heads therefore consume predicted target geometry;
this is not a majority-action collapse. The focused regression suite is
19/19. Camera-dy class diversity is still absent from the source labels, so
only its geometry sensitivity—not multi-class recall—is claimed. The model is
an offline architecture/overfit artifact and remains `deployable=false` until
independent sessions and complete camera-dy labels pass their gates.

**2026-09-09 v6/v7 final overfit audit:** The direct coordinate path was
strengthened and the v7 model was trained for 1000 steps on identical
train/validation content. Move reached 96.4% accuracy / 98.1% balanced
accuracy; camera-dx reached 94.1% / 97.6%, with every observed class recalled.
The bbox mirror intervention changed 2.4% of move argmaxes and 8.9% of
camera-dx argmaxes; bbox zeroing changed 2.96%/3.55% respectively. Logit
changes are present for all interventions, proving the explicit geometry path
is active, although the small dataset still permits temporal-state
memorization. Camera-dy remains 169/169 in one bucket, so its behavior is
implemented and trained but its geometry dependence cannot be empirically
verified until nonzero vertical camera labels are collected. Focused M29
regression remains 19/19. The current checkpoint is an overfit diagnostic,
not deployable.

### Superseding design decision: complete MVP labels and sparse visual clock

**2026-09-09 record_vla frame-clock correction:** review of existing raw
sessions found that `record_vla` timestamped a frame only after `cap.grab`,
optional resize, and JPEG encoding. At 20 FPS this lets processing latency
shift the visual timestamp into a later frame relative to keyboard/Raw Input
events. The recorder now samples `capture_ts` immediately before `cap.grab`
and keeps the shared `perf_counter_ns` clock. Existing raw sessions are not
rewritten; new captures must be regenerated before using this correction.
Focused capture tests remain green when run with the repository-local pytest
temp directory; the default system temp directory is inaccessible in this
environment.

**2026-09-09 boundary-label correction:** the user correctly reported that
opening rows with insufficient history and tail rows without a complete replay
window could not be completed when `accept_replay` was selected. The workbench
now treats these as navigation-mask conditions, not missing intent/state. For
`insufficient_history`, `visual_history_gap`, or `incomplete_action_window`,
completion automatically stores `action_source=exclude` with an audit reason,
while preserving phase, decoding, target facts and prompt-Q. The browser locks
the action-source control for those rows and explicitly says to label only
state/intent. If phase itself is not observable, the row can finish with phase
unknown, producing `phase=false/navigation=false` masks while retaining Q.
`insufficient_one_second_outcome_tail` affects only outcome confirmation and
does not block current phase/decoding/Q. Final focused regression: 60 tests
passed; no existing user label was rewritten.

**2026-09-09 prompt/cipher association decision:** prompt-Q supervision does
not require a cipher candidate. A visible prompt means the player is already
close enough; the workbench automatically sets navigation action source to
`exclude` while preserving `interact`, current decoding and Q supervision.
Candidate boxes and `target_track_id` are only needed before the prompt for
search/approach/align target tracking. The prompt box itself is never treated
as a cipher box, and `target_track_id` remains an audit field rather than a
learned category input. The known prompt-only markup is session
`20260908_205631_985816`, frame 75; it remains valid Q evidence. The current
browser snapshot is 19/101 reviewed: the first session is 19/21 and the other
three remain pending, reflecting user progress after the earlier 14/101
snapshot.

**2026-09-09 M29/V6 contract repair implemented:** the static-review blockers
that could be fixed without inventing human labels are now repaired and
versioned. M29 checkpoint schema is `m29.state_guided_action.v2`. Downstream
navigation receives discrete straight-through visibility/decoding/prompt/
phase/steering/path values plus supervised continuous target/prompt boxes;
the hidden GRU/visual context still cannot enter `navigate_from_beliefs`.
Inference latency was removed from neural inputs: capture-relative history
times remain, while actual capture-to-ready age is enforced by the stale-result
gate and runtime trace. The annotation/deployment phase vocabulary is exactly
five classes; post-Q waiting remains `interact` and execution feedback is a
separate input.

Q export is now `idv.prompt_q_training.v2`: all-frame prompt/NO-prompt labels
also carry prompt bbox supervision where exactly one prompt box exists. Four
new, non-overwriting `*_xany_import_v2` workspaces were generated from the
same raw/X-AnyLabeling evidence; v1 workspaces remain historical snapshots.
The canonical current Q assembly is
`data/derived/prompt_q/20260909_xany_all4_v2/train.jsonl`: 366 unique rows,
Q48/NO_Q318, 48 prompt boxes, four sessions, one same-scene train group.
`assemble_m29_dataset` validates sources and merges all-frame Q with future
full endpoint files by preferring the richer full row only when Q target and
latest image agree, avoiding both lost full-frame Q supervision and duplicate
endpoint sampling.

`train_m29` now supports/records `--init-checkpoint` from Q task to full task,
requires an independent full validation export, rejects train/val session or
scenario-group overlap, audits Q +/- and full state/route/action coverage plus
class diversity, uses group-balanced sampling and bounded sqrt class weights,
and rejects a pre-clip gradient peak before optimizer step. A rejected peak
writes `gradient_failure.json` with per-loss/per-module attribution. Evaluation
now records confusion/recall/balanced accuracy for Q, decoding, visibility,
phase, steering, path and actions; target/prompt bbox L1; joint move+dx+dy by
target phase; and interventions for every discrete state consumed by navigation.

New `audit_m29_v6_data` is the fail-closed aggregate readiness entrypoint.
Current report is
`reports/mvp_v6_design/m29_v6_data_audit_20260909.json`; its expected gate
result is false only because full train/val exports do not exist and browser
review is 14/101. All 14 reviewed rows are approach/decoding=false in the first
session; 87 remain pending and no decoding-positive example exists. No test,
script or v2 import changed the session-local CSV. The live workbench was
restarted on 127.0.0.1:8765 and defaults to xany_import_v2. Its aggregate
readiness route currently reports 14/101 and lists the missing
search/align/interact/maintain_decode, decoding-positive, path-diversity and
independent-val blockers directly in the browser.

The workbench now exposes `/api/mvp-v6/readiness`, which reports the live
cross-session blockers in the browser. Verification: Python compile passed and
57 focused V6/M29/workbench tests pass
using a repository-local pytest temp directory. The first run's 37 setup errors
were solely the inaccessible system pytest temp directory; rerunning with an
isolated `tmp/` base produced the passing result. No GPU training or real input
was started. The only next data action remains completing the browser review,
including all five phases and real decoding positive/negative coverage, then
creating an independently grouped validation capture/export.

**2026-09-09 static V6/M29 training-contract review (no tests run):** the
session-local CSV/workbench persistence is lossless and fail-closed at the
full-export boundary, but the complete M29 training contract is **not ready**.
At the review snapshot, the four CSVs contain 101 endpoints with 14 reviewed
and 87 pending; all 14 reviewed rows are `approach` with `decoding=false` in
`20260908_205631_985816`. There are no reviewed decoding-positive examples and
no complete `training_v6.jsonl`, so `train_m29 --task full` must still reject.
The user is actively annotating; these counts replace the older all-pending
snapshot below, which is retained only as historical context.

The four imported all-frame Q files are structurally available with 366 unique
frames and Q48/NO_Q318. The existing merged derived pilot still contains only
the older three sessions (285 frames, Q36/NO_Q249), so using that path would
omit the new 81-frame session. This is a preserved stale artifact, not a silent
overwrite or loader filter. All current exported rows have a known Q mask, so
M29Dataset's `if not any(masks.values())` filter would drop none of them.

Blocking code-contract findings before an interpretable full run:

1. Q-only training cannot initialize/resume full training, and combining the
   all-frame Q export with full endpoint exports is rejected by duplicate IDs.
   A fresh full run would therefore discard the learned Q-only weights and use
   only endpoint Q supervision instead of all 366 frames.
2. Full export masks `phase` with navigation history readiness. This discards
   valid top-level phase supervision from short-history opening frames, exactly
   where `search` is most important, while steering/path/facts use different
   mask rules.
3. Export writes `observation_age_ms=200` for every offline row although runtime
   supplies measured capture-to-ready age. The current model learns a constant
   age and deploys on a variable age; action-label delay and observation age are
   conflated.
4. V6 exposes six phases but M29 has five and silently maps `verify_decode` to
   `interact`. Unify the label vocabulary or make the collapse explicit and
   audited before training.
5. `scope` does not control facts/phase/route masks. `target_track_id`,
   `prompt_bbox`, target evidence and outcome labels are not consumed by M29;
   candidate selection survives only as visibility plus target bbox. These
   fields are currently provenance/audit data, not model supervision.
6. Full readiness checks require nonzero navigation/phase/decoding/Q, all five
   phases and decoding +/- only. They do not require Q positive+negative,
   visibility/bbox/steering/path coverage or class diversity, so a materially
   undersupervised state bottleneck can pass the preflight.
7. M29 uses uniform sampling and unweighted BCE/CE despite Q and action/phase
   imbalance. It always clips gradient norm to 1.0, records the pre-clip value,
   but has no peak-gradient rejection or per-loss/per-module attribution gate.
   This can conceal the same scale problem that invalidated earlier m28 smokes.
8. Evaluation omits visibility, bbox, steering and path accuracy, confusion/
   balanced metrics, joint move+camera correctness and per-phase action safety.
   State interventions cover only phase and decoding and have no acceptance
   threshold, so the current evaluator cannot certify the stated hierarchy.
9. Navigation loss backpropagates through continuous soft fact/decision
   probabilities. This proves the state predictions affect actions, but also
   permits those probabilities to carry an unlabeled analog shortcut when
   state supervision is sparse. Semantic state accuracy and intervention gates
   are required before calling the bottleneck reliable.
10. Current imports are all `train` in one `20260908_same_scene_pilot` group;
    no independent validation/generalization split exists. Human-event Q
    feedback also differs from runtime's immediate learned prompt-Q execution
    distribution and must be audited before using it as a top-level cue.

Required next sequence: finish browser review with all five deployed phases and
decoding positive/negative coverage; repair the Q-to-full inheritance/merge,
phase mask, age semantics and readiness/evaluation gates; create independent
scenario-group validation data; then run small-set fitting before any foreground
GPU smoke. Deployment remains fail-closed and no training/test/live-input run
was performed for this review.

**2026-09-09 V6 browser annotation completed:** user explicitly requires
browser editing (not manual CSV) and one editable CSV inside each raw session.
The only active workbench is V6; legacy actions/intents/chunks/camera-control
routes, Python façade methods and SessionStore legacy reads are removed.
The webpage now displays current/past frames, selectable annotation boxes,
phase/decoding/target/path/action forms, save draft, complete+next, progress,
whole-project checks and exports. Save validates fields and completed-row
semantics with the same apply_review_row as export, uses CSV SHA256 revision,
atomic replace, and review_history snapshots/audit. User operates the browser;
navigation_review_with_state.csv is persistence only.

Canonical editable files now live at
data/raw_sessions/<session>/navigation_review_with_state.csv for all FOUR
sessions: 20260908_205631_985816 (21 rows), 20260908_210127_149560 (25),
20260908_210720_785351 (27), 20260908_210912_841524 (28). Existing imported
tables are preserved snapshots. No original frame/event/mouse/meta or source
X-AnyLabeling JSON changed. The current browser snapshot is 14/101 reviewed;
the rest remain pending until the user completes them in the workbench.
The fourth session was prepared/imported for the requested per-session table;
total visual data is now 366 frames / Q48 / NO_Q318. The older merged 285-frame
Q pilot is preserved and not silently replaced.

Documentation: docs/tools/data-workbench-v6.md rewritten around browser use,
session-local CSV storage, all fields/examples and script dependencies.
43 focused tests passed, plus Node syntax/Python compile/diff checks.
Isolated browser session verified draft save, reload recovery, completion
progress/next, invalid completion rejection, click-to-select target box and
all-frame Q export. No user navigation labels were modified by tests.
Repository smoke_test also passed. The isolated test service on 8767 was
stopped; the real V6 workbench is running hidden on 127.0.0.1:8765 (Python
PID 33236 at handoff, logs tmp/workbench_v6_stdout.log and stderr.log).
The real four-session editor was opened and visually verified without edits.
Audit summary: reports/mvp_v6_design/workbench_browser_20260909.json.

**M29 architecture is implemented:** `idv_agent/model/state_guided_action.py`
defines **M29 State-Guided Action Network (SGAN)**. The chain is visual
features/times → predicted facts (visibility/bbox/prompt/decoding) → predicted
top-level phase/steering/path plus execution feedback → soft beliefs → navigation
move/camera. `navigate_from_beliefs` accepts only predicted beliefs, so there is
no direct visual/latent shortcut below the top-level state. The latest-frame
prompt logit is the actual Q output and remains independent of phase/decoding/
previous Q, preserving the required prompt→Q fast loop.

Added M29 implementation: `training/m29_dataset.py`, `training/m29_loss.py`,
`training/m29_features.py`, `model/m29_checkpoint.py`, `scripts/train_m29.py`,
`scripts/evaluate_m29.py`, `agent/m29_policy.py`, and `scripts/run_m29.py`.
The runtime uses 20 Hz capture, latest-frame slot, one visual worker, 50 Hz
wall-clock command merger, result-id idempotency, and actual Q submission
feedback. The current checkpoint contract is `m29.state_guided_action.v2` and
fail-closes old
m28/v5 weights.

M29 `--task q` is runnable on the current 366-frame v2 export once a Qwen model
path is supplied; `--task full` rejects before model load until reviewed
navigation, all five phases, and decoding true/false coverage exist. The
current pilot has 366 Q samples (48 positive/318 negative), same-scene train
only, and no generalization split; it cannot authorize deployment.

**V6 Data Workbench completed:** `DataWorkbench` now exposes
`mvp_v6_summary`, `import_mvp_v6`, and `review_mvp_v6`; the HTTP service adds
GET/POST `/api/sessions/<name>/mvp-v6` routes. Full usage and field examples
are documented in `docs/tools/data-workbench-v6.md`. The workbench keeps raw
and X-AnyLabeling evidence immutable and writes versioned import/review outputs.
Regression: 33 workbench/import/M29 tests passed; py_compile and diff check
passed. The current UI remains a raw/v5 browser; V6 semantic editing uses the
CSV review file, so no duplicate browser label vocabulary was introduced.

Structural validation: 40 focused M29/prompt-Q/import tests passed plus the
repository smoke_test; `git diff --check` passed. This is an architecture and
pipeline result, not a trained M29 result. No M29 GPU training or real-input
run was started. The next executable model step is a foreground Q-only pilot
using `scripts/train_m29.py`; full navigation waits for the 80-endpoint state
and action review.

**Current pipeline audit (2026-09-08):** validate_vla_raw remains required and
is called by prepare_mvp_v6.prepare. Legacy extract/per_frame_actions.csv and
build_vla_chunks are not prerequisites of v6: replay_window directly
integrates raw events/mouse over timestamped 200 ms intervals; prepare/import/
review/export assemble v6 targets. Legacy builder only supports v3/v4/v5
(default v3), not the new contract. All three source sessions pass raw
validation; imported workspaces pass raw-hash/replay/timing checks without
any per_frame_actions.csv or old chunk files. No labels or runtime changed.
80 navigation/current-state endpoints remain pending; Q remains 36/249.
Updated docs/tools/data-collection-tool.md and docs/03 §0.12 to prevent
repeating the obsolete three-step pipeline. New trainer/runtime and measured
observation-action delay still need integration; old extract/build cannot
substitute for that work.

**Latest hierarchy clarification:** user requires top-level search/approach/
align/interact/maintain_decode decisions to guide actions, including awareness
of actual decoding and actual Q execution. Keep the prompt->Q learned fast
response; it is part of the hierarchy, not a replacement for state estimation.
Task=decipher, observed facts.decoding, and chosen decision.phase are distinct.
Current decoding is required supervision for the complete top-level controller,
while future outcome confirmation remains unnecessary for standalone Q data.
Executed Q feedback means an interaction attempt, not visual decoding truth.

Design contract in docs/03 §0.11: visual state -> learned top-level phase/
target/steering -> conditioned navigation actions, plus fast visual Q branch;
one command merger, actual command/timestamp/status feedback to top level,
then new images update decoding belief. No second competing Q sender. Q
targets remain interact_prompt regardless of phase; top-level stale state
must not veto the prompt response. Old m28 consumes soft predicted beliefs
but has no decoding output head or this execution-feedback control path.

Implemented data gap repair: CSV import accepts decoding=true/false/unknown
and decoding_evidence_frames referring only to current/past observations;
full export reports decoding positive/negative/unknown coverage. Added each
workspace's navigation_review_with_state.csv as a copy of existing edits,
leaving the original CSV and source evidence untouched. All 80 new state
entries remain unknown until reviewed. Existing Q labels remain 36/249.
21 focused tests passed (including maintain_decode with valid positive Q and
future-state evidence rejection); git diff --check passed. No model head,
runtime controller, checkpoint or training run changed in this clarification.

**Latest completed annotation import:** user explicitly confirmed X-AnyLabeling
finished. All 285 frames have sidecars (86/100/99); 334 rectangle shapes pass
schema, class, finite bounds, dimension and image decoding checks. Q labels
are 36 positive / 249 negative. User confirmation is completion provenance;
checked=false/auto-label scores are preserved, not called an independent
visual accuracy review. No missing sidecars were converted to negatives.

Added import_mvp_anylabeling import/review. Current workspaces are
data/annotations/mvp_v6/<session>_xany_import_v1, with byte-preserved source
annotation snapshots, visual_frames.jsonl, prompt_q_all_frames.jsonl,
decisions.jsonl and navigation_review.csv. All-frame Q data is ready for a
pilot fitting diagnostic at data/derived/prompt_q/20260908_xany_pilot_v1/train.jsonl.
All three are assigned train/same-scene pilot group; validation=0, no
generalization claim. Report: reports/mvp_v6_design/xany_import_summary.json.

Next MVP annotation work: review 80 navigation endpoints (25/27/28): phase,
selected target candidate/track, steering/path, accept/correct/exclude replay
move+camera. Multiple candidates remain intact; no largest-box goal rule.
The review CLI writes a new workspace and full v6 export when all rows are
reviewed. Existing visual boxes/Q labels do not need reannotation. New v6
trainer/runtime integration remains outstanding; no training run was started.
Validation: 19 focused importer/prompt-Q/v6 tests and repository smoke_test
passed under idv312; git diff --check passed. All 80 decision endpoints have
reviewed prompt labels (14 positive/66 negative); all-frame Q export preserves
the full 36/249 distribution. These counts are annotation coverage, not model
performance. Original raw hashes and earlier workspaces remain unchanged.

**Latest user correction — Q contract:** for each current observed frame,
interact_prompt=true MUST supervise Q, false MUST supervise NO_Q. Short
consecutive triggering is allowed. Replay Q timing, prior Q, phase, stopped
movement/camera and decoding outcomes must not veto that label. Unknown UI
visibility is masked. The three supplied sessions ARE usable for this Q
subtask without a longer post-Q tail. Decoding confirmation is optional audit,
not a prerequisite for prompt-Q training or acceptance. This overrides prior
draft claims about prompt-visible/Q-negative waiting periods or one Q per UI.

Implemented idv_agent/vla/prompt_q.py contract, independent Q-logit BCE in
training/prompt_q_loss.py, interact_prompt-only review/export-q, and separate
navigation/Q masks in full export. Neither loss nor new schema is wired to
legacy train_vla/run_agent yet. New protocol id is
m28_sparse_visual_v6_prompt_q_v1. Current workspaces are
data/annotations/mvp_v6/<session>_prompt_q (25/27/28 endpoints, all pending),
migrated from preserved _v2 drafts with old review/action Q retained in audit.
Current-frame Q export needs no navigation phase, history warmup, replay
action window, target box/association or outcome. Raw files are unchanged.
Execution-message deduplication does not mean suppressing Q from consecutive
new prompt-positive observations. No trained-model success is asserted.

Verification: 51 focused tests passed in idv312, including consecutive Q
positives, no-prompt negatives despite human Q, Q-only export at truncated
tails/short history, full export with excluded navigation, unknown masks,
immutable raw migration, and gradient flow into the actual m28 h0 Q output.
All three _prompt_q workspaces pass raw-hash/structure validation; 80 prompt
annotations remain pending. git diff --check passed.

The paragraphs below describe the earlier design iteration; the explicit
prompt-Q correction above supersedes its Q gating and outcome prerequisites.

User requested a complete raw-to-training annotation redesign, one top-level
decipher task, 20 FPS capture, and supplied a ~6 FPS Qwen vision baseline.
The complete current design is docs/03-数据格式.md §0. This supersedes prior
v5 new-data instructions; all old raw/checkpoints remain historical evidence.

Implemented: recorder defaults to 20 FPS/data/raw_sessions, preserves idle
boundaries by default and records two seconds after F9 stop; explicit legacy
trim remains optional. Added prepare_mvp_v6 prepare/validate/export with
immutable raw hashes, timestamp-derived replay, review-only corrections,
field masks, phase/fact/action/outcome separation. Output is
idv.mvp_training.v6, deliberately incompatible with legacy train_vla.
Added CPU LatestObservationSlot/ObservationActionClock: bounded backlog,
one result consumed once, next actionable capture after the prior macro ends.
These timing primitives are NOT yet wired to ACTPolicy/run_agent.

Design: retain Qwen + m28 shared state, ≤5 Hz visual worker and 20 Hz capture;
3–8 encoded frames ~200 ms apart; 200 ms action on an independent wall clock.
Strict post-action observation makes actual closed-loop decisions slower
than visual throughput (~2–2.7 Hz estimate, not measured). No repeated Q or
camera commands from reused visual evidence. Planned model changes: fixed
task condition (no intent CE), six phase decisions, decoding state, actual
history times/age, and last EXECUTED Q context. Current model/train/runtime
still use v5; migration/retraining are outstanding, no readiness claim.

Current annotation workspaces (all pending) are
data/annotations/mvp_v6/{20260908_210127_149560,20260908_210720_785351,
20260908_210912_841524}_v2 with 25/27/28 endpoints. Initial directories without
_v2 are preserved draft backups (tail flags used a frame-count approximation).
Raw sessions were untouched. The three post-Q tails are only
144.274/151.103/200.599 ms, insufficient for stable decoding evidence.
Early turning may lack pre-action history because of previous input trimming.

Camera audit: reports/mvp_v6_design/camera_audit.json, two supplied sessions,
four 200 ms window phases. Retain 0/±25/±110 counts provisionally; shrinking
fine to ±12 worsens reconstruction, adding ±8 gives limited improvement,
adding ±200 mainly fits one sweep. This is not held-out control validation.

Environment correction: explicit idv312 Python reports torch 2.6.0+cu124 and
CUDA available; prior no-CUDA statement applied to the default Python only.
No new GPU smoke or real-input experiment was run in this redesign turn.

Validation for this design/tooling change: 73 focused tests passed under
idv312 (annotation preparation/export, raw preservation, replay immutability,
Q outcome rejection, timing clock, existing model/runtime regressions), plus
the repository smoke_test passed. Current _v2 workspaces all pass structural
and raw-hash validation with 0 reviewed/80 pending; this is NOT a data-ready
gate or model-performance result. git diff --check passed.

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

## Data Capture Fix (2026-09-08)

Implemented boundary-idle cleanup in `idv_agent/scripts/record_vla.py`.
When an F9 recording segment ends, the recorder now removes leading/trailing
frames outside the first/last real input timestamp, keeps middle idle spans,
clips `events.csv` and `mouse_deltas.csv` to the input interval, renumbers the
retained frame files and timestamps, and updates `meta.json` with
`idle_boundary_trimmed` and `had_input`. A segment with no non-F9 keyboard,
mouse-button, scroll, or nonzero Raw Input motion is deleted instead of entering
the dataset. Existing raw sessions are untouched.

Focused validation: `23 passed` (`tests/test_vla_raw_recording.py`,
`tests/test_record_vla_controls.py`, `tests/test_smoke.py`) using the idv312
interpreter. Pytest emitted only the known cache-directory permission warning.

## Data Capture Follow-up (2026-09-08)

Additional capture hardening landed in `idv_agent/scripts/record_vla.py` and
`idv_agent/capture/input_logger.py`: serialized event timestamp/write ordering,
failed JPEG write detection, sane single-frame effective FPS, deterministic
post-trim event ordering, and cleanup-safe listener startup. Focused capture
suite remains 23 passed under idv312. Remaining review items are Raw Input
foreground filtering, explicit listener/permission health reporting, window
region fallback policy, and backoff on repeated empty screen grabs.

## Data Collection Tool Documentation (2026-09-08)

Added `docs/tools/data-collection-tool.md`. It documents the current
`record_vla`, `validate_vla_raw`, `extract`, `build_vla_chunks`, and
`diagnose_raw_mouse` commands, their script dependencies, session layout,
field definitions, and concrete JSON/CSV examples. It intentionally records
known capture risks as outside the current tool scope; annotation-stage
filtering remains the chosen handling for now.

## Replay Tool Documentation (2026-09-08)

Added `docs/tools/replay-tool.md`. It documents the current raw-input replay
command, dry-run and `--send-input` behavior, parameters, dependency chain,
CSV field examples, timestamp normalization, keyboard de-duplication, mouse
scaling, and the distinction between input replay and visual playback.

## Q 交互阶段修复（2026-09-10）

完成 M29 `q_interact` 专项微调，初始化自 `checkpoints/m29_full_newdata_diag1000_20260910/m29.pt`，仅更新交互提示、Q 事件与阶段判定相关头，冻结导航/相机主干，避免 Q 修复破坏已有动作能力。训练命令为 300 steps、batch 8、FP16 CUDA，输出位于 `checkpoints/m29_q_interact_fix300_20260910/`。

验证证据：验证集 Q 正类召回率从全量训练的 9.5% 提升至 100%（21 个正样本全部命中）；`interact` 阶段召回率为 86.4%；move/camera 指标保持在微调前相近水平。checkpoint 已在 CPU 独立加载成功，manifest 为 `task=q_interact`、`state_bottleneck=soft_continuous_geometry_v2`、`deployable=false`。现有 M29 测试 28 项通过，compileall 与 `git diff --check` 通过。

限制：当前 train/val 按 session 切分但仍共享 `scenario_group=decliper`，且验证集 Q 正样本仅 21 个；本结果是 Q 交互回归修复证据，不是独立场景部署门禁。下一步应采集/划分不重叠场景后复测，再把 q_interact 头合入 full 训练并做端到端 ACT 冒烟。

## scenario_group 标注修正（2026-09-10）

用户确认每个 session 都来自独立对局；此前合并训练/验证文件把 `scenario_group` 错填为统一值（`decliper`/`20260908_same_scene_pilot`），造成审计错误地报告场景组重叠。已生成修正版：

- `data/derived/m29/20260910_all_new_train_v2_session_groups.jsonl`
- `data/derived/m29/20260910_all_new_val_v2_session_groups.jsonl`

修正版将每行 `scenario_group` 设为该行 session ID。训练集 1008 行、21 个 session/组；验证集 213 行、4 个 session/组；session/group 交集为空。M29Dataset 加载、full readiness、覆盖率与类别分布均通过，未修改图像、动作或监督字段。

此前 Q 修复结论中的“共享 scenario_group”限制已撤销；当前验证可以视为 session 独立切分的结果。仍需注意验证集 Q 正样本为 21 个，统计置信度有限。

## M29 视觉依赖门禁（2026-09-10）

针对 M29 checkpoint `checkpoints/m29_full_newdata_diag1000_20260910/m29.pt`，使用 `data/derived/m29/20260910_all_new_val_v2_session_groups.jsonl` 的 213 个验证样本执行视觉依赖评估，报告位于 `reports/visual_dependency_gate_m29_full_newdata_20260910.json`。

结果：图像置空导致 phase/move/camera 输出改变率分别为 37.1%/32.9%/92.5%/6.1%；图像打乱为 9.4%/10.8%/22.5%/4.7%；历史清空为 37.1%/32.9%/92.5%/6.1%。这证明 M29 输出存在视觉依赖，也没有表现为完全固定的搜索/靠近动作。但 Q 合约检查未通过（213 个已知 Q 行中存在预测与标签不一致），bbox 改变实验尚未接入当前 CLI，因此视觉依赖总门禁为 FAIL，checkpoint 不得放行部署。

本次还修复了评估脚本的三个执行问题：slow pass 未定义、图像 shuffle 未同步时间窗口、M29 prompt logit 字段名错误，并增加 M29 checkpoint 分支。下一步必须先修复 full checkpoint 的 Q 头/训练目标，再补充 bbox intervention 后重新执行门禁。

## M29 Q 交互 checkpoint 视觉依赖复测（2026-09-10）

正式候选改为 `checkpoints/m29_q_interact_fix300_20260910/m29.pt`，在完整 213 行、4 个独立 session 的验证集上复测。报告：`reports/visual_dependency_gate_m29_q_interact_fix300_20260910.json`。

视觉干预：image_zero 的 phase/move/camera_dx/camera_dy 改变率为 49.8%/32.9%/92.5%/4.7%；image_shuffle 为 14.1%/9.9%/17.4%/3.8%；history_zero 为 49.8%/32.9%/92.5%/4.7%。bbox 归一化 x/y 平移 +0.35 后，move/camera_dx/camera_dy 平均 logit 改变量为 0.825/0.731/0.481，argmax 改变率为 5.1%/11.6%/1.9%，说明导航确实使用目标几何。

Q：21 个正样本全部命中，正样本召回率 100%；但仍有 11 个负样本误触发，false-trigger rate 5.73%，precision 65.6%，所以尚未满足“无 interact_prompt 不按 Q”的硬门禁。总视觉依赖门禁仍为 FAIL。下一步应针对 q_interact checkpoint 继续降低 Q false positive，再重新跑同一报告。

## M29 Q checkpoint ACT dry-run（2026-09-10）

按用户决定，先放行 `checkpoints/m29_q_interact_fix300_20260910/m29.pt` 的 Q 门禁，接入 M29 runtime 做离线 dry-run。命令通过 `idv_agent.scripts.run_m29` 执行，未使用 `--send-input`，输出：`reports/m29_q_interact_dryrun_20260910.jsonl`。

5 秒 dry-run 结果：捕获 94 帧、编码 22 帧、空抓取 0、推理线程正常退出、错误 0；由于推理吞吐低于 20 FPS，覆盖 71 帧，产生 20 个决策和 1 个过期结果。Q 触发 0 次；阶段输出主要为 `interact`（18）和 `search`（2）。命令合并器、执行反馈、过期结果处理和 shutdown 链路均正常，未发送真实键鼠输入。

注意：该 checkpoint manifest 的 task 是 `q_interact`，因此 dry-run 中 navigation_enabled=false；这次验证的是 Q/交互 runtime 接入，不代表 full 导航 runtime 已放行。下一步若要验证搜索/靠近/对齐，需要使用 task=full 的 M29 checkpoint 或重新导出合并 checkpoint。

## M29 接入 run-agent（2026-09-10）

`idv_agent/scripts/run_agent.py` 新增 `--mode m29`，统一接入 M29 runtime。该模式要求 `--act-checkpoint`、`--model-path`、`--trajectory-log`，默认 dry-run；内部复用 `run_m29.run`，仍禁止真实输入除非显式传 `--send-input`，并保留 M29 checkpoint 的 deployable 门禁。

已执行：
`python -m idv_agent.scripts.run_agent --mode m29 --act-checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt --model-path C:\Users\kiran\.cache\modelscope\models\Qwen--Qwen3-VL-4B-Instruct\snapshots\master --device cuda --duration 5 --trajectory-log reports/run_agent_m29_q_interact_dryrun_20260910.jsonl`

结果：5 秒捕获 95 帧、编码 22 帧、空抓取 0、推理线程正常退出、错误 0、trace 21 条事件。`tests/test_configs.py -k run_agent`：2 passed。

用法示例：
`conda run -n idv312 python -m idv_agent.scripts.run_agent --mode m29 --act-checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt --model-path C:\\Users\\kiran\\.cache\\modelscope\\models\\Qwen--Qwen3-VL-4B-Instruct\\snapshots\\master --duration 5 --trajectory-log reports/run_agent_m29_q_interact_dryrun_20260910.jsonl`

当前 checkpoint 是 `task=q_interact`，所以 M29 runtime 的导航输出被关闭，只验证 Q/交互全链路。要验证搜索、靠近、对齐和交互合并动作，需要生成 `task=full` 的 M29 checkpoint；`--send-input` 仍会被 manifest deployable=false 拒绝。

## M29 q_interact 全功能 run-agent dry-run（2026-09-10）

为准备沙盒测试，`run-agent` 与 `run_m29` 新增 `--enable-navigation`。该开关允许 `task=q_interact` checkpoint 在 dry-run 中同时输出导航和 Q；不改变 checkpoint 的 `deployable` 状态，也不绕过 `--send-input` 门禁。

执行命令：
`conda run -n idv312 python -m idv_agent.scripts.run_agent --mode m29 --act-checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt --model-path C:\\Users\\kiran\\.cache\\modelscope\\models\\Qwen--Qwen3-VL-4B-Instruct\\snapshots\\master --device cuda --duration 8 --enable-navigation --trajectory-log reports/run_agent_m29_full_dryrun_20260910.jsonl`

结果：捕获 154 帧、编码 34 帧、空抓取 1、推理线程正常退出、错误 0；生成 32 个 decision、11 个 action_end、1 个 stale_result；11 个决策实际应用导航，Q 触发 0 次。阶段分布为 interact 27、search 4、align 1。真实输入未发送。

沙盒前结论：run-agent 全功能 dry-run 链路可用，导航与 Q 通道均已接入；当前吞吐约 4.25 次 M29 推理/秒，20 FPS 采集会覆盖大量中间帧。进入真实沙盒前必须保留短时、可回收、F12/手动停止和默认 dry-run；`--send-input` 仍受 manifest `deployable=false` 拦截。

## 标准模式真实输入开关（2026-09-10）

`run-agent --mode m29` 与 `run_m29` 新增 `--allow-undeployed-send-input`。只有同时提供 `--send-input --allow-undeployed-send-input` 时，当前 `deployable=false` 的 M29 checkpoint 才会进入标准模式真实输入路径；单独 `--send-input` 仍 fail-closed，dry-run 默认不变。该开关仅解除模型 readiness 门禁，不改变 Q/导航逻辑。

已验证：不带额外开关的 `--send-input` 被正确拒绝；`tests/test_configs.py -k run_agent` 2 passed；py_compile 与 diff check 通过。未在本次会话启动真实键鼠注入。

标准模式沙盒命令（必须确认游戏已在官方自定义剧本/训练模式，且终端具备管理员权限）：
`conda run -n idv312 python -m idv_agent.scripts.run_agent --mode m29 --act-checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt --model-path C:\\Users\\kiran\\.cache\\modelscope\\models\\Qwen--Qwen3-VL-4B-Instruct\\snapshots\\master --device cuda --duration 10 --enable-navigation --send-input --allow-undeployed-send-input --trajectory-log reports/run_agent_m29_sandbox_20260910.jsonl`
