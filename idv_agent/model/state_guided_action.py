"""M29 State-Guided Action Network (SGAN).

Visual facts -> top-level decision -> navigation, with no latent shortcut to
navigation. The latest-frame prompt logit is the sole Q output, and is also
observed by the top level. Ground-truth labels are never forward arguments.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

M29_SCHEMA = "m29.state_guided_action.v3"
PHASES = ("search", "approach", "align", "interact", "maintain_decode")
# "unknown" is an annotation mask, not a deployable semantic class. Keeping it
# as a logit without positive supervision would create a random downstream state.
VISIBILITY = ("visible", "highlight_only", "absent", "occluded")
STEERING = ("target_center", "path_follow", "search_sweep", "hold")
PATHS = ("direct", "detour_left", "detour_right", "blocked")
CAMERA_COMMANDS = (-110, -25, 0, 25, 110)
FACT_DIM = len(VISIBILITY) + 4 + 4 + 1 + 1
BELIEF_DIM = FACT_DIM + len(PHASES) + len(STEERING) + len(PATHS)


def _categorical_belief(logits: torch.Tensor) -> torch.Tensor:
    """Continuous semantic belief used by navigation.

    Navigation needs uncertainty to survive the state-to-action boundary. A
    straight-through argmax made the deployed value one-hot and discarded
    useful alternatives whenever a fact was ambiguous.
    """
    return logits.softmax(-1)


def _binary_belief(logits: torch.Tensor) -> torch.Tensor:
    return logits.sigmoid()


@dataclass
class PredictedFacts:
    visibility_logits: torch.Tensor
    bbox: torch.Tensor
    prompt_bbox: torch.Tensor
    decoding_logits: torch.Tensor
    prompt_logits: torch.Tensor

    def probabilities(self) -> torch.Tensor:
        # Navigation sees probability-valued semantic state and continuous
        # geometry. Do not hard-gate geometry by a potentially wrong visibility
        # argmax: uncertainty is part of the learned condition.
        visibility = _categorical_belief(self.visibility_logits)
        prompt = _binary_belief(self.prompt_logits)[:, None]
        return torch.cat((visibility, self.bbox, self.prompt_bbox,
                          _binary_belief(self.decoding_logits)[:, None], prompt), dim=-1)


@dataclass
class PredictedDecision:
    phase_logits: torch.Tensor
    steering_logits: torch.Tensor
    path_logits: torch.Tensor

    def probabilities(self) -> torch.Tensor:
        return torch.cat((_categorical_belief(self.phase_logits),
                          _categorical_belief(self.steering_logits),
                          _categorical_belief(self.path_logits)), dim=-1)


@dataclass
class NavigationOutput:
    move_logits: torch.Tensor
    camera_dx_logits: torch.Tensor
    camera_dy_logits: torch.Tensor


@dataclass
class StateGuidedOutput:
    facts: PredictedFacts
    decision: PredictedDecision
    navigation: NavigationOutput
    beliefs: torch.Tensor
    top_level_context: torch.Tensor

    @property
    def q_logits(self) -> torch.Tensor:
        # No second Q classifier can disagree with the visual prompt output.
        return self.facts.prompt_logits


class StateGuidedActionNetwork(nn.Module):
    architecture_name = "M29 State-Guided Action Network"

    def __init__(self, feature_dim: int = 4096, hidden_dim: int = 128):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 4:
            raise ValueError("invalid M29 feature/hidden dimensions")
        self.feature_dim, self.hidden_dim = feature_dim, hidden_dim
        self.register_buffer("feature_center", torch.zeros(feature_dim))
        self.visual_norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        self.visual_projection = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
        # Only observation-relative capture times are model inputs. Runtime
        # inference latency is enforced by the stale-result gate and traced,
        # rather than becoming a hardware-speed shortcut in the policy.
        self.time_projection = nn.Linear(1, hidden_dim)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.visibility = nn.Linear(hidden_dim, len(VISIBILITY))
        self.bbox = nn.Linear(hidden_dim, 4)
        self.decoding = nn.Linear(hidden_dim, 1)
        # Current-image-only fast response: no timestamp, previous Q or phase.
        self.prompt_trunk = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.prompt_q = nn.Linear(hidden_dim, 1)
        self.prompt_bbox = nn.Linear(hidden_dim, 4)
        self.task = nn.Parameter(torch.zeros(8))  # fixed task=decipher; no intent classification
        # Local obstacles cannot be reconstructed from one cipher bbox alone.
        # Visual context is available to the TOP level for route selection,
        # but never to the downstream navigation action decoder.
        self.decision_trunk = nn.Sequential(nn.Linear(FACT_DIM + 2 + 8 + hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.phase = nn.Linear(hidden_dim, len(PHASES))
        self.steering = nn.Linear(hidden_dim, len(STEERING))
        self.path = nn.Linear(hidden_dim, len(PATHS))
        # Keep the semantic belief as the mandatory condition, but retain the
        # continuous predicted-state representation that produced it. The
        # previous version discarded this 128-D state and left navigation with
        # only 27 low-bandwidth values, making target geometry/action history
        # unrecoverable after state prediction.
        action_input_dim = BELIEF_DIM + hidden_dim
        self.navigation_trunk = nn.Sequential(nn.Linear(action_input_dim, hidden_dim), nn.GELU(),
                                             nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.camera_trunk = nn.Sequential(nn.Linear(action_input_dim, hidden_dim), nn.GELU(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.geometry_trunk = nn.Sequential(nn.Linear(8, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.geometry_move = nn.Linear(hidden_dim, 9)
        self.geometry_camera_dx = nn.Linear(hidden_dim, len(CAMERA_COMMANDS))
        self.geometry_camera_dy = nn.Linear(hidden_dim, len(CAMERA_COMMANDS))
        # Direct coordinate path prevents the geometry MLP from becoming an
        # optional weak residual that a temporal memorization path can ignore.
        self.geometry_direct_move = nn.Linear(8, 9)
        self.geometry_direct_camera_dx = nn.Linear(8, len(CAMERA_COMMANDS))
        self.geometry_direct_camera_dy = nn.Linear(8, len(CAMERA_COMMANDS))
        self.move = nn.Linear(hidden_dim, 9)
        self.camera_dx = nn.Linear(hidden_dim, len(CAMERA_COMMANDS))
        self.camera_dy = nn.Linear(hidden_dim, len(CAMERA_COMMANDS))

    def decide_from_facts(self, facts: PredictedFacts, execution_feedback: torch.Tensor,
                          top_level_context: torch.Tensor | None = None) -> PredictedDecision:
        probabilities = facts.probabilities()
        if execution_feedback.shape != (probabilities.shape[0], 2):
            raise ValueError("execution_feedback must be [B,2]: Q sent validity, normalized age")
        if not bool(torch.isfinite(execution_feedback).all()) or bool(((execution_feedback < 0) | (execution_feedback > 1)).any()):
            raise ValueError("execution feedback must be finite in [0,1]")
        if top_level_context is None:
            top_level_context = probabilities.new_zeros((probabilities.shape[0], self.hidden_dim))
        if top_level_context.shape != (probabilities.shape[0], self.hidden_dim):
            raise ValueError("invalid top-level visual context")
        hidden = self.decision_trunk(torch.cat((probabilities, execution_feedback.to(probabilities),
                                                self.task[None].expand(probabilities.shape[0], -1), top_level_context), dim=-1))
        return PredictedDecision(self.phase(hidden), self.steering(hidden), self.path(hidden))

    def navigate_from_beliefs(self, beliefs: torch.Tensor,
                              predicted_state: torch.Tensor | None = None) -> NavigationOutput:
        """Navigate from semantic belief plus continuous predicted state.

        ``predicted_state`` is the shared temporal state used to produce the
        predicted facts/decision; it is not a raw visual-feature shortcut.
        A zero context keeps the intervention/test API usable for belief-only
        probes.
        """
        if beliefs.ndim != 2 or beliefs.shape[-1] != BELIEF_DIM:
            raise ValueError(f"beliefs must be [B,{BELIEF_DIM}]")
        if predicted_state is None:
            predicted_state = beliefs.new_zeros((beliefs.shape[0], self.hidden_dim))
        if predicted_state.shape != (beliefs.shape[0], self.hidden_dim):
            raise ValueError("predicted_state must match [B,hidden_dim]")
        # Prevent the continuous temporal state from becoming a memorization
        # shortcut. During training it is partially dropped, forcing action
        # heads to use the semantic belief and explicit target geometry.
        if self.training:
            # Deterministic attenuation keeps padded-history invariance and
            # makes the regularizer reproducible for the small overfit gate.
            predicted_state = predicted_state * 0.65
        action_input = torch.cat((beliefs, predicted_state), dim=-1)
        move_hidden = self.navigation_trunk(action_input)
        camera_hidden = self.camera_trunk(action_input)
        geometry_hidden = self.geometry_trunk(torch.cat((beliefs[:, 4:8], beliefs[:, 8:12]), dim=-1))
        geometry_scale = 2.0
        direct_geometry = torch.cat((beliefs[:, 4:8], beliefs[:, 8:12]), dim=-1)
        return NavigationOutput(
            self.move(move_hidden) + geometry_scale * self.geometry_move(geometry_hidden)
            + 10.0 * self.geometry_direct_move(direct_geometry),
            self.camera_dx(camera_hidden) + geometry_scale * self.geometry_camera_dx(geometry_hidden)
            + 10.0 * self.geometry_direct_camera_dx(direct_geometry),
            self.camera_dy(camera_hidden) + geometry_scale * self.geometry_camera_dy(geometry_hidden)
            + 10.0 * self.geometry_direct_camera_dy(direct_geometry))

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor,
                relative_times_s: torch.Tensor,
                execution_feedback: torch.Tensor) -> StateGuidedOutput:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim or not 1 <= features.shape[1] <= 8:
            raise ValueError("features must be [B,1..8,feature_dim]")
        batch, steps = features.shape[:2]
        if valid_mask.shape != (batch, steps) or relative_times_s.shape != (batch, steps):
            raise ValueError("invalid M29 temporal input shape")
        valid = valid_mask.bool()
        lengths = valid.sum(-1)
        if not bool((lengths > 0).all()) or not torch.equal(valid, torch.arange(steps, device=features.device)[None] < lengths[:, None]):
            raise ValueError("M29 expects right-padded nonempty visual history")
        times = relative_times_s[valid]
        if not bool(torch.isfinite(features[valid]).all()) or not bool(torch.isfinite(times).all()) or bool((times > 0).any()):
            raise ValueError("history features/times must be finite and causal")
        rows = torch.arange(batch, device=features.device)
        if not bool((relative_times_s[rows, lengths-1] == 0).all()):
            raise ValueError("latest frame relative time must be zero")
        if steps > 1 and bool(((relative_times_s[:, 1:] <= relative_times_s[:, :-1]) & valid[:, 1:]).any()):
            raise ValueError("valid history times must strictly increase")
        # Head arithmetic stays FP32 on Turing; only frozen Qwen uses FP16.
        with torch.autocast(device_type=features.device.type, enabled=False):
            safe_features = features.float().masked_fill(~valid[:, :, None], 0)
            normalized = self.visual_norm(safe_features - self.feature_center)
            latest = normalized[rows, lengths-1]
            safe_times = relative_times_s.float().masked_fill(~valid, 0)
            timing = (safe_times.clamp(-5, 0)/5)[:, :, None]
            sequence = self.visual_projection(normalized) + self.time_projection(timing)
            packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, last = self.temporal(packed)
            state = self.state_norm(last[-1])
            raw_box = self.bbox(state).sigmoid()
            xy = raw_box[:, :2]
            box = torch.cat((xy, xy + (1-xy)*raw_box[:, 2:]), dim=-1)
            prompt_hidden = self.prompt_trunk(latest)
            raw_prompt_box = self.prompt_bbox(prompt_hidden).sigmoid()
            prompt_xy = raw_prompt_box[:, :2]
            prompt_box = torch.cat((prompt_xy, prompt_xy + (1-prompt_xy)*raw_prompt_box[:, 2:]), dim=-1)
            facts = PredictedFacts(self.visibility(state), box, prompt_box,
                                   self.decoding(state).squeeze(-1),
                                   self.prompt_q(prompt_hidden).squeeze(-1))
            decision = self.decide_from_facts(facts, execution_feedback, state)
            beliefs = torch.cat((facts.probabilities(), decision.probabilities()), dim=-1)
            return StateGuidedOutput(facts, decision, self.navigate_from_beliefs(beliefs, state), beliefs, state)
