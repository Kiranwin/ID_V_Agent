from dataclasses import replace
import inspect

import pytest
import torch

from idv_agent.model.state_guided_action import (
    StateGuidedActionNetwork, PredictedFacts, PredictedDecision, FACT_DIM, PHASES,
)
from idv_agent.training.m29_loss import compute_m29_loss
from idv_agent.agent.m29_policy import ExecutionFeedback, M29CommandMerger, M29Decision, M29Policy
from idv_agent.agent.observation_clock import CapturedObservation


def inputs(batch=4):
    return {"features": torch.randn(batch, 4, 8), "valid_mask": torch.ones(batch, 4, dtype=torch.bool),
            "relative_times_s": torch.tensor([[-.6, -.4, -.2, 0.]]).repeat(batch, 1),
            "execution_feedback": torch.zeros(batch, 2)}


def test_m29_has_no_visual_shortcut_below_predicted_state():
    assert list(inspect.signature(StateGuidedActionNetwork.navigate_from_beliefs).parameters) == ["self", "beliefs", "predicted_state"]
    torch.manual_seed(9)
    model = StateGuidedActionNetwork(8, 16)
    args = inputs()
    first = model(**args)
    # Every downstream fact/phase can be intervened on without touching images.
    for field, count in (("phase_logits", 5), ("steering_logits", 4), ("path_logits", 4)):
        changed = replace(first.decision, **{field: torch.full((4,count), -10.)})
        getattr(changed, field)[:, 0] = 10
        nav = model.navigate_from_beliefs(torch.cat((first.facts.probabilities(), changed.probabilities()), -1), first.top_level_context)
        assert not torch.allclose(first.navigation.move_logits, nav.move_logits)
        assert not torch.allclose(first.navigation.camera_dx_logits, nav.camera_dx_logits)
    facts = replace(first.facts, decoding_logits=torch.where(
        first.facts.decoding_logits >= 0,
        torch.full((4,), -10.),
        torch.full((4,), 10.),
    ))
    changed = model.decide_from_facts(facts, args["execution_feedback"], first.top_level_context)
    assert not torch.allclose(first.decision.phase_logits, changed.phase_logits)
    nav = model.navigate_from_beliefs(torch.cat((facts.probabilities(), changed.probabilities()), -1), first.top_level_context)
    assert not torch.allclose(first.navigation.camera_dx_logits, nav.camera_dx_logits)


def test_execution_feedback_affects_top_level_but_never_fast_q():
    torch.manual_seed(11)
    model = StateGuidedActionNetwork(8, 16)
    args = inputs()
    first = model(**args)
    args["execution_feedback"] = torch.ones(4,2)
    second = model(**args)
    assert torch.equal(first.q_logits, second.q_logits)
    assert torch.equal(first.facts.decoding_logits, second.facts.decoding_logits)
    assert not torch.allclose(first.decision.phase_logits, second.decision.phase_logits)
    assert not torch.allclose(first.navigation.move_logits, second.navigation.move_logits)
    assert first.q_logits.data_ptr() == first.facts.prompt_logits.data_ptr()


def test_latest_q_is_independent_of_temporal_history():
    model = StateGuidedActionNetwork(8, 16)
    args = inputs()
    first = model(**args)
    args["features"][:, :3] += 10
    args["relative_times_s"][:, :3] *= 2
    second = model(**args)
    assert torch.equal(first.q_logits, second.q_logits)
    assert not torch.allclose(first.facts.decoding_logits, second.facts.decoding_logits)


def test_padding_is_not_visual_evidence_and_future_time_is_rejected():
    model = StateGuidedActionNetwork(8, 16)
    args = inputs(1)
    first = model(**args)
    args["features"] = torch.cat((args["features"], torch.full((1,2,8), float('nan'))), 1)
    args["valid_mask"] = torch.tensor([[1,1,1,1,0,0]], dtype=torch.bool)
    args["relative_times_s"] = torch.cat((args["relative_times_s"], torch.zeros(1,2)), 1)
    second = model(**args)
    assert torch.allclose(first.navigation.move_logits, second.navigation.move_logits)
    args["relative_times_s"][0,0] = .1
    with pytest.raises(ValueError, match="causal"):
        model(**args)


def test_navigation_gradient_reaches_decoding_and_phase_predictors():
    model = StateGuidedActionNetwork(8,16)
    output = model(**inputs())
    loss = torch.nn.functional.cross_entropy(output.navigation.move_logits, torch.tensor([0,1,2,3]))
    loss.backward()
    for head in (model.phase, model.decoding, model.steering, model.visibility):
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0


def test_learned_navigation_responds_to_phase_intervention():
    # Learn downstream semantics with identical visual facts. This is a
    # controlled architectural test, NOT real-game training/generalization.
    torch.manual_seed(4)
    model = StateGuidedActionNetwork(8,32)
    facts = PredictedFacts(torch.zeros(5,4), torch.full((5,4),.5), torch.full((5,4),.5),
                           torch.zeros(5), torch.zeros(5))
    decision = PredictedDecision(torch.eye(5)*12, torch.zeros(5,4), torch.zeros(5,4))
    beliefs = torch.cat((facts.probabilities(), decision.probabilities()), -1).detach()
    move_target = torch.tensor([0,1,7,0,0])
    camera_target = torch.tensor([4,2,1,2,2])
    params = (list(model.navigation_trunk.parameters()) + list(model.camera_trunk.parameters())
              + list(model.move.parameters()) + list(model.camera_dx.parameters()))
    optimizer = torch.optim.Adam(params,lr=.02)
    for _ in range(150):
        nav = model.navigate_from_beliefs(beliefs)
        loss = torch.nn.functional.cross_entropy(nav.move_logits,move_target)+torch.nn.functional.cross_entropy(nav.camera_dx_logits,camera_target)
        optimizer.zero_grad();loss.backward();optimizer.step()
    original = model.navigate_from_beliefs(beliefs)
    assert torch.equal(original.move_logits.argmax(-1),move_target)
    assert torch.equal(original.camera_dx_logits.argmax(-1),camera_target)
    changed = beliefs.clone()
    changed[:, FACT_DIM:FACT_DIM+5] = beliefs.roll(1,0)[:, FACT_DIM:FACT_DIM+5]
    altered = model.navigate_from_beliefs(changed)
    assert torch.equal(altered.move_logits.argmax(-1),move_target.roll(1))
    assert (altered.camera_dx_logits.argmax(-1) != camera_target).any()


def decision(sequence, capture, ready, q=True, nav=False):
    return M29Decision(sequence,capture,ready,q,1,25,0,"maintain_decode",.9,.9,nav)


def test_fast_q_repeats_and_feedback_records_actual_send_without_phase_veto():
    feedback = ExecutionFeedback(); commands=[]
    merger = M29CommandMerger(lambda items: commands.extend(items) or True, feedback)
    merger.tick(170_000_000,decision(1,0,160_000_000))
    merger.tick(200_000_000,decision(1,0,160_000_000)) # duplicate transport
    merger.tick(370_000_000,decision(2,200_000_000,360_000_000))
    merger.tick(570_000_000,decision(3,400_000_000,560_000_000,q=False))
    assert [c.kind for c in commands if c.code=="key:q"] == ["press","release","press","release"]
    assert feedback.last_q_sent_ns == 370_000_000
    assert feedback.snapshot(200_000_000)[0] == 1
    assert feedback.snapshot(100_000_000) == [0,0]
    assert feedback.snapshot(400_000_000)[0] == 1


def test_navigation_release_clock_works_without_more_model_results():
    feedback = ExecutionFeedback(); commands=[]
    merger = M29CommandMerger(lambda items: commands.extend(items) or True,feedback)
    merger.tick(170_000_000,decision(1,0,160_000_000,q=False,nav=True))
    assert merger.held == {"key:w"}
    merger.tick(370_000_000)
    assert not merger.held
    assert commands[-1].kind == "release"


def test_failed_send_is_not_successful_q_feedback():
    feedback=ExecutionFeedback()
    merger=M29CommandMerger(lambda commands: False,feedback)
    with pytest.raises(RuntimeError,match="send failed"):
        merger.tick(170_000_000,decision(1,0,160_000_000))
    assert feedback.last_q_sent_ns is None
    assert feedback.last_status == "send_failed"


def test_runtime_consumes_new_top_level_feedback_and_fast_q_while_warming():
    class Encoder:
        def image(self,image):return torch.ones(1,8)*image
    core=StateGuidedActionNetwork(8,16)
    feedback=ExecutionFeedback()
    policy=M29Policy(core,Encoder(),feedback)
    first=policy.observe(CapturedObservation(0,0,1),now_ns=170_000_000)
    assert not first.navigation_ready
    feedback.acknowledge(at_ns=170_000_000,q_sent=True,status="submitted")
    seen=[]
    hook=core.register_forward_pre_hook(lambda module,args,kwargs: seen.append(kwargs['execution_feedback'].clone()),with_kwargs=True)
    policy.observe(CapturedObservation(1,200_000_000,1),now_ns=370_000_000)
    third=policy.observe(CapturedObservation(2,400_000_000,1),now_ns=570_000_000)
    hook.remove()
    assert seen[0][0,0] == 1
    assert third.navigation_ready
