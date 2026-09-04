from __future__ import annotations


def test_base_act_loader_does_not_require_m2(monkeypatch):
    import idv_agent.scripts.train_vla as train_vla

    calls = []
    class Adapter:
        def to(self, device):
            self.device = device
            return self
    adapter_value = Adapter()
    monkeypatch.setattr(train_vla, "load_qwen3vl_backbone", lambda *args, **kwargs: calls.append((args, kwargs)) or (adapter_value, "processor"))
    adapter, processor = train_vla._load_act_base_backbone("base", dtype="fp32", device="cpu")
    assert (adapter, processor) == (adapter_value, "processor")
    assert calls[0][1]["apply_lora"] is False


def test_runtime_loader_uses_base_loader_and_never_passes_m2(monkeypatch):
    import torch
    import idv_agent.scripts.benchmark_act_realtime as realtime

    class Adapter:
        hidden_size = 4
        visual_projection = torch.nn.Linear(1, 1)
        condition_projection = torch.nn.Linear(1, 1)
        def eval(self): return self
    checkpoint = {"adapter": {"visual_projection": Adapter.visual_projection.state_dict(),
                                "condition_projection": Adapter.condition_projection.state_dict()},
                  "core": {}}
    seen = []
    monkeypatch.setattr(realtime, "_load_act_base_backbone", lambda *args, **kwargs: seen.append((args, kwargs)) or (Adapter(), "processor"))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(realtime.SharedFastSlowVLA, "load_state_dict", lambda *args, **kwargs: None)
    args = type("Args", (), {"model_path": "base", "checkpoint": "act.pt", "temporal_dim": 4})()
    realtime._load_model(args, torch.device("cpu"))
    assert seen == [(("base",), {"dtype": torch.float32, "device": torch.device("cpu")})]


def test_act_policy_checkpoint_api_does_not_require_m2_init_checkpoint():
    import inspect
    from idv_agent.agent.act_policy import ACTPolicy

    parameter = inspect.signature(ACTPolicy.from_checkpoint).parameters["init_checkpoint"]
    assert parameter.default is None
