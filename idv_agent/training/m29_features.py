"""Frozen Qwen feature interface shared by M29 training and inference."""
from collections import OrderedDict

import torch
from PIL import Image


class M29FeatureEncoder:
    def __init__(self, adapter, *, cache_size=1024):
        self.adapter = adapter.eval()
        for parameter in adapter.parameters():
            parameter.requires_grad = False
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def image(self, image):
        with torch.no_grad():
            raw = self.adapter.encode_raw_frames([image], micro_batch_size=1)
            return self.adapter.project_raw_visual_features(raw).float().detach()

    def __call__(self, paths):
        outputs = []
        for path in paths:
            if path in self.cache:
                self.cache.move_to_end(path)
                feature = self.cache[path]
            else:
                with Image.open(path) as source:
                    feature = self.image(source.convert("RGB"))[0].cpu()
                self.cache[path] = feature
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            outputs.append(feature)
        return torch.stack(outputs)


def load_m29_encoder(model_path, device):
    from idv_agent.model.qwen_backbone_adapter import load_frozen_qwen3vl_backbone
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    adapter, _ = load_frozen_qwen3vl_backbone(model_path, dtype=dtype, device=device)
    return M29FeatureEncoder(adapter)
