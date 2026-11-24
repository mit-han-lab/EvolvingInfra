# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""The DeepSeek-V4 vision wrapper (models/deepseek_v4_vl.py) leaves DeepSeek-V4.1's vision tower,
which the text model builds itself, to the text model. The text model and the wrapper's own tower
are stand-ins here; needs the engine and torch."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("decoupled_engine")
vl = pytest.importorskip("decoupled_engine.srt.models.deepseek_v4_vl")
from torch import nn  # noqa: E402


class TextLM(nn.Module):
    """Like deepseek_v4.DeepseekV4ForCausalLM: a V4.1 checkpoint with vision layers gets its tower."""

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.loaded = []
        if config.model_type == "deepseek_v41" and config.vision_n_layers > 0:
            self.vision = nn.Linear(2, 2)
            self.image_start = nn.Parameter(torch.zeros(config.hidden_size))

    def load_weights(self, weights):
        self.loaded = [name for name, _ in weights]

    def pad_input_ids(self, input_ids, mm_inputs):
        return ("text model", input_ids)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, pp_proxy_tensors=None):
        return "text model forward"


@pytest.fixture
def wrapper(monkeypatch):
    monkeypatch.setattr(vl, "_DeepseekV4TextLM", TextLM)
    monkeypatch.setattr(vl, "DeepseekV4VisionTower", lambda config, prefix="": nn.Identity())
    monkeypatch.setattr(vl, "DeepseekV4VisionAligner", lambda config, prefix="": nn.Identity())

    def make(model_type):
        return vl.DeepseekV4ForCausalLM(SimpleNamespace(model_type=model_type, vision_n_layers=2, hidden_size=8))
    return make


def test_v41_keeps_the_text_models_vision_tower(wrapper):
    model = wrapper("deepseek_v41")
    assert model.in_model_vision and not model.is_multimodal
    assert model.image_start is model.language_model.image_start, "no second tower"
    model.load_weights([("vision.w", torch.zeros(1)), ("image_start", torch.zeros(8)), ("model.w", torch.zeros(1))])
    assert model.language_model.loaded == ["vision.w", "image_start", "model.w"]
    assert model.pad_input_ids([1, 2], None) == ("text model", [1, 2])
    assert model(None, None, None) == "text model forward"


def test_v4_vision_checkpoints_keep_the_wrappers_tower(wrapper):
    model = wrapper("deepseek_v4")
    assert model.is_multimodal and not model.in_model_vision
    assert "image_start" in dict(model.named_parameters())
