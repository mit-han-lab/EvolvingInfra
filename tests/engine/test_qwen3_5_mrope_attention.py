# SPDX-FileCopyrightText: Copyright (c) 2026 The evolving-infra Authors
# SPDX-License-Identifier: MIT
"""Qwen3.5 full attention on CUDA rotates Q and K by their multimodal positions.

A multimodal Qwen3.5 model passes (3, T) positions (temporal, height, width); image and video
tokens differ between the three rows. Build one full-attention layer of a small config
around a stand-in attention that records Q and K, run ``self_attention`` and compare Q and K
with GemmaRMSNorm followed by the interleaved multimodal rotation, written out in torch.

Needs a CUDA GPU and the engine (decoupled_engine); skips otherwise.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
qwen3_5 = pytest.importorskip("decoupled_engine.srt.models.qwen3_5")
from decoupled_engine.srt.layers.layernorm import GemmaRMSNorm  # noqa: E402
from decoupled_engine.srt.layers.rotary_embedding import get_rope  # noqa: E402
from decoupled_engine.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

HEADS, KV_HEADS, HEAD_DIM = 4, 2, 128
PARTIAL = 0.25  # rotary dimension 32: 16 frequencies
SECTION = [6, 5, 5]  # temporal, height, width frequencies, interleaved
THETA = 10_000_000.0
EPS = 1e-6
DEVICE = "cuda"
DTYPE = torch.bfloat16


def build_layer():
    """A full-attention layer whose projections are identities and whose attention records Q, K."""
    layer = qwen3_5.Qwen3_5AttentionDecoderLayer.__new__(qwen3_5.Qwen3_5AttentionDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.attn_output_gate = True
    layer.num_heads, layer.num_kv_heads, layer.head_dim = HEADS, KV_HEADS, HEAD_DIM
    layer.q_size, layer.kv_size = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    layer.alt_stream = None
    gen = torch.Generator().manual_seed(0)
    for name in ("q_norm", "k_norm"):
        norm = GemmaRMSNorm(HEAD_DIM, eps=EPS).to(DEVICE, DTYPE)
        norm.weight.weight_loader(norm.weight, (0.1 * torch.randn(HEAD_DIM, generator=gen)).to(DEVICE, DTYPE))
        setattr(layer, name, norm)
    layer.rotary_emb = get_rope(
        head_size=HEAD_DIM, rotary_dim=HEAD_DIM, max_position=4096, base=THETA,
        rope_scaling={"rope_type": "default", "mrope_section": SECTION, "mrope_interleaved": True},
        partial_rotary_factor=PARTIAL, is_neox_style=True, dtype=DTYPE,
    ).to(DEVICE)
    layer.qkv_proj = lambda hidden_states: (hidden_states, None)
    layer.o_proj = lambda x: (x, None)
    seen = {}

    def attn(q, k, v, forward_batch):
        seen["q"], seen["k"] = q.clone(), k.clone()
        return torch.zeros(q.shape[0], layer.q_size, dtype=q.dtype, device=q.device)

    layer.attn = attn
    return layer, seen


def reference(layer, qkv, positions):
    """GemmaRMSNorm per head, then NeoX rotation of the first 32 dims; frequency j follows
    height when j % 3 == 1 and j < 3 * SECTION[1], width when j % 3 == 2 and j < 3 * SECTION[2],
    else time."""
    x = qkv.float()
    T = x.shape[0]
    q = x[:, : 2 * HEADS * HEAD_DIM].view(T, HEADS, 2 * HEAD_DIM)[..., :HEAD_DIM]
    k = x[:, 2 * HEADS * HEAD_DIM: 2 * HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM].view(T, KV_HEADS, HEAD_DIM)

    def norm(t, layer_norm):
        w = layer_norm.weight.float()
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS) * (1.0 + w)

    rot = int(HEAD_DIM * PARTIAL)
    half = rot // 2
    inv_freq = 1.0 / (THETA ** (torch.arange(0, rot, 2, device=DEVICE, dtype=torch.float32) / rot))
    j = torch.arange(half, device=DEVICE)
    axis = torch.zeros(half, dtype=torch.long, device=DEVICE)
    axis[(j % 3 == 1) & (j < 3 * SECTION[1])] = 1
    axis[(j % 3 == 2) & (j < 3 * SECTION[2])] = 2
    angle = positions.float()[axis].T * inv_freq  # (T, half)
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]

    def rotate(t):
        x1, x2, rest = t[..., :half], t[..., half:rot], t[..., rot:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1)

    return rotate(norm(q, layer.q_norm)), rotate(norm(k, layer.k_norm))


def run(positions, mode):
    layer, seen = build_layer()
    T = positions.shape[1]
    gen = torch.Generator().manual_seed(1)
    qkv = torch.randn(T, 2 * HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM, generator=gen).to(DEVICE, DTYPE)
    layer.self_attention(positions=positions, hidden_states=qkv, forward_batch=SimpleNamespace(forward_mode=mode))
    q_ref, k_ref = reference(layer, qkv, positions)
    return seen["q"].float().view(T, HEADS, HEAD_DIM), seen["k"].float().view(T, KV_HEADS, HEAD_DIM), q_ref, k_ref


def test_prefill_rotates_image_tokens_by_height_and_width():
    # 3 text tokens, a 2 x 3 image (one temporal position, height 0..1, width 0..2), 2 text tokens
    t = [0, 1, 2, 3, 3, 3, 3, 3, 3, 6, 7]
    h = [0, 1, 2, 3, 3, 3, 4, 4, 4, 6, 7]
    w = [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7]
    positions = torch.tensor([t, h, w], dtype=torch.int64, device=DEVICE)
    q, k, q_ref, k_ref = run(positions, ForwardMode.EXTEND)
    torch.testing.assert_close(q, q_ref, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(k, k_ref, atol=3e-2, rtol=2e-2)


def test_decode_rotates_text_tokens_by_their_position():
    positions = torch.tensor([17, 250, 3001], dtype=torch.int64, device=DEVICE).repeat(3, 1)
    q, k, q_ref, k_ref = run(positions, ForwardMode.DECODE)
    torch.testing.assert_close(q, q_ref, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(k, k_ref, atol=3e-2, rtol=2e-2)
