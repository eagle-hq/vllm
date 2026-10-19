# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gated RMS norm with one weight shared by all groups of a row.

The Qwen GDN spec-decode path normalizes ``[tokens, heads, head_dim]`` with a
``head_dim`` weight and an output gate that is a strided view of the input
projection. Flattening the gate per head copied it; with a shared weight the
kernel takes one row per token and reads the gate in place.

The kernel and compile-key tests need Triton; the call-site and warmup tests
run anywhere.
"""

import types

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.warmup import qwen_triton_warmup
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonPointerInputVariant,
)
from vllm.third_party.flash_linear_attention.ops import layernorm_guard
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    LayerNormFwdKernel,
    calc_rows_per_block,
    layer_norm_fwd,
)
from vllm.triton_utils import HAS_TRITON

ON_GPU = torch.cuda.is_available()
DEVICE = "cuda" if ON_GPU else "cpu"
needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs Triton")

HEADS, HEAD_DIM = 4, 32
# [q, k, v, z] per token: the gate z is the last HEADS * HEAD_DIM columns.
QKV_WIDTH = 3 * HEADS * HEAD_DIM


@pytest.fixture(autouse=True)
def _cpu_compute_units(monkeypatch):
    if not ON_GPU:
        monkeypatch.setattr(layernorm_guard, "num_compute_units", lambda *_: 148)


def _inputs(num_tokens: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(num_tokens, HEADS, HEAD_DIM, generator=gen)
    proj = torch.randn(num_tokens, QKV_WIDTH + HEADS * HEAD_DIM + 8, generator=gen)
    weight = torch.randn(HEAD_DIM, generator=gen)
    bias = torch.randn(HEAD_DIM, generator=gen)
    to = dict(device=DEVICE, dtype=torch.bfloat16)
    proj = proj.to(**to)
    gate = proj[:, QKV_WIDTH : QKV_WIDTH + HEADS * HEAD_DIM]
    return (
        x.to(**to),
        gate.view(num_tokens, HEADS, HEAD_DIM),
        weight.to(**to),
        bias.to(**to),
    )


def _per_head_rows(x, gate, weight, bias, norm_before_gate, is_rms_norm):
    """Reference: one row per (token, head), contiguous gate copy."""
    out, _, _ = layer_norm_fwd(
        x.reshape(-1, HEAD_DIM),
        weight,
        bias,
        1e-6,
        z=gate.reshape(-1, HEAD_DIM),
        group_size=HEAD_DIM,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )
    return out.view_as(x)


@needs_triton
@pytest.mark.parametrize("num_tokens", [1, 3, 16])
@pytest.mark.parametrize("norm_before_gate", [True, False])
@pytest.mark.parametrize("is_rms_norm", [True, False])
@torch.inference_mode()
def test_shared_weight_matches_per_head_rows(num_tokens, norm_before_gate, is_rms_norm):
    x, gate, weight, bias = _inputs(num_tokens)
    if is_rms_norm:
        bias = None
    gate_2d = gate.flatten(-2)
    assert gate_2d.data_ptr() == gate.data_ptr()  # a view, not a copy
    out, mean, rstd = layer_norm_fwd(
        x.view(num_tokens, -1),
        weight,
        bias,
        1e-6,
        z=gate_2d,
        group_size=HEAD_DIM,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )
    assert rstd.shape == (HEADS * num_tokens,)
    ref = _per_head_rows(x, gate, weight, bias, norm_before_gate, is_rms_norm)
    torch.testing.assert_close(out.view_as(x), ref, atol=0, rtol=0)


@needs_triton
@torch.inference_mode()
def test_shared_weight_is_not_a_per_group_weight():
    """A full-row weight still gives each group its own slice."""
    x, gate, weight, _ = _inputs(3, seed=1)
    full = torch.randn(HEADS * HEAD_DIM, generator=torch.Generator().manual_seed(2))
    full = full.to(device=DEVICE, dtype=torch.bfloat16)
    out_full, _, _ = layer_norm_fwd(
        x.view(3, -1),
        full,
        None,
        1e-6,
        z=gate.flatten(-2),
        group_size=HEAD_DIM,
        is_rms_norm=True,
    )
    out_rows = torch.empty_like(x)
    for h in range(HEADS):
        out_rows[:, h], _, _ = layer_norm_fwd(
            x[:, h].contiguous(),
            full[h * HEAD_DIM : (h + 1) * HEAD_DIM].contiguous(),
            None,
            1e-6,
            z=gate[:, h].contiguous(),
            is_rms_norm=True,
        )
    torch.testing.assert_close(out_full.view_as(x), out_rows, atol=0, rtol=0)


def _norm(group_size=None):
    return types.SimpleNamespace(
        weight=torch.ones(HEAD_DIM),
        bias=None,
        eps=1e-6,
        group_size=group_size,
        norm_before_gate=True,
        activation="silu",
    )


def _capture_layer_norm(monkeypatch):
    calls = []

    def fake(x, weight, bias, eps, *, z, out, group_size, **kw):
        calls.append(dict(x=x, weight=weight, z=z, out=out, group_size=group_size))

    monkeypatch.setattr(layernorm_guard, "layer_norm_fwd", fake)
    return calls


@pytest.mark.parametrize("num_tokens", [1, 5])
def test_gdn_gated_norm_reads_the_gate_in_place(monkeypatch, num_tokens):
    calls = _capture_layer_norm(monkeypatch)
    proj = torch.randn(num_tokens, QKV_WIDTH + HEADS * HEAD_DIM)
    gate = proj[:, QKV_WIDTH:].view(num_tokens, HEADS, HEAD_DIM)
    x = torch.randn(num_tokens, HEADS, HEAD_DIM)
    layer = types.SimpleNamespace(norm=_norm())
    QwenGatedDeltaNetAttention._rms_norm_gated_cuda(layer, x, gate, x)
    (call,) = calls
    assert call["z"].data_ptr() == gate.data_ptr()
    assert call["z"].shape == (num_tokens, HEADS * HEAD_DIM)
    if num_tokens > 1:
        assert call["z"].stride(0) == proj.stride(0)
    assert call["x"].shape == call["out"].shape == (num_tokens, HEADS * HEAD_DIM)
    assert call["weight"].shape == (HEAD_DIM,)
    assert call["group_size"] == HEAD_DIM


@pytest.mark.parametrize("case", ["group_smaller_than_head", "heads_not_adjacent"])
def test_gdn_gated_norm_falls_back_to_per_head_rows(monkeypatch, case):
    calls = _capture_layer_norm(monkeypatch)
    num_tokens = 4
    proj = torch.randn(num_tokens, QKV_WIDTH + HEADS * HEAD_DIM)
    gate = proj[:, QKV_WIDTH:].view(num_tokens, HEADS, HEAD_DIM)
    norm = _norm()
    if case == "group_smaller_than_head":
        norm = _norm(group_size=HEAD_DIM // 2)
    else:
        # Each head's gate is followed by padding: no single row per token.
        gate = proj.new_empty(num_tokens, HEADS, 2 * HEAD_DIM)[..., :HEAD_DIM]
    x = torch.randn(num_tokens, HEADS, HEAD_DIM)
    layer = types.SimpleNamespace(norm=norm)
    QwenGatedDeltaNetAttention._rms_norm_gated_cuda(layer, x, gate, x)
    (call,) = calls
    assert call["z"].shape == (num_tokens * HEADS, HEAD_DIM)
    assert call["x"].shape == (num_tokens * HEADS, HEAD_DIM)


def _runtime_key(kernel, num_tokens, row_stride, z_row_stride, shared):
    return kernel.dispatch(
        x_dtype=torch.bfloat16,
        y_dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16,
        bias_dtype=None,
        z_dtype=torch.bfloat16,
        mean_dtype=None,
        rstd_dtype=torch.float32,
        x_aligned=True,
        y_aligned=True,
        weight_aligned=True,
        bias_aligned=True,
        z_aligned=True,
        mean_aligned=True,
        rstd_aligned=True,
        stride_x_row=row_stride,
        stride_y_row=row_stride,
        stride_z_row=z_row_stride,
        M=num_tokens,
        N=128,
        BLOCK_N=128,
        ROWS_PER_BLOCK=calc_rows_per_block(num_tokens, torch.device(DEVICE)),
        norm_before_gate=True,
        is_rms_norm=True,
        activation="silu",
        shared_weight=shared,
    )


@needs_triton
@pytest.mark.parametrize("z_row_stride", [4128, 4120])
def test_shared_weight_warmup_keys_cover_runtime(z_row_stride):
    kernel = LayerNormFwdKernel()
    keys = set(
        kernel.get_warmup_keys(
            max_num_tokens=512,
            rows_per_token=1,
            group_size=128,
            x_dtype=torch.bfloat16,
            weight_dtype=torch.bfloat16,
            device=torch.device(DEVICE),
            norm_before_gate=True,
            is_rms_norm=True,
            activation="silu",
            row_stride=12 * 128,
            z_row_stride=z_row_stride,
            shared_weight=True,
        )
    )
    for t in range(1, 513):
        assert _runtime_key(kernel, t, 12 * 128, z_row_stride, True) in keys
    # The per-group-weight variant is a different kernel.
    assert _runtime_key(kernel, 16, 12 * 128, z_row_stride, False) not in keys
    assert all(key.SHARED_WEIGHT for key in keys)


def _warmup_config(gate_row_stride):
    return qwen_triton_warmup._QwenGDNWarmupConfig(
        h=4,
        hv=12,
        k=128,
        v=128,
        conv_kernel_size=4,
        conv_state=torch.empty(0),
        conv_dtype=torch.bfloat16,
        norm_weight_dtype=torch.bfloat16,
        norm_before_gate=True,
        norm_activation="silu",
        a_log=torch.empty(0),
        dt_bias=torch.empty(0),
        state_stride_token=0,
        state_dtype=torch.float32,
        gate_row_stride=gate_row_stride,
    )


@pytest.mark.parametrize("gate_row_stride", [None, 4128])
def test_qwen_warmup_warms_the_shared_weight_layout(monkeypatch, gate_row_stride):
    calls = []
    monkeypatch.setattr(
        layernorm_guard, "warmup_layer_norm_fwd", lambda **kw: calls.append(kw)
    )
    qwen_triton_warmup._warm_gated_rms_norm_kernel(
        torch.device("cpu"), _warmup_config(gate_row_stride), 64, torch.bfloat16
    )
    assert calls[0]["rows_per_token"] == 12
    assert not calls[0].get("shared_weight", False)
    if gate_row_stride is None:
        assert len(calls) == 1
        return
    assert len(calls) == 2
    shared = calls[1]
    assert shared["rows_per_token"] == 1
    assert shared["group_size"] == 128
    assert shared["row_stride"] == 12 * 128
    assert shared["z_row_stride"] == gate_row_stride
    assert shared["shared_weight"] is True


def test_qwen_warmup_config_reads_the_gate_stride():
    layer = types.SimpleNamespace(
        enable_fused_gdn_spec_decode=True,
        in_proj_qkvz=types.SimpleNamespace(weight=torch.empty(4128, 8)),
    )
    assert qwen_triton_warmup._gate_row_stride(layer) == 4128
    layer.enable_fused_gdn_spec_decode = False
    assert qwen_triton_warmup._gate_row_stride(layer) is None


def test_compile_passes_shared_weight():
    kernel = LayerNormFwdKernel()
    seen = []
    kernel.kernel = types.SimpleNamespace(warmup=lambda *a, **kw: seen.append(kw))
    for shared in (False, True):
        key = LayerNormFwdKernel.CompileKey(
            x_dtype=torch.bfloat16,
            y_dtype=torch.bfloat16,
            weight_dtype=torch.bfloat16,
            bias_dtype=None,
            z_dtype=torch.bfloat16,
            mean_dtype=None,
            rstd_dtype=torch.float32,
            input_variant=TritonPointerInputVariant.from_alignment(
                x=True, y=True, weight=True, bias=True, z=True, mean=True, rstd=True
            ),
            stride_x_row=16,
            stride_y_row=16,
            stride_z_row=16,
            M=16,
            N=128,
            BLOCK_N=128,
            ROWS_PER_BLOCK=1,
            NORM_BEFORE_GATE=True,
            IS_RMS_NORM=True,
            ACTIVATION="silu",
            SHARED_WEIGHT=shared,
        )
        kernel.compile(key)
    assert [kw["SHARED_WEIGHT"] for kw in seen] == [False, True]


def test_qwen_warmup_config_carries_the_gate_stride():
    layer = types.SimpleNamespace(
        num_k_heads=16,
        num_v_heads=48,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        tp_size=4,
        kv_cache=(torch.zeros(2, 3, 8), torch.zeros(2, 4, 4)),
        norm=types.SimpleNamespace(
            weight=torch.ones(128), norm_before_gate=True, activation="silu"
        ),
        A_log=torch.zeros(12),
        dt_bias=torch.zeros(12),
        enable_fused_gdn_spec_decode=True,
        in_proj_qkvz=types.SimpleNamespace(weight=torch.empty(4128, 8)),
    )
    config = qwen_triton_warmup._qwen_gdn_warmup_config({"layer": layer})
    assert config is not None
    assert config.hv == 12
    assert config.gate_row_stride == 4128
