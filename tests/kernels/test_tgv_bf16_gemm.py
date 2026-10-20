# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers import utils as layer_utils
from vllm.utils.flashinfer import is_flashinfer_cutedsl_bf16_gemm_supported


def _require_sm100_cutedsl() -> None:
    if not torch.accelerator.is_available():
        pytest.skip("CUDA is required")
    if not is_flashinfer_cutedsl_bf16_gemm_supported():
        pytest.skip("FlashInfer BF16 cute-dsl backend is unavailable")


@pytest.mark.parametrize(
    "m,n,pdl,expected",
    [
        (2, 1024, True, True),
        (16, 5120, True, True),
        (1, 5120, True, False),
        (17, 5120, True, False),
        (0, 5120, True, False),
        (8, 1023, True, False),
        (8, 5120, False, False),
    ],
)
def test_use_tgv_bf16_gemm(m: int, n: int, pdl: bool, expected: bool) -> None:
    assert layer_utils.use_tgv_bf16_gemm(m, n, pdl) is expected


# (n, k): Qwen3-style attention, MLP and GDN projections at TP 4. The L2
# prefetch is on where k is past the tactic's shared-memory ring (k 4352,
# 5120) and off where the ring holds all of it (k 1536, 1024).
@pytest.mark.parametrize(
    "n,k",
    [
        (3584, 5120),
        (5120, 1536),
        (8704, 5120),
        (5120, 4352),
        (4120, 5120),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("m", [1, 2, 4, 8, 12, 16])
@pytest.mark.parametrize("use_bias", [False, True])
def test_tgv_bf16_gemm(n: int, k: int, m: int, use_bias: bool) -> None:
    _require_sm100_cutedsl()
    from vllm.model_executor.kernels.linear.cute_dsl.tgv_bf16 import tgv_bf16_gemm

    torch.manual_seed(m * 7919 + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
    bias = (
        torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.1 if use_bias else None
    )

    actual = tgv_bf16_gemm(x, weight, bias, pdl=True)
    expected = F.linear(
        x.float(), weight.float(), None if bias is None else bias.float()
    )
    torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=2e-2)

    # The prefetch only moves weight bytes into L2 earlier: same result.
    without_prefetch = tgv_bf16_gemm(x, weight, bias, pdl=True, l2_prefetch=False)
    assert torch.equal(actual, without_prefetch)
    without_pdl = tgv_bf16_gemm(x, weight, bias, pdl=False)
    assert torch.equal(actual, without_pdl)


def test_flashinfer_cutedsl_routes_decode_to_tgv() -> None:
    _require_sm100_cutedsl()
    from vllm.model_executor.kernels.linear.cute_dsl.tgv_bf16 import tgv_bf16_gemm

    torch.manual_seed(0)
    x = torch.randn(2, 4, 5120, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(8704, 5120, device="cuda", dtype=torch.bfloat16) * 0.02

    actual = layer_utils.cuda_flashinfer_bf16_gemm_impl(
        x, weight, None, True, "flashinfer_cutedsl"
    )
    assert actual.shape == (2, 4, 8704)
    assert torch.equal(
        actual.view(8, 8704), tgv_bf16_gemm(x.view(8, 5120), weight, None, pdl=True)
    )


def test_flashinfer_cutedsl_routes_prefill_to_torch(monkeypatch) -> None:
    """Rows past FlashInfer's CuTe DSL range run torch's GEMM directly, not
    FlashInfer's cuBLASLt fallback behind its autotuner."""
    max_m = 32
    monkeypatch.setattr(layer_utils, "flashinfer_cutedsl_bf16_max_m", lambda: max_m)
    monkeypatch.setattr(
        layer_utils,
        "_get_flashinfer_bf16_backend",
        lambda backend: layer_utils._FlashInferBf16Backend(
            flashinfer_backend="cute-dsl",
            is_supported=lambda: True,
            can_implement=lambda x, weight, bias: True,
        ),
    )

    def no_flashinfer(*args, **kwargs):
        raise AssertionError("FlashInfer must not run a prefill-sized GEMM")

    monkeypatch.setattr(layer_utils, "flashinfer_bf16_mm", no_flashinfer)
    torch.manual_seed(0)
    x = torch.randn(1, max_m + 1, 256, dtype=torch.bfloat16)
    weight = torch.randn(512, 256, dtype=torch.bfloat16) * 0.02
    actual = layer_utils.cuda_flashinfer_bf16_gemm_impl(
        x, weight, None, True, "flashinfer_cutedsl"
    )
    assert torch.equal(actual, F.linear(x, weight))
