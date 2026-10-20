# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query-head metadata must not become a cache-layout merge constraint."""
from dataclasses import replace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
)


@pytest.mark.parametrize(
    "spec_type", [FullAttentionSpec, MLAAttentionSpec, SinkFullAttentionSpec]
)
def test_merge_ignores_query_heads_but_checks_cache_layout(spec_type):
    first = spec_type(
        block_size=64,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        num_query_heads=24,
    )
    second = replace(first, num_query_heads=40)
    merged = spec_type.merge([first, second])
    assert merged.num_kv_heads == 4
    assert merged.num_query_heads is None
    with pytest.raises(AssertionError, match="same attention spec"):
        spec_type.merge([first, replace(second, num_kv_heads=8)])
