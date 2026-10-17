# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The packed GDN decode path reads the b/a gates as strided views.

``forward_cuda`` used to copy b and a into contiguous tensors before the core
op on every step; the packed recurrent decode kernel takes a token stride for
both, so only the other paths copy them now.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types

import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def _source_tree(fn) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def test_forward_cuda_passes_gates_uncopied() -> None:
    tree = _source_tree(QwenGatedDeltaNetAttention.forward_cuda)
    core_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "qwen_gdn_attention_core"
    ]
    assert len(core_calls) == 1
    gate_args = core_calls[0].args[1:3]
    assert [type(arg) for arg in gate_args] == [ast.Name, ast.Name]
    assert [arg.id for arg in gate_args] == ["b", "a"]


def test_forward_core_copies_gates_after_packed_decode() -> None:
    tree = _source_tree(QwenGatedDeltaNetAttention._forward_core)
    packed_return = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", "") == "_forward_core_decode_non_spec"
    )
    copies = {
        node.targets[0].id: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", "") == "contiguous"
    }
    assert packed_return < copies["b"]
    assert packed_return < copies["a"]


def test_packed_decode_gets_strided_gates(monkeypatch) -> None:
    metadata = object.__new__(GDNAttentionMetadata)
    metadata.spec_sequence_masks = None
    metadata.num_prefills = 0
    metadata.num_decodes = 2
    context = types.SimpleNamespace(attn_metadata={"layer.0": metadata})
    monkeypatch.setattr(qwen_gdn_linear_attn, "get_forward_context", lambda: context)

    seen = {}

    def decode_non_spec(**kwargs):
        seen.update(kwargs)

    layer = types.SimpleNamespace(
        prefix="layer.0",
        enable_packed_recurrent_decode=True,
        _forward_core_decode_non_spec=decode_non_spec,
    )
    projection = torch.randn(2, 40)
    b, a = projection[:, 32:36], projection[:, 36:40]
    assert not b.is_contiguous()
    QwenGatedDeltaNetAttention._forward_core(
        layer,
        mixed_qkv=projection[:, :32],
        b=b,
        a=a,
        core_attn_out=torch.empty(2, 4, 8),
    )
    assert seen["b"] is b
    assert seen["a"] is a
