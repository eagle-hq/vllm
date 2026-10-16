# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hooks that let a speculator own verification (tree or other non-chain drafts).

Every hook is a no-op by default: a speculator that does not override it gets
the behavior vLLM had before the hook existed.
"""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.speculator import (
    BaseSpeculator,
    DraftModelSpeculator,
)


def _calls_in(func, attr: str) -> list[ast.Call]:
    """Method calls ``<anything>.<attr>(...)`` in the source of ``func``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


# --- make_rejection_sampler -------------------------------------------------


class _FakeRejectionSampler:
    def __init__(self, sampler, speculative_config, device, watermark_key=None):
        self.args = (sampler, speculative_config, device, watermark_key)


def _runner(speculator, monkeypatch) -> GPUModelRunner:
    monkeypatch.setattr(model_runner_module, "RejectionSampler", _FakeRejectionSampler)
    monkeypatch.setattr(
        model_runner_module, "speculative_target_watermark_key", lambda config: "key"
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.sampler = object()
    runner.speculator = speculator
    runner.speculative_config = object()
    runner.device = "cpu"
    runner.vllm_config = SimpleNamespace(watermark_config=None)
    return runner


def test_speculator_hook_default_keeps_the_chain_rejection_sampler():
    assert BaseSpeculator.make_rejection_sampler(object(), object()) is None  # type: ignore[arg-type]
    # Built-in subclasses do not override it.
    assert "make_rejection_sampler" not in DraftModelSpeculator.__dict__


class _Declines:
    def make_rejection_sampler(self, sampler):
        return None


@pytest.mark.parametrize("speculator", [_Declines(), None])
def test_default_rejection_sampler_is_built_unless_the_speculator_supplies_one(
    speculator, monkeypatch
):
    runner = _runner(speculator, monkeypatch)
    built = GPUModelRunner._build_rejection_sampler(runner)
    assert isinstance(built, _FakeRejectionSampler)
    assert built.args == (runner.sampler, runner.speculative_config, "cpu", "key")


def test_speculator_supplied_rejection_sampler_replaces_the_default(monkeypatch):
    custom = object()
    seen = []

    class Supplies:
        def make_rejection_sampler(self, sampler):
            seen.append(sampler)
            return custom

    runner = _runner(Supplies(), monkeypatch)
    assert GPUModelRunner._build_rejection_sampler(runner) is custom
    assert seen == [runner.sampler]


def test_load_model_builds_its_rejection_sampler_through_the_hook():
    # load_model needs a GPU and a real model, so check the call site in source:
    # the speculative branch must go through _build_rejection_sampler.
    assert len(_calls_in(GPUModelRunner.load_model, "_build_rejection_sampler")) == 1
    constructed = [
        node
        for node in ast.walk(
            ast.parse(textwrap.dedent(inspect.getsource(GPUModelRunner.load_model)))
        )
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RejectionSampler"
    ]
    assert constructed == []
