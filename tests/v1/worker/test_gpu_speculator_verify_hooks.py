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

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
import vllm.v1.worker.gpu.spec_decode.adaptive_verification as adaptive_module
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.ngram.speculator import NgramGPUSpeculator
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


def test_native_decode_is_sticky_by_request_identity_and_drains_pending_state():
    from vllm.v1.worker.gpu.spec_decode.native_decode import NativeDecodeState

    state = NativeDecodeState(((4, 8),))
    # A speculative forward leaves an accepted state to consume on the next step.
    assert state.select(["a", "b"], set(), {"a", "b"}) == (False, False)
    # Crossing the policy boundary stops drafting, but finishes pending verification.
    assert state.select(["b", "a", "c", "d"], set(), {"a", "b"}) == (True, False)
    # One target-only transition consumes the accepted recurrent-state offset.
    assert state.select(["a", "b"], {"c", "d"}, set()) == (True, False)
    # Filtering/compaction cannot resume a stale drafter when the batch shrinks.
    assert state.select(["b"], {"a"}, set()) == (True, True)
    assert state.select(["b"], set(), set()) == (True, True)
    # A new request merged into an OFF batch is also disabled, without a draft prefill.
    assert state.select(["new", "b"], set(), set()) == (True, True)
    assert state.select(["new"], {"b"}, set()) == (True, True)
    # Only request finish resets the contract (including reuse of the same ID).
    assert state.select(["new"], {"new"}, set()) == (False, False)


@pytest.mark.parametrize("native_target", [False, True])
def test_suspended_batch_publishes_no_drafts_without_calling_the_speculator(
    monkeypatch, native_target
):
    from unittest.mock import Mock

    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    batch = InputBatch.make_dummy(2, 2, InputBuffers(2, 4, torch.device("cpu")))
    batch.skip_draft = True
    batch.native_target = native_target
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.execute_model_state = SimpleNamespace(
        input_batch=batch,
        attn_metadata={},
        slot_mappings_by_layer={},
        hidden_states=torch.zeros(2, 4),
        aux_hidden_states=None,
        dp_sync_state=None,
        finished_req_ids=set(),
        ec_connector_output=None,
        cudagraph_stats=None,
        num_spec_tokens_to_schedule=3,
    )
    runner.is_last_pp_rank = True
    runner.pcp_manager = runner.pp_handler = runner.aux_output_connector = None
    runner.speculator = _MinimalSpeculator()
    runner.speculator.supports_mm_inputs = False
    runner.model = SimpleNamespace(compute_logits=lambda x: x)
    output = SimpleNamespace(sampled_token_ids=torch.ones(2, 1, dtype=torch.int32))
    runner.sample = lambda *args: (output, torch.ones(2), torch.zeros(2))
    runner.prompt_logprobs_worker = SimpleNamespace(
        compute_prompt_logprobs=lambda *args: {},
        compute_prompt_token_id_logprobs=lambda *args: {},
    )
    runner.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
        draft_tokens=torch.full((2, 3), 123),
    )
    runner.main_stream = runner.output_copy_stream = None
    runner.check_ep_fault = False
    runner.kv_connector = Mock()
    runner.postprocess_sampled = lambda *args: None
    runner.sampler = object()
    runner.num_speculative_steps = 3
    published: list[torch.Tensor] = []
    runner.draft_tokens_handler = SimpleNamespace(
        set_draft_tokens=lambda batch, tokens: published.append(tokens.clone())
    )
    runner.eplb = Mock()

    monkeypatch.setattr(model_runner_module, "AsyncOutput", lambda **kw: kw)

    def unexpected_draft(*args):
        pytest.fail("OFF must not prepare hidden states or run the drafter")

    monkeypatch.setattr(
        model_runner_module, "get_drafter_hidden_states", unexpected_draft
    )
    commits = []
    runner.speculator.post_accept = lambda *args: commits.append("verify")
    GPUModelRunner.sample_tokens(runner, None)
    assert commits == ([] if native_target else ["verify"])
    assert len(published) == 1
    assert published[0].shape == (2, 0)
    assert torch.all(runner.req_states.draft_tokens == 123)


def test_native_decode_builds_separate_target_metadata_and_graphs(monkeypatch):
    from unittest.mock import Mock

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.speculator = SimpleNamespace(native_decode_batch_sizes=())
    runner.scheduler_config = SimpleNamespace(async_scheduling=False)
    runner.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_dbo=False,
    )
    runner.cache_config = SimpleNamespace(
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        use_replayssm=False,
        use_kda_recoverssm=False,
    )
    runner.speculative_config = SimpleNamespace(native_decode_batch_sizes=[(4, 8)])
    runner.vllm_config = SimpleNamespace(
        speculative_config=runner.speculative_config,
        kv_transfer_config=None,
        parallel_config=runner.parallel_config,
        cache_config=runner.cache_config,
        scheduler_config=runner.scheduler_config,
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
    )
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.adaptive_verification = None
    runner.kv_cache_config = object()
    runner.device = "cpu"
    runner.kernel_block_sizes = [16, 16]
    runner.lora_capture_cases = [0]
    original_spec = runner.vllm_config.speculative_config
    # An excluded draft-only group need not use the target kernel block size.
    init = Mock(return_value=([["native"], []], None, [16, 32]))
    monkeypatch.setattr(model_runner_module, "init_attn_backend", init)
    manager = Mock(return_value="native graphs")
    monkeypatch.setattr(model_runner_module, "ModelCudaGraphManager", manager)
    GPUModelRunner._init_native_decode(runner, {"target"}, "mode")
    assert init.called, "Serialized OFF ranges must initialize native metadata"
    assert init.call_args.args[1].speculative_config is None
    assert init.call_args.kwargs["active_layer_names"] == {"target"}
    assert manager.call_args.kwargs["decode_query_len"] == 1
    assert runner.native_cudagraph_manager == "native graphs"
    assert runner.native_attn_groups == [["native"], []]
    assert runner.vllm_config.speculative_config is original_spec


@pytest.mark.parametrize("full_graph", [False, True])
def test_native_execution_uses_native_metadata_graphs_and_no_verify_hook(
    monkeypatch, full_graph
):
    from unittest.mock import Mock

    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.spec_decode.native_decode import NativeDecodeState

    batch = InputBatch.make_dummy(2, 2, InputBuffers(2, 4, torch.device("cpu")))
    batch.req_ids = ["a", "b"]
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"a": 1, "b": 1}
    scheduled.total_num_scheduled_tokens = 2
    desc = SimpleNamespace(
        num_tokens=2,
        cg_mode=CUDAGraphMode.FULL if full_graph else CUDAGraphMode.NONE,
        num_ubatches=1,
        num_active_loras=0,
    )
    outputs = torch.zeros(2, 4)
    groups_seen = []
    graphs_seen = []
    native_manager = SimpleNamespace(run_fullgraph=lambda desc: outputs)

    def prepare_attn(*args, **kwargs):
        groups_seen.append(args[4])
        return {}

    runner = SimpleNamespace(
        update_pp_decode_requests=lambda: None,
        finish_requests=lambda s: None,
        free_states=lambda s: None,
        add_requests=lambda s: None,
        update_requests=lambda s: None,
        block_tables=Mock(),
        aux_output_connector=None,
        native_decode_state=NativeDecodeState(((2, 8),)),
        native_cudagraph_manager=native_manager,
        cudagraph_manager=object(),
        native_attn_groups=["native"],
        attn_groups=["verify"],
        gather_batch_req_state=lambda *a: (
            SimpleNamespace(num_tokens=2, has_prefill=False),
            1,
        ),
        pcp_manager=None,
        lora_config=None,
        is_encoder_decoder=False,
        dp_size=1,
        dp_rank=0,
        parallel_config=object(),
        ubatch_runner=None,
        decode_query_len=4,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
        prepare_inputs=lambda *a: batch,
        prepare_attn=lambda b: ([], []),
        model_state=SimpleNamespace(
            preprocess_state=lambda *a: None,
            prepare_attn=prepare_attn,
            prepare_inputs=lambda *a: {},
        ),
        kv_cache_config=object(),
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=None)),
        dcp_size=1,
        uses_inputs_embeds=False,
        is_first_pp_rank=True,
        is_last_pp_rank=True,
        speculator=SimpleNamespace(
            prepare_verify_inputs=lambda *a, **kw: pytest.fail(
                "Native batch ran verify hook"
            )
        ),
        eplb=Mock(),
        step_timing=Mock(),
        kv_connector=Mock(),
        vllm_config=object(),
        model_config=object(),
        model=lambda **kw: outputs,
        use_aux_hidden_state_outputs=False,
    )

    def dispatch(manager, *args, **kwargs):
        graphs_seen.append(manager)
        return desc, None

    from contextlib import nullcontext

    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", dispatch)
    monkeypatch.setattr(
        model_runner_module, "build_slot_mappings_by_layer", lambda *a: {}
    )
    monkeypatch.setattr(
        model_runner_module, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    GPUModelRunner.execute_model(runner, scheduled)
    assert groups_seen == [["native"]]
    assert graphs_seen == [native_manager]
    assert runner.execute_model_state.input_batch.native_target
    assert runner.execute_model_state.input_batch.skip_draft


def test_native_graphs_capture_even_when_speculative_graphs_are_already_ready(
    monkeypatch,
):
    from contextlib import nullcontext
    from unittest.mock import Mock

    captures = []
    native = SimpleNamespace(
        needs_capture=lambda: True,
        capture=lambda *args, **kwargs: captures.append(args[5]),
    )
    runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(needs_capture=lambda: False),
        native_cudagraph_manager=native,
        native_attn_groups=["native"],
        attn_groups=["verify"],
        model_state=SimpleNamespace(
            supports_mm_inputs=False, capture_inner_cudagraphs=lambda *a: None
        ),
        maybe_setup_dummy_loras=lambda *a: nullcontext(),
        lora_config=None,
        input_buffers=object(),
        pcp_manager=None,
        model=object(),
        intermediate_tensors=None,
        block_tables=object(),
        kv_cache_config=object(),
        use_aux_hidden_state_outputs=False,
        speculator=None,
        adaptive_verification=None,
        kv_connector=Mock(),
    )
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: (1, 1))
    monkeypatch.setattr(
        model_runner_module, "create_lora_capture_hook", lambda *a: None
    )
    assert GPUModelRunner.needs_cudagraph_capture(runner)
    GPUModelRunner.capture_model(runner, profile_only=True)
    assert captures == [["native"]]


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("scheduler_config", "async_scheduling", True),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "mamba_cache_mode", "align"),
        ("cache_config", "use_replayssm", True),
        ("cache_config", "use_kda_recoverssm", True),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_dbo", True),
        ("compilation_config", "cudagraph_mode", CUDAGraphMode.PIECEWISE),
        ("compilation_config", "cudagraph_mode", CUDAGraphMode.FULL_AND_PIECEWISE),
    ],
)
def test_native_decode_rejects_uncoordinated_state_and_cache_reuse(
    section, field, value
):
    from vllm.v1.worker.gpu.spec_decode.native_decode import native_decode_config

    config = SimpleNamespace(
        speculative_config=object(),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(async_scheduling=False),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            use_replayssm=False,
            use_kda_recoverssm=False,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_dbo=False,
        ),
    )
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="Native draft suspension requires"):
        native_decode_config(config)


def test_native_decode_preserves_unscheduled_and_preempted_request_lifetimes():
    from vllm.v1.worker.gpu.spec_decode.native_decode import NativeDecodeState

    state = NativeDecodeState(((2, 8),))
    assert state.select(["a", "b"], set(), set()) == (True, True)
    # Not scheduled is not finished. An unrelated batch may still speculate.
    assert state.select(["other"], set(), {"other"}) == (False, False)
    assert state.select([], set(), set()) == (False, False)
    # Resume the preempted ID without resurrecting its stale draft KV.
    assert state.select(["a"], set(), set()) == (True, True)
    # Merging a speculative request drains its own most recent accepted state.
    assert state.select(["a", "other"], set(), set()) == (True, False)
    assert state.select(["other"], {"a", "b"}, set()) == (True, True)
    assert state.select([], {"other"}, set()) == (False, False)
    assert not state.disabled and not state.pending_state


@pytest.mark.parametrize("override", ["0", "1"])
def test_native_decode_configuration_requires_v2(monkeypatch, override):
    from vllm.config import VllmConfig

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", override)
    config = SimpleNamespace(
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=SimpleNamespace(
            method="eagle3", native_decode_batch_sizes=[(4, 8)]
        ),
    )
    if override == "0":
        with pytest.raises(
            ValueError, match="Native draft suspension requires Model Runner V2"
        ):
            VllmConfig.use_v2_model_runner.fget(config)
    else:
        assert VllmConfig.use_v2_model_runner.fget(config)


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


# --- prepare_verify_inputs / post_accept ------------------------------------


class _MinimalSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):  # pragma: no cover - not called
        raise NotImplementedError


def _call_lines(func, dotted: str) -> list[int]:
    """Line numbers of calls whose callee prints as ``dotted`` (e.g. self.a.b)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == dotted
    )


def test_verify_hooks_do_nothing_by_default():
    speculator = _MinimalSpeculator.__new__(_MinimalSpeculator)
    positions = torch.arange(6)
    batch = SimpleNamespace(positions=positions)
    model_inputs = {"positions": positions}
    for dummy_run in (False, True):
        result = speculator.prepare_verify_inputs(
            batch, model_inputs, dummy_run=dummy_run
        )  # type: ignore[arg-type]
        assert result is None
    assert torch.equal(positions, torch.arange(6))
    assert speculator.post_accept(batch, None, None, None) is None  # type: ignore[arg-type]
    for builtin in (DraftModelSpeculator, NgramGPUSpeculator):
        for hook in ("prepare_verify_inputs", "post_accept"):
            assert hook not in builtin.__dict__


def test_execute_model_prepares_verify_inputs_between_model_inputs_and_forward():
    execute = GPUModelRunner.execute_model
    (slot_mapping_line,) = _call_lines(execute, "self.prepare_attn")
    (metadata_line,) = _call_lines(execute, "self.model_state.prepare_attn")
    (inputs_line,) = _call_lines(execute, "self.model_state.prepare_inputs")
    (hook_line,) = _call_lines(execute, "self.speculator.prepare_verify_inputs")
    (forward_line,) = _call_lines(execute, "self.step_timing.forward_start")
    assert slot_mapping_line < metadata_line < inputs_line < hook_line < forward_line
    # only runs with a speculator, and only where the attention metadata is built
    assert "verify_inputs = self.speculator is not None" in inspect.getsource(execute)


def test_sample_tokens_calls_post_accept_between_sampling_and_state_commit():
    func = GPUModelRunner.sample_tokens
    (sample_line,) = _call_lines(func, "self.sample")
    (hook_line,) = _call_lines(func, "self.speculator.post_accept")
    (commit_line,) = _call_lines(func, "self.postprocess_sampled")
    (propose_line,) = _call_lines(func, "self.speculator.propose")
    assert sample_line < hook_line < commit_line < propose_line


# --- uses_varlen_verify_graphs -------------------------------------------------


def _graph_runner(speculator, adaptive=None) -> GPUModelRunner:
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.speculator = speculator
    runner.adaptive_verification = adaptive
    runner.attn_groups = ["groups"]
    runner.vllm_config = "vllm_config"
    runner.decode_query_len = 9
    return runner


ALWAYS = (AttentionCGSupport.ALWAYS, None)


def test_varlen_verify_graphs_are_off_by_default():
    assert BaseSpeculator.uses_varlen_verify_graphs is False
    assert "uses_varlen_verify_graphs" not in DraftModelSpeculator.__dict__
    for speculator in (None, _MinimalSpeculator.__new__(_MinimalSpeculator)):
        runner = _graph_runner(speculator)
        assert GPUModelRunner._use_varlen_verify_graphs(runner, None, ALWAYS) is False


def test_adaptive_verification_still_selects_varlen_graphs():
    runner = _graph_runner(None, adaptive=object())
    assert GPUModelRunner._use_varlen_verify_graphs(runner, None, ALWAYS) is True


class _VarlenSpeculator(_MinimalSpeculator):
    uses_varlen_verify_graphs = True


def test_speculator_opt_in_checks_backend_support_then_selects_varlen(monkeypatch):
    seen = {}

    def fake_unsupported(attn_groups, vllm_config, max_query_len, checked_layer_names):
        seen["args"] = (attn_groups, vllm_config, max_query_len, checked_layer_names)

    monkeypatch.setattr(
        adaptive_module, "get_varlen_cudagraph_unsupported_backend", fake_unsupported
    )
    runner = _graph_runner(_VarlenSpeculator.__new__(_VarlenSpeculator))
    assert GPUModelRunner._use_varlen_verify_graphs(runner, {"layer"}, ALWAYS) is True
    assert seen["args"] == (["groups"], "vllm_config", 9, {"layer"})


def test_speculator_opt_in_rejects_a_backend_without_varlen_graph_support(monkeypatch):
    monkeypatch.setattr(
        adaptive_module,
        "get_varlen_cudagraph_unsupported_backend",
        lambda *args, **kwargs: ("SomeBackend", 4),
    )
    runner = _graph_runner(_VarlenSpeculator.__new__(_VarlenSpeculator))
    with pytest.raises(ValueError, match="SomeBackend allows at most 4"):
        GPUModelRunner._use_varlen_verify_graphs(runner, None, ALWAYS)


def test_speculator_opt_in_rejects_a_model_state_without_always_support(monkeypatch):
    monkeypatch.setattr(
        adaptive_module,
        "get_varlen_cudagraph_unsupported_backend",
        lambda *args, **kwargs: None,
    )
    runner = _graph_runner(_VarlenSpeculator.__new__(_VarlenSpeculator))
    with pytest.raises(ValueError, match="MambaBackend allows none"):
        GPUModelRunner._use_varlen_verify_graphs(
            runner, None, (AttentionCGSupport.UNIFORM_BATCH, "MambaBackend")
        )


def test_initialize_kv_cache_feeds_the_varlen_decision_to_the_graph_manager():
    func = GPUModelRunner.initialize_kv_cache
    assert len(_call_lines(func, "self._use_varlen_verify_graphs")) == 1
    source = inspect.getsource(func)
    assert "varlen_decode=varlen_verify_graphs" in source
    assert "if varlen_verify_graphs:" in source
    assert "varlen_decode=self.adaptive_verification" not in source
