# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import random
import typing

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

torch.manual_seed(42)
random.seed(44)

test_size_elements = 1024 * 1024


def symm_mem_allreduce_worker(local_rank: int, world_size: int, q: mp.Queue):
    monkeypatch = pytest.MonkeyPatch()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))

    with monkeypatch.context() as m, set_current_vllm_config(config):
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        symm_mem_comm = cuda_communicator.symm_mem_comm
        if symm_mem_comm is None or symm_mem_comm.disabled:
            # can't use skip under multiprocessing
            q.put("SymmMemCommunicator is not available or disabled.")
            return

        inp_direct_symm_mem = torch.randint(
            1, 23, (test_size_elements,), dtype=dtype, device=device
        )
        if not symm_mem_comm.should_use_symm_mem(inp_direct_symm_mem):
            # can't use skip under multiprocessing
            q.put("SymmMemCommunicator isn't used for this world and input size.")
            return

        original_inp_direct_symm_mem = inp_direct_symm_mem.clone()
        out_direct_symm_mem = symm_mem_comm.all_reduce(inp_direct_symm_mem)
        assert out_direct_symm_mem is not None

        group = get_tp_group().device_group
        dist.all_reduce(original_inp_direct_symm_mem, group=group)
        torch.testing.assert_close(
            out_direct_symm_mem, original_inp_direct_symm_mem, atol=2.5, rtol=0.1
        )

        # Test tensor_model_parallel_all_reduce which should use symm_mem
        inp_tensor_parallel = torch.randint(
            -23, 1, (test_size_elements,), dtype=dtype, device=device
        )
        original_inp_tensor_parallel = inp_tensor_parallel.clone()
        out_tensor_parallel = tensor_model_parallel_all_reduce(inp_tensor_parallel)
        dist.all_reduce(original_inp_tensor_parallel, group=group)
        torch.testing.assert_close(
            out_tensor_parallel, original_inp_tensor_parallel, atol=2.5, rtol=0.1
        )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_symm_mem_allreduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pipeline_parallel_size
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    q = mp.get_context("spawn").Queue()
    mp.spawn(symm_mem_allreduce_worker, args=(world_size, q), nprocs=world_size)
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
        if val is not None:
            pytest.skip(val)


def symm_mem_two_streams_worker(local_rank: int, world_size: int, port: int, q):
    """All-reduces issued from two caller streams while one rank lags at a time
    (the startup pattern: profiling, warmup and capture streams, one rank
    JIT-building a kernel). Before the communicator ran every op on its own
    stream, B300s returned wrong sums from iteration 49 on: max error 8e26
    at TP8 (multimem) and 6.1 at TP2 (two-shot) over 5000 iterations. The
    barrier can also deadlock (pytorch/pytorch#189228)."""
    import time

    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))
    with set_current_vllm_config(config):
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": str(port),
            }
        )
        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        comm = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        ).symm_mem_comm
        if comm is None or comm.disabled:
            q.put("skip: SymmMemCommunicator is not available or disabled.")
            return
        x = torch.randn(32 * 4096, dtype=torch.bfloat16, device=device)
        golden = x.clone()
        dist.all_reduce(golden, group=get_tp_group().device_group)
        streams = [torch.cuda.Stream(device=device) for _ in range(2)]
        outs = []
        for i in range(300):
            if local_rank == i % world_size and i % 7 == 0:
                time.sleep(0.02)
            with torch.cuda.stream(streams[i % 2]):
                outs.append(comm.all_reduce(x))
            if i % 50 == 49:
                torch.accelerator.synchronize()
                # Every result, not only the last: a copy out of the shared
                # buffer that runs after the next op wrote it is wrong too.
                # Reduction order differs from NCCL by a few bf16 ulps; the
                # cross-stream race corrupts elements by 10 to 1000.
                for out in outs:
                    torch.testing.assert_close(out, golden, atol=0.5, rtol=0.02)
                outs.clear()
        torch.accelerator.synchronize()
        q.put("ok")


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.parametrize("tp_size", [2, 8])  # two-shot and multimem on Blackwell
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_symm_mem_allreduce_from_two_streams_does_not_deadlock(tp_size):
    if tp_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 12000 + random.randint(0, 999)
    procs = [
        ctx.Process(target=symm_mem_two_streams_worker, args=(r, tp_size, port, q))
        for r in range(tp_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=240)
    hung = [p.pid for p in procs if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.kill()
    cleanup_dist_env_and_memory()
    assert not hung, f"symm-mem all-reduce from two streams deadlocked: {hung}"
    results = [q.get(timeout=5) for _ in procs]
    if any(r.startswith("skip") for r in results):
        pytest.skip(results[0])
    assert results == ["ok"] * tp_size
    assert all(p.exitcode == 0 for p in procs)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_dp_with_symm_mem_allreduce(monkeypatch: pytest.MonkeyPatch):
    world_size = 4
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    # Verify that the DataParallel runs without error
    engine_args = EngineArgs(
        model="distilbert/distilgpt2",
        enforce_eager=True,
        enable_prefix_caching=True,
        data_parallel_size=2,
        tensor_parallel_size=2,
        data_parallel_backend="mp",
    )
    LLMEngine.from_engine_args(engine_args)
