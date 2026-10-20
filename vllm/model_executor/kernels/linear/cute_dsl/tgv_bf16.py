# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from FlashInfer's flashinfer/gemm/kernels/tgv_gemm_cute_ext.py
# (Copyright (c) 2026, NVIDIA CORPORATION; Copyright (c) 2026 by FlashInfer team;
# Apache-2.0), with a cluster exit barrier for 2-CTA launches and a warp that
# prefetches the weight into L2 during the PDL wait.
"""CuTe DSL TGV BF16 GEMM, a low-latency GEMM for decode-sized M on SM10x.

Computes ``out[M, N] = x[M, K] @ weight[N, K].T (+ bias[N])`` with bf16 inputs,
fp32 accumulation and bf16 output. The kernel computes the transposed product
(kernel M = weight rows), so the weight is the TMA-streamed A operand and the
activations are B.

Warp specialization (8 warps, 256 threads per CTA):
  Warp 0    DMA_A   TMA-loads A (weight) tiles; starts before the PDL wait
  Warp 1    DMA_B   TMA-loads B (activation) tiles after griddepcontrol.wait
  Warp 2    MMA     tcgen05.mma into TMEM; owns alloc/dealloc
  Warp 3    L2PF    (l2_prefetch_tiles > 0 only) L2-prefetches the A k-tiles
                    beyond the smem ring while DMA_B waits on PDL
  Warps 4-7 EPILOG  TMEM -> RMEM -> bf16 cast -> st.global
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack, make_fake_stream

_CTA_K = 128

# Budget of the weight L2 prefetch (TgvGemmKernel.l2_prefetch_warp).
# Size: with one CTA per SM, each CTA prefetches at most its share of half the
# L2, so the prefetched tiles fit next to what the running kernels keep there
# and are still resident when the refills read them.
# Rate: the grid issues at the rate streaming reads sustain, 80% of the peak
# HBM rate (a large device copy reaches 6.5 of 8 TB/s on B300); unpaced
# prefetches queue behind the predecessor's traffic and slow the GEMM down.
# On B300 this gives 448 KB per CTA and ~3 KB per SM clock.
_L2_PREFETCH_L2_SHARE = 0.5
_L2_PREFETCH_HBM_EFFICIENCY = 0.8


@functools.cache
def _device_budget(device_index: int) -> tuple[int, int, int]:
    """(SM count, L2 prefetch bytes per CTA, L2 prefetch bytes per SM clock)."""
    props = torch.cuda.get_device_properties(device_index)
    num_sms = props.multi_processor_count
    per_cta = int(props.L2_cache_size * _L2_PREFETCH_L2_SHARE) // num_sms
    # Peak HBM bytes per second (double data rate) over the SM clock, both in kHz.
    peak = 2 * props.memory_clock_rate * props.memory_bus_width // 8
    per_clk = int(peak * _L2_PREFETCH_HBM_EFFICIENCY) // props.clock_rate
    return num_sms, per_cta, max(1, per_clk)


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes. Mirrors the original WorkTileInfo."""

    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32
    K_idx_end: cutlass.Int32


class TgvGemmKernel:
    """Low-latency Blackwell GEMM kernel rewritten with cute_ext primitives,
    keeping the raw-mbarrier 7-warp specialization from the C++ kernel
    in ``include/flashinfer/gemm/tgv_gemm.cuh``.
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        cta_m: int = 64,
        cta_n: int = 8,
        cta_k: int = _CTA_K,
        num_ab_stage: int = 8,
        use_2cta: bool = False,
        use_pdl: bool = False,
        pdl_launch: bool | None = None,
        pdl_count: int = -1,
        has_bias: bool = False,
        out_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        l2_prefetch_tiles: int = 0,
        l2_prefetch_bytes_per_clk: int = 1,
    ):
        self.acc_dtype = acc_dtype
        self.out_dtype = out_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.use_2cta = use_2cta
        self.use_pdl = use_pdl
        self.pdl_launch = pdl_launch if pdl_launch is not None else use_pdl
        self.pdl_count = pdl_count
        # has_bias: when True, kernel reads a (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        # bias tensor (M-broadcast over N,L), converts it to fp32 in RMEM, and
        # adds it to the accumulator before the bf16 cast. When False, all
        # bias-related code is elided via cutlass.const_expr.
        self.has_bias = has_bias
        # l2_prefetch_tiles: k-tiles past the smem ring that warp 3
        # L2-prefetches while DMA_B waits on PDL (0 = off). Needs a K-major A
        # and PDL.
        if l2_prefetch_tiles and (not use_pdl or cta_m % 32 != 0):
            raise ValueError("TGV L2 prefetch needs use_pdl and cta_m % 32 == 0")
        self.l2_prefetch_tiles = l2_prefetch_tiles
        # SM clocks per grid CTA between two of one CTA's tile prefetches.
        self.l2_prefetch_clk_per_cta_tile = max(
            1, round(cta_m * cta_k * 2 / l2_prefetch_bytes_per_clk)
        )

        # 1-CTA: cta_n ∈ [8, 256] step 8 (bf16 tcgen05.mma atom limit).
        # 2-CTA: cta_n ∈ [16, 256] step 16 (bf16 K-major cluster mma).
        min_n, step_n = (16, 16) if use_2cta else (8, 8)
        if cta_n < min_n or cta_n > 256 or cta_n % step_n != 0:
            raise ValueError(
                f"cta_n={cta_n} invalid for use_2cta={use_2cta}: "
                f"bf16 K-major mma requires N ∈ [{min_n}, 256] step {step_n}"
            )

        # Fixed configuration matching the C++ / DSL kernels.
        self.threads_per_cta = 256  # 8 warps (warp 3 unused unless L2 prefetch)
        if use_2cta:
            # 2-CTA cluster along M; joint MMA tile = (cta_m*2, cta_n).
            self.cluster_shape = (2, 1, 1)
            self.mma_tiler_mn = (cta_m * 2, cta_n)
            self.cta_group = tcgen05.CtaGroup.TWO
            self.tma_op = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM
        else:
            # 1 SM mode, 1x1 cluster, no multicast.
            self.cluster_shape = (1, 1, 1)
            self.mma_tiler_mn = (cta_m, cta_n)
            self.cta_group = tcgen05.CtaGroup.ONE
            self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD

    def __repr__(self) -> str:
        return (
            f"TgvGemmKernel_cta{self.cta_m}x{self.cta_n}x{self.cta_k}"
            f"_2cta{int(self.use_2cta)}_pdl{int(self.use_pdl)}"
            f"_bias{int(self.has_bias)}"
            f"_out{self.out_dtype.__name__.lower()}"
            # The L2-prefetch variant gets its own kernel name.
            + (f"_l2pf{self.l2_prefetch_tiles}" if self.l2_prefetch_tiles else "")
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        b: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        c: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        # (Gemm_M, Gemm_N, Gemm_L):(1,0,0); unused when has_bias=False
        bias: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # Each CTA processes one (CTA_M, CTA_N) output tile, no persistence.
        # Round the (M, N) tile counts up to a full cluster so 2-CTA launches
        # don't leave an odd M-CTA without its peer (e.g. M=2880, cta_m=64 →
        # 45 M-tiles, invalid for cluster=(2,1,1)). cute_ext OOB-protects the
        # extra CTA's GMEM stores.
        grid = cute.round_up(
            (
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n),
                c.layout.shape[2],
            ),
            self.cluster_shape,
        )
        self.kernel(a, b, c, bias).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        # (Gemm_M, Gemm_N, Gemm_L):(1,0,0); unused when has_bias=False
        mBias: cute.Tensor,
    ):
        """Device-side dispatcher: build MMA descriptor, allocate SMEM/TMEM/
        mbarriers, init barriers, dispatch into dma_a/dma_b/mma/epilog warps.
        """
        DMA_Stage = self.num_ab_stage

        # ---- Tiled MMA ----
        # make_trivial_tiled_mma picks the largest tcgen05.mma atom that fits
        # mma_tiler_mn. 1-CTA bf16 (64,8): Mma_M=(16,4)=64, Mma_N=8, Mma_K=16.
        # 2-CTA bf16 K-major (128,N): Mma_M=128 split across cluster, Mma_K=16.
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type  # bf16
        c_dtype = mC.element_type  # bf16
        d_layout = utils.LayoutEnum.from_tensor(mC)

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            a_major,
            b_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)  # 1 (1-CTA) or 2 (2-CTA)

        # NumMma_K = CTA_K/Mma_K — inner-K trip count for the MMA warp.
        # bf16 default: Mma_K=16, CTA_K=128 → mma_inst_tile_k = 8.
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)  # CTA tile of A
        # 2-CTA splits B across cluster M (each CTA loads Mma_N/2 cols).
        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)  # CTA tile of C

        # ---- WorkTileInfo (static 1-tile-per-CTA scheduler) ----
        # blockIdx maps directly to (M_tile, N_tile, batch). K_idx_start/end
        # carry over from C++ for future split-K; here every CTA reduces the
        # full K range (= ceil(K/CTA_K) tiles, e.g. 12 for K=1536).
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # In 2-CTA mode: identify which CTA within the (2,1,1) cluster owns
        # the leader role (issues MMAs / multicasts commits). In 1-CTA mode
        # every CTA is its own leader.
        leader_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.use_2cta):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
            is_leader = cta_rank_in_cluster == 0
        else:
            cta_rank_in_cluster = leader_rank
            is_leader = cutlass.Boolean(True)

        # ---- SMEM A/B (DMA_Stage-staged, swizzled ComposedLayout) ----
        # alignment=1024 covers TMA's natural alignment requirements. 2-CTA
        # halves the M extent of sA and the N extent of sB internally (the SS
        # atom splits A across cluster M and B across cluster M).
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — Sw<3,4,3>
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mnk_tiler,
            ab_dtype,
            DMA_Stage,
        )  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — Sw<3,4,3>

        # sA per-CTA M extent = cta_m for both modes; sB per-CTA N extent =
        # cta_n (1-CTA) or cta_n/2 (2-CTA, SS atom splits B across cluster M).
        # ((Mma_M_per_cta, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            a_smem_layout_staged,
            alignment=1024,
        )
        # ((Mma_N_per_cta, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            b_smem_layout_staged,
            alignment=1024,
        )

        # ---- TMEM accumulator layout (manual alloc; see MMA warp) ----
        # cute_ext.make_tmem_layout_acc accounts for cta_group when handed
        # the cluster MMA tile, so the resulting layout is already per-CTA.
        # acc_layout per-CTA: ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        # = (((16,4), cta_n), 1, 1, 1) for both 1-CTA and 2-CTA modes.
        # TMEM is still allocated by hand (cute.arch.alloc_tmem below), which
        # is faster.
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma,
            self.mma_tiler_mn,
            acc_stage=1,
        )

        # ---- Raw mbarriers (Int64 SMEM arrays) + tmem_base_ptr Int32 slot ----
        # Two flavors of barriers exist on Hopper/Blackwell:
        #   1. Named barriers (bar.arv/bar.sync) — Ampere-era, 16 hardware
        #      barriers per SM, only handle intra-CTA thread sync.
        #   2. Mbarriers (mbarrier.* PTX) — 64-bit in SMEM per barrier,
        #      software-programmable. Used here for (a) thread sync within a
        #      CTA, (b) TMA transaction-count tracking (TMA→SM signaling),
        #      and (c) 1-arrival "wake the consumer warp" patterns.
        # Allocated as 1D tensors; .iterator gives a Pointer[Int64] supporting
        # `bar + stage` arithmetic. Arrival counts are in the module docstring.
        # With L2 prefetch, one extra slot (bar_full + DMA_Stage) is a
        # 1-arrival "PDL wait released" flag that DMA_B arrives on and the
        # prefetch warp polls (mbarrier.test_wait; nobody blocks on it).
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage + (1 if self.l2_prefetch_tiles else 0)),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(DMA_Stage),
            alignment=8,
        )
        bar_tma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_mma_epilog_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        bar_tmem_alloc_arr = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        # alloc_tmem writes the TMEM base address to this slot; both MMA and
        # EPILOG read it back via retrieve_tmem_ptr.
        tmem_base_arr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )

        bar_full = bar_full_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_empty = bar_empty_arr.iterator  # Pointer[Int64], DMA_Stage
        bar_tma_epilog = bar_tma_epilog_arr.iterator  # Pointer[Int64]
        bar_mma_epilog = bar_mma_epilog_arr.iterator  # Pointer[Int64]
        bar_tmem_alloc = bar_tmem_alloc_arr.iterator  # Pointer[Int64]
        tmem_base_ptr = tmem_base_arr.iterator  # Pointer[Int32]

        # ---- Barrier init (1 thread, PTX mbarrier.init is single-thread) ----
        # bar_full arrival = 2 * num_mma_ctas:
        #   1-CTA: 2 (own DMA_A + DMA_B)
        #   2-CTA: 4 (own + peer DMA_A/DMA_B, routed via SM100_TMA_LOAD_2SM
        #          onto the leader's bar). In 2-CTA the peer's copy is unused.
        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                if cutlass.const_expr(self.l2_prefetch_tiles > 0):
                    cute.arch.mbarrier_init(bar_full + DMA_Stage, 1)  # DMA_B
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(bar_tma_epilog, 32)  # whole DMA_B warp
                cute.arch.mbarrier_init(bar_mma_epilog, 1)  # MMA tcgen05.commit
                cute.arch.mbarrier_init(
                    bar_tmem_alloc, 32 + 128
                )  # MMA + 4 EPILOG warps

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.use_2cta):
            # Cluster-wide sync: ensure peer's barriers are visible before any
            # cross-CTA arrives. Splitting arrive/wait lets the per-CTA tile
            # setup below overlap with the cluster barrier. (Implicit
            # intra-CTA sync — cluster.arrive requires all threads to reach.)
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        work_tile_info = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k),
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        # ---- CTA-to-Value maps for TMA (constexpr) ----
        # Encodes "which cluster CTA is responsible for which slice", i.e.
        # the CTA-coord → logical-multicast-id layout. For our 1×1 cluster
        # (no multicast) it's an identity layout; cute_ext.tma_load still
        # needs this explicit layout to construct the TMA descriptor.
        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")

        # ---- Per-CTA tile views ----
        # local_tile(t, tiler, coord) = zipped_divide(t, tiler)[coord]; modes
        # passed `None` stay free. Per-CTA M/N extents:
        #   gA M extent = cta_m   (both modes; SS atom splits cluster M)
        #   gB N extent = cta_n   (1-CTA) or cta_n//2 (2-CTA)
        #   gD M×N      = cta_m × cta_n (cluster N is identical for both CTAs)
        gA_tile = cute.local_tile(  # (cta_m, cta_k, Tiles_K)
            mA,
            a_tiler_mk,
            (work_tile_info.M_idx, None, work_tile_info.L_idx),
        )
        # gB n-index: 1-CTA just bidy; 2-CTA each cluster N_idx covers
        # num_mma_ctas tiles of b_tiler, each CTA loads one half.
        if cutlass.const_expr(self.use_2cta):
            gB_n_idx = work_tile_info.N_idx * num_mma_ctas + cta_rank_in_cluster
        else:
            gB_n_idx = work_tile_info.N_idx
        gB_tile = cute.local_tile(  # (cta_n//num_mma_ctas, cta_k, Tiles_K)
            mB,
            b_tiler_nk,
            (gB_n_idx, None, work_tile_info.L_idx),
        )
        gD_tile = cute.local_tile(  # (cta_m, cta_n)
            mC,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )
        # gBias_tile: same (cta_m, cta_n) shape as gD_tile but stride (1, 0)
        # — the same M element is repeated across N. local_tile preserves the
        # (1, 0, 0) stride from mBias, so this works automatically.
        gBias_tile = cute.local_tile(  # (cta_m, cta_n) stride (1, 0)
            mBias,
            c_tiler_mn,
            (work_tile_info.M_idx, work_tile_info.N_idx, work_tile_info.L_idx),
        )

        if cutlass.const_expr(self.use_2cta):
            # Cluster sync deferred to here — tile-view setup above runs in
            # parallel with the cluster barrier latency.
            cute.arch.cluster_wait()

        # ---- Warp dispatch (warp 3 idle unless L2 prefetch; 256-thread parity) ----
        if warp_idx == 0:
            self.dma_a_warp(
                bar_full,
                bar_empty,
                leader_rank,
                gA_tile,
                sA,
                a_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                is_leader,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx == 3:
            if cutlass.const_expr(self.l2_prefetch_tiles > 0):
                self.l2_prefetch_warp(
                    mA,
                    work_tile_info.M_idx * self.cta_m,
                    work_tile_info.L_idx,
                    bar_full + DMA_Stage,
                )
        elif warp_idx >= 4:
            # Epilog tid is 128..255 in the CTA; offset to 0..127 for partition.
            epi_tid = tidx - 128
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD_tile,
                gBias_tile,
                epi_tid,
                c_dtype,
                d_layout,
            )

        if cutlass.const_expr(self.use_2cta):
            # Cluster-wide exit barrier (sgl-project/sglang#32907): cluster
            # sync previously existed only at kernel entry, so one CTA could
            # retire while its peer still had in-flight peer-redirected
            # mbarrier arrives / multicast tcgen05.commit / TMEM dealloc
            # targeting it -> CUDBG_EXCEPTION_CLUSTER_BLOCK_NOT_PRESENT
            # (Xid 13 "CTA Not Present"). All 8 warps (256 threads) of both
            # cluster CTAs arrive here after their dispatch branch returns,
            # so no CTA can exit until the peer's tail work has landed.
            # relaxed arrive (matches the entry barrier at L378): this fence is
            # only about CTA lifetime, not inter-CTA smem visibility -- every
            # cross-CTA datum above already flows through mbarrier phases.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    # ====================================================================
    # DMA_A WARP — TMA-loads A tiles into sA[..., stage], one per K-iter.
    # 1-CTA: SM90_TMA_LOAD into local sA, arrives on local bar_full.
    # 2-CTA: SM100_TMA_LOAD_2SM (cluster-aware); peer's complete_tx routes
    #        onto the LEADER's bar_full. arrive_and_expect_tx is also
    #        peer-redirected to leader_rank.
    # ====================================================================
    @cute.experimental.jit
    def dma_a_warp(
        self,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        leader_rank: cutlass.Int32,  # used for the 2-CTA arrive-peer redirect
        gA_tile: cute.Tensor,  # (CTA_M, CTA_K, Tiles_K) — this CTA's A strip
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        a_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        # Phase-bit walkthrough for `bar_empty` (DMA_Stage=2 example).
        # Init phase = 0; phase flips on each arrival-count round. We wait
        # on the *old* phase about to flip away.
        #   k_tile  stage  old empty_phase  → wait for flip
        #     0       0          1                0→1 (init=0≠1, passes)
        #     1       1          1                0→1
        #     2       0          0                1→0 (slot reused)
        #     3       1          0                1→0
        #     4       0          1                0→1
        # Flip empty_phase once per DMA_Stage iters.
        empty_phase = cutlass.Int32(1)
        pdl_count = self.pdl_count

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            # cute_ext.tma_load only manages TX bytes, NEVER the arrival
            # count (in the pipeline path producer_commit handles that).
            # bar_full[stage] needs 2 (1-CTA) or 4 (2-CTA) arrivals; we
            # arrive ourselves via the fused mbarrier.arrive.expect_tx PTX,
            # paired with update_expect_tx=False so TX bytes aren't
            # double-counted. peer_cta_rank_in_cluster=leader_rank in 2-CTA
            # routes the arrive onto the leader's bar; None (1-CTA) lands
            # on the local CTA's bar.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    # 1 stage of A SMEM = sizeof(bf16) × CTA_M × CTA_K
                    cute.size_in_bytes(
                        sA.element_type,
                        cute.slice_(sA.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gA_tile[None, None, k_tile],  # (CTA_M, CTA_K) GMEM slice
                sA[None, None, None, stage],  # ((Mma_M,Mma_K),NumMma_M,NumMma_K)
                (bar_full + stage).value,  # Pointer→ir.Value bridge
                cta_v_map=a_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

            # PDL: launch dependent grid at pdl_count (default -1 = end).
            if cutlass.const_expr(self.use_pdl):  # noqa: SIM102 (compile-time)
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # producer tail: drain the remaining bar_empty arrivals from MMA so the
        # CTA stays alive until MMA's last tcgen05.commit lands — otherwise the
        # arrive can hit freed SMEM (illegal memory access).
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    # ====================================================================
    # L2PF WARP (warp 3, only when l2_prefetch_tiles > 0) — L2-prefetches
    # this CTA's A k-tiles [DMA_Stage, DMA_Stage + l2_prefetch_tiles) while
    # the PDL wait blocks.
    #
    # DMA_A streams A (weights) before the PDL wait, but only DMA_Stage
    # k-tiles fit in the smem ring; the rest used to stream from HBM after
    # the wait. If the preceding kernel is latency-bound (all-reduce,
    # act_and_mul, ...), HBM idles meanwhile, so we pull those tiles into
    # L2 and the post-wait refills hit L2.
    #
    # - prefetch.global.L2, one 128 B line per lane per row, k-tile by
    #   k-tile (the order DMA_A will consume them).
    # - Starts after an estimate of the ring's own fill time and is paced so
    #   the whole grid issues at about the HBM rate (grid_ctas *
    #   l2_prefetch_clk_per_cta_tile clocks per tile per CTA).
    #   Unpaced prefetches issued while HBM is still busy (the previous
    #   GEMM's tail, back-to-back GEMMs, a short predecessor) were measured
    #   to slow the GEMM down instead.
    # - Stops as soon as DMA_B passes griddepcontrol.wait (bar_released):
    #   from then on the refill TMA loads fetch the data themselves.
    # Prefetches only move bytes into L2; results are unchanged.
    # ====================================================================
    @cute.experimental.jit
    def l2_prefetch_warp(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        m_base: cutlass.Int32,  # first A row of this CTA
        l_idx: cutlass.Int32,
        bar_released,  # Pointer[Int64], 1-arrival, arrived by DMA_B
    ):
        DMA_Stage = self.num_ab_stage
        lane = cute.arch.lane_idx()
        gemm_m = cute.size(mA, mode=[0])
        gemm_k = cute.size(mA, mode=[1])
        k_lo = DMA_Stage * self.cta_k
        k_hi = k_lo + self.l2_prefetch_tiles * self.cta_k
        if k_hi > gemm_k:
            k_hi = gemm_k
        if k_hi > k_lo:
            gx, gy, gz = cute.arch.grid_dim()
            interval = cutlass.Int64(gx * gy * gz * self.l2_prefetch_clk_per_cta_tile)
            t_next = cute.arch.clock64() + interval * DMA_Stage
            go = cutlass.Boolean(True)
            for k0 in cutlass.range(k_lo, k_hi, self.cta_k, unroll=1):
                while go and cute.arch.clock64() < t_next:
                    if cute.arch.mbarrier_test_wait(bar_released, 0):
                        go = cutlass.Boolean(False)
                t_next = t_next + interval
                if go:  # noqa: SIM102 (test the barrier only while going)
                    if cute.arch.mbarrier_test_wait(bar_released, 0):
                        go = cutlass.Boolean(False)
                if go:
                    for i in cutlass.range_constexpr(self.cta_m // 32):
                        m = m_base + i * 32 + lane
                        if m < gemm_m:
                            # 128 B lines = 64 bf16 along K.
                            for j in cutlass.range_constexpr(self.cta_k // 64):
                                kk = k0 + j * 64
                                if kk < gemm_k:
                                    ptr = mA[m, None, l_idx].iterator + kk
                                    cute.arch.prefetch(ptr.llvm_ptr, cache_level="L2")

    # ====================================================================
    # DMA_B WARP — same as DMA_A but for B; also drives the PDL
    # griddepcontrol.wait and signals the epilog when all activation loads
    # are issued.
    # 2-CTA: SM100_TMA_LOAD_2SM + arrive redirected to leader_rank.
    # ====================================================================
    @cute.experimental.jit
    def dma_b_warp(
        self,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        bar_tma_epilog,  # Pointer[Int64], 1 entry (32-arrival)
        leader_rank: cutlass.Int32,
        gB_tile: cute.Tensor,  # (CTA_N, CTA_K, Tiles_K) — this CTA's B strip
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        b_cta_v_map: cute.Layout,
        k_tile_count: cutlass.Int32,
    ):
        DMA_Stage = self.num_ab_stage

        # PDL griddepcontrol.wait: B is the activation, produced by an
        # upstream kernel. Block until that kernel signals completion. A
        # is weights (already resident), so DMA_A doesn't need this.
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        # Tell the L2 prefetch warp the wait is over (it stops).
        if cutlass.const_expr(self.l2_prefetch_tiles > 0):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(bar_full + DMA_Stage)

        # See dma_a_warp for empty_phase + fused arrive-and-expect-tx logic.
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)

            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    # 1 stage of B SMEM = sizeof(bf16) × CTA_N × CTA_K
                    cute.size_in_bytes(
                        sB.element_type,
                        cute.slice_(sB.layout, (None, None, None, 0)),
                    ),
                    peer_cta_rank_in_cluster=leader_rank if self.use_2cta else None,
                )
            cute_ext.tma_load(
                gB_tile[None, None, k_tile],  # (CTA_N, CTA_K) GMEM slice
                sB[None, None, None, stage],  # ((Mma_N,Mma_K),NumMma_N,NumMma_K)
                (bar_full + stage).value,
                cta_v_map=b_cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )

            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

        # 32-thread arrive on bar_tma_epilog: signals epilog "activations
        # issued" so the bias-load can fire after all B tiles have landed.
        # Only needed when has_bias=True; elided otherwise.
        if cutlass.const_expr(self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)

        # producer tail: drain the remaining bar_empty arrivals from MMA so the
        # CTA stays alive until MMA's last tcgen05.commit lands — otherwise the
        # arrive can hit freed SMEM (illegal memory access).
        for k_tile in cutlass.range(DMA_Stage, unroll=1):
            stage = (k_tile + k_tile_count) % DMA_Stage
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == (DMA_Stage - 1):
                empty_phase = empty_phase ^ 1

    # ====================================================================
    # MMA WARP — owns the TMEM accumulator and issues every tcgen05.mma.
    # 1-CTA: every CTA's MMA warp runs the K-loop on its own TMEM.
    # 2-CTA: BOTH CTAs alloc/dealloc TMEM (cluster-coherent via
    #        is_two_cta=True); only the leader runs the K-loop; commits
    #        multicast to both CTAs (mask=0b11) so each one's bar_empty /
    #        bar_mma_epilog still gets signaled.
    # ====================================================================
    @cute.experimental.jit
    def mma_warp(
        self,
        is_leader: cutlass.Boolean,
        bar_full,  # Pointer[Int64], DMA_Stage entries
        bar_empty,  # Pointer[Int64], DMA_Stage entries
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot for TMEM addr
        acc_layout: cutlass.Constexpr,  # TMEM accumulator layout
        mma_inst_tile_k: cutlass.Constexpr,  # NumMma_K — inner loop count
        k_tile_count: cutlass.Int32,  # Tiles_K — outer loop count
    ):
        DMA_Stage = self.num_ab_stage

        # ---- TMEM allocation (manual) ----
        # We allocate TMEM explicitly here rather than via the convenience
        # helper ``cute_ext.allocate(tmem)`` so that we can *hide the TMEM
        # allocation latency*: the alloc must happen here in MMA's prolog
        # (we need the TMEM ptr before the first tcgen05.mma), but it
        # serializes against the next CTA's own alloc on the back-to-back
        # PDL launch path. By manually emitting
        # ``relinquish_tmem_alloc_permit`` right after our alloc completes,
        # the next CTA can start ITS allocation in parallel with this
        # CTA's MMA work rather than stalling at its own prolog waiting
        # for us. cute_ext.allocate(tmem) does not expose this early-
        # relinquish control — worth ~3% on the PDL hot path.
        #
        # TMEM on SM100 has 128 lanes × 512 columns × 4B = 256KB total. We
        # allocate half (256 of 512 cols) so the next CTA can prefetch its
        # alloc on the other half. For CTA_M=64 the accumulator only uses
        # 64 lanes (16 lanes × 4 subpartitions), well within half-of-TMEM.
        # 2-CTA: alloc/relinquish/dealloc are cluster-coherent (is_two_cta=True).
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # phase 0: 32 of 160
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)

        # Bind acc_layout to the just-allocated TMEM ptr. Drop the trailing
        # AccStage=1 mode so the MMA atom sees (MMA, Mma_M, Mma_N).
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        acc_view = tAcc[
            None, None, None, 0
        ]  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N)

        if is_leader:
            # MMA atom: each cute_ext.dot call = 1 tcgen05.mma. ACCUMULATE:
            #   False → C  = A*B  (first inst of first k-tile only)
            #   True  → C += A*B
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            # 2-CTA: mask=0b11 multicasts the commit to both CTAs' bar_empty.
            commit_mask = 0b11 if self.use_2cta else None

            full_phase = cutlass.Int32(0)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % DMA_Stage
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)

                # Inner-K loop: NumMma_K straight-line tcgen05.mma instructions.
                # cute_ext.dot expects rank-3 operands in the MMA fragment
                # profile (MMA_atom, REST_M, REST_K). Slicing
                # `sA[None,None,k_block,stage]` leaves rank 2
                # ((Mma_M, Mma_K), NumMma_M), so pad explicitly via
                # cute.append_ones.
                for k_block in range(mma_inst_tile_k):
                    if k_block == 0:
                        # First inst of the very first k_tile clears TMEM.
                        mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    else:
                        mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                    a_frag = cute.append_ones(
                        sA[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    b_frag = cute.append_ones(
                        sB[None, None, k_block, stage],
                        up_to_rank=3,
                    )
                    cute_ext.dot(mma_atom, a_frag, b_frag, acc_view)

                # CRITICAL: tcgen05.commit MUST be inside elect_one(). Unlike
                # the C++ ``cutlass::arch::umma_arrive`` helper (which has an
                # internal elect_one_sync), the DSL's tcgen05.commit has no
                # guard — without this, all 32 lanes commit (32× redundant
                # arrivals on bar_empty). This was a major perf bug encountered
                # during development.
                with cute.arch.elect_one():
                    tcgen05.commit(bar_empty + stage, commit_mask, self.cta_group)

                if (k_tile % DMA_Stage) == (DMA_Stage - 1):
                    full_phase = full_phase ^ 1

            # Wake the epilog (1-thread arrive on the 1-arrival barrier).
            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, commit_mask, self.cta_group)

        # ---- TMEM dealloc: wait for own EPILOG's tcgen05.ld to retire, free.
        # bar_tmem_alloc phase 1 fires after EPILOG's tcgen05.ld is observable
        # (post fence_view_async_tmem_load). 2-CTA dealloc is per-CTA (no
        # cross-CTA handshake) because each CTA owns its own physical TMEM
        # half; the cluster-shared accumulator is just a logical view.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)  # phase 1: 32 of 160
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=self.use_2cta)

    # ====================================================================
    # EPILOG WARPS — TMEM → RMEM → bf16 cast → direct st.global to GMEM.
    # ====================================================================
    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,  # Pointer[Int64], 1 entry (32-arrival)
        bar_mma_epilog,  # Pointer[Int64], 1 entry
        bar_tmem_alloc,  # Pointer[Int64], 1 entry, 160-arrival
        tmem_base_ptr,  # Pointer[Int32] — SMEM slot from MMA
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,  # (CTA_M, CTA_N) — this CTA's output tile
        gBias_tile: cute.Tensor,  # (CTA_M, CTA_N) with stride (1,0) — bias broadcast
        epi_tid: cutlass.Int32,  # 0..127 within the 4 EPILOG warps
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
    ):
        # Sync with MMA's alloc_tmem (phase 0 of bar_tmem_alloc): 128 arrivals
        # from this warp + 32 from MMA = 160. tmem_base_ptr is uninitialized
        # before the wait clears.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        # Bind acc_layout to the TMEM ptr written by alloc_tmem.
        # acc_layout shape: ((Mma_M, Mma_N), NumMma_M, NumMma_N, AccStage)
        #                 = (((16,4), Mma_N), 1, 1, 1).
        # acc_view drops the 1×1×1 outer modes → (Mma_M, Mma_N).
        #
        # Concrete TMEM layout for 1-CTA bf16 M64 N8:
        #   tmem_[32b](0x0...) o (((16,4),8),1,1,1):(((65536,2097152),1),0,0,0)
        #   TMEM addr = [31:16=dp_lane, 15:0=column]:
        #     stride between dp lanes = 1<<16 = 65536           (Mma_M_per_subp)
        #     stride between subparts = 65536 × 32 = 2097152    (NumSubp=4)
        #     stride between cols     = 1                       (N is contiguous)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        tCtAcc = cute.make_tensor(
            tmem_ptr, acc_layout
        )  # ((Mma_M_per_cta, Mma_N), NumMma_M, NumMma_N, AccStage)
        acc_view = tCtAcc[((None, None), 0, 0, 0)]  # (Mma_M_per_cta, Mma_N)

        # ---- t2r tiled-copy + per-thread RMEM layout ----
        # ``sm100_utils.get_tmem_load_op`` picks the best tcgen05.ld atom
        # for the (M, N, K) tile and output dtype. For CTA_M=64, M/N-major
        # output, the chosen atom is SM100_TMEM_LOAD_16dp256b1x — the 16dp
        # variant is required because each MMA subpartition only uses 16
        # of the 32 lanes per subpartition (CTA_M=64 split across 4 subps =
        # 16 lanes each). Other tcgen05.ld atoms would also be functional;
        # 16dp is just optimal here.
        # epi_tile = full CTA tile: N=8 fits in one tcgen05.ld instance
        # (16dp × 8col → 4 regs per thread × 128 threads). Larger N would
        # need a sub-tile loop here.
        epi_tile = (self.cta_m, self.cta_n)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            d_layout,
            c_dtype,
            self.acc_dtype,
            epi_tile,
            self.use_2cta,
        )
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(copy_atom_t2r, acc_view)

        # flat_divide → (CTA_M, CTA_N, Rest_M=1, Rest_N=1); the rank-4 shape
        # is required because make_t2r_rmem_layout internally does [...,0,0].
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # rmem_layout matches one t2r copy's destination. The TMEM-side
        # partition (CpyS) per-subpartition view is shape ((CpyS_N=8,
        # CpyS_M=16), NumCpy_M=1, NumCpy_N=Mma_N/CpyS_N) — i.e. each
        # tcgen05.ld.16dp256bit.x1 instance moves 16dp×8col elements per
        # subpartition. Per the PTX page for tcgen05.ld.16dp256bit, thread
        # 0 of an SM100_TMEM_LOAD_16dp256b1x copy receives 4 registers at
        # output coordinates (0,0), (0,1), (8,0), (8,1) — i.e. a (2,2)
        # per-thread value-tile, hence CpyD = (2,2). The dst rmem layout
        # mirrors this CpyD shape but with stride (1, 2) so the registers
        # are stored contiguously.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(  # fp32, per-thread
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(  # bf16, per-thread
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        # ---- Bias setup (GMEM→RMEM via partition_and_copy, predicate auto) ----
        # gBias_tile has the same (CTA_M, CTA_N) shape as gD_tile but with
        # stride (1, 0) — the bias value depends only on the M coord. Each
        # lane needs to read the same (m, n) coord it writes back to gD so
        # bias and accumulator align register-for-register.
        #
        # partition_and_copy slices the GMEM src using
        # `tiled_copy.layout_src_tv_tiled`. With t2r that's the TMEM (CpyS)
        # side, which is 16dp×8col=128 elements/group — doesn't match rBias's
        # CpyD=4 elements/thread. `make_tiled_copy_D` builds a new TiledCopy
        # whose src/dst TV layouts both equal t2r's *dst* (CpyD) layout, so
        # the GMEM src gets sliced to CpyD per thread — matching rBias. OOB
        # is handled by the cute_ext lowering's auto-predBounds (same story
        # as the st.global to mC).
        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            gBias_epi = cute.flat_divide(gBias_tile, epi_tile)
            tiled_copy_g2r = cute.make_tiled_copy_D(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                tiled_copy_t2r,
            )
            thr_g2r = tiled_copy_g2r.get_slice(epi_tid)
            rBias = cute_ext.allocate(  # bias dtype (e.g. bf16)
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(  # converted to fp32
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )

        # ---- Wait for data: B loads done + MMA done ----
        # Only needed when has_bias=True (DMA_B signals bar_tma_epilog, epilog
        # waits before issuing the bias load so it hides behind MMA). Without
        # bias the bar is unused — neither arrive nor wait fires.
        if cutlass.const_expr(self.has_bias):
            cute.arch.mbarrier_wait(bar_tma_epilog, 0)  # all activations issued

        # ---- Bias load: GMEM → RMEM (predicated ld.global) and convert to fp32 ----
        # Done before the MMA wait so the load latency hides behind MMA.
        if cutlass.const_expr(self.has_bias):
            cute_ext.partition_and_copy(thr_g2r, gBias_epi[None, None, 0, 0], rBias)
            rBiasAcc.store(rBias.load().to(self.acc_dtype))

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)  # accumulator ready

        # TMEM → RMEM (one tcgen05.ld for the 64×8 tile).
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)

        # tcgen05.ld is async — fence makes the result visible to (a) the
        # rAcc.load() below and (b) MMA's dealloc_tmem after we arrive
        # on bar_tmem_alloc phase 1.
        cute.arch.fence_view_async_tmem_load()

        # Phase 1 of bar_tmem_alloc: 128 from this warp + 32 from MMA = 160.
        # MMA's mbarrier_wait(bar_tmem_alloc, 1) clears and it dealloc's.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Add bias in fp32 before the dtype cast.
        if cutlass.const_expr(self.has_bias):
            rAcc.store(rAcc.load() + rBiasAcc.load())

        # fp32 → bf16 cast in RMEM.
        rD.store(rAcc.load().to(c_dtype))

        # RMEM → GMEM: cute_ext.partition_and_copy does two things —
        #   (a) PARTITION: splits non-RMEM operands by thr_copy's thread-value
        #       layout (RMEM operands skipped — already per-thread). Here only
        #       gD_epi (the dst) gets partitioned; rD stays as-is.
        #   (b) INSTRUCTION: picked from (src, dst) memspace pair —
        #         TMEM→RMEM = tcgen05.ld     (atom-driven)
        #         RMEM→GMEM = st.global     (simt_auto_vec_copy)
        #         GMEM→SMEM = cp.async       (async_op=True)
        # We REUSE thr_t2r so each lane writes to the same (m, n) GMEM coords
        # where it read from TMEM. A different copy atom would scatter
        # registers to the wrong addresses.
        #
        # OOB STORES are handled automatically by the cute_ext lowering —
        # simt_auto_vec_copy infers predBounds from the destination tensor's
        # MemRef shape (propagated from mC through local_tile / flat_divide).
        # Lanes whose (m, n) coord is past mC's shape simply don't issue an
        # st.global. Verified with a non-aligned M=63 test: writes to in-bounds rows
        # 0..62 land correctly, the M=63 row + the rest of the (64,8) tile
        # stay untouched. So the C++ / vanilla-DSL kernels' explicit predicate
        # dance (make_identity_tensor + elem_less + basic_copy_if) isn't
        # needed; the explicit `predicated_tensor_origin` marker isn't either.
        cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])


# =====================================================================
# BMM wrappers — feed (L, M, K) / (L, K, N) / (L, M, N) PyTorch-shaped
# cute tensors and select-permute them into the kernel's (M, K, L) /
# (N, K, L) / (M, N, L) view.
# =====================================================================
@cute.experimental.jit
def _bmm_no_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (L, M, K) from PyTorch
    b: cute.Tensor,  # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,  # (L, M, N) from PyTorch (permuted)
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # Dummy bias aliasing C — never dereferenced when has_bias=False.
    bias = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2]))
    gemm_op(a, b, c, bias, stream)


@cute.experimental.jit
def _bmm_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,  # (L, M, K) from PyTorch
    b: cute.Tensor,  # (L, K, N) from PyTorch (permuted)
    c: cute.Tensor,  # (L, M, N) from PyTorch (permuted)
    bias: cute.Tensor,  # (L, M, N):(0,1,0) — M-broadcast via as_strided
    stream: cuda.CUstream,
):
    a = cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0]))
    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0]))
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # After [1,2,0] permute: (M, N, L):(1, 0, 0) — matches C++ make_layout_Bias.
    bias = cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0]))
    gemm_op(a, b, c, bias, stream)


# Compiled kernels by config. The kernel marks its layouts dynamic, so one
# binary serves every (M, N, K) with this module's fixed operand layouts.
_COMPILED: dict[tuple, object] = {}


def _compile(
    cta_m: int,
    cta_n: int,
    num_ab_stage: int,
    use_2cta: bool,
    pdl: bool,
    has_bias: bool,
    l2_prefetch_tiles: int,
    l2_prefetch_bytes_per_clk: int,
):
    gemm = TgvGemmKernel(
        cta_m=cta_m,
        cta_n=cta_n,
        num_ab_stage=num_ab_stage,
        use_2cta=use_2cta,
        use_pdl=pdl,
        has_bias=has_bias,
        l2_prefetch_tiles=l2_prefetch_tiles,
        l2_prefetch_bytes_per_clk=l2_prefetch_bytes_per_clk,
    )
    weight = torch.empty(8, 128, dtype=torch.bfloat16, device="cuda")
    x = torch.empty(64, 128, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(64, 8, dtype=torch.bfloat16, device="cuda")
    bias = torch.empty(8, dtype=torch.bfloat16, device="cuda") if has_bias else None
    args = _to_cute(x, weight, out, bias)
    fn = _bmm_bias if has_bias else _bmm_no_bias
    return cute_ext.compile(fn, gemm, *args, make_fake_stream())


def _to_cute(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple:
    """PyTorch (M, K) x, (N, K) weight, (M, N) out -> (L, ...) kernel views.

    The kernel computes out.T = weight @ x.T: kernel A = weight (K-major),
    kernel B = x.T (K-major), kernel C = out.T (kernel M = N contiguous).
    """
    a = from_dlpack(weight.unsqueeze(0), assumed_align=32).mark_layout_dynamic(
        leading_dim=2
    )
    b = from_dlpack(
        x.unsqueeze(0).transpose(1, 2), assumed_align=32
    ).mark_layout_dynamic(leading_dim=1)
    c = from_dlpack(
        out.unsqueeze(0).transpose(1, 2), assumed_align=32
    ).mark_layout_dynamic(leading_dim=1)
    if bias is None:
        return a, b, c
    m, n = out.shape
    # Per-output-feature bias = kernel-M bias, broadcast over kernel N and L.
    bias_3d = bias.as_strided(size=(1, n, m), stride=(0, 1, 0))
    return (
        a,
        b,
        c,
        from_dlpack(bias_3d, assumed_align=2).mark_layout_dynamic(leading_dim=1),
    )


# (cta_m, cta_n, num_ab_stage, use_2cta), smallest output tile first.
_TACTIC_LADDER = (
    (64, 16, 12, True),
    (64, 32, 11, True),
    (64, 64, 9, True),
    (64, 128, 7, True),
    (128, 16, 6, False),
    (128, 32, 6, True),
    (128, 64, 5, True),
)


def _grid_ctas(m: int, n: int, cta_m: int, cta_n: int, use_2cta: bool) -> int:
    # Kernel M is the output's N, kernel N its M.
    num_mma_ctas = 2 if use_2cta else 1
    return num_mma_ctas * -(n // -(cta_m * num_mma_ctas)) * -(m // -cta_n)


@functools.cache
def _pick_tactic(m: int, n: int, num_sms: int) -> tuple[int, int, int, bool]:
    """The first ladder tactic whose grid fits one wave of the SMs, else the
    tactic with the fewest CTAs."""
    best, best_ctas = None, None
    for tactic in _TACTIC_LADDER:
        ctas = _grid_ctas(m, n, tactic[0], tactic[1], tactic[3])
        if ctas <= num_sms:
            return tactic
        if best_ctas is None or ctas < best_ctas:
            best, best_ctas = tactic, ctas
    assert best is not None
    return best


def tgv_bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    pdl: bool,
    l2_prefetch: bool = True,
) -> torch.Tensor:
    """``x[M, K] @ weight[N, K].T (+ bias[N])`` in bf16 on SM10x.

    ``x`` and ``weight`` must be contiguous bf16 with 32-byte aligned data and
    K a multiple of 8, and M >= 1. With ``pdl`` the kernel streams the weight
    while it waits on the previous kernel; ``l2_prefetch`` (PDL only) also
    prefetches the weight tiles past its shared-memory ring into L2 during
    that wait. The prefetch never changes the result.
    """
    m, k = x.shape
    n = weight.shape[0]
    num_sms, per_cta, per_clk = _device_budget(x.device.index)
    cta_m, cta_n, num_ab_stage, use_2cta = _pick_tactic(m, n, num_sms)
    l2_prefetch_tiles = (
        per_cta // (cta_m * _CTA_K * x.element_size())
        if l2_prefetch and pdl and k > num_ab_stage * _CTA_K
        else 0
    )
    key = (
        cta_m,
        cta_n,
        num_ab_stage,
        use_2cta,
        pdl,
        bias is not None,
        l2_prefetch_tiles,
        per_clk,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = _COMPILED[key] = _compile(*key)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x.device)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(*_to_cute(x, weight, out, bias), stream)
    return out
