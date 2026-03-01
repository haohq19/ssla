# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ops.utils import prepare_chunk_indices
from ops.utils.op import exp, safe_exp
from utils.ela_utils import input_guard

from ..common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from ..utils.cumsum import chunk_local_cumsum


BH_LIST = [1, 16]


# This kernel is used to calculate the attention map of each chunk
# It calculates the no-diagonal attention map between sub-chunks inside a chunk
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps, num_stages=num_stages)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"]
)
@triton.jit()
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    NC: tl.constexpr,
):
    # q, k, g: (T, H, D)
    # A: (T, H, BT)
    # i_t: chunk index
    # i_c: sub-chunk index
    # i_h: head chunk index
    i_t, i_c, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    
    # i_n: sequence index
    # i_t: chunk index within the sequence
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos   # sequence length

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BH, BC, BC], dtype=tl.float32)
    # for i_k in range(tl.cdiv(K, BK)):
        # o_k = i_k * BK + tl.arange(0, BK)
        # m_k = o_k < K
    # q: (T, H, K)
    p_q = tl.make_block_ptr(q + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))
    # g: (T, H, K)
    p_g = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))
    # k: (T, H, K)    
    p_k = tl.make_block_ptr(k + bos*H*K, (H, K, T), (K, 1, H*K), (i_h * BH, 0, i_t * BT + i_j * BC), (BH, BK, BC), (1, 0, 2))
    p_gk = tl.make_block_ptr(g + bos*H*K, (H, K, T), (K, 1, H*K), (i_h * BH, 0, i_t * BT + i_j * BC), (BH, BK, BC), (1, 0, 2))
    # gn is 
    p_gn = tl.make_block_ptr(g + (bos + i_t * BT + i_i * BC) * H*K, (H, K), (K, 1), (i_h * BH, 0), (BH, BK), (1, 0))
    # p_gn = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k

    # (BH, BK)
    b_gn = tl.load(p_gn, boundary_check=(0, 1))
    # (BH, BC, BK)
    b_q = tl.load(p_q, boundary_check=(0, 1, 2))
    b_g = tl.load(p_g, boundary_check=(0, 1, 2))
    b_qg = b_q * exp(b_g - b_gn[:, None, :]) * scale
    # (BH, BK, BC)
    b_k = tl.load(p_k, boundary_check=(0, 1, 2))
    b_gk = tl.load(p_gk, boundary_check=(0, 1, 2))
    b_kg = b_k * exp(b_gn[:, :, None] - b_gk)
    # (BH, BC, BC)
    b_A += tl.dot(b_qg, b_kg)

    # A: (T, H, BT)
    p_A = tl.make_block_ptr(A + bos*H*BT, (H, T, BT), (BT, H*BT, 1), (i_h * BH, i_t * BT + i_i * BC, i_j * BC), (BH, BC, BC), (1, 2, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1, 2))


# This kernel is used to calculate the attention map of each chunk
# It calculates the diagonal attention map inside each sub-chunk
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BT"]
)
@triton.jit()
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):  
    # q, k, g: (T, H, D)
    # A: (T, H, BT)
    # BT: chunk size
    # BC: sub-chunk size
    # i_t is the chunk index
    # i_i is the sub-chunk index
    # i_h is the head index
    i_t, i_i, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_j = i_i
    # i_n is the sequence index
    # i_t is the chunk index within the sequence
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    # o_k = tl.arange(0, BK)
    # m_k = o_k < K
    # m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    # o_A = (bos + i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_j * BC
    
    # q: (T, H, D)
    p_q = tl.make_block_ptr(q + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))
    p_g = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))
    # p_k = k + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k     # (D,)
    # p_gk = g + (bos + i_t * BT + i_j * BC) * H*K + i_h * K + o_k    # (D,)

    b_q = tl.load(p_q, boundary_check=(0, 1, 2))
    b_g = tl.load(p_g, boundary_check=(0, 1, 2))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_k = tl.make_block_ptr(k + (bos + i_t*BT + i_j*BC + j)*H*K, (H, K), (K, 1), (i_h * BH, 0), (BH, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + (bos + i_t*BT + i_j*BC + j)*H*K, (H, K), (K, 1), (i_h * BH, 0), (BH, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
        b_gk = tl.load(p_gk, boundary_check=(0, 1)).to(tl.float32)
        # b_A: (BH, BC, BK) * (BH, 1, BK) -> (BH, BC, BK) ->(sum) -> (BH, BC)
        b_A = tl.sum(b_q * b_k[:, None, :] * exp(b_g - b_gk[:, None, :]), 2)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)

        p_A =  tl.make_block_ptr(
            A + bos*H*BT + (i_j * BC + j) , 
            (H, T), 
            (BT, H * BT), 
            (i_h * BH, i_t * BT + i_i * BC), 
            (BH, BC), 
            (0, 1)
        )
        tl.store(p_A, b_A, boundary_check=(0, 1))


# TODO
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['BT'],
)
@triton.jit()
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    # i_tg is the global chunk index
    i_tg = i_t
    # i_n is the sequence index
    # i_t is the chunk index within the sequence
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    # NT = tl.cdiv(T, BT)

    # lower triangular mask. (BT, BT)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BH, BT, BV], dtype=tl.float32)
    # for i_k in range(tl.cdiv(K, BK)):
    p_q = tl.make_block_ptr(q + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    p_g = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    # h: (N, H, K, V)
    p_h = tl.make_block_ptr(h + i_tg*H*K*V, (H, K, V), (K*V, V, 1), (i_h * BH, 0, 0), (BH, BK, BV), (2, 1, 0))

    # [BH, BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1, 2))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BH, BT, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1, 2))
    # [BH, BT, BK]
    b_qg = (b_q * exp(b_g)).to(b_q.dtype)
    # [BH, BK, BV]
    b_h = tl.load(p_h, boundary_check=(0, 1, 2))
    # works but dkw, owing to divine benevolence
    # [BH, BT, BV]
    b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    
    p_v = tl.make_block_ptr(v + bos*H*V, (H, T, V), (V, H*V, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BV), (1, 2, 0))
    p_o = tl.make_block_ptr(o + bos*H*V, (H, T, V), (V, H*V, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BV), (1, 2, 0))
    p_A = tl.make_block_ptr(A + bos*H*BT, (H, T, BT), (BT, H*BT, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BT), (1, 2, 0))
    # [BH, BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1, 2))
    # [BH, BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1, 2))
    b_A = tl.where(m_s[None, :, :], b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1, 2))


# TODO
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BK', 'NC', 'BT'],
)
@triton.jit()
def chunk_gla_bwd_kernel_intra(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    cu_seqlens,
    chunk_indices,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    NC: tl.constexpr,
):
    i_t, i_i, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    # This part calculates dq of the no-diagonal sub-chunks (i_j < i_i)
    p_g = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))
    # [BH, BC, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1, 2))
    b_dq = tl.zeros([BH, BC, BK], dtype=tl.float32)
    if i_i > 0:
        # gn is the g_cumsum of the first g_cumsum of the i_i-th sub-chunk
        p_gn = tl.make_block_ptr(g + (bos + i_t * BT + i_i * BC) * H*K, (H, K), (K, 1), (i_h * BH, 0), (BH, BK), (1, 0))
        # [BH, BK]
        b_gn = tl.load(p_gn, boundary_check=(0, 1))
        # loop over previous sub-chunks with i_j < i_i (no-diagonal sub-chunks)
        for i_j in range(0, i_i):
            # 
            p_k = tl.make_block_ptr(k + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_j * BC, 0), (BH, BC, BK), (1, 2, 0))
            p_gk = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_j * BC, 0), (BH, BC, BK), (1, 2, 0))
            p_dA = tl.make_block_ptr(dA + bos*H*BT, (H, T, BT), (BT, H*BT, 1), (i_h * BH, i_t * BT + i_i * BC, i_j * BC), (BH, BC, BC), (1, 2, 0))
            # [BH, BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1, 2))
            b_gk = tl.load(p_gk, boundary_check=(0, 1, 2))
            b_kg = (b_k * exp(b_gn[:, None, :] - b_gk))  # K[j] * (B[i][0] / B[j])
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1, 2))
            # [BC, BK]
            b_dq += tl.dot(b_dA, b_kg)  # (dQ[i] * B[i]/B[i][0])
        b_dq *= exp(b_g - b_gn[:, None, :])

    # This part calculates dq of the diagonal sub-chunk (i_j == i_i)
    o_i = tl.arange(0, BC)  # offsets inside the i_i-th sub-chunk
    p_dq = tl.make_block_ptr(dq + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BH, BC]
        p_dA = tl.make_block_ptr(
            dA + bos*H*BT + i_i * BC + j, 
            (H, T), 
            (BT, H*BT), 
            (i_h * BH, i_t * BT + i_i * BC), 
            (BH, BC),
            (0, 1)
        )
        b_dA = tl.load(p_dA, boundary_check=(0, 1))
        # [BH, BK]
        p_kj = tl.make_block_ptr(
            k + (bos + i_t * BT + i_i * BC + j)*H*K,
            (H, K),
            (K, 1),
            (i_h * BH, 0),
            (BH, BK),
            (1, 0)
        )
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)
        p_gkj = tl.make_block_ptr(
            g + (bos + i_t * BT + i_i * BC + j)*H*K,
            (H, K),
            (K, 1),
            (i_h * BH, 0),
            (BH, BK),
            (1, 0)
        )  
        b_gkj = tl.load(p_gkj, boundary_check=(0, 1)).to(tl.float32)
        # [BH, BC, BK]
        m_i = o_i[None, :, None] >= j
        # [BC, BK]
        # (SY 09/17) important to not use bf16 here to have a good precision.
        # (BH, BC, 1) * (BH, 1, BK) * exp((BH, BC, BK) - (BH, 1, BK)) -> (BH, BC, BK)
        b_dq += tl.where(m_i, b_dA[:, :, None] * b_kj[:, None, :] * exp(b_g - b_gkj[:, None, :]), 0.)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1, 2))

    # tl.debug_barrier()

    # This part calculates dk of the no-diagonal sub-chunks (i_j > i_i)
    p_k = tl.make_block_ptr(k + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))
    p_gk = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))

    # [BH, BC, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1, 2))
    b_gk = tl.load(p_gk, boundary_check=(0, 1, 2))
    b_dk = tl.zeros([BH, BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    # loop over the following sub-chunks with i_j > i_i (no-diagonal sub-chunks)
    if i_i < NC - 1:
        # p_gn is the g_cumsum of the last element of the i_i-th sub-chunk B[i][-1]
        p_gn = tl.make_block_ptr(g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H*K, (H, K), (K, 1), (i_h * BH, 0), (BH, BK), (1, 0))

        # [BH, BK]
        b_gn = tl.load(p_gn, boundary_check=(0, 1))
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_j * BC, 0), (BH, BC, BK), (1, 2, 0))
            p_gq = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_j * BC, 0), (BH, BC, BK), (1, 2, 0))
            p_dA = tl.make_block_ptr(dA + bos*H*BT, (H, BT, T), (BT, 1, H*BT), (i_h * BH, i_i * BC, i_t * BT + i_j * BC), (BH, BC, BC), (1, 0, 2))
            # [BH, BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1, 2))
            b_gq = tl.load(p_gq, boundary_check=(0, 1, 2))
            b_qg = b_q * safe_exp(b_gq - b_gn[:, None, :])
            # [BH, BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1, 2))
            # [BH, BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= exp(b_gn[:, None, :] - b_gk)
    # This part calculates dk of the diagonal sub-chunk (i_j == i_i)    
    p_dk = tl.make_block_ptr(dk + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT + i_i * BC, 0), (BH, BC, BK), (1, 2, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_dA = tl.make_block_ptr(
            dA + (bos + i_t*BT + i_i*BC + j)*H*BT,
            (H, BT),
            (BT, 1),
            (i_h * BH, i_i * BC),
            (BH, BC),
            (1, 0),
        )
        # [BH, BC]
        b_dA = tl.load(p_dA, boundary_check=(0, 1)).to(tl.float32)
        # [BH, BK]
        # b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        p_qj = tl.make_block_ptr(
            q + (bos + i_t*BT + i_i*BC + j)*H*K,
            (H, K),
            (K, 1),
            (i_h * BH, 0),
            (BH, BK),
            (1, 0),
        )
        b_qj = tl.load(p_qj, boundary_check=(0, 1)).to(tl.float32)
        p_gqj = tl.make_block_ptr(
            g + (bos + i_t*BT + i_i*BC + j)*H*K,
            (H, K),
            (K, 1),
            (i_h * BH, 0),
            (BH, BK),
            (1, 0),
        )
        b_gqj = tl.load(p_gqj, boundary_check=(0, 1)).to(tl.float32)
        # [BH, BC, BK]
        m_i = o_i[None, :, None] <= j
        # (BH, BC, 1) * (BH, 1, BK) * exp((BH, 1, BK) - (BH, BC, BK)) -> (BH, BC, BK)
        b_dk += tl.where(m_i, b_dA[:, :, None] * b_qj[:, None, :] * exp(b_gqj[:, None, :] - b_gk), 0.)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1, 2))


# TODO
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BV', 'BT'],
)
@triton.jit()
def chunk_gla_bwd_kernel_dA(
    v,
    do,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)

    T = eos - bos

    b_dA = tl.zeros([BH, BT, BT], dtype=tl.float32)
    
    p_do = tl.make_block_ptr(do + bos*H*V, (H, T, V), (V, H*V, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BV), (1, 2, 0))
    p_v = tl.make_block_ptr(v + bos*H*V, (H, V, T), (V, 1, H*V), (i_h * BH, 0, i_t * BT), (BH, BV, BT), (1, 0, 2))
    b_v = tl.load(p_v, boundary_check=(0, 1, 2))
    b_do = tl.load(p_do, boundary_check=(0, 1, 2))
    b_dA += tl.dot(b_do, b_v)
    p_dA = tl.make_block_ptr(dA + bos*H*BT, (H, T, BT), (BT, H*BT, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BT), (1, 2, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s[None, :, :], b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1, 2))


@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['BT'],
)
@triton.jit()
def chunk_gla_bwd_kernel_dv(
    k,
    g,
    A,
    do,
    dh,
    dv,
    cu_seqlens,
    chunk_indices,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
):  
    # i_t is the chunk index
    # i_h is the head index
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    # i_tg is the global chunk index
    i_tg = i_t
    # i_n is the sequence index
    # i_t is the chunk index within the sequence
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    # NT = tl.cdiv(T, BT)

    # TODO
    p_A = tl.make_block_ptr(A + bos*H*BT, (H, BT, T), (BT, 1, H*BT), (i_h * BH, 0, i_t * BT), (BH, BT, BT), (1, 0, 2))
    p_do = tl.make_block_ptr(do + bos*H*V, (H, T, V), (V, H*V, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BV), (1, 2, 0))
    p_dv = tl.make_block_ptr(dv + bos*H*V, (H, T, V), (V, H*V, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BV), (1, 2, 0))

    b_A = tl.load(p_A, boundary_check=(0, 1, 2))
    m_A = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]
    b_A = tl.where(m_A[None, :, :], b_A, 0.)
    b_do = tl.load(p_do, boundary_check=(0, 1, 2))
    # (SY 09/17) important to disallow tf32 here to maintain a good precision.
    # [BH, BT, BT] * [BH, BT, BV] -> [BH, BT, BV]
    b_dv = tl.dot(b_A, b_do.to(b_A.dtype), allow_tf32=False)
    

    # TODO
    # for i_k in range(tl.cdiv(K, BK)):
    # o_k = i_k * BK + tl.arange(0, BK)
    # m_k = o_k < K

    p_k = tl.make_block_ptr(k + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    p_gk = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    # gn is the last g_cumsum in the chunk, (BH, BK)
    p_gn = tl.make_block_ptr(g + (bos + min(T, i_t * BT + BT) - 1)*H*K, (H, K), (K, 1), (i_h * BH, 0), (BH, BK), (1, 0))
    # p_gn = g + (bos + min(i_t * BT + BT, T) - 1)*H*K + i_h * K + o_k
    p_dh = tl.make_block_ptr(dh + i_tg*H*K*V, (H, K, V), (K*V, V, 1), (i_h * BH, 0, 0), (BH, BK, BV), (2, 1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1, 2))
    b_gk = tl.load(p_gk, boundary_check=(0, 1, 2))
    # (BH, BT, BK)
    b_gn = exp(tl.load(p_gn, boundary_check=(0, 1))[:, None, :] - b_gk)
    b_k = (b_k * b_gn).to(b_k.dtype)
    b_dh = tl.load(p_dh, boundary_check=(0, 1, 2))
    # (BH, BT, BV)
    # (SY 09/17) it is ok to have bf16 interchunk gradient contribution here
    b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1, 2))


# TODO
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['BT'],
)
@triton.jit()
def chunk_gla_bwd_kernel_inter(
    q,
    k,
    v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dq2,
    dk2,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    i_tg = i_t
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    
    # TODO here
    # o_k = i_k * BK + tl.arange(0, BK)
    # m_k = o_k < K

    p_gk = tl.make_block_ptr(g + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    # gn is the g_cumsum of the last element in the i_t-th chunk, (BH, BK), B[i][-1]
    p_gn = tl.make_block_ptr(g + (bos + min(T, i_t * BT + BT) - 1)*H*K, (H, K), (K, 1), (i_h * BH, 0), (BH, BK), (1, 0))
    # p_gn = g + (bos + min(T, i_t * BT + BT)-1) * H*K + i_h * K + o_k
    b_gn = tl.load(p_gn, boundary_check=(0, 1))
    b_dq = tl.zeros([BH, BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BH, BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BH, BK], dtype=tl.float32)

    # for i_v in range(tl.cdiv(V, BV)):
    p_v = tl.make_block_ptr(v + bos*H*V, (H, T, V), (V, H*V, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BV), (1, 2, 0))
    p_do = tl.make_block_ptr(do + bos*H*V, (H, T, V), (V, H*V, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BV), (1, 2, 0))
    p_h = tl.make_block_ptr(h + i_tg*H*K*V, (H, V, K), (K*V, 1, V), (i_h * BH, 0, 0), (BH, BV, BK), (2, 0, 1))
    p_dh = tl.make_block_ptr(dh + i_tg*H*K*V, (H, V, K), (K*V, 1, V), (i_h * BH, 0, 0), (BH, BV, BK), (2, 0, 1))
    # [BH, BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1, 2))
    b_do = tl.load(p_do, boundary_check=(0, 1, 2))
    # [BH, BV, BK]
    b_h = tl.load(p_h, boundary_check=(0, 1, 2))  # H^T
    b_dh = tl.load(p_dh, boundary_check=(0, 1, 2))
    # [BH, BK]
    b_dgk += tl.sum(b_h * b_dh, axis=1) # ?
    # [BH, BT, BK]   # [BH, BT, BV] * [BH, BV, BK] -> [BH, BT, BK]
    b_dq += tl.dot(b_do, b_h.to(b_do.dtype))  # d(Q[i] * B[i]/B[i-1][-1]) = dO H^T 
    b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))  # d(K[i] * B[i][-1]/B[i]) = V dH^T

    b_dgk *= exp(b_gn)  # ? what is dgk
    b_dq *= scale
    b_gk = tl.load(p_gk, boundary_check=(0, 1, 2))
    b_dq = b_dq * exp(b_gk) # dQ[i] = d(Q[i] * B[i]/B[i-1][-1]) * (B[i]/B[i-1][-1])
    b_dk = b_dk * exp(b_gn[:, None, :] - b_gk)  # dK[i] = d(K * B[i][-1]/B[i]) * (B[i][-1]/B[i])

    p_q = tl.make_block_ptr(q + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    p_k = tl.make_block_ptr(k + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    p_dq = tl.make_block_ptr(dq + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    p_dk = tl.make_block_ptr(dk + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1, 2))
    b_k = tl.load(p_k, boundary_check=(0, 1, 2))
    b_dgk += tl.sum(b_dk * b_k, axis=1)  # ?
    b_dq += tl.load(p_dq, boundary_check=(0, 1, 2))  # + dQ from intra
    b_dk += tl.load(p_dk, boundary_check=(0, 1, 2))  # + dK from intra
    b_dg = b_q * b_dq - b_k * b_dk
    # tl.debug_barrier()
    b_dg = b_dg - tl.cumsum(b_dg, axis=1) + tl.sum(b_dg, axis=1)[:, None, :] + b_dgk[:, None, :]
    # Buggy due to strange triton compiler issue.
    # m_s = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], 1., 0.)
    # b_dg = tl.dot(m_s, b_dg, allow_tf32=False) + b_dgk[None, :]
    p_dq = tl.make_block_ptr(dq2 + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    p_dk = tl.make_block_ptr(dk2 + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    p_dg = tl.make_block_ptr(dg + bos*H*K, (H, T, K), (K, H*K, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BK), (1, 2, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1, 2))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1, 2))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1, 2))


def chunk_gla_fwd_intra_gk(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
):  
    # q: shape (T, H, K)
    # k: shape (T, H, K)
    # g: shape (T, H, K)
    T, H, K = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))    # BT is the chunk size
    
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)   # (NT, 2), the first col is the seq_id of the chunk, the second col is the chunk_id within the sequence
    NT = len(chunk_indices)   # NT is the number of chunks
    BC = min(16, BT)          # BC is the sub-chunk size
    NC = triton.cdiv(BT, BC)  # NC is the number of sub-chunks in a chunk
    BK = max(16, triton.next_power_of_2(K))

    A = q.new_empty(T, H, BT, dtype=torch.float)   # A is the attention map for each chunk
    def grid(meta): return (NT, NC * NC, triton.cdiv(H, meta['BH']))
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q,
        k,
        g,
        A,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )

    def grid(meta): return (NT, NC, triton.cdiv(H, meta['BH']))
    chunk_gla_fwd_A_kernel_intra_sub_intra[grid](
        q,
        k,
        g,
        A,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
    )
    # remove cases for K > 256
    # you should assert K <= 256 before calling this function
    return A


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
):  
    # q, v, g: shape (T, H, K)
    # A: shape (T, H, BT)
    # h: shape (N, H, K, V)
    T, H, K, V = *q.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)  # NT is the number of chunks

    # o: shape (T, H, V)
    o = torch.empty_like(v)
    def grid(meta): return (NT, triton.cdiv(H, meta['BH']))
    chunk_gla_fwd_kernel_o[grid](
        q,
        v,
        g,
        h,
        o,
        A,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return o


def chunk_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
):
    T, H, V = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)
    BV = max(16, triton.next_power_of_2(V))

    dA = v.new_empty(T, H, BT, dtype=torch.float)
    def grid(meta): return (NT, triton.cdiv(H, meta['BH']))
    chunk_gla_bwd_kernel_dA[grid](
        v,
        do,
        dA,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dA


def chunk_gla_bwd_dv(
    k: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
):  
    T, H, K, V = *k.shape, do.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)  # number of chunks

    dv = torch.empty_like(do)
    def grid(meta): return (NT, triton.cdiv(H, meta['BH']))
    chunk_gla_bwd_kernel_dv[grid](
        k,
        g,
        A,
        do,
        dh,
        dv,
        cu_seqlens,
        chunk_indices,
        H=H,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BT=BT,
    )
    return dv


def chunk_gla_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
):
    T, H, K = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)
    BK = max(16, triton.next_power_of_2(K))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    def grid(meta): return (NT, NC, triton.cdiv(H, meta['BH']))
    chunk_gla_bwd_kernel_intra[grid](
        q,
        k,
        g,
        dA,
        dq,
        dk,
        cu_seqlens,
        chunk_indices,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    return dq, dk


def chunk_gla_bwd_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
):
    T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)

    dg = torch.empty_like(g)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    def grid(meta): return (NT, triton.cdiv(H, meta['BH']))
    chunk_gla_bwd_kernel_inter[grid](
        q,
        k,
        v,
        h,
        g,
        do,
        dh,
        dq,
        dk,
        dq2,
        dk2,
        dg,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dq2, dk2, dg


def chunk_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # q, k, v, g: (T, H, D)
    T = q.shape[0]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))  # BT: chunk size of T
    # g_cumsum: local cumulative sum of g for each chunk
    g_cumsum = chunk_local_cumsum(g, BT, cu_seqlens=cu_seqlens)

    # h is the initial hidden state of each chunk
    # ht is the final hidden state of each sequence
    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )

    # the intra A is kept in fp32
    # the computation has very marginal effect on the entire throughput
    # A is the attention map for each chunk
    # A: (T, H, BT)
    A = chunk_gla_fwd_intra_gk(
        q=q,
        k=k,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=g_cumsum,
        A=A,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    return g_cumsum, A, h, ht, o


def chunk_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: Optional[torch.Tensor],
    scale: float,
    initial_state: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
):
    T = q.shape[0]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, BT, cu_seqlens=cu_seqlens)

    if h is None:
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            states_in_fp32=True
        )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        states_in_fp32=True
    )

    # dv
    dv = chunk_gla_bwd_dv(
        k=k,
        g=g_cumsum,
        A=A,
        do=do,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )

    # dq dk in fp32
    dA = chunk_gla_bwd_dA(
        v=v,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    dq, dk = chunk_gla_bwd_dqk_intra(
        q=q,
        k=k,
        g=g_cumsum,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    dq, dk, dg = chunk_gla_bwd_dqkg(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g_cumsum,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    return dq, dk, dv, dg, dh0


class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    ):
        T = q.shape[0]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))

        g_cumsum, A, h, ht, o = chunk_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size
        )
        # recompute g_cumsum in bwd pass
        if g.dtype != torch.float:
            g_cumsum = None
        else:
            g = None
        ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o, ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, k, v, g, g_cumsum, initial_state, A = ctx.saved_tensors
        chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
        dq, dk, dv, dg, dh0 = chunk_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=g_cumsum,
            scale=scale,
            h=None,
            A=A,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size
        )
        return dq.to(q), dk.to(k), dv.to(v), dg, None, dh0, None, None


@torch.compiler.disable
def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.LongTensor,
    scale: Optional[int] = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[T, H, K]`.
        k (torch.Tensor):
            keys of shape `[T, H, K]`.
        v (torch.Tensor):
            values of shape `[T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[T, H, K]`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        scale (Optional[int]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
   
    if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
        raise ValueError(
            f"The number of initial states is expected to be equal to the number of input sequences, "
            f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
        )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    o, final_state = ChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state, cu_seqlens)
    return o, final_state
