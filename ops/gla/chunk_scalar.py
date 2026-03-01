# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ops.utils import prepare_chunk_indices
from ops.utils.op import exp, safe_exp
from utils.ela_utils import input_guard

from ..common.chunk_h_scalar import chunk_bwd_dh_scalar, chunk_fwd_h_scalar
from ..utils.cumsum import chunk_local_cumsum


BH_LIST = [1, 4, 16, 64, 256]


# This kernel is used to calculate the attention map of each chunk
# It calculates the no-diagonal sub-chunks inside a chunk
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
def chunk_gla_fwd_A_kernel_intra_sub_inter_scalar(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    NC: tl.constexpr,
):
    # q, k, g: (T, H)
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
    # q: (T, H)
    p_q = tl.make_block_ptr(q + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))
    # g: (T, H)
    p_g = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))
    # k: (T, H)    
    p_k = tl.make_block_ptr(k + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_j * BC), (BH, BC), (0, 1))
    p_gk = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_j * BC), (BH, BC), (0, 1))
    # gn is the g_cumsum of the first g_cumsum of the i_i-th sub-chunk
    p_gn = tl.make_block_ptr(g + (bos + i_t * BT + i_i * BC) * H, (H,), (1,), (i_h * BH,), (BH,), (0,))
    # p_gn = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k

    # (BH)
    b_gn = tl.load(p_gn, boundary_check=(0,))
    # (BH, BC)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_qg = b_q * exp(b_g - b_gn[:, None]) * scale
    # (BH, BC)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_kg = b_k * exp(b_gn[:, None] - b_gk)
    # (BH, BC, BC)
    b_A += b_qg[:, :, None] * b_kg[:, None, :]  # (BH, BC, 1) * (BH, 1, BC) -> (BH, BC, BC)

    # A: (T, H, BT)
    p_A = tl.make_block_ptr(A + bos*H*BT, (H, T, BT), (BT, H*BT, 1), (i_h * BH, i_t * BT + i_i * BC, i_j * BC), (BH, BC, BC), (1, 2, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1, 2))


# This kernel is used to calculate the attention map of each chunk
# It calculates the diagonal sub-chunks
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BT"]
)
@triton.jit()
def chunk_gla_fwd_A_kernel_intra_sub_intra_scalar(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
):  
    # q, k, g: (T, H)
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

    # q, g: (T, H)
    p_q = tl.make_block_ptr(q + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))
    p_g = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))

    # b_q, b_g: (BH, BC)
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        p_k = tl.make_block_ptr(k + (bos + i_t*BT + i_j*BC + j)*H, (H,), (1,), (i_h * BH,), (BH,), (0,))
        p_gk = tl.make_block_ptr(g + (bos + i_t*BT + i_j*BC + j)*H, (H,), (1,), (i_h * BH,), (BH,), (0,))
        b_k = tl.load(p_k, boundary_check=(0,)).to(tl.float32)
        b_gk = tl.load(p_gk, boundary_check=(0,)).to(tl.float32)
        # b_A: (BH, BC) * (BH, 1) -> (BH, BC)
        b_A = b_q * b_k[:, None] * exp(b_g - b_gk[:, None])
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

# Calculate output by qS and Av
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['BT'],
)
@triton.jit()
def chunk_gla_fwd_kernel_o_scalar(
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
    BT: tl.constexpr,
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

    # lower triangular mask, (BT, BT)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BH, BT], dtype=tl.float32)
    p_q = tl.make_block_ptr(q + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_g = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    # h: (N, H)
    p_h = tl.make_block_ptr(h + i_tg*H, (H,), (1,), (i_h * BH,), (BH,), (0,))

    # [BH, BT]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [BH, BT]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    # [BH, BT]
    b_qg = (b_q * exp(b_g)).to(b_q.dtype)
    # [BH]
    b_h = tl.load(p_h, boundary_check=(0,))
    # works but dkw, owing to divine benevolence
    # [BH, BT]
    b_o += b_qg * b_h.to(b_qg.dtype)[:, None]
    
    p_v = tl.make_block_ptr(v + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_o = tl.make_block_ptr(o + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_A = tl.make_block_ptr(A + bos*H*BT, (H, T, BT), (BT, H*BT, 1), (i_h * BH, i_t * BT, 0), (BH, BT, BT), (1, 2, 0))
    # [BH, BT]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BH, BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1, 2))
    b_A = tl.where(m_s[None, :, :], b_A, 0.).to(b_v.dtype)
    # b_o += tl.dot(b_A, b_v, allow_tf32=False)
    b_o += tl.sum(b_A * b_v[:, None, :], axis=2)  # [BH, BT] += [BH, BT, BT] @ [BH, BT, 1]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))



@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
    ],
    key=['NC', 'BT'],
)
@triton.jit()
def chunk_gla_bwd_kernel_intra_scalar(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    cu_seqlens,
    chunk_indices,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
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
    p_g = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))
    # [BH, BC]
    b_g = tl.load(p_g, boundary_check=(0, 1))
    b_dq = tl.zeros([BH, BC], dtype=tl.float32)
    if i_i > 0:
        # gn is the g_cumsum of the first g_cumsum of the i_i-th sub-chunk
        p_gn = tl.make_block_ptr(g + (bos + i_t * BT + i_i * BC) * H, (H,), (1,), (i_h * BH,), (BH,), (0,))
        # [BH]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        # loop over previous sub-chunks with i_j < i_i (no-diagonal sub-chunks)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_j * BC), (BH, BC), (0, 1))
            p_gk = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_j * BC), (BH, BC), (0, 1))
            p_dA = tl.make_block_ptr(dA + bos*H*BT, (H, T, BT), (BT, H*BT, 1), (i_h * BH, i_t * BT + i_i * BC, i_j * BC), (BH, BC, BC), (1, 2, 0))
            # [BH, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = (b_k * exp(b_gn[:, None] - b_gk))  # K[j] * (B[i][0] / B[j])
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1, 2))
            # [BH, BC]
            # b_dq += tl.dot(b_dA, b_kg)  # (dQ[i] * B[i]/B[i][0])
            b_dq += tl.sum(b_dA * b_kg[:, None, :], axis=2)  # [BH, BC] += [BH, BC, BC] @ [BH, BC, 1]
        b_dq *= exp(b_g - b_gn[:, None])

    # This part calculates dq of the diagonal sub-chunk (i_j == i_i)
    o_i = tl.arange(0, BC)  # offsets inside the i_i-th sub-chunk
    p_dq = tl.make_block_ptr(dq + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))
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
        # [BH]
        p_kj = tl.make_block_ptr(
            k + (bos + i_t * BT + i_i * BC + j)*H,
            (H,),
            (1,),
            (i_h * BH,),
            (BH,),
            (0,)
        )
        b_kj = tl.load(p_kj, boundary_check=(0,)).to(tl.float32)
        p_gkj = tl.make_block_ptr(
            g + (bos + i_t * BT + i_i * BC + j)*H,
            (H,),
            (1,),
            (i_h * BH,),
            (BH,),
            (0,)
        )  
        b_gkj = tl.load(p_gkj, boundary_check=(0,)).to(tl.float32)
        # [1, BC]
        m_i = o_i[None, :] >= j
        # (BH, BC) * (BH, 1) * exp((BH, BC) - (BH, 1)) -> (BH, BC)
        b_dq += tl.where(m_i, b_dA * b_kj[:, None] * exp(b_g - b_gkj[:, None]), 0.)
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    # tl.debug_barrier()

    # This part calculates dk of the no-diagonal sub-chunks (i_j > i_i)
    p_k = tl.make_block_ptr(k + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))
    p_gk = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))

    # [BH, BC]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dk = tl.zeros([BH, BC], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    # loop over the following sub-chunks with i_j > i_i (no-diagonal sub-chunks)
    if i_i < NC - 1:
        # p_gn is the g_cumsum of the last element of the i_i-th sub-chunk B[i][-1]
        p_gn = tl.make_block_ptr(g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H, (H,), (1,), (i_h * BH,), (BH,), (0,))

        # [BH]
        b_gn = tl.load(p_gn, boundary_check=(0,))
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_j * BC), (BH, BC), (0, 1))
            p_gq = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_j * BC), (BH, BC), (0, 1))
            p_dA = tl.make_block_ptr(dA + bos*H*BT, (H, BT, T), (BT, 1, H*BT), (i_h * BH, i_i * BC, i_t * BT + i_j * BC), (BH, BC, BC), (1, 0, 2))
            # [BH, BC]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.load(p_gq, boundary_check=(0, 1))
            b_qg = b_q * safe_exp(b_gq - b_gn[:, None])
            # [BH, BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1, 2))
            # [BH, BC]
            # b_dk += tl.dot(b_dA, b_qg)
            b_dk += tl.sum(b_dA * b_qg[:, None, :], axis=2)  # [BH, BC] += [BH, BC, BC] @ [BH, BC, 1]
        b_dk *= exp(b_gn[:, None] - b_gk)
    # This part calculates dk of the diagonal sub-chunk (i_j == i_i)    
    p_dk = tl.make_block_ptr(dk + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT + i_i * BC), (BH, BC), (0, 1))
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
        # [BH]
        p_qj = tl.make_block_ptr(
            q + (bos + i_t*BT + i_i*BC + j)*H,
            (H,),
            (1,),
            (i_h * BH,),
            (BH,),
            (0,),
        )
        b_qj = tl.load(p_qj, boundary_check=(0,)).to(tl.float32)
        p_gqj = tl.make_block_ptr(
            g + (bos + i_t*BT + i_i*BC + j)*H,
            (H,),
            (1,),
            (i_h * BH,),
            (BH,),
            (0,),
        )
        b_gqj = tl.load(p_gqj, boundary_check=(0,)).to(tl.float32)
        # [BH, BC]
        m_i = o_i[None, :] <= j
        # (BH, BC) * (BH, 1) * exp((BH, 1) - (BH, BC)) -> (BH, BC)
        b_dk += tl.where(m_i, b_dA * b_qj[:, None] * exp(b_gqj[:, None] - b_gk), 0.)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


# 
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BT'],
)
@triton.jit()
def chunk_gla_bwd_kernel_dA_scalar(
    v,
    do,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)

    T = eos - bos

    b_dA = tl.zeros([BH, BT, BT], dtype=tl.float32)
    
    p_do = tl.make_block_ptr(do + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_v = tl.make_block_ptr(v + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    # b_v: (BH, BT)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # b_do: (BH, BT)
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dA += b_do[:, :, None] * b_v[:, None, :]  # (BH, BT, 1) * (BH, 1, BT) -> (BH, BT, BT)
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
def chunk_gla_bwd_kernel_dv_scalar(
    k,
    g,
    A,
    do,
    dh,
    dv,
    cu_seqlens,
    chunk_indices,
    H: tl.constexpr,
    BT: tl.constexpr,
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
    p_do = tl.make_block_ptr(do + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_dv = tl.make_block_ptr(dv + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))

    b_A = tl.load(p_A, boundary_check=(0, 1, 2))
    m_A = tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :]
    b_A = tl.where(m_A[None, :, :], b_A, 0.)
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_dv = tl.sum(b_A * b_do[:, None, :], axis=2)  # [BH, BT] += [BH, BT, BT] @ [BH, BT, 1]


    p_k = tl.make_block_ptr(k + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_gk = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    # gn is the last g_cumsum in the chunk, (BH)
    p_gn = tl.make_block_ptr(g + (bos + min(T, i_t * BT + BT) - 1)*H, (H,), (1,), (i_h * BH,), (BH,), (0,))
    # p_gn = g + (bos + min(i_t * BT + BT, T) - 1)*H*K + i_h * K + o_k
    p_dh = tl.make_block_ptr(dh + i_tg*H, (H,), (1,), (i_h * BH,), (BH,), (0,))

    # (BH, BT)
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    # (BH, BT)
    b_gn = exp(tl.load(p_gn, boundary_check=(0,))[:, None] - b_gk)
    b_k = (b_k * b_gn).to(b_k.dtype)
    b_dh = tl.load(p_dh, boundary_check=(0,))
    # (BH, BT) * (BH, 1) -> (BH, BT) 
    b_dv += b_k * b_dh.to(b_k.dtype)[:, None]
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in BH_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['BT'],
)
@triton.jit()
def chunk_gla_bwd_kernel_inter_scalar(
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
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    i_tg = i_t
    i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    
    p_gk = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    # gn is the g_cumsum of the last element in the i_t-th chunk, (BH,), B[i][-1]
    p_gn = tl.make_block_ptr(g + (bos + min(T, i_t * BT + BT) - 1)*H, (H,), (1,), (i_h * BH,), (BH,), (0,))
    b_gn = tl.load(p_gn, boundary_check=(0,))
    b_dq = tl.zeros([BH, BT], dtype=tl.float32)
    b_dk = tl.zeros([BH, BT], dtype=tl.float32)
    b_dgk = tl.zeros([BH], dtype=tl.float32)

    p_v = tl.make_block_ptr(v + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_do = tl.make_block_ptr(do + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_h = tl.make_block_ptr(h + i_tg*H, (H,), (1,), (i_h * BH,), (BH,), (0,))
    p_dh = tl.make_block_ptr(dh + i_tg*H, (H,), (1,), (i_h * BH,), (BH,), (0,))
    # [BH, BT]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [BH]
    b_h = tl.load(p_h, boundary_check=(0,))  # H^T
    b_dh = tl.load(p_dh, boundary_check=(0,))
    # [BH]
    b_dgk += b_h * b_dh
    # [BH, BT]  ([BH, BT, 1] * [BH, 1, 1] -> [BH, BT, 1])
    b_dq += b_do * b_h.to(b_do.dtype)[:, None]  # d(Q[i] * B[i]/B[i-1][-1]) = dO H^T 
    b_dk += b_v * b_dh.to(b_v.dtype)[:, None]  # d(K[i] * B[i][-1]/B[i]) = V dH^T

    # [BH]
    b_dgk *= exp(b_gn)  # ? what is dgk
    b_dq *= scale
    # [BH, BT]
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    b_dq = b_dq * exp(b_gk) # dQ[i] = d(Q[i] * B[i]/B[i-1][-1]) * (B[i]/B[i-1][-1])
    b_dk = b_dk * exp(b_gn[:, None] - b_gk)  # dK[i] = d(K * B[i][-1]/B[i]) * (B[i][-1]/B[i])

    p_q = tl.make_block_ptr(q + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_k = tl.make_block_ptr(k + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_dq = tl.make_block_ptr(dq + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_dk = tl.make_block_ptr(dk + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    # [BH]
    b_dgk += tl.sum(b_dk * b_k, axis=1)  # ?
    b_dq += tl.load(p_dq, boundary_check=(0, 1))  # + dQ from intra
    b_dk += tl.load(p_dk, boundary_check=(0, 1))  # + dK from intra
    b_dg = b_q * b_dq - b_k * b_dk
    # tl.debug_barrier()
    # [BH, BT]
    b_dg = b_dg - tl.cumsum(b_dg, axis=1) + tl.sum(b_dg, axis=1)[:, None] + b_dgk[:, None]
    # Buggy due to strange triton compiler issue.
    # m_s = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], 1., 0.)
    # b_dg = tl.dot(m_s, b_dg, allow_tf32=False) + b_dgk[None, :]
    p_dq = tl.make_block_ptr(dq2 + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_dk = tl.make_block_ptr(dk2 + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    p_dg = tl.make_block_ptr(dg + bos*H, (H, T), (1, H), (i_h * BH, i_t * BT), (BH, BT), (0, 1))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_intra_gk_scalar(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
):  
    # q: shape (T, H)
    # k: shape (T, H)
    # g: shape (T, H)
    T, H = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))    # BT is the chunk size
    
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)   # (NT, 2), the first col is the seq_id of the chunk, the second col is the chunk_id within the sequence
    NT = len(chunk_indices)   # NT is the number of chunks
    BC = min(16, BT)          # BC is the sub-chunk size
    NC = triton.cdiv(BT, BC)  # NC is the number of sub-chunks in a chunk

    A = q.new_empty(T, H, BT, dtype=torch.float)   # A is the attention map for each chunk
    def grid(meta): return (NT, NC * NC, triton.cdiv(H, meta['BH']))
    chunk_gla_fwd_A_kernel_intra_sub_inter_scalar[grid](
        q,
        k,
        g,
        A,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        BT=BT,
        BC=BC,
        NC=NC,
    )

    def grid(meta): return (NT, NC, triton.cdiv(H, meta['BH']))
    chunk_gla_fwd_A_kernel_intra_sub_intra_scalar[grid](
        q,
        k,
        g,
        A,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        BT=BT,
        BC=BC,
    )
    return A


def chunk_gla_fwd_o_gk_scalar(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64
):  
    # q, v, g: shape (T, H)
    # A: shape (T, H, BT)
    # h: shape (N, H)
    T, H = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)  # NT is the number of chunks

    # o: shape (T, H, V)
    o = torch.empty_like(v)
    def grid(meta): return (NT, triton.cdiv(H, meta['BH']))
    chunk_gla_fwd_kernel_o_scalar[grid](
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
        BT=BT,
    )
    return o


def chunk_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
):
    T, H = v.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)

    dA = v.new_empty(T, H, BT, dtype=torch.float)
    def grid(meta): return (NT, triton.cdiv(H, meta['BH']))
    chunk_gla_bwd_kernel_dA_scalar[grid](
        v,
        do,
        dA,
        cu_seqlens,
        chunk_indices,
        scale,
        H=H,
        BT=BT,
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
    T, H = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)  # number of chunks

    dv = torch.empty_like(do)
    def grid(meta): return (NT, triton.cdiv(H, meta['BH']))
    chunk_gla_bwd_kernel_dv_scalar[grid](
        k,
        g,
        A,
        do,
        dh,
        dv,
        cu_seqlens,
        chunk_indices,
        H=H,
        BT=BT,
    )
    return dv


def chunk_gla_bwd_dqk_intra_scalar(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64
):
    T, H = q.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    BC = min(16, BT)

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    def grid(meta): return (NT, NC, triton.cdiv(H, meta['BH']))
    chunk_gla_bwd_kernel_intra_scalar[grid](
        q,
        k,
        g,
        dA,
        dq,
        dk,
        cu_seqlens,
        chunk_indices,
        H=H,
        BT=BT,
        BC=BC,
        NC=NC,
    )
    return dq, dk


def chunk_gla_bwd_dqkg_scalar(
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
    T, H = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(chunk_indices)

    dg = torch.empty_like(g)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    def grid(meta): return (NT, triton.cdiv(H, meta['BH']))
    chunk_gla_bwd_kernel_inter_scalar[grid](
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
        BT=BT,
    )
    return dq2, dk2, dg


def chunk_gla_scalar_fwd(
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
    # q, k, v, g: (T, H)
    T = q.shape[0]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))  # BT: chunk size of T
    # g_cumsum: local cumulative sum of g for each chunk
    g_cumsum = chunk_local_cumsum(g, BT, cu_seqlens=cu_seqlens)

    # h is the initial hidden state of each chunk
    # ht is the final hidden state of each sequence
    h, ht = chunk_fwd_h_scalar(
        k=k,
        v=v,
        g=g_cumsum,
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
    A = chunk_gla_fwd_intra_gk_scalar(
        q=q,
        k=k,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    o = chunk_gla_fwd_o_gk_scalar(
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


def chunk_gla_scalar_bwd(
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
        h, _ = chunk_fwd_h_scalar(
            k=k,
            v=v,
            g=g_cumsum,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            states_in_fp32=True
        )
    dh, dh0 = chunk_bwd_dh_scalar(
        q=q,
        k=k,
        v=v,
        g=g_cumsum,
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
    dq, dk = chunk_gla_bwd_dqk_intra_scalar(
        q=q,
        k=k,
        g=g_cumsum,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_size=BT
    )
    dq, dk, dg = chunk_gla_bwd_dqkg_scalar(
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


class ChunkGLAScalarFunction(torch.autograd.Function):

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
        chunk_size = min(16, max(16, triton.next_power_of_2(T)))

        g_cumsum, A, h, ht, o = chunk_gla_scalar_fwd(
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
        dq, dk, dv, dg, dh0 = chunk_gla_scalar_bwd(
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
def chunk_gla_scalar(
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
            queries of shape `[T, H]`.
        k (torch.Tensor):
            keys of shape `[T, H]`.
        v (torch.Tensor):
            values of shape `[T, H]`.
        g (torch.Tensor):
            Forget gates of shape `[T, H]`.
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
            Outputs of shape `[T, H]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H]` if `output_final_state=True` else `None`.
    """
   
    if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
        raise ValueError(
            f"The number of initial states is expected to be equal to the number of input sequences, "
            f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
        )
    if scale is None:
        scale = 1.0
    o, final_state = ChunkGLAScalarFunction.apply(q, k, v, g, scale, initial_state, output_final_state, cu_seqlens)
    return o, final_state
