from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ops.utils import prepare_chunk_offsets
from ops.utils.op import exp

BH_LIST = [1, 16]


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps, num_stages=num_stages)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV', 'USE_G', 'USE_GK', 'USE_GV']
)
@triton.jit()
def chunk_fwd_kernel_h(
    k,
    v,
    h,
    g,
    gk,
    gv,
    h0,
    ht,
    cu_seqlens, # cumulative sequence lengths
    chunk_offsets,  # cumulative number of chunks of each sequence
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):  
    # i_n: sequence index, i_h: head chunk index
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos  # length of the sequence
    NT = tl.cdiv(T, BT)  # number of chunks in the sequence
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)  # beginning of the output hidden state for the sequence

    # [BK, BV]
    b_h = tl.zeros([BH, BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # h0: [N, H, K, V]
        p_h0 = tl.make_block_ptr(h0 + i_n * H*K*V, (H, K, V), (K*V, V, 1), (i_h*BH, 0, 0), (BH, BK, BV), (2, 1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1, 2)).to(tl.float32)

    for i_t in range(NT):
        # k: [T, H, K], v: [T, H, V]
        p_k = tl.make_block_ptr(k + bos*H*K, (H, K, T), (K, 1, H*K), (i_h*BH, 0, i_t*BT), (BH, BK, BT), (1, 0, 2))
        p_v = tl.make_block_ptr(v + bos*H*V, (H, T, V), (V, H*V, 1), (i_h*BH, i_t*BT, 0), (BH, BT, BV), (1, 2, 0))

        o_h = ((boh + i_t)*H + i_h*BH).to(tl.int64) * K*V
        p_h = tl.make_block_ptr(h + o_h, (H, K, V), (K*V, V, 1), (0, 0, 0), (BH, BK, BV), (2, 1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1, 2))  # store the initial hidden state of each chunk
        # [BH, K, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1, 2))
        # [BH, BT, V]
        b_v = tl.load(p_v, boundary_check=(0, 1, 2))
        last_idx = min((i_t + 1) * BT, T) - 1  # last index of the chunk within the sequence

        # scalar decay
        if USE_G:
            # g is cumsum of decay in the chunk, [T, H]
            # b_g_last is the last value of g in the chunk, [BH,]
            p_g_last = g + bos*H + last_idx*H + i_h*BH + tl.arange(0, BH)
            b_g_last = tl.load(p_g_last, mask=(i_h*BH + tl.arange(0, BH) < H), other=0.)
            b_h *= exp(b_g_last)[:, None, None]  # decay of the hidden state of last chunk
            #  b_g is the g in the current chunk, [BT, BH]
            p_g = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (ii_h * BH, i_t*BT), (BH, BT), (0, 1))
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_v = (b_v * exp(b_g_last[:, None] - b_g)[:, :, None]).to(b_v.dtype)

        # vector decay, h = Diag(gk) @ h
        if USE_GK:
            # gk is cumsum of decay in the chunk, [T, H, K]
            # b_gk_last is the last value of gk in the chunk, [BH, K]
            # b_gk is the gk in the current chunk, [BH, K, BT]
            p_gk = tl.make_block_ptr(gk + bos*H*K, (H, K, T), (K, 1, H*K), (i_h*BH, 0, i_t*BT), (BH, BK, BT), (1, 0, 2))
            p_gk_last = tl.make_block_ptr(gk + (bos + last_idx) * H*K, (H, K), (K, 1), (i_h*BH, 0), (BH, BK), (1, 0))
            # p_gk_last = gk + (bos + last_idx)*H*K + i_h*BH*K + tl.arange(0, BH)*K + tl.arange(0, K)

            b_gk_last = tl.load(p_gk_last, boundary_check=(0, 1))
            b_h *= exp(b_gk_last)[:, :, None]

            b_gk = tl.load(p_gk, boundary_check=(0, 1, 2))
            b_k = (b_k * exp(b_gk_last[:, :, None] - b_gk)).to(b_k.dtype)

        # vector decay, h = h @ Diag(gv)
        if USE_GV:
            # gv is cumsum of decay in the chunk, [T, H, V]
            # b_gv_last is the last value of gv in the chunk, [BH, V]
            # b_gv is the gv in the current chunk, [BH, BT, V]
            p_gv = tl.make_block_ptr(gv + bos*H*V, (H, T, V), (V, H*V, 1), (i_h*BH, i_t*BT, 0), (BH, BT, BV), (1, 2, 0))
            p_gv_last = tl.make_block_ptr(gv + (bos + last_idx) * H*V, (H, V), (V, 1), (i_h*BH, 0), (BH, BV), (1, 0))
            # p_gv_last = gv + (bos + last_idx) * H*V + i_h * V + i_v * BV + tl.arange(0, BV)

            b_gv_last = tl.load(p_gv_last, boundary_check=(0, 1))
            b_h *= exp(b_gv_last)[:, None, :]

            b_gv = tl.load(p_gv, boundary_check=(0, 1, 2))
            b_v = (b_v * exp(b_gv_last[:, None, :] - b_gv)).to(b_v.dtype)

        b_h += tl.dot(b_k, b_v)  # [BH, K, V] += [BH, K, T] @ [BH, T, V]

    if STORE_FINAL_STATE:
        # ht: [N, H, K, V]
        p_ht = tl.make_block_ptr(ht + i_n*H*K*V, (H, K, V), (K*V, V, 1), (i_h*BH, 0, 0), (BH, BK, BV), (2, 1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1, 2))


@triton.heuristics({
    'STORE_INITIAL_STATE_GRADIENT': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps, num_stages=num_stages)
        for BH in BH_LIST
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'BK', 'BV', 'USE_G', 'USE_GK', 'USE_GV']
)
@triton.jit()
def chunk_bwd_kernel_dh(
    q,
    g,
    gk,
    gv,
    do,
    dh,
    dht,
    dh0,
    cu_seqlens,
    chunk_offsets,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
):  
    # calculate the grads of hidden state from the output gradients
    # i_n: sequence index, i_h: head chunk index
    i_n, i_h = tl.program_id(0), tl.program_id(1)

    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)  # beginning of the output hidden state for the sequence

    # [BH, K, V]
    b_dh = tl.zeros([BH, BK, BV], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_n * H*K*V, (H, K, V), (K*V, V, 1), (i_h * BH, 0, 0), (BH, BK, BV), (2, 1, 0))
        b_dh += tl.load(p_dht, boundary_check=(0, 1, 2)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        o_dh = ((boh + i_t) * H).to(tl.int64) * K*V
        p_dh = tl.make_block_ptr(dh + o_dh, (H, K, V), (K*V, V, 1), (i_h*BH, 0, 0), (BH, BK, BV), (2, 1, 0))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1, 2))
        last_idx = min(i_t * BT + BT, T) - 1
        # [BH, K, BT]
        p_q = tl.make_block_ptr(q + bos*H*K, (H, K, T), (K, 1, H*K), (i_h*BH, 0, i_t*BT), (BH, BK, BT), (1, 0, 2))
        p_do = tl.make_block_ptr(do + bos*H*V, (H, T, V), (V, H*V, 1), (i_h*BH, i_t*BT, 0), (BH, BT, BV), (1, 2, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1, 2))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BH, BT, V]
        b_do = tl.load(p_do, boundary_check=(0, 1, 2))

        if USE_G:
            # b_g_last: last value of g in the chunk, [BH,]
            # b_g: g in the current chunk, [BH, BT]
            p_g_last = g + bos*H + last_idx*H + i_h*BH + tl.arange(0, BH)
            p_g = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h*BH, i_t*BT), (BH, BT), (1, 0))
            # p_g = g + (bos + i_t*BT + tl.arange(0, BT)) * H + i_h
            b_g_last = tl.load(p_g_last, mask=(i_h*BH + tl.arange(0, BH) < H), other=0.)
            b_g = tl.load(p_g, boundary_check=(0, 1))
            b_q = (b_q * exp(b_g)[:, None, :]).to(b_q.dtype)

            b_dh *= exp(b_g_last)[:, None, None]  # decay of the hidden state of last chunk

        if USE_GK:
            # b_gk_last is the last value of gk in the chunk, [BH, K]
            # b_gk is the gk in the current chunk, [BH, K, BT]
            p_gk = tl.make_block_ptr(gk + bos*H*K, (H, K, T), (K, 1, H*K), (i_h*BH, 0, i_t*BT), (BH, BK, BT), (1, 0, 2))
            p_gk_last = tl.make_block_ptr(gk + (bos + last_idx) * H*K, (H, K), (K, 1), (i_h*BH, 0), (BH, BK), (1, 0))

            b_gk = tl.load(p_gk, boundary_check=(0, 1, 2))
            b_q = (b_q * exp(b_gk)).to(b_q.dtype)

            b_gk_last = tl.load(p_gk_last, boundary_check=(0, 1))
            b_dh *= exp(b_gk_last)[:, :, None]

        if USE_GV:
            # b_gv_last is the last value of gv in the chunk, [BH, V]
            # b_gv is the gv in the current chunk, [BH, BT, V]
            p_gv = tl.make_block_ptr(gv + bos*H*V, (H, T, V), (V, H*V, 1), (i_h*BH, i_t*BT, 0), (BH, BT, BV), (1, 2, 0))
            p_gv_last = tl.make_block_ptr(gv + (bos + last_idx) * H*V, (H, V), (V, 1), (i_h*BH, 0), (BH, BV), (1, 0))

            b_gv = tl.load(p_gv, boundary_check=(0, 1, 2))
            b_do = (b_do * exp(b_gv)).to(b_do.dtype)

            b_gv_last = tl.load(p_gv_last, boundary_check=(0, 1))
            b_dh *= exp(b_gv_last)[:, None, :]

        b_dh += tl.dot(b_q, b_do)

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_n * H*K*V, (H, K, V), (K*V, V, 1), (i_h*BH, 0, 0), (BH, BK, BV), (2, 1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1, 2))


def chunk_fwd_h(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    h0: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
    chunk_size: int = 64,
    states_in_fp32: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # chunk_offsets: the cumulative number of chunks for each sequence
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NS: the total number of chunks in the batch
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    N, NS = len(cu_seqlens) - 1, chunk_offsets[-1].item()
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    # h: the output hidden state for each chunk
    # for each chunk, h has shape [H, K, V]
    h = k.new_empty(NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    # ht: the final hidden state for each sequence
    # for each sequence, ht has shape [H, K, V]
    ht = k.new_empty(N, H, K, V, dtype=torch.float) if output_final_state else None
    def grid(meta): return (N, triton.cdiv(H, meta['BH']))
    # the limit of blocks in each dimension is 2 ** 32 - 1, 65535 and 65535
    # use the first dimension for N to avoid exceeding the limit
    chunk_fwd_kernel_h[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        gk=gk,
        gv=gv,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return h, ht


def chunk_bwd_dh(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gv: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.Tensor],
    chunk_size: int = 64,
    states_in_fp32: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, H, K, V = *k.shape, v.shape[-1]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NS: the total number of chunks in the batch
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    N, NS = len(cu_seqlens) - 1, chunk_offsets[-1].item()
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))

    dh = k.new_empty(NS, H, K, V, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta): return (N, triton.cdiv(H, meta['BH']))
    chunk_bwd_kernel_dh[grid](
        q=q,
        g=g,
        gk=gk,
        gv=gv,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        USE_GK=gk is not None,
        USE_GV=gv is not None,
    )
    return dh, dh0
