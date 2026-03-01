from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ops.utils import prepare_chunk_offsets
from ops.utils.op import exp

BH_LIST = [1, 4, 16, 64, 256]


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
    key=['BT']
)
@triton.jit()
def chunk_fwd_kernel_h_scalar(
    k,
    v,
    h,
    g,
    h0,
    ht,
    cu_seqlens, # cumulative sequence lengths
    chunk_offsets,  # cumulative number of chunks of each sequence
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):  
    # i_n: sequence index, i_h: head chunk index
    i_n, i_h = tl.program_id(0), tl.program_id(1)
    
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos  # length of the sequence
    NT = tl.cdiv(T, BT)  # number of chunks in the sequence
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)  # beginning of the output hidden state for the sequence

    # [BH]
    b_h = tl.zeros([BH], dtype=tl.float32)
    if USE_INITIAL_STATE:
        # h0: [N, H], b_h: [BH]
        p_h0 = tl.make_block_ptr(h0 + i_n * H, (H,), (1,), (i_h*BH,), (BH,), (0,))
        b_h = tl.load(p_h0, boundary_check=(0,)).to(tl.float32)

    for i_t in range(NT):
        # k: [T, H], v: [T, H]
        p_k = tl.make_block_ptr(k + bos*H, (H, T), (1, H), (i_h*BH, i_t*BT), (BH, BT), (0, 1))
        p_v = tl.make_block_ptr(v + bos*H, (H, T), (1, H), (i_h*BH, i_t*BT), (BH, BT), (0, 1))

        o_h = ((boh + i_t)*H + i_h*BH).to(tl.int64)
        p_h = tl.make_block_ptr(h + o_h, (H,), (1,), (0,), (BH,), (0,))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0,))  # store the initial hidden state of each chunk
        # [BH, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BH, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        last_idx = min((i_t + 1) * BT, T) - 1  # last index of the chunk within the sequence

        # g is cumsum of decay in the chunk, [T, H]
        # b_g is the g in the current chunk, [BH, BT]
        # b_g_last is the last value of g in the chunk, [BH]
        p_g = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h*BH, i_t*BT), (BH, BT), (0, 1))
        p_g_last = tl.make_block_ptr(g + (bos + last_idx) * H, (H,), (1,), (i_h*BH,), (BH,), (0,))

        b_g_last = tl.load(p_g_last, boundary_check=(0,))
        b_h *= exp(b_g_last)

        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_k = (b_k * exp(b_g_last[:, None] - b_g)).to(b_k.dtype)

        b_h += tl.sum(b_k * b_v, axis=1)  # [BH] += [BH, BT] * [BH, BT]

    if STORE_FINAL_STATE:
        # ht: [N, H]
        p_ht = tl.make_block_ptr(ht + i_n*H, (H,), (1,), (i_h*BH,), (BH,), (0,))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0,))


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
    key=['BT']
)
@triton.jit()
def chunk_bwd_kernel_dh_scalar(
    q,
    g,
    do,
    dh,
    dht,
    dh0,
    cu_seqlens,
    chunk_offsets,
    scale,
    H: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
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

    # [BH]
    b_dh = tl.zeros([BH], dtype=tl.float32)
    if USE_FINAL_STATE_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_n * H, (H,), (1,), (i_h * BH,), (BH,), (0,))
        b_dh += tl.load(p_dht, boundary_check=(0,)).to(tl.float32)

    for i_t in range(NT - 1, -1, -1):
        o_dh = ((boh + i_t) * H).to(tl.int64)
        p_dh = tl.make_block_ptr(dh + o_dh, (H,), (1,), (i_h*BH,), (BH,), (0,))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0,))
        last_idx = min(i_t * BT + BT, T) - 1
        # [BH, BT]
        p_q = tl.make_block_ptr(q + bos*H, (H, T), (1, H), (i_h*BH, i_t*BT), (BH, BT), (0, 1))
        p_do = tl.make_block_ptr(do + bos*H, (H, T), (1, H), (i_h*BH, i_t*BT), (BH, BT), (0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)
        # [BH, BT]
        b_do = tl.load(p_do, boundary_check=(0, 1))

        # b_g_last is the last value of g in the chunk, [BH]
        # b_g is the g in the current chunk, [BH, BT]
        p_g = tl.make_block_ptr(g + bos*H, (H, T), (1, H), (i_h*BH, i_t*BT), (BH, BT), (0, 1))
        p_g_last = tl.make_block_ptr(g + (bos + last_idx) * H, (H,), (1,), (i_h*BH,), (BH,), (0,))

        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_q = (b_q * exp(b_g)).to(b_q.dtype)
        b_g_last = tl.load(p_g_last, boundary_check=(0,))
        b_dh *= exp(b_g_last)

        b_dh +=  tl.sum(b_q * b_do, axis=1)  # [BH] += [BH, 1, BT] @ [BH, BT, 1]

    if STORE_INITIAL_STATE_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_n * H, (H,), (1,), (i_h*BH,), (BH,), (0,))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0,))


def chunk_fwd_h_scalar(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    h0: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
    chunk_size: int = 64,
    states_in_fp32: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, H = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # chunk_offsets: the cumulative number of chunks for each sequence
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NS: the total number of chunks in the batch
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    N, NS = len(cu_seqlens) - 1, chunk_offsets[-1].item()

    # h: the output hidden state for each chunk
    # for each chunk, h has shape [H]
    h = k.new_empty(NS, H, dtype=k.dtype if not states_in_fp32 else torch.float)
    # ht: the final hidden state for each sequence
    # for each sequence, ht has shape [H]
    ht = k.new_empty(N, H, dtype=torch.float) if output_final_state else None
    def grid(meta): return (N, triton.cdiv(H, meta['BH']))
    # the limit of blocks in each dimension is 2 ** 32 - 1, 65535 and 65535
    # use the first dimension for N to avoid exceeding the limit
    chunk_fwd_kernel_h_scalar[grid](
        k=k,
        v=v,
        h=h,
        g=g,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        H=H,
        BT=BT,
    )
    return h, ht


def chunk_bwd_dh_scalar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    scale: float,
    cu_seqlens: Optional[torch.Tensor],
    chunk_size: int = 64,
    states_in_fp32: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    T, H = k.shape
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    # N: the actual number of sequences in the batch with either equal or variable lengths
    # NS: the total number of chunks in the batch
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    N, NS = len(cu_seqlens) - 1, chunk_offsets[-1].item()

    dh = k.new_empty(NS, H, dtype=k.dtype if not states_in_fp32 else torch.float)
    dh0 = torch.empty_like(h0, dtype=torch.float) if h0 is not None else None

    def grid(meta): return (N, triton.cdiv(H, meta['BH']))
    chunk_bwd_kernel_dh_scalar[grid](
        q=q,
        g=g,
        do=do,
        dh=dh,
        dht=dht,
        dh0=dh0,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        H=H,
        BT=BT,
    )
    return dh, dh0
