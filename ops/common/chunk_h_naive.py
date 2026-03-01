# chunk_h_pytorch.py
from typing import Optional, Tuple, List
import torch
from ops.utils import prepare_chunk_offsets


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def _seq_ranges(cu_seqlens: torch.LongTensor) -> List[Tuple[int, int]]:
    cu = cu_seqlens.tolist()
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def chunk_h_naive(
    k: torch.Tensor,                 # [T, H, K]
    v: torch.Tensor,                 # [T, H, V]
    g: Optional[torch.Tensor],       # [T, H]     or None
    gk: Optional[torch.Tensor],      # [T, H, K]  or None
    gv: Optional[torch.Tensor],      # [T, H, V]  or None
    h0: Optional[torch.Tensor],      # [N, H, K, V] or None
    output_final_state: bool,
    cu_seqlens: torch.LongTensor,
    chunk_size: int = 64,
    states_in_fp32: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Pure-PyTorch forward version of chunk_h.
    Autograd will handle backward automatically.
    """
    T, H, K = k.shape
    V = v.shape[-1]
    BT = min(chunk_size, max(16, _next_pow2(max(1, T))))

    chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    N = len(cu_seqlens) - 1  # number of sequences
    NS = int(chunk_offsets[-1].item())  # number of chunks
    
    out_dtype = k.dtype if not states_in_fp32 else torch.float32
    h = k.new_empty((NS, H, K, V), dtype=out_dtype)  # states for each chunk (starting state)
    ht = k.new_empty((N, H, K, V), dtype=torch.float32) if output_final_state else None  # final states for each sequence
    

    for i_seq, (bos, eos) in enumerate(_seq_ranges(cu_seqlens)):
        
        if h0 is not None:
            hidden = h0[i_seq].to(torch.float32)
        else:
            hidden = torch.zeros((H, K, V), device=k.device, dtype=torch.float32)

        T = eos - bos
        NT = (T + BT - 1) // BT  # number of chunks in this sequence
        boh = int(chunk_offsets[i_seq].item())  # boh is the chunk index of the first chunk of this sequence
        i_t = 0

        for i_t in range(NT):
            chunk_start, chunk_end = bos + i_t * BT, min(bos + (i_t + 1) * BT, eos)
            last_idx = chunk_end - 1

            # store entering state
            h[boh + i_t] = hidden.to(out_dtype)

            # slice
            k_c = k[chunk_start: chunk_end].to(torch.float32)  # [L, H, K]
            v_c = v[chunk_start: chunk_end].to(torch.float32)  # [L, H, V]

            if g is not None:
                g_last = g[last_idx].to(torch.float32)   # [H]
                g_c = g[chunk_start: chunk_end].to(torch.float32)  # [L, H]
                hidden = hidden * torch.exp(g_last).view(H, 1, 1)
                v_c = v_c * torch.exp(g_last[None, :] - g_c)[:, :, None]
            if gk is not None:
                gk_last = gk[last_idx].to(torch.float32) # [H, K]
                gk_c = gk[chunk_start: chunk_end].to(torch.float32) # [L, H, K]
                hidden = hidden * torch.exp(gk_last)[:, :, None]
                k_c = k_c * torch.exp(gk_last[None, :, :] - gk_c)
            if gv is not None:
                gv_last = gv[last_idx].to(torch.float32) # [H, V]
                gv_c = gv[chunk_start: chunk_end].to(torch.float32) # [L, H, V]
                hidden = hidden * torch.exp(gv_last)[:, None, :]
                v_c = v_c * torch.exp(gv_last[None, :, :] - gv_c)

            hidden = hidden + torch.einsum("thk,thv->hkv", k_c, v_c)

            i_t += 1

        if output_final_state:
            ht[i_seq] = hidden

    return h, ht
