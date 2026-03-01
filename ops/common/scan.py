from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function

BC_LIST = [16, 64, 256]

@triton.autotune(
    configs=[
        triton.Config({'BC': BC}, num_warps=num_warps)
        for BC in BC_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['C']
)
@triton.jit()
def sequential_scan_fwd(
    u,
    g,
    h,
    cu_seqlens,
    C,
    BC: tl.constexpr,
):
    i_b, i_c = tl.program_id(0), tl.program_id(1)  # batch index, chunk index
    bos, eos = tl.load(cu_seqlens + i_b).to(tl.int32), tl.load(cu_seqlens + i_b + 1).to(tl.int32)
    T = eos - bos   # sequence length

    # offset = tl.arange(0, BC) + bos * C + i_c * BC
    b_h = tl.zeros([BC,], dtype=tl.float32)

    for t in range(T):
        p_u = tl.make_block_ptr(u + (bos + t) * C, (C,), (1,), (i_c * BC), (BC,), (0,))
        p_g = tl.make_block_ptr(g + (bos + t) * C, (C,), (1,), (i_c * BC), (BC,), (0,))
        p_h = tl.make_block_ptr(h + (bos + t) * C, (C,), (1,), (i_c * BC), (BC,), (0,))

        b_u = tl.load(p_u, boundary_check=(0,)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32) 
        b_h = b_h * b_g + b_u
        tl.store(p_h, b_h.to(h.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({'BC': BC}, num_warps=num_warps)
        for BC in BC_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['C']
)
@triton.jit()
def sequential_scan_bwd(
    do,
    g,
    h,
    du,    
    dg,
    cu_seqlens,
    C, 
    BC: tl.constexpr,
):
    i_b, i_c = tl.program_id(0), tl.program_id(1)
    bos, eos = tl.load(cu_seqlens + i_b).to(tl.int32), tl.load(cu_seqlens + i_b + 1).to(tl.int32)
    T = eos - bos   # sequence length
  
    b_dh = tl.zeros([BC,], dtype=tl.float32)

    for t in range(T-1, -1, -1):
        p_do = tl.make_block_ptr(do + (bos + t) * C, (C,), (1,), (i_c * BC,), (BC,), (0,))
        p_g = tl.make_block_ptr(g + (bos + t) * C, (C,), (1,), (i_c * BC,), (BC,), (0,))
        p_h_m_1 = tl.make_block_ptr(h + bos * C, (T, C), (C, 1), (t - 1, i_c * BC,), (1, BC), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0,)).to(tl.float32)            
        b_dh += b_do
        b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)   
        b_h_m_1 = tl.load(p_h_m_1, boundary_check=(0, 1), padding_option="zero").to(tl.float32).reshape(BC,)
        b_dg = b_dh * b_h_m_1

        p_dg = tl.make_block_ptr(dg + (bos + t) * C, (C,), (1,), (i_c * BC,), (BC,), (0,))
        p_du = tl.make_block_ptr(du + (bos + t) * C, (C,), (1,), (i_c * BC,), (BC,), (0,))
        tl.store(p_dg, b_dg.to(dg.dtype.element_ty), boundary_check=(0,))                
        tl.store(p_du, b_dh.to(du.dtype.element_ty), boundary_check=(0,))
        b_dh = b_dh * b_g    
    


class SequentialScan(Function):
    @staticmethod
    # @torch.amp.custom_fwd
    def forward(ctx, u: torch.Tensor, g: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        T, C = u.shape
        NT = len(cu_seqlens) - 1
        # assert C % 256 == 0, 'Hidden dimension must be multiple of 256'
        u = u.contiguous()
        g = g.contiguous()
        cu_seqlens = cu_seqlens.contiguous()

        h = torch.zeros_like(u).contiguous()

        def grid(meta): return (NT, triton.cdiv(C, meta['BC']))
        sequential_scan_fwd[grid](
            u,
            g,
            h,
            cu_seqlens,
            C,
        )

        ctx.save_for_backward(u, g, h)    
        ctx.cu_seqlens = cu_seqlens
        return h
            
    @staticmethod
    # @torch.amp.custom_bwd
    def backward(ctx, do: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        u, g, h = ctx.saved_tensors 
        cu_seqlens = ctx.cu_seqlens
        T, C = u.shape
        NT = len(cu_seqlens) - 1

        du = torch.zeros_like(u).contiguous()
        dg = torch.zeros_like(g).contiguous()
 
        def grid(meta): return (NT, triton.cdiv(C, meta['BC']))
        sequential_scan_bwd[grid](
            do,
            g,
            h,
            du,
            dg,
            cu_seqlens,
            C,
        )
        return du, dg, None


@torch.compiler.disable
def scan(
    u: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    r"""
    Performs a causal scan with forget gates over the input queries.
    h_t = g_t * h_{t-1} + u_t
    Args:
        u (torch.Tensor):
            queries of shape `[T, C]`.
        g (torch.Tensor):
            Forget gates of shape `[T, C]`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        h (torch.Tensor):
            Outputs of shape `[T, C]`.
    """
  
    h = SequentialScan.apply(u, g, cu_seqlens)
    return h

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# from torch.autograd import gradcheck
# def _check_grad():
#     T, C = 64, 8
#     seqlens = torch.tensor([T//3, T//3, T-2*(T//3)], dtype=torch.int32)
#     # seqlens = torch.tensor([T], dtype=torch.int32)
#     u = torch.randn(T, C, dtype=torch.float, requires_grad=True)
#     g = torch.randn(T, C, dtype=torch.float, requires_grad=True)
#     g = torch.exp(F.logsigmoid(g))
#     cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens, dim=0)])
#     u = u.cuda()
#     g = g.cuda()
#     cu_seqlens = cu_seqlens.cuda()

#     test = gradcheck(
#         scan,
#         (u, g, cu_seqlens),
#         eps=1e-1,
#         atol=1e-3,
#         rtol=1e-3
#     )
#     print('gradcheck passed?', test)

# if __name__ == '__main__':
#     _check_grad()

# if __name__ == '__main__':
#     T, C = 6, 1
#     seqlens = torch.tensor([T//3, T//3, T-2*(T//3)], dtype=torch.int32)
#     u = torch.randn(T, C, dtype=torch.float)
#     g = torch.randn(T, C, dtype=torch.float)
#     cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens, dim=0)])
#     u = u.cuda()
#     g = g.cuda()
#     cu_seqlens = cu_seqlens.cuda()

#     h = scan(u, g, cu_seqlens)
#     print(h)