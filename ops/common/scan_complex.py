import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.autograd import Function


@triton.jit()
def fwd_sequential_scan_complex(
    v_real,
    v_imag,
    g_real,
    g_imag,
    h_real,
    h_imag,
    cu_seqlens,
    C,
    BC: tl.constexpr,
):
    
    i_b, i_c = tl.program_id(0), tl.program_id(1)  # batch index, chunk index
    bos, eos = tl.load(cu_seqlens + i_b).to(tl.int32), tl.load(cu_seqlens + i_b + 1).to(tl.int32)
    T = eos - bos   # sequence length

    offset = tl.arange(0, BC) + bos * C + i_c * BC
    _h_real = tl.zeros([BC,], dtype=tl.float32)
    _h_imag = tl.zeros([BC,], dtype=tl.float32)

    for _ in range(T):        
        _v_real = tl.load(v_real + offset).to(tl.float32)                
        _v_imag = tl.load(v_imag + offset).to(tl.float32)
        
        _g_real = tl.load(g_real + offset).to(tl.float32) 
        _g_imag = tl.load(g_imag + offset).to(tl.float32) 
        
        _h_real_new = _h_real * _g_real - _h_imag * _g_imag + _v_real
        _h_imag_new = _h_real * _g_imag + _h_imag * _g_real + _v_imag 
                
        tl.store(h_real + offset, _h_real_new.to(h_real.dtype.element_ty))
        tl.store(h_imag + offset, _h_imag_new.to(h_imag.dtype.element_ty))
        _h_real = _h_real_new
        _h_imag = _h_imag_new
        offset += C


@triton.jit()
def bwd_sequential_scan_complex(
    do_real,
    do_imag,
    v_real,
    v_imag,
    g_real,
    g_imag,
    h_real,
    h_imag,
    dv_real,
    dv_imag,
    dg_real,    
    dg_imag,
    cu_seqlens,
    C, 
    BC: tl.constexpr,
):
    i_b, i_c = tl.program_id(0), tl.program_id(1)
    bos, eos = tl.load(cu_seqlens + i_b).to(tl.int32), tl.load(cu_seqlens + i_b + 1).to(tl.int32)
    T = eos - bos   # sequence length
  
    offset = tl.arange(0, BC) + (bos + T - 1) * C + i_c * BC  # last time step
    _dh_real = tl.zeros([BC,], dtype=tl.float32)
    _dh_imag = tl.zeros([BC,], dtype=tl.float32)

    for _ in range(T-1, -1, -1):
        _do_real = tl.load(do_real + offset).to(tl.float32)            
        _do_imag = tl.load(do_imag + offset).to(tl.float32)          
        
        _dh_real += _do_real
        _dh_imag += _do_imag
        
        _g_real = tl.load(g_real + offset).to(tl.float32)   
        _g_imag = tl.load(g_imag + offset).to(tl.float32) 

        _h_m_1_real = tl.load(h_real + offset - C, mask=offset >= (bos * C + C), other=0.0).to(tl.float32)
        _h_m_1_imag = tl.load(h_imag + offset - C, mask=offset >= (bos * C + C), other=0.0).to(tl.float32)
                
        _dg_real = (_dh_real * _h_m_1_real + _dh_imag * _h_m_1_imag) 
        _dg_imag = (_dh_imag * _h_m_1_real - _dh_real * _h_m_1_imag) 

        tl.store(dg_real + offset, _dg_real.to(dg_real.dtype.element_ty))                
        tl.store(dg_imag + offset, _dg_imag.to(dg_imag.dtype.element_ty))                
        tl.store(dv_real + offset, _dh_real.to(dv_real.dtype.element_ty))
        tl.store(dv_imag + offset, _dh_imag.to(dv_imag.dtype.element_ty))

        _dh_real_new = _dh_real * _g_real + _dh_imag * _g_imag 
        _dh_imag_new = _dh_imag * _g_real - _dh_real * _g_imag
        
        _dh_real = _dh_real_new
        _dh_imag = _dh_imag_new
        
        offset -= C        



class SequentialScanComplex(Function):
    @staticmethod
    @torch.amp.custom_fwd
    def forward(ctx, v_real, v_imag, g_real, g_imag, cu_seqlens):
        T, C = v_real.shape
        NT = len(cu_seqlens) - 1
        num_warps = 8
        assert C % 256 == 0, 'Hidden dimension must be multiple of 256'
        v_real = v_real.contiguous()
        v_imag = v_imag.contiguous()
        g_real = g_real.contiguous()
        g_imag = g_imag.contiguous()
        cu_seqlens = cu_seqlens.contiguous()

        h_real = torch.zeros_like(v_real).contiguous()
        h_imag = torch.zeros_like(v_imag).contiguous()
                                    
        fwd_sequential_scan_complex[(NT, int(C/256))](
            v_real,
            v_imag,
            g_real,
            g_imag,
            h_real,
            h_imag,
            cu_seqlens,
            C, 
            BC=256,
            num_warps=num_warps
        )

        ctx.save_for_backward(v_real, v_imag, g_real, g_imag, h_real, h_imag)    
        ctx.cu_seqlens = cu_seqlens
        return h_real, h_imag
            
    @staticmethod
    @torch.amp.custom_bwd
    def backward(ctx, do_real, do_imag):
        v_real, v_imag, g_real, g_imag, h_real, h_imag = ctx.saved_tensors 
        cu_seqlens = ctx.cu_seqlens
        T, C = v_real.shape
        NT = len(cu_seqlens) - 1

        dv_real = torch.zeros_like(v_real).contiguous()
        dv_imag = torch.zeros_like(v_imag).contiguous()
        dg_real = torch.zeros_like(g_real).contiguous()
        dg_imag = torch.zeros_like(g_imag).contiguous()
        
        num_warps = 8

        bwd_sequential_scan_complex[(NT,  int(C/256))](
            do_real, 
            do_imag,
            v_real, 
            v_imag,
            g_real,
            g_imag, 
            h_real, 
            h_imag,
            dv_real,
            dv_imag,
            dg_real,
            dg_imag,
            cu_seqlens,
            C, 
            BC=256,
            num_warps=num_warps
        )
        return dv_real, dv_imag, dg_real, dg_imag, None


complex_scan = SequentialScanComplex.apply