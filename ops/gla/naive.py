# naive pytorch implementation of GLA
# used for verification

import torch

def naive_gla_parallel(q, k, v, g, cu_seqlens, scale=None):
    """
    Naive pytorch implementation of GLA
    Parallel form: O = (((Q*B)(K/B)^T))*M)V
    q, k: (T, H, K) 
    g: (T, H, K), log decay terms 
    v: [T, H, V], 
    cu_seqlens: [N + 1]
    """
    assert q.shape == k.shape == g.shape
    T, H, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype
    if scale is None:
        scale = K ** -0.5

    outputs = []
    for i_seq in range(len(cu_seqlens) - 1):
        idx_seq_start = cu_seqlens[i_seq].item()
        idx_seq_end = cu_seqlens[i_seq + 1].item()
        
        qn = q[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H, K]
        kn = k[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H, K]
        gn = g[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H, K]
        vn = v[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H, V]
         # H first: [H, Tn, K]
        qn, kn, gn, vn = qn.transpose(0, 1), kn.transpose(0, 1), gn.transpose(0, 1), vn.transpose(0, 1)  # [H,Tn,K], [H,Tn,V]
        Tn = qn.shape[1]
        gn_cumsum = torch.cumsum(gn, dim=1)  # [H, Tn, K]

        A = (qn[:, :, None, :] * kn[:, None, :, :] 
             * torch.exp(gn_cumsum[:, :, None, :] - gn_cumsum[:, None, :, :])
             ).sum(dim=-1)  # [H, Tn, Tn]
        A = A * scale

        M = torch.tril(torch.ones(Tn, Tn, dtype=torch.bool, device=A.device))
        A = A.masked_fill(~M, 0)

        on = torch.matmul(A, vn)      # [H, Tn, V]
        outputs.append(on.transpose(0, 1).to(dtype=dtype))  # [Tn, H, V]

    return torch.cat(outputs, dim=0)


def naive_gla_recurrent(q, k, v, g, cu_seqlens, scale=None, initial_state=None, output_final_state=False):
    """
    Naive pytorch implementation of GLA
    Recurrent form: S_t = S_{t-1}*a_t + k_t (v_t)^T
                    o_t = q_t * S_t
    Inputs:
        q, k: (T, H, K) 
        g: (T, H, K), log decay terms 
        v: [T, H, V], 
        cu_seqlens: [N + 1]
        initial_state: [N, H, K, V], initial S_0 for each sequence
        output_final_state: whether to output the final state for each sequence
    return: o: [T, H, V], final_state: [N, H, K, V] (optional)
    """
    assert q.shape == k.shape == g.shape
    T, H, K = q.shape
    V = v.shape[-1]
    device = q.device
    dtype = q.dtype
    if scale is None:
        scale = K ** -0.5

    outputs = []
    final_states = []

    for i_seq in range(len(cu_seqlens) - 1):
        idx_seq_start = cu_seqlens[i_seq].item()
        idx_seq_end = cu_seqlens[i_seq + 1].item()

        qn = q[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H, K]
        kn = k[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H, K]
        vn = v[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H, V]
        gn = g[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H, K]
        Tn = qn.shape[0]

        if initial_state is not None:
            S_0 = initial_state[i_seq].to(torch.float32)  # [H, K, V]
        else:
            S_0 = torch.zeros(H, K, V, device=device, dtype=torch.float32)  # [H, K, V]
        
        S_t = torch.zeros(H, K, V, device=device, dtype=torch.float32)
        S_t += S_0
        on = []
        for t in range(Tn):
            decay_t = torch.exp(gn[t])  # [H, K]
            S_t = decay_t[:, :, None] * S_t + kn[t][:, :, None] * vn[t][:, None, :]  # [H, K, 1] * [H, 1, V] -> [H, K, V]

            a_t = qn[t] * scale  # [H, K]
            o_t = torch.einsum('hk,hkv->hv', a_t, S_t).to(dtype)  # [H, V]
            on.append(o_t)

        outputs.append(torch.stack(on, dim=0))  # [Tn, H, V]
        if output_final_state:
            final_states.append(S_t)           # [H, K, V]

    o = torch.cat(outputs, dim=0)  # [T, H, V]
    final_state = None
    if output_final_state:
        final_state = torch.stack(final_states, dim=0)
    else:
        final_state = None
    return o, final_state


if __name__ == "__main__":
    T, H, K, V = 512, 16, 64, 64
    dtype = torch.float32
    device = torch.device('cuda:0')
    torch.manual_seed(0)
    q = torch.randn(T, H, K, device=device, dtype=dtype)
    k = torch.randn(T, H, K, device=device, dtype=dtype)
    v = torch.randn(T, H, V, device=device, dtype=dtype)
    g = torch.randn(T, H, K, device=device, dtype=dtype) - 1.0
    cu_seqlens = torch.tensor([0, T//3, T//3 + T//4, T], device='cuda', dtype=torch.long)

    o1 = naive_gla_parallel(q, k, v, g, cu_seqlens)
    o2, final_state = naive_gla_recurrent(q, k, v, g, cu_seqlens)

    print("o1 vs o2:", torch.allclose(o1, o2, atol=1e-2, rtol=1e-2))
    print("o1:", o1)
    print("o2:", o2)