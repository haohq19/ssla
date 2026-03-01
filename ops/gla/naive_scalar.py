# naive pytorch implementation of GLA
# used for verification

import torch


def naive_gla_recurrent_scalar(q, k, v, g, cu_seqlens, scale=None, initial_state=None, output_final_state=False):
    """
    Naive pytorch implementation of GLA
    Recurrent form: S_t = S_{t-1}*a_t + k_t (v_t)^T
                    o_t = q_t * S_t
    Inputs:
        q, k: (T, H) 
        g: (T, H), log decay terms 
        v: [T, H], 
        cu_seqlens: [N + 1]
        initial_state: [N, H], initial S_0 for each sequence
        output_final_state: whether to output the final state for each sequence
    return: o: [T, H], final_state: [N, H] (optional)
    """
    assert q.shape == k.shape == g.shape
    T, H = q.shape
    device = q.device
    dtype = q.dtype
    if scale is None:
        scale = 1.0

    outputs = []
    final_states = []

    for i_seq in range(len(cu_seqlens) - 1):
        idx_seq_start = cu_seqlens[i_seq].item()
        idx_seq_end = cu_seqlens[i_seq + 1].item()
    
        qn = q[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H]
        kn = k[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H]
        vn = v[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H]
        gn = g[idx_seq_start: idx_seq_end].to(torch.float32)  # [Tn, H]
        Tn = qn.shape[0]

        if initial_state is not None:
            S_0 = initial_state[i_seq].to(torch.float32)  # [H]
        else:
            S_0 = torch.zeros(H, device=device, dtype=torch.float32)  # [H]
        
        S_t = torch.zeros(H, device=device, dtype=torch.float32)
        S_t += S_0
        on = []
        for t in range(Tn):
            decay_t = torch.exp(gn[t])  # [H]
            S_t = decay_t * S_t + kn[t] * vn[t]  # [H]

            a_t = qn[t] * scale  # [H]
            o_t = a_t * S_t
            on.append(o_t.to(dtype))
        outputs.append(torch.stack(on, dim=0))  # [Tn, H]
        if output_final_state:
            final_states.append(S_t)           # [H]

    o = torch.cat(outputs, dim=0)  # [T, H]
    final_state = None
    if output_final_state:
        final_state = torch.stack(final_states, dim=0)
    else:
        final_state = None
    return o, final_state