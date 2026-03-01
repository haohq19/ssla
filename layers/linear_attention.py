import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ops.common.scan import scan
from utils.async_utils import group_events

class LinearAttention(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            frame_size: Tuple[int, int],
        ):
        super(LinearAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.frame_size = frame_size
        self.frame_output_size = frame_size
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.hidden_size)


        if self.input_size != self.hidden_size:
            self.input_proj = nn.Linear(self.input_size, self.hidden_size, bias=False)
            self.residual_proj = nn.Linear(self.input_size, self.hidden_size, bias=False)
        else:
            self.input_proj = None
            self.residual_proj = None

    def forward(self, inputs):
        """
        :param inputs: a dictionary as returned by group_events.
        :param init_state: a List of Tuple[Tensor,Tensor] with the grouped initial hidden states and cell states to be passed to the lstm cells.
        :param grouped_input: True if the input is already passed as grouped by group_events, False otherwise
        :return:
        """
        gr_inputs   = inputs["events"]
        gr_time     = inputs["time"]
        gr_w        = inputs["w"]
        gr_h        = inputs["h"]
        gr_batch_id = inputs["batch_id"]
        gr_lengths  = inputs["lengths"]
        B           = inputs["batch_size"]
        residual = gr_inputs

        if self.input_proj is not None:
            gr_x = self.input_proj(gr_inputs)   # (E, hidden_size)
        else:
            gr_x = gr_inputs 

        ev_batch_id = gr_batch_id.repeat_interleave(gr_lengths)
        lengths = torch.bincount(ev_batch_id, minlength=B)
        time_sort_idx = torch.argsort(gr_time, stable=True)
        sort_idx = time_sort_idx[torch.argsort(ev_batch_id[time_sort_idx], stable=True)]
        inv_sort_idx = torch.empty_like(sort_idx)
        inv_sort_idx[sort_idx] = torch.arange(sort_idx.numel(), device=sort_idx.device)

        x = gr_x[sort_idx]   # (E, hidden_size)
        nz_lengths = lengths[lengths > 0]
        cu_seqlens = torch.cat(
            [torch.zeros(1, device=lengths.device, dtype=lengths.dtype),
            nz_lengths.cumsum(dim=0)],
            dim=0
        )

        # linear attention
        q = self.q_proj(x)
        v = self.v_proj(x)
        g = F.sigmoid(self.g_proj(x))
        h = scan(
            u=v,
            g=g,
            cu_seqlens=cu_seqlens,
        )
        o = q * h
        o = self.o_proj(o)

        outputs = o[inv_sort_idx]

        if self.residual_proj:
            residual = self.residual_proj(residual)
        outputs = outputs + residual
        outputs = self.norm(outputs)

        return {
            "events"     : outputs,
            "time"       : gr_time, 
            "w"          : gr_w, 
            "h"          : gr_h, 
            "batch_id"   : gr_batch_id,
            "lengths"    : gr_lengths, 
            "batch_size" : B
        }


    def compute_flops(self, inputs):
        flops = 0
        # attention proj
        flops += 3 * num_patch_events * self.hidden_size * (2 * self.hidden_size - 1)  # q, v, g
        # sigmoid
        flops += num_patch_events * self.hidden_size * 4  # exp, add, div, mul
        # attention, h = g * h + u
        flops += num_patch_events * self.hidden_size * 2  # element-wise multiply and addition
        # q * h
        flops += num_patch_events * self.hidden_size    # element-wise multiply
        # output proj
        flops += num_patch_events * self.hidden_size * (2 * self.hidden_size - 1)
        # input & residual proj
        flops += num_group_events * self.hidden_size * (2 * self.input_size - 1) if self.input_proj else 0
        flops += num_group_events * self.hidden_size * (2 * self.input_size - 1) if self.residual_proj else 0
        # residual connection
        flops += num_group_events * self.hidden_size
        # layer norm
        flops += num_group_events * (8 * self.hidden_size + 1)
        return flops