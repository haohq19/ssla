import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.async_utils import scatter_group_events_to_patch, gather_patch_events_to_group, get_patch2pixel_lut
from ops.common.scan import scan


class MosAttention(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            frame_size: Tuple[int, int],
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int] = (1,1),
            scatter_proj: bool = True,
            gather_proj: bool = True,
        ):
        super(MosAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.frame_size = frame_size
        self.frame_output_size = frame_size # to be consistent with other layers
        self.kernel_size = kernel_size
        self.stride = stride

        self.scatter_proj = scatter_proj
        self.gather_proj = gather_proj

        num_patch_coords = self.kernel_size[0] * self.kernel_size[1]
        scatter_proj_weights_size = self.input_size * self.hidden_size
        gather_proj_weights_size = self.hidden_size * self.hidden_size      

        if self.scatter_proj:
            scatter_proj_weights = torch.nn.Parameter(torch.zeros([num_patch_coords, scatter_proj_weights_size], dtype=torch.float32))
        else:
            scatter_proj_weights = torch.nn.Parameter(torch.zeros([1, scatter_proj_weights_size], dtype=torch.float32))
        self.register_parameter(name='scatter_conv_weights', param=scatter_proj_weights)
        if self.gather_proj:
            gather_proj_weights = torch.nn.Parameter(torch.zeros([num_patch_coords, gather_proj_weights_size], dtype=torch.float32))
        else:
            gather_proj_weights = torch.nn.Parameter(torch.zeros([1, gather_proj_weights_size], dtype=torch.float32))
        self.register_parameter(name='gather_conv_weights', param=gather_proj_weights)
        self._initialize_weights()

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm = nn.LayerNorm(self.hidden_size)

        patch2pixel_lut = get_patch2pixel_lut(self.frame_size, self.kernel_size, self.stride)
        self.register_buffer('patch2pixel_lut', patch2pixel_lut)
        patch_outcoord = torch.zeros([patch2pixel_lut.shape[0], 2], dtype=torch.int)  # to be consistent with previous versions
        self.register_buffer('patch_outcoord', patch_outcoord)    # to be consistent with previous versions

        if self.input_size != self.hidden_size:
            self.input_proj = nn.Linear(self.input_size, self.hidden_size, bias=False)
        else:
            self.input_proj = None

    @torch.no_grad()
    def _initialize_weights(self):
        std = math.sqrt(2.0 / (self.input_size + self.hidden_size))     # Xavier init
        with torch.no_grad():
            self.scatter_conv_weights.normal_(0, std)
            self.gather_conv_weights.normal_(0, std)


    def forward(self, inputs):
        """
        :param inputs: a dictionary as returned by group_events.
        :param init_state: a List of Tuple[Tensor,Tensor] with the grouped initial hidden states and cell states to be passed to the lstm cells.
        :param grouped_input: True if the input is already passed as grouped by group_events, False otherwise
        :return:
        """

        # hard checks (remove later)
        assert inputs["events"].shape[0] == inputs["time"].shape[0]
        assert int(inputs["lengths"].sum().item()) == inputs["events"].shape[0]
        assert inputs["lengths"].min().item() >= 1

        residual    = inputs["events"]
        gr_time     = inputs["time"]
        gr_w        = inputs["w"]
        gr_h        = inputs["h"]
        gr_batch_id = inputs["batch_id"]
        gr_lengths  = inputs["lengths"]
        B           = inputs["batch_size"]

        # scatter to patch events
        patch_inputs = scatter_group_events_to_patch(
            inputs,
            frame_size=self.frame_size,
            patch2pixel_lut=self.patch2pixel_lut,
        )

        patch_lengths = patch_inputs["lengths"]
        patch_events  = patch_inputs["events"]
        patch_pos_id  = patch_inputs["pos_id"]
        
        # scatter proj
        patch_events = self._apply_scatter_proj(patch_events, patch_pos_id)

        # linear attention
        q = self.q_proj(patch_events)
        v = self.v_proj(patch_events)
        g = F.sigmoid(self.g_proj(patch_events))
        cu_seqlens = torch.cumsum(torch.cat([torch.tensor([0], device=patch_lengths.device), patch_lengths]), dim=0)
        h = scan(
            u=v,
            g=g,
            cu_seqlens=cu_seqlens.unique(sorted=True),
        )
        o = q * h
        o = self.o_proj(o)

        # gather proj
        o = self._apply_gather_proj(o, patch_pos_id)
        
        # gather back to group events
        outputs = gather_patch_events_to_group(
            o,
            patch_inputs['orig_event_idx'],
            num_pixel_events=residual.shape[0],
        )

        if self.input_proj:
            residual = self.input_proj(residual)
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

    def _apply_scatter_proj(self, patch_events, patch_pos_id):
        new_patch_events = torch.empty([patch_events.shape[0], self.hidden_size], device=patch_events.device, dtype=torch.float32)
        proj_weights = self.scatter_conv_weights.reshape(-1, self.hidden_size, self.input_size)
        if self.scatter_proj:
            for i in range(proj_weights.shape[0]):
                new_patch_events[patch_pos_id == i] = F.linear(patch_events[patch_pos_id == i], proj_weights[i])
        else:
            new_patch_events = F.linear(patch_events, proj_weights[0])
        return new_patch_events

    def _apply_gather_proj(self, patch_events, patch_pos_id):
        new_patch_events = torch.empty([patch_events.shape[0], self.hidden_size], device=patch_events.device, dtype=torch.float32)
        proj_weights = self.gather_conv_weights.reshape(-1, self.hidden_size, self.hidden_size)
        if self.gather_proj:
            for i in range(proj_weights.shape[0]):
                new_patch_events[patch_pos_id == i] = F.linear(patch_events[patch_pos_id == i], proj_weights[i])
        else:
            new_patch_events = F.linear(patch_events, proj_weights[0])

        return new_patch_events

    def compute_flops(self, inputs):
        gr_events = inputs["events"]
        num_group_events = gr_events.shape[0]
        num_patch_events = gr_events.shape[0] * self.kernel_size[0] * self.kernel_size[1]
        flops = 0
        # scatter proj
        flops += num_patch_events * self.hidden_size * (2 * self.input_size - 1)
        
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
        
        # gather proj
        flops += num_patch_events * self.hidden_size * (2 * self.hidden_size - 1)
        # gather back
        flops += num_patch_events * self.hidden_size
        
        # input proj
        flops += num_group_events * self.hidden_size * (2 * self.input_size - 1) if self.input_proj else 0
        # residual connection
        flops += num_group_events * self.hidden_size
        
        # layer norm
        flops += num_group_events * (8 * self.hidden_size + 1)
        return flops