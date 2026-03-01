# copied from FARSE-CNN: https://github.com/AIRLab-POLIMI/farse-cnn

import torch
import torch.nn as nn
from utils.farsecnn_utils import compute_rf2pixels, group_events, gather_receptive_fields


class AsyncSparseModule(nn.Module):
    def __init__(self, frame_size, kernel_size=(1,1), stride=(1,1), max_events_per_rf=512):
        super(AsyncSparseModule, self).__init__()
        self.frame_size = frame_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.register_buffer('max_events_per_rf', torch.tensor(max_events_per_rf))

        rf2pixel_lut, rf_outcoord = compute_rf2pixels(self.frame_size, self.kernel_size, self.stride)
        self.register_buffer('rf2pixel_lut', rf2pixel_lut)
        self.register_buffer('rf_outcoord', rf_outcoord)

        output_maxcoord = rf_outcoord.max(dim=0)[0].tolist()  # max output coordinates in <x,y> format
        self.frame_output_size = (output_maxcoord[1] + 1, output_maxcoord[0] + 1)

    def prepare_inputs(self, inputs, grouped_events):
        if not grouped_events:
            inputs = self.group_events(inputs)

        if not (self.kernel_size[0]==1 and self.kernel_size[1]==1 and\
                self.stride[0]==1 and self.stride[1]==1):
            # gather pixel events into receptive fields
            inputs = self.gather_receptive_fields(inputs)

        return inputs

    def group_events(self, inputs):
        with torch.no_grad():
            grouped_inputs = group_events(inputs, self.frame_size)
        return grouped_inputs

    def gather_receptive_fields(self, inputs):
        gathered_inputs = gather_receptive_fields(inputs, self.frame_size, self.rf2pixel_lut, self.rf_outcoord)
        return gathered_inputs


    def compute_flops(self, inputs, grouped_events=True):
        # fallback function for modules that do not implement compute_flops, or that have no flops cost
        return 0