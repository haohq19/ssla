# copied from FARSE-CNN: https://github.com/AIRLab-POLIMI/farse-cnn

import math
import torch
from torch_scatter import scatter_max, scatter_mean
from .async_sparse_module import AsyncSparseModule

class SparsePool(AsyncSparseModule):
    def __init__(self, pool_func, *args, kernel_size=(1,1), stride=None, **kwargs):
        stride = stride if stride else kernel_size
        super(SparsePool, self).__init__(*args, kernel_size=kernel_size, stride=stride, **kwargs)
        self.pool_func = pool_func

    def forward(self, inputs, grouped_events=True):
        inputs = self.prepare_inputs(inputs, grouped_events)

        batch_size = inputs["batch_size"]
        rf_batch_id = inputs["batch_id"]
        rf_lengths = inputs["lengths"]
        rf_h = inputs["h"]
        rf_w = inputs["w"]
        rf_events = inputs["events"]
        rf_time = inputs["time"]
        rf_pos_id = inputs["pos_id"]

        max_v = rf_time.max() + 1  # add one just to avoid rare edge cases
        offsets = torch.arange(rf_lengths.shape[0], device=rf_time.device, dtype=rf_time.dtype).repeat_interleave(rf_lengths)
        offsets *= max_v
        rf_time += offsets

        new_rf_time, indices = torch.unique_consecutive(rf_time, return_inverse=True)

        pool_res = self.pool_func(rf_events, indices.unsqueeze(1).expand(-1, rf_events.shape[-1]), dim=0, dim_size=(indices.max()+1))
        if isinstance(pool_res, tuple):
            new_rf_events = pool_res[0]
        else:
            new_rf_events = pool_res

        rf_groups = torch.arange(rf_lengths.shape[0], device=rf_lengths.device,
                                 dtype=rf_lengths.dtype).repeat_interleave(rf_lengths)
        new_rf_groups = torch.zeros(indices.max() + 1, device=rf_groups.device, dtype=rf_groups.dtype)
        new_rf_groups.scatter_(0, indices, rf_groups)
        _, new_rf_lengths = torch.unique_consecutive(new_rf_groups, return_counts=True)

        new_offsets = torch.arange(new_rf_lengths.shape[0], device=new_rf_time.device,
                                   dtype=new_rf_time.dtype).repeat_interleave(new_rf_lengths)
        new_offsets *= max_v
        new_rf_time -= new_offsets

        return {"events": new_rf_events, "time": new_rf_time, "lengths": new_rf_lengths,
                "batch_id": rf_batch_id, "h": rf_h, "w": rf_w, "batch_size": batch_size}

    def compute_flops(self, inputs, grouped_events=True, elementwise_agg_flops=0):
        x = self.prepare_inputs(inputs, grouped_events)

        rf_lengths = x["lengths"]
        rf_events = x["events"]
        rf_time = x["time"]

        in_size = rf_events.shape[-1]

        # aggregation of simultaneous inputs at same cells
        max_v = rf_time.max() + 1
        offsets = torch.arange(rf_lengths.shape[0], device=rf_time.device, dtype=rf_time.dtype).repeat_interleave(rf_lengths)
        offsets *= max_v
        rf_time = rf_time + offsets
        agg_rf_time, count = torch.unique_consecutive(rf_time, return_counts=True)
        agg_n = rf_time.shape[0] - agg_rf_time.shape[0]  # number of inputs that have been aggregated
        flops = agg_n * in_size * elementwise_agg_flops
        return flops


class SparseMaxPool(SparsePool):
    def __init__(self, *args, **kwargs):
        pool_func = scatter_max
        super(SparseMaxPool, self).__init__(pool_func, *args, **kwargs)

    def compute_flops(self, inputs, grouped_events=True):
        return super(SparseMaxPool, self).compute_flops(inputs, grouped_events=grouped_events, elementwise_agg_flops=1)

class SparseAvgPool(SparsePool):
    def __init__(self, *args, **kwargs):
        pool_func = scatter_mean
        super(SparseAvgPool, self).__init__(pool_func, *args, **kwargs)

    def compute_flops(self, inputs, grouped_events=True):
        return super(SparseAvgPool, self).compute_flops(inputs, grouped_events=grouped_events, elementwise_agg_flops=2)

class SparseAdaptiveMaxPool(SparseMaxPool):
    def __init__(self, frame_size, output_size):
        if output_size[0] > frame_size[0] or output_size[1] > frame_size[1]:
            raise ValueError("SparseAdaptiveMaxPool cannot be used to unpool the input frame.")

        kernel_size = [math.ceil(frame_size[0]/output_size[0]), math.ceil(frame_size[1]/output_size[1])]
        super(SparseAdaptiveMaxPool, self).__init__(frame_size, kernel_size=kernel_size)

class SparseAdaptiveAvgPool(SparseAvgPool):
    def __init__(self, frame_size, output_size):
        if output_size[0] > frame_size[0] or output_size[1] > frame_size[1]:
            raise ValueError("SparseAdaptiveMaxPool cannot be used to unpool the input frame.")

        kernel_size = [math.ceil(frame_size[0] / output_size[0]), math.ceil(frame_size[1] / output_size[1])]
        super(SparseAdaptiveAvgPool, self).__init__(frame_size, kernel_size=kernel_size)
