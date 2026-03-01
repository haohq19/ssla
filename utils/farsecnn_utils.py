import math
from typing import List

import torch
import torch.nn.functional as F


def truncate_packed_sequence(data: torch.Tensor, seqlens: torch.Tensor, threshold: int, others: torch.Tensor = None):
    """    
    Args:
        data: (L, D) Tensor,
        seqlens: (N,) Tensor
        threshold: int
        others: Tensor with first dimension L to be truncated in the same way as data
        
    Returns:
        new_data: (L_new, D)
        new_lengths: (N,)
    """
    offsets = torch.cumsum(seqlens, dim=0) - seqlens
    repeated_offsets = torch.repeat_interleave(offsets, seqlens)
    global_indices = torch.arange(data.shape[0], device=data.device)
    intra_indices = global_indices - repeated_offsets

    mask = intra_indices < threshold
    new_data = data[mask]
    new_lengths = torch.clamp(seqlens, max=threshold)
    
    if others is not None:
        new_others = others[mask]
        return new_data, new_lengths, new_others

    return new_data, new_lengths


def normalize_range(values, min=None, max=None, preserve_pad=True, pad_value=0, pad_start_idx=None):
    # unless values for min and max are specified, the min and max value for each sequence are used
    # conversion is performed in-place by modifying the input tensor's values
    # Note: if pad_start_idx is not provided, it is computed assuming that padded values are at the end of each slice (padding starts after last non-pad index)

    indices = torch.arange(values.shape[1], device=values.device, dtype=torch.long).unsqueeze(0).expand(values.shape[0], -1)
    if pad_start_idx is None:
        pad_start_idx = (values.ne(pad_value).long() * indices).max(dim=1)[0] + 1
    pad_start_idx = pad_start_idx.unsqueeze(-1).expand(-1,indices.shape[1])

    if min is not None:
        if torch.is_tensor(min):
            assert min.shape[0]==values.shape[0], "Shape mismatch for values tensor and min tensor"
        else:
            min = torch.as_tensor([min]*values.shape[0], device=values.device)
    else:
        min = torch.where((indices < pad_start_idx), values, torch.tensor(float('inf'), device=values.device, dtype=values.dtype)).min(dim=1)[0]

    if max is not None:
        if torch.is_tensor(max):
            assert max.shape[0]==values.shape[0], "Shape mismatch for values tensor and max tensor"
        else:
            max = torch.as_tensor([max]*values.shape[0], device=values.device)
    else:
        max = torch.where((indices < pad_start_idx), values, torch.tensor(float('-inf'), device=values.device, dtype=values.dtype)).max(dim=1)[0]

    min = min.unsqueeze(-1).expand(-1, values.shape[1])
    max = max.unsqueeze(-1).expand(-1, values.shape[1])

    values[:] = (values - min) / (max - min + 1e-8)

    if preserve_pad:
        values[:] = torch.where((indices < pad_start_idx), values, torch.tensor(pad_value, device=values.device, dtype=values.dtype))

    return values



'''
Util functions for frame padding
'''

def compute_padding_1d(in_size, kernel_size, stride, mode='same'):
    if mode=='same':
        # amount of padding so that out_size = ceil(in_size/stride)
        return math.ceil((math.ceil(in_size/stride) - 1)*stride + kernel_size - in_size)
    elif mode=='minimal':
        # minimal amount of padding so that each value in the input frame is included in at least one receptive field
        return math.ceil((in_size - kernel_size) / stride) * stride - in_size + kernel_size

def compute_output_size_1d(padded_in_size, kernel_size, stride):
    return math.floor((padded_in_size - kernel_size)/stride + 1)



''' 
General functions for event data
'''

def compute_rf2pixels(frame_size, kernel_size, stride):
    """
    Computes:
    - a [n_rf, n_coords_per_rf] tensor containing for each receptive field, the indices of pixels contained in it (ascending from top-left to bot-right)
    - a [n_rf, 2] tensor containing for each receptive field, the set of coordinates in the output frame that is formed by the convolution as a <x,y> tensor

    Padding is automatically applied using function compute_padding_1d with mode 'same', and the output frame size if adjusted accordingly.
    Padded elements in the rf2p lookup table contain value -1.

    :rtype (torch.Tensor, torch.Tensor)
    """
    x = torch.arange(frame_size[1]).repeat([frame_size[0], 1])
    y = torch.arange(frame_size[0]).repeat([frame_size[1], 1]).transpose(0, 1)
    g = x + y * frame_size[1]

    pad_h = compute_padding_1d(frame_size[0], kernel_size[0], stride[0])
    pad_top, pad_bot = math.floor(pad_h / 2), math.ceil(pad_h / 2)

    pad_w = compute_padding_1d(frame_size[1], kernel_size[1], stride[1])
    pad_left, pad_right = math.floor(pad_w / 2), math.ceil(pad_w / 2)

    g = F.pad(g, (pad_left, pad_right, pad_top, pad_bot), mode='constant', value=-1)

    g = g.unfold(0, kernel_size[0], stride[0]) \
        .unfold(1, kernel_size[1], stride[1])

    rf2p = g.reshape([g.shape[0] * g.shape[1], g.shape[2] * g.shape[3]])

    x_out = torch.arange(g.shape[1]).repeat([g.shape[0], 1]).unsqueeze(-1)
    y_out = torch.arange(g.shape[0]).repeat([g.shape[1], 1]).transpose(0, 1).unsqueeze(-1)
    outcoord = torch.cat([x_out, y_out], dim=-1)
    outcoord = outcoord.reshape([outcoord.shape[0] * outcoord.shape[1], outcoord.shape[2]])

    return rf2p, outcoord

def group_events(inputs, frame_size, max_events_per_px=512):
    events, lengths = inputs
    events = events.float()
    batch_size = events.shape[0]

    nonpad_mask = torch.arange(events.shape[1], device=events.device).unsqueeze(0).expand(events.shape[0],
                                                                                          -1) < lengths.unsqueeze(1)
    events = events[nonpad_mask]

    batch_id = torch.arange(lengths.shape[0], device=events.device).repeat_interleave(lengths)

    ids = batch_id * frame_size[0] * frame_size[1] + events[:, 1] * frame_size[1] + events[:, 0]
    order_idx = torch.sort(ids, stable=True)[1]

    _, ids_unique_idx, gr_lengths = ids[order_idx].unique_consecutive(return_inverse=True, return_counts=True)

    filter_idx = (gr_lengths - max_events_per_px).clamp(min=0)
    filter_idx = filter_idx.cumsum(dim=0)
    gr_lengths = gr_lengths.clamp(max=max_events_per_px)
    filter_idx = filter_idx.repeat_interleave(gr_lengths)
    filter_idx = filter_idx + torch.arange(filter_idx.shape[0], device=filter_idx.device)

    events = events[order_idx]

    gr_w, gr_h, gr_batch_id = torch.empty(ids_unique_idx.max() + 1, device=events.device, dtype=torch.int64), \
                              torch.empty(ids_unique_idx.max() + 1, device=events.device, dtype=torch.int64), \
                              torch.empty(ids_unique_idx.max() + 1, device=events.device, dtype=torch.int64)

    gr_w[ids_unique_idx] = events[:, 0].type(torch.int64)
    gr_h[ids_unique_idx] = events[:, 1].type(torch.int64)
    gr_batch_id[ids_unique_idx] = batch_id

    events = events[filter_idx]
    gr_time = events[:, 2].type(torch.int64)
    gr_events = events[:, 3:]

    return {
        "events"    : gr_events, 
        "time"      : gr_time, 
        "batch_id"  : gr_batch_id, 
        "h"         : gr_h, 
        "w"         : gr_w,
        "lengths"   : gr_lengths, 
        "batch_size": batch_size,
    }


def gather_receptive_fields(inputs, frame_size, rf2pixel_lut, rf_outcoord):
    gr_batch_id = inputs["batch_id"]  # [Hao, batch_id of each group]
    gr_lengths = inputs["lengths"]  # [Hao, lengths of each group, no empty group is included]
    gr_h = inputs["h"]  # [Hao, gr_h and gr_w are the location of groups, not the location of each grouped event]
    gr_w = inputs["w"]  # [Hao, so, if one group has 5 events, there will be only one gr_h and one gr_w for that group]
    gr_events = inputs["events"]  # [Hao, each event have its own feature]
    gr_time = inputs["time"]  # [Hao, each event have its own timestamp]
    batch_size = inputs["batch_size"]

    gr_ids = (gr_batch_id * frame_size[0] * frame_size[1]) + (gr_h * frame_size[1]) + gr_w

    bs_id = torch.arange(batch_size, device=gr_events.device).repeat_interleave(rf2pixel_lut.shape[0]).unsqueeze(1)  # shape [num_rf] # bs_id is the batch-wise offsets for the RFs
    rf_outcoord = torch.cat([rf_outcoord.repeat([batch_size, 1]), bs_id], dim=-1)
    # rf_outcoord is repeated for each sample in the batch, and the batch id is added as third coordinate

    bs_id = bs_id.repeat([1, rf2pixel_lut.shape[1]])  # shape [num_rf, num_rfcoords]
    rf2pixel_lut = rf2pixel_lut.repeat([batch_size, 1])  # shape [num_rf, num_rfcoords]
    bs_id[rf2pixel_lut == -1] = 0
    rf2pixel_lut += bs_id * frame_size[0] * frame_size[1]
    # rf2pixel_lut maps receptive fields of the whole batch to pixels (rf of different samples are independent)
    # padded pixels have index -1

    num_ids = max(gr_ids.max().item(), rf2pixel_lut.max().item()) + 2  # includes the -1 padding id [Hao, from -1, 0, so here is +2]
    ids_set = torch.zeros([num_ids], device=gr_ids.device, dtype=torch.bool)
    ids_set[gr_ids + 1] = True
    nonempty_ids = ids_set[rf2pixel_lut + 1]
    nonempty_rf = nonempty_ids.any(dim=1)
    num_nonempty_rf = nonempty_rf.sum()

    rf2px_flat = rf2pixel_lut.view(-1)
    rf2px_flat = rf2px_flat[nonempty_ids.view(-1)]  # [Hao, the length of rf2px_flat = the length of rf_events?: no]
    # flat lookup table, filtered to contain only nonempty pixel positions

    nonempty_ids = nonempty_ids[nonempty_rf]  # discard empty rfs, shape [num_nonempty_rf, num_rfcoords]

    ids_idx = ids_set.cumsum(dim=0) - 1  # exploits the fact that gr_ids are in ascending order. [Hao, the index of the i_th G in the gr_h, gr_w, gr_lengths] 
    lookup_idx = ids_idx[rf2px_flat + 1]  # lookup_idx: i_th px -> index in gr_*

    rf_groups = torch.arange(num_nonempty_rf, device=nonempty_rf.device)
    rf_groups = rf_groups.repeat_interleave(nonempty_ids.sum(dim=1))
    # for each consecutive sequence id in the flat lookup table, which rf it belongs to
    # (ascending number from 0 to num_nonempty_rf)

    rf_lengths = torch.zeros(num_nonempty_rf, device=gr_lengths.device, dtype=gr_lengths.dtype)
    rf_lengths.scatter_add_(0, rf_groups, gr_lengths[lookup_idx])
    # <num_nonempty_rf>, for each nonempty rf, the length of the contained sequence

    start_ids = gr_lengths.cumsum(dim=0)
    start_ids = start_ids.roll(1)
    start_ids[0] = 0

    flat_lookup_idx = start_ids[lookup_idx]  # selects the first element of every sequence in the flat tensor of sequences
    repeats = gr_lengths[lookup_idx]
    flat_lookup_idx = torch.repeat_interleave(flat_lookup_idx, repeats)  # repeat indices for the length of each sequence
    offsets = torch.ones(flat_lookup_idx.shape[0], dtype=flat_lookup_idx.dtype, device=flat_lookup_idx.device)
    offsets[torch.cat([torch.tensor([0], device=repeats.device), repeats.cumsum(dim=0)[:-1]])] = \
                                            torch.cat([torch.tensor([0], device=repeats.device), (1 - repeats[:-1])])
    offsets = offsets.cumsum(dim=0)

    flat_lookup_idx += offsets  # indices that select all events of sequences in rf order

    rf_time = gr_time[flat_lookup_idx]
    rf_events = gr_events[flat_lookup_idx]
    # flat rf sequences (unsorted)

    offsets = torch.arange(rf_lengths.shape[0], device=rf_time.device, dtype=rf_time.dtype).repeat_interleave(rf_lengths)
    offsets *= (rf_time.max() + 1)  # add one just to avoid rare edge cases

    rf_time += offsets
    rf_time, sort_idx = torch.sort(rf_time)
    rf_time -= offsets

    rf_events = rf_events[sort_idx]
    # flat rf sequences (sorted)

    num_rfcoords = rf2pixel_lut.shape[1]
    rf_pos_id = torch.arange(num_rfcoords, device=nonempty_ids.device).repeat([num_nonempty_rf])
    rf_pos_id = rf_pos_id[nonempty_ids.view(-1)]
    rf_pos_id = torch.repeat_interleave(rf_pos_id, repeats)  # repeat pos ids for the length of each sequence
    rf_pos_id = rf_pos_id[sort_idx]

    nonempty_rf_outcoord = rf_outcoord[nonempty_rf]
    # <num_nonempty_rf, 3>, for each nonempty rf, the out coordinates of the rf in form <x,y,b_id>
    rf_w = nonempty_rf_outcoord[:, 0]
    rf_h = nonempty_rf_outcoord[:, 1]
    rf_batch_id = nonempty_rf_outcoord[:, 2]

    return {"events": rf_events, "time": rf_time, "pos_id": rf_pos_id, "lengths": rf_lengths,
            "h": rf_h, "w": rf_w, "batch_id": rf_batch_id, "batch_size": batch_size}


def ungroup_events_spatial(events, lengths, rf_idx, batch_size=None, frame_size=None):
    """
    Ungroups the values from the flat sequence of events into a dense frame, selecting the last value of each sequence.
    """
    rf_batch_id, rf_h, rf_w = rf_idx
    if batch_size is None:
        batch_size = rf_batch_id.max() + 1
    if frame_size is None:
        frame_size = (rf_h.max() + 1, rf_w.max() + 1)
    feature_size = events.shape[-1]

    spatial_out = torch.zeros([batch_size, *frame_size, feature_size], device=events.device)

    spatial_out[rf_batch_id, rf_h, rf_w] = events[lengths.cumsum(dim=0) - 1]
    return spatial_out
