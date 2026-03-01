# functions for spatial sparse linear attention operations.
# modified from FARSE-CNN: https://github.com/AIRLab-POLIMI/farse-cnn
from typing import Tuple, Dict
import math

import torch
import torch.nn.functional as F


def group_events(
        inputs: Tuple[torch.Tensor, torch.Tensor],
        frame_size: Tuple[int, int],
        max_events_per_px: int = 512
    ) -> Dict[str, torch.Tensor]:
    """
    Groups events by pixel location, and limits the number of events per pixel.
    Args:
        inputs: Tuple of `(events, lengths)`
            - events: `(B, N, C)` tensor of events with elements `[x, y, t, features]`
            - lengths: `(B,)` tensor of lengths
        frame_size: `(H, W)` Tuple of frame size
        max_events_per_px: maximum number of events per pixel to keep, if more events are present, only the latest ones are kept.
    Returns:
        dict:
            - embeddings: `(E, D_emb)` tensor of features of grouped events
            - timestamps: `(E,)` tensor of grouped event timestamps
            - batch_id: `(E,)` tensor of batch indices
            - h: `(G,)` tensor of group y coordinates
            - w: `(G,)` tensor of group x coordinates
            - lengths: `(G,)` tensor of number of events per group
            - batch_size: int
    """
    # input check
    assert isinstance(inputs, tuple) and len(inputs) == 2, "inputs should be a tuple of (events, lengths)"
    assert isinstance(frame_size, tuple) and len(frame_size) == 2, "frame_size should be a tuple of (H, W)"
    events, lengths = inputs
    assert events.dim() == 3 and events.size(2) >= 4, "events should be a (B, N, C) tensor"
    assert lengths.dim() == 1 and lengths.size(0) == events.size(0), "lengths should be a (B,) tensor"
    
    # group events
    events = events.float()
    B, N, _ = events.shape
    H, W = frame_size

    nonpad_mask = torch.arange(N, device=events.device).unsqueeze(0).expand(B, -1) < lengths.unsqueeze(1)   # (B, N)
    events = events[nonpad_mask]    # (E,), E is the total number of non-padded events in the batch

    batch_id = torch.arange(B, device=events.device).repeat_interleave(lengths) # (E,)

    pixel_id = batch_id * (H * W) + events[:, 1] * W + events[:, 0] # (E,)
    order_indices = torch.sort(pixel_id, stable=True)[1]    # (E,)

    # gr_id: (E,), which group the event belongs to
    # gr_lengths: (G,), number of events group
    _, gr_id, gr_lengths = pixel_id[order_indices].unique_consecutive(return_inverse=True, return_counts=True)

    filter_idx = torch.clamp(gr_lengths - max_events_per_px, min=0)      # (G,), number of events to be filtered per group
    gr_lengths = torch.clamp(gr_lengths, max=max_events_per_px)          # (G,), updated group lengths after filtering
    filter_idx = filter_idx.cumsum(dim=0).repeat_interleave(gr_lengths)  # (E',), offset indices of events in each group to keep
    filter_idx = filter_idx + torch.arange(filter_idx.shape[0], device=filter_idx.device)   # (E',), absolute indices of events to keep

    events = events[order_indices]  # grouped events (E,) 

    gr_w = torch.empty(gr_id.max() + 1, device=events.device, dtype=torch.int64)    # (G,)
    gr_h = torch.empty(gr_id.max() + 1, device=events.device, dtype=torch.int64)    # (G,)
    gr_batch_id = torch.empty(gr_id.max() + 1, device=events.device, dtype=torch.int64) # (G,)

    gr_w[gr_id] = events[:, 0].type(torch.int64)    # (G,)
    gr_h[gr_id] = events[:, 1].type(torch.int64)    # (G,)
    gr_batch_id[gr_id] = batch_id                   # (G,)

    events = events[filter_idx]                     # (E',)
    gr_timestamps = events[:, 2].type(torch.int64)  # (E',)
    gr_embeddings = events[:, 3:]                   # (E', D_emb)

    return {
        "events"     : gr_embeddings, 
        "time"       : gr_timestamps, 
        "batch_id"   : gr_batch_id, 
        "h"          : gr_h, 
        "w"          : gr_w,
        "lengths"    : gr_lengths, 
        "batch_size" : B,
    }


def _compute_padding_1d(in_size, kernel_size, stride, mode='same'):
    if mode=='same':
        # amount of padding so that out_size = ceil(in_size/stride)
        return math.ceil((math.ceil(in_size/stride) - 1) * stride + kernel_size - in_size)
    elif mode=='minimal':
        # minimal amount of padding so that each value in the input frame is included in at least one receptive field
        return math.ceil((in_size - kernel_size) / stride) * stride - in_size + kernel_size


def get_patch2pixel_lut(
        frame_size: Tuple[int, int], 
        kernel_size: Tuple[int, int], 
        stride: Tuple[int, int],
    ) -> torch.Tensor:
    """
    Computes the patch to pixel lookup table for given frame size, kernel size and stride.
    Use 'same' padding mode. Padded elements in the lookup table contain value -1.
    Args:
        frame_size: `(H, W)` tuple of input frame size
        kernel_size: `(kh, kw)` tuple of kernel size
        stride: `(sh, sw)` tuple of stride size
    Returns:
      `(num_patches, patch_area)` tensor containing for each receptive field, the indices of pixels contained in it (ascending from top-left to bot-right)
    """
    x = torch.arange(frame_size[1]).repeat([frame_size[0], 1])
    y = torch.arange(frame_size[0]).repeat([frame_size[1], 1]).transpose(0, 1)
    g = x + y * frame_size[1]
    pad_h = _compute_padding_1d(frame_size[0], kernel_size[0], stride[0])
    pad_top, pad_bot = math.floor(pad_h / 2), math.ceil(pad_h / 2)
    pad_w = _compute_padding_1d(frame_size[1], kernel_size[1], stride[1])
    pad_left, pad_right = math.floor(pad_w / 2), math.ceil(pad_w / 2)
    g = F.pad(g, (pad_left, pad_right, pad_top, pad_bot), mode='constant', value=-1)
    g = g.unfold(0, kernel_size[0], stride[0]).unfold(1, kernel_size[1], stride[1])
    patch2pixel_lut = g.reshape([g.shape[0] * g.shape[1], g.shape[2] * g.shape[3]])

    return patch2pixel_lut


def scatter_group_events_to_patch(
    pixel_inputs: Dict[str, torch.Tensor],
    frame_size: Tuple[int, int],
    patch2pixel_lut: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    pixel-level grouped event sequences -> patch-level event sequences.
    Args:
        pixel_inputs: dict containing:
            - batch_id:  `(G,)`
            - lengths:   `(G,)`
            - h:         `(G,)`
            - w:         `(G,)`
            - events:    `(E_pixel, C)`
            - time:      `(E_pixel,)`
            - batch_size: int
        frame_size: `(H, W)`
        patch2pixel_lut: `(num_patch, patch_area)` tensor mapping each patch to pixel indices

    Returns
      - events:        `(E_patch, C)`
      - pos_id:        `(E_patch,)`
      - lengths:       `(num_active_patch,)`
      - batch_size      int
      - orig_event_idx `(E_patch,)`
    """
    # inputs
    pixel_batch_id = pixel_inputs["batch_id"]
    pixel_lengths  = pixel_inputs["lengths"]
    pixel_h        = pixel_inputs["h"]
    pixel_w        = pixel_inputs["w"]
    pixel_events   = pixel_inputs["events"]
    pixel_time     = pixel_inputs["time"]
    B              = pixel_inputs["batch_size"]
    H, W = frame_size
    device = pixel_events.device
    num_patch, patch_area = patch2pixel_lut.shape

    # pixel id of each group
    pixel_id = pixel_batch_id * (H * W) + pixel_h * W + pixel_w   # (G,)

    # batch id of patches in all batches
    patch_batch_id = torch.arange(B, device=device).repeat_interleave(num_patch)    # (B * num_patch,)
    # batch id of all pixels in all batches
    patch_batch_grid = patch_batch_id.unsqueeze(1).repeat(1, patch_area)            # (B * num_patch, patch_area)
    patch2pixel = patch2pixel_lut.repeat(B, 1)                                      # (B * num_patch, patch_area)
    
    patch_batch_grid[patch2pixel == -1] = 0                                         # ignore padded pixels
    # global pixel id of all pixels in all batches, with padding pixels set to -1
    patch2pixel = patch2pixel + patch_batch_grid * (H * W)                          # (B * num_patch, patch_area)


    num_pixel_ids = max(pixel_id.max().item(), patch2pixel.max().item()) + 2    # +1 for padding (-1), +1 for indexing
    active_pixel_mask = torch.zeros(num_pixel_ids, device=device, dtype=torch.bool)
    active_pixel_mask[pixel_id + 1] = True   # mark existing pixel ids (+1 for padding)

    active_patch_pixel_mask = active_pixel_mask[patch2pixel + 1]    # (B * num_patch, patch_area), mask of active pixels in each patch
    active_patch_mask = active_patch_pixel_mask.any(dim=1)          # (B * num_patch,), mask of active patches
    active_patch_pixel_mask = active_patch_pixel_mask[active_patch_mask]  # (num_active_patch, patch_area)
    patch2pixel = patch2pixel[active_patch_mask]                          # (num_active_patch, patch_area)

    pixel_id_to_group_idx = active_pixel_mask.cumsum(dim=0) - 1  # (num_pixel_ids,), map pixel id to group index
    flat_pixel_ids = patch2pixel[active_patch_pixel_mask]        # flattened active pixel ids in all patches
    pixel_group_idx = pixel_id_to_group_idx[flat_pixel_ids + 1]  # flattened group indices of active pixels in all patches

    num_active_pixels_per_patch = active_patch_pixel_mask.sum(dim=1)  # (num_active_patch,)
    num_active_patches = active_patch_pixel_mask.shape[0]
    # patch id of each active pixel/group
    patch_indices = torch.arange(num_active_patches, device=device).repeat_interleave(num_active_pixels_per_patch)
    patch_lengths = torch.zeros(num_active_patches, device=device, dtype=pixel_lengths.dtype)   # (num_active_patch,)
    patch_lengths.scatter_add_(0, patch_indices, pixel_lengths[pixel_group_idx])                # (num_active_patch,)

    pixel_event_start_idx = pixel_lengths.cumsum(dim=0).roll(1) 
    pixel_event_start_idx[0] = 0    # start index of each group in pixel events

    base_event_idx = pixel_event_start_idx[pixel_group_idx] # start index of each group in patched event pixels
    repeats = pixel_lengths[pixel_group_idx]                # number of events in each group in patched event pixels

    # repeated start index of grouped events in patched event pixels
    patch_event_to_pixel_event_idx = torch.repeat_interleave(
        base_event_idx, repeats
    )

    # position offsets of index of each group in patched event pixels
    offsets = torch.ones_like(patch_event_to_pixel_event_idx)
    segment_starts = torch.cat([torch.tensor([0], device=device), repeats.cumsum(dim=0)[:-1]])
    offsets[segment_starts] = torch.cat([torch.tensor([0], device=device), (1 - repeats[:-1])])
    offsets = offsets.cumsum(dim=0)
    patch_event_to_pixel_event_idx += offsets   # pixel index of each event in patched event 

    # scatter patch events (unsorted)
    patch_time = pixel_time[patch_event_to_pixel_event_idx]
    patch_events = pixel_events[patch_event_to_pixel_event_idx]

    # sort events inside each patch by time
    patch_event_patch_id = torch.arange(patch_lengths.shape[0], device=device).repeat_interleave(patch_lengths)
    time_stride = patch_time.max() + 1
    sort_key = patch_time + patch_event_patch_id * time_stride
    sort_key, sort_idx = torch.sort(sort_key)

    # patch_time = sort_key - patch_event_patch_id * time_stride
    patch_events = patch_events[sort_idx]
    patch_event_to_pixel_event_idx = patch_event_to_pixel_event_idx[sort_idx]   # map: index in patch events -> index in group events

    # compute position id (relative position in each patch) inside patch
    patch_pos_id = torch.arange(patch_area, device=device).repeat(active_patch_pixel_mask.shape[0])
    patch_pos_id = patch_pos_id[active_patch_pixel_mask.reshape(-1)]
    patch_pos_id = torch.repeat_interleave(patch_pos_id, repeats)[sort_idx]

    return {
        "events"        : patch_events,
        "pos_id"        : patch_pos_id,
        "lengths"       : patch_lengths,
        "batch_size"    : B,
        "orig_event_idx": patch_event_to_pixel_event_idx,
    }


def gather_patch_events_to_group(
    patch_events: torch.Tensor,
    patch_event_to_pixel_event_idx: torch.Tensor,
    num_pixel_events: int,
):
    """
    Gather patch-level event sequences back to pixel-level event sequences by summing features of events that belong to the same pixel event.

    Args:
        patch_events: `(E_patch, C)`
        patch_event_to_pixel_event_idx: `(E_patch,)`
        num_pixel_events: int

    Returns:
        pixel_events: `(E_pixel, C)`
    """
    device = patch_events.device
    C = patch_events.shape[1]

    pixel_events = torch.zeros(num_pixel_events, C, device=device)
    pixel_events.index_add_(0, patch_event_to_pixel_event_idx, patch_events)
    return pixel_events
