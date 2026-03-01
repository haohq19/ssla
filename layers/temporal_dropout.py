# copied from FARSE-CNN: https://github.com/AIRLab-POLIMI/farse-cnn

import torch
from .async_sparse_module import AsyncSparseModule

class TemporalDropout (AsyncSparseModule):
    def __init__(self, window_size, *args, **kwargs):
        super(TemporalDropout, self).__init__(*args, **kwargs)
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0.")
        self.window_size = window_size


    def forward(self, inputs, grouped_events=True):
        """
        :param inputs: a Tuple[Tensor,Tensor] with events and unpadded lengths if events are not already grouped, otherwise a dictionary as returned by group_events.
        :param grouped_events: True if the input is already passed as grouped by group_events, False otherwise
        """

        inputs = self.prepare_inputs(inputs, grouped_events)

        batch_id = inputs["batch_id"]
        lengths = inputs["lengths"]
        h = inputs["h"]
        w = inputs["w"]
        events = inputs["events"]
        time = inputs["time"]
        batch_size = inputs["batch_size"]

        with torch.no_grad():
            start_idx = lengths.cumsum(dim=0)
            start_idx = start_idx.roll(1)
            start_idx[0] = 0

            keep_idx_lengths = (lengths / float(self.window_size)).ceil().long() #numbers of events to keep for each sequence
            keep_start_idx = keep_idx_lengths.cumsum(dim=0)
            keep_start_idx = keep_start_idx.roll(1)
            keep_start_idx[0] = 0

            offsets = keep_start_idx.repeat_interleave(keep_idx_lengths)
            keep_idx = torch.arange(offsets.shape[0], device=keep_idx_lengths.device)
            keep_idx -= offsets # consecutive aranges
            keep_idx *= self.window_size

            ####
            keep_window_offset = (lengths % self.window_size - 1).remainder(self.window_size)
            keep_window_offset = keep_window_offset.repeat_interleave(keep_idx_lengths)
            keep_idx += keep_window_offset
            ###

            offsets = start_idx.repeat_interleave(keep_idx_lengths)
            keep_idx += offsets
            keep_idx = keep_idx.long()

        events = events[keep_idx]
        time = time[keep_idx]
        lengths = keep_idx_lengths

        return {"events":events, "time":time, "lengths":lengths,
                "batch_id":batch_id, "h":h, "w":w, "batch_size":batch_size}