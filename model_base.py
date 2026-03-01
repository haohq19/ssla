from typing import Optional, Any, Dict, List, Tuple
import numpy as np

import torch
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.utilities import grad_norm

from utils.data_utils import get_dataset

class BaseModel(LightningModule):
    def __init__(self, 
            batch_size: int = 1, 
            dataset: str = 'Gen1',
            window_size: float = 0.1,
            start_lr: float = 1e-3,
            weight_decay: float = 1e-2,
            num_epochs: int = 40,
            num_workers: int = 2,
            do_validation: bool = True,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.window_size = window_size
        self.start_lr = start_lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.do_validation = do_validation

        assert dataset in ['Gen1', 'NCaltech101'], f'Unknown dataset %s' % dataset
        if self.dataset == 'Gen1':
            self.frame_size = [240, 304]
            self.num_classes = 2
        elif self.dataset == 'NCaltech101':
            self.frame_size = [180, 240]
            self.num_classes = 101
        else:
            raise ValueError(f'Unknown dataset %s' % self.dataset)

        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    def setup(self, stage=None):
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(
            self.dataset,
            window_size=self.window_size,
            do_validation=self.do_validation,
        )
        self.classes = self.test_dataset.classes


    def train_dataloader(self):
        # Shape of each batch is (bs, max_length, 4) where max_length is the maximum number of events in the samples of the batch
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                                           num_workers=self.num_workers, collate_fn=self.pad_batches)

    def val_dataloader(self):
        # Shape of each batch is (bs, max_length, 4) where max_length is the maximum number of events in the samples of the batch
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                                           num_workers=self.num_workers, collate_fn=self.pad_batches)

    def test_dataloader(self):
        # Shape of each batch is (bs, max_length, 4) where max_length is the maximum number of events in the samples of the batch
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, 
                                           num_workers=self.num_workers, collate_fn=self.pad_batches)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.start_lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.num_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({'train/'+k:v for k,v in grad_norm(self, norm_type=2).items()})
        return

    def log_metrics(self, metrics):
        self.log_dict(metrics)

    def pad_batches(self, data: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """
        Inputs:
        - data: is a list of len `batch_size` containing sample tuples `(input, label)`
        
        Returns:
        - A batch dictionary with padded events, labels and their lengths
        """

        events = [np.stack([sample[0][f].astype(np.float32) for f in sample[0].dtype.names], axis=-1) for sample in data]
        events_lens = [len(ev) for ev in events]
        max_len = max(events_lens)
        events = [np.pad(ev, ((0, max_len - ln), (0, 0)), mode='constant', constant_values=0) for
                  ln, ev in zip(events_lens, events)]

        labels = [sample[1] for sample in data]
        labels_lens = [len(l) for l in labels]
        max_len = max(labels_lens)
        labels = [np.pad(lb, ((0, max_len - ln), (0, 0)), mode='constant', constant_values=0) for
                  ln, lb in zip(labels_lens, labels)]

        events = torch.as_tensor(np.stack(events, axis=0))
        labels = torch.as_tensor(np.stack(labels, axis=0))
        events_lens = torch.as_tensor(events_lens)
        labels_lens = torch.as_tensor(labels_lens)

        if self.dataset == 'Gen1':
            # <t, x, y, p> -> <x, y, t, p>
            events = events[:, :, [1,2,0,3]]
            # convert p to [-1,+1]
            events[:, :, -1] = events[:, :, -1] * 2 - 1
        elif self.dataset == 'NCaltech101':
            # <t, x, y, p> -> <x, y, t, p>
            events = events[:, :, [1,2,0,3]]
            # convert p to [-1,+1]
            events[:, :, -1] = events[:, :, -1] * 2 - 1

        batch = {
            "events"        : events, 
            "labels"        : labels, 
            "lengths"       : events_lens, 
            "labels_lengths": labels_lens
        }
        return batch
    