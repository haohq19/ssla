from copy import deepcopy

import torch
import pytorch_lightning as pl


class EMACallback(pl.Callback):
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.ema_state_dict = {}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.ema_state_dict = {k: v.clone().detach() for k, v in pl_module.state_dict().items()}

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        with torch.no_grad():
            for k, v in pl_module.state_dict().items():
                if v.dtype.is_floating_point:
                    ema_v = self.ema_state_dict[k]
                    ema_v.copy_(ema_v * self.decay + v.detach() * (1 - self.decay))

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.original_state_dict = deepcopy(pl_module.state_dict())
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.load_state_dict(self.original_state_dict, strict=False)
        del self.original_state_dict

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.on_validation_start(trainer, pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.on_validation_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict):
        checkpoint['ema_state_dict'] = self.ema_state_dict

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict):
        if 'ema_state_dict' in checkpoint:
            self.ema_state_dict = checkpoint['ema_state_dict']