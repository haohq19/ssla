import os
import argparse
import yaml
from pprint import pprint
import datetime

import torch
torch.set_float32_matmul_precision('high') 

import lightning
from lightning.pytorch.callbacks import ModelCheckpoint

from model_mos import MosModel
from utils.model_utils import EMACallback

def main(train_cfg_path):
    # config
    with open(train_cfg_path, "r") as f:
        train_cfg = yaml.safe_load(f)
    
    pprint("Training configuration:")
    pprint(train_cfg)

    model_cfg_path = train_cfg.get('model_cfg')
    with open(model_cfg_path, "r") as f:
        model_config = yaml.safe_load(f)
    pprint("Model configuration:")
    pprint(model_config)

    # set random seed
    lightning.seed_everything(seed=train_cfg.get('seed', 0))

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("No GPU available for training")

    # parsing training config
    dataset_name = train_cfg.get('dataset_name')
    batch_size = train_cfg.get('batch_size', 1)
    
    grad_bs = train_cfg.get('gradient_batch_size', 1)
    if batch_size > grad_bs or (grad_bs % batch_size) != 0:
        new_grad_bs = batch_size * max(grad_bs // batch_size, 1)
        print(f"grad_bs = %d is not a multiple of bs = %d. Setting grad_bs = %d" % grad_bs, batch_size, new_grad_bs)
        grad_bs = new_grad_bs
    accumulate_grad = grad_bs // batch_size


    # model
    model = MosModel(
        model_config=model_config,
        hot_pixel_thres=train_cfg.get('hot_pixel_thres', 50),
        conf_thres=train_cfg.get('conf_thres', 0.001),
        iou_thres=train_cfg.get('iou_thres', 0.65),
        batch_size=batch_size,
        dataset=dataset_name, 
        window_size=train_cfg.get('window_size', 0.1), 
        start_lr=train_cfg.get('learning_rate', 1e-3), 
        weight_decay=train_cfg.get('weight_decay', 1e-2),
        num_epochs=train_cfg.get('max_epoch', 40),
        num_workers=train_cfg.get('num_workers', 2),
        do_validation=train_cfg.get('do_validation', True), 
    ).to(device)

    # callbacks
    # use date-time as part of checkpoint name
    dt_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dirpath = os.path.join(train_cfg.get('checkpoint_dir', 'checkpoints/'), f"{dataset_name}-{dt_string}")
    os.makedirs(checkpoint_dirpath, exist_ok=True)
    print(f"Saving checkpoints to %s" % checkpoint_dirpath)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        monitor='val/loss' if train_cfg.get('do_validation', True) else 'train/loss',
        save_top_k=10,
        save_last=True,
        save_weights_only=False,
    )
    callbacks = [checkpoint_callback]
    use_ema = train_cfg.get('use_ema', False)
    if use_ema:
        ema_callback = EMACallback()
        callbacks.append(ema_callback)
    
    trainer = lightning.Trainer(
        accelerator="gpu",
        strategy="ddp",
        num_nodes=1,
        devices="auto",
        max_epochs=train_cfg.get('max_epoch', 40),
        accumulate_grad_batches=accumulate_grad,
        callbacks=callbacks,
        deterministic=True,
        logger=True,
        limit_val_batches=1.0,
        num_sanity_val_steps=2,
        fast_dev_run=False,
        sync_batchnorm=train_cfg.get('sync_bn', False)
    )

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cfg", default=None,
                        help="Path to a config file for the training run.")
    args = parser.parse_args()
    train_cfg_path = args.train_cfg
    main(train_cfg_path)

