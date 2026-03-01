import argparse
import yaml

import torch
torch.set_float32_matmul_precision('high') 

import lightning

from model_mos import MosModel
from utils.model_utils import EMACallback


def main(test_cfg_path):
    with open(test_cfg_path, "r") as stream:
        test_cfg = yaml.safe_load(stream)
    model_cfg_path = test_cfg.get('model_cfg')
    with open(model_cfg_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    # set random seed 
    lightning.seed_everything(seed=test_cfg.get('seed', 0))

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("No GPU available for testing.")

    dataset = test_cfg.get('dataset_name')
    batch_size = test_cfg.get('batch_size', 1)
    window_size = test_cfg.get('window_size', 0.1)
    existing_ckpt = test_cfg.get('resume_ckpt')
    test_on_val = test_cfg.get('test_on_val', False)
    if test_on_val:
        print("Warning! Testing on validation set.")

    callbacks = []
    use_ema = test_cfg.get('use_ema', False)
    if use_ema:
        ema_callback = EMACallback()
        callbacks.append(ema_callback)

    model = MosModel(
        model_config=model_config, 
        hot_pixel_thres=test_cfg.get('hot_pixel_thres', 50),
        conf_thres=test_cfg.get('conf_thres', 0.001),
        iou_thres=test_cfg.get('iou_thres', 0.65),
        batch_size=batch_size, 
        dataset=dataset, 
        window_size=window_size, 
        num_workers=test_cfg.get('num_workers', 2),
    ).to(device)

    trainer = lightning.Trainer(
        accelerator="gpu",
        num_nodes=1,
        devices=1,
        callbacks=callbacks,
        deterministic=True,
        logger=True,
    )
    if test_on_val:
        trainer.validate(model, ckpt_path=existing_ckpt)
    else:
        trainer.test(model, ckpt_path=existing_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_cfg", default='configs/test_cfg_ncaltech101_b.yaml',
                        help="Path to a config file for the training run.")
    args = parser.parse_args()
    test_cfg_path = args.test_cfg

    main(test_cfg_path)

