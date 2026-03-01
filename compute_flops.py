# compute FLOPS for MosModel on datasets
from typing import Tuple, Optional
import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset

from model_mos import MosModel
from layers.mos_attention import MosAttention
from utils.io.psee_loader import PSEELoader
from utils.io.h5_events_tools import _load_events_h5


class Gen1(Dataset):
    
    def __init__(self, root: str, split: str = 'test', max_events: Optional[int]=None):
        super(Gen1, self).__init__()
        self.root = root
        self.split = split  # 'train', 'val', 'test'
        assert self.split in ['train', 'val', 'test'], "split must be 'train', 'val' or 'test'"
        self.max_events = max_events

        self.classes = ['car', "pedestrian"]
        self.files = self._get_files()

    def _get_files(self):
        dat_dir = os.path.join(self.root, 'data', self.split)
        dat_files = sorted(os.listdir(dat_dir))
        files = []
        for dat_filename in dat_files:
                files.append(os.path.join(dat_dir, dat_filename))
        return files

    def __getitem__(self, idx):
        """
        Returns:
            - events (timestamp, x, y, polarity) 
        """
        dat_file = self.files[idx]
        loader = PSEELoader(dat_file)
        num_events = loader.event_count()
        if self.max_events is not None:
            num_events = min(self.max_events, num_events)
        events = loader.load_n_events(num_events)
        events = np.stack([events[f].astype(np.float32) for f in events.dtype.names], axis=-1)
        # convert to (x, y, t, p)
        events = events[:, [1,2,0,3]]
        # convert p to [-1,+1]
        events[:, -1] = events[:, -1] * 2 - 1
        return events


    def __len__(self):
        return len(self.files)


class NCaltech101(Dataset):
    def __init__(self, root: str, split: str = "train", max_events: Optional[int]=None):
        super().__init__()
        self.root = root
        self.split = split
        self.max_events = max_events
        assert split in ["train", "val", "test"], "split must be one of ['train', 'val', 'test']"
        # a bit stupid
        if split == 'train':
            self.load_dir = os.path.join(self.root, "training")
        elif split == 'val':
            self.load_dir = os.path.join(self.root, "validation")
        elif split == 'test':
            self.load_dir = os.path.join(self.root, "testing")
        
        self.load_dir = Path(self.load_dir)
        self.classes = sorted([d.name for d in self.load_dir.glob("*")])
        self.files = sorted(list(self.load_dir.rglob("*.h5")))


    def __getitem__(self, idx: int):
        f_path = self.files[idx]
        # events = self.load_events(f_path)   # [t, x, y, p]
        ev_dict = _load_events_h5(str(f_path), self.max_events)
        
        t = np.asarray(ev_dict["t"])
        x = np.asarray(ev_dict["x"])
        y = np.asarray(ev_dict["y"])
        p = np.asarray(ev_dict["p"])

        events = np.stack([x.astype(np.float32), y.astype(np.float32), t.astype(np.float32), p.astype(np.float32)], axis=-1)
        return events

    def __len__(self):
        return len(self.files)



def compute_flops(net, dataset, device = torch.device('cpu'), max_files: Optional[int]=-1) -> Tuple[list, list]:
    flops_per_layer = [0] * (len(net.backbone) + 1)  # +1 for head
    n_ev_per_layer = [0] * (len(net.backbone) + 1)  # +1 for head

    count = 0
    num_files = len(dataset)
    for i in tqdm(range(num_files)):
        count += 1
        if max_files > 0 and count > max_files:
            break
        events = dataset[i]
        events = torch.as_tensor(events).unsqueeze(0).to(device)  # (1, N, 4)
        lengths = torch.tensor([events.shape[1]], device=device)

        x = net.preprocess_inputs(events, lengths)

        with torch.no_grad():
            # backbone
            for i, l in enumerate(net.backbone):
                n_ev_per_layer[i] = n_ev_per_layer[i] + x['events'].shape[0]
                flops_per_layer[i] = flops_per_layer[i] + l.compute_flops(x)
                if not isinstance(l, MosAttention):
                    x = l(x)
            # head
            n_ev_last_layer = x['events'].shape[0]
            n_ev_per_layer[-1] = n_ev_per_layer[-1] + n_ev_last_layer
            flops_per_layer[-1] = flops_per_layer[-1] + net.head.compute_flops() * n_ev_last_layer

    return flops_per_layer, n_ev_per_layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cfg", default='configs/model/MOS_B.yaml', help="Path to model config file.")
    parser.add_argument("--dataset", default='Gen1', help="Name of the dataset to compute the FLOPS.")
    parser.add_argument("--split", default='test', help="Dataset split to use: 'train', 'val', 'test'.")
    args = parser.parse_args()

    model_cfg_path = args.model_cfg
    dataset_name = args.dataset
    split = args.split
    with open(model_cfg_path, "r") as f:
        model_config = yaml.safe_load(f)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Use config: {model_cfg_path}")
    net = MosModel(model_config, dataset=dataset_name).to(device)

    if dataset_name == 'Gen1':
        dataset = Gen1(root='./data/Gen1', split=split, max_events=36000000)
    elif dataset_name == 'NCaltech101':
        dataset = NCaltech101(root='./data/NCaltech101', split=split, max_events=50000)
    else:
        raise NotImplementedError(f"Dataset %s not implemented for FLOPS computation." % dataset_name)
    

    print(f"Compute on dataset: {dataset}")

    print("Computing FLOPS...")
    flops_per_layer, n_ev_per_layer = compute_flops(net, dataset, device, max_files=-1)
    in_events = n_ev_per_layer[0]

    print("============ FLOPS report ============")
    print(f"Total FLOPS:\t %d" % sum(flops_per_layer))
    print(f"Total FLOPS/ev:\t %.4f M" % (sum(flops_per_layer)/in_events/1e6))

    for i, (f,ev) in enumerate(zip(flops_per_layer,n_ev_per_layer)):
        if i == len(net.backbone):
            print(f"%s\t\tFLOPS/ev:%.4f M\t#events received:%d" % (net.head.__class__.__name__, (f/in_events/1e6), ev))
        else:
            print(f"%s\t\tFLOPS/ev:%.4f M\t#events received:%d" % (net.backbone[i].__class__.__name__, (f/in_events/1e6), ev))