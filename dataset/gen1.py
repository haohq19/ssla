from typing import Tuple, Optional
import os
import json
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset

from utils.io.psee_loader import PSEELoader


def random_event_downsample(events, keep_ratio_range=(0.8, 1.0)):
    """
    Randomly downsample events by Bernoulli thinning.

    Args:
        events: structured numpy array with fields ['t','x','y','p', ...]
        keep_ratio_range: (r_min, r_max), fraction of events to keep

    Returns:
        downsampled events
    """
    if events.shape[0] == 0:
        return events

    r_min, r_max = keep_ratio_range
    assert 0.0 < r_min <= r_max <= 1.0

    keep_ratio = np.random.uniform(r_min, r_max)
    mask = np.random.rand(events.shape[0]) < keep_ratio
    return events[mask]



class Gen1(Dataset):
    
    def __init__(
        self, 
        root: str,
        split: str = 'train', 
        window_size: float = 0.1, 
        hflip_p: float = 0.5, 
        downsample_ratio_range: Tuple[float, float] = (0.8, 1.0), 
        valid_idx_path: Optional[str] = None,
    ):
        super(Gen1, self).__init__()
        self.root = root
        self.split = split  # 'train', 'val', 'test'
        assert self.split in ['train', 'val', 'test'], "split must be 'train', 'val' or 'test'"

        self.classes = ['car', "pedestrian"]
        self.chunks = self._chunk_by_bbox()
        self.window_size = window_size * 1e6  # expects window_size in seconds, converts in microseconds
        self.hflip_p = hflip_p
        self.downsample_ratio_range = downsample_ratio_range

        if valid_idx_path:
            if os.path.exists(valid_idx_path):
                with open(valid_idx_path, "r") as fp:
                    deserialized_data = json.load(fp)

                valid_idx = deserialized_data["valid_idx"]
            else:
                valid_idx = self._filter_invalid_samples()
                data_to_serialize = {"valid_idx": valid_idx, "window_size":self.window_size}

                with open(valid_idx_path, "w") as fp:
                    json.dump(data_to_serialize, fp)
        else:
            print("No valid_idx_path provided, valid indices will not be stored.")
            valid_idx = self._filter_invalid_samples()

        self.chunks = [self.chunks[i] for i in valid_idx]


    @property
    def sensor_size(self) -> Tuple[int, int]:
        return (240, 304)


    @property
    def num_classes(self):
        return len(self.classes)


    def _chunk_by_bbox(self):
        bbox_dir = os.path.join(self.root, 'bbox', self.split)
        bbox_files = sorted(os.listdir(bbox_dir))
        dat_dir = os.path.join(self.root, 'data', self.split)
        dat_files = sorted(os.listdir(dat_dir))

        chunks = []

        for bbox_filename, dat_filename in zip(bbox_files, dat_files):
            # e.g. 
            # bbox: 17-03-30_12-53-58_1037500000_1097500000_bbox.npy
            # dat: 17-03-30_12-53-58_1037500000_1097500000_td.dat
            # check they correspond to the same recording
            
            bbox = np.load(os.path.join(bbox_dir, bbox_filename))
            assert bbox_filename.split('_bbox')[0] == dat_filename.split('_td')[0], \
                "Mismatched bbox {} and dat files {}".format(bbox_filename.split('_bbox')[0], dat_filename.split('_td')[0])
            unique_ts, unique_indices = np.unique(bbox['ts'], return_index=True)
            unique_indices = np.append(unique_indices, bbox.shape[0])

            for i, ts in enumerate(unique_ts):
                target = bbox[unique_indices[i]:unique_indices[i + 1]][['x', 'y', 'w', 'h', 'class_id']]
                # (num_boxes, 5), (l, t, w, h, class_id)
                # target = target.astype(np.float32)
                # target[:, 0] = target[:, 0] + target[:, 2] / 2  # convert to center x
                # target[:, 1] = target[:, 1] + target[:, 3] / 2  # convert to center y
                target = list(map(tuple, target))  # convert to list of tuples
                target = np.array(target, dtype=np.float32)
                
                # crop to fov filter
                x_left = target[:, 0]
                y_top = target[:, 1]
                x_right = target[:, 0] + target[:, 2]
                y_bottom = target[:, 1] + target[:, 3]

                height, width = self.sensor_size
                x_left_cropped = np.clip(x_left , a_min=0, a_max=width - 1)
                y_top_cropped = np.clip(y_top, a_min=0, a_max=height - 1)
                x_right_cropped = np.clip(x_right, a_min=0, a_max=width - 1)
                y_bottom_cropped = np.clip(y_bottom, a_min=0, a_max=height - 1)

                x_center = (x_left_cropped + x_right_cropped) / 2
                y_center = (y_top_cropped + y_bottom_cropped) / 2
                w_cropped = x_right_cropped - x_left_cropped
                h_cropped = y_bottom_cropped - y_top_cropped
                target[:, 0] = x_center
                target[:, 1] = y_center
                target[:, 2] = w_cropped
                target[:, 3] = h_cropped

                # filter out bboxes
                # prophesee_bbox_filter
                # Default values taken from: https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/0393adea2bf22d833893c8cb1d986fcbe4e6f82d/src/psee_evaluator.py#L23-L24
                min_box_diag = 30
                min_box_side = 10   
                diag_ok = w_cropped ** 2 + h_cropped ** 2 >= min_box_diag ** 2
                side_ok = (w_cropped >= min_box_side) & (h_cropped >= min_box_side)
                # remove_faulty_huge_bbox_filter
                # only in train set
                # Default values taken from: https://github.com/uzh-rpg/RVT/blob/master/scripts/genx/preprocess_dataset.py
                if self.split == 'train':
                    max_box_width = width * 9 // 10
                else:
                    max_box_width = width  # no filtering in val/test
                width_ok = w_cropped <= max_box_width
                # filter out invalid boxes
                keep = diag_ok & side_ok & width_ok
                target = target[keep]

                chunks.append({'path': os.path.join(dat_dir, dat_filename),
                               'ts': ts,
                               'target': target})

        return chunks


    def _filter_invalid_samples(self):
        """
        Removes chunks with empty sequences of events, and sequences with unordered timestamps
        """
        invalid_idx = []

        for i in tqdm(range(len(self.chunks))):
            ev, _ = self.__getitem__(i)
            if ev.shape[0] == 0:
                invalid_idx.append(i)
            t = ev['t']
            if (t != sorted(t)).any():
                invalid_idx.append(i)

        return [i for i in range(len(self.chunks)) if i not in invalid_idx]


    def __getitem__(self, idx):
        """
        Returns:
            - events (timestamp, x, y, polarity) 
            - label (x, y, w, h, class_id)
        """
        chunk_data = self.chunks[idx]

        f = chunk_data['path']
        loader = PSEELoader(f)

        # load a window of length window_size before the bounding box timestamp
        loader.seek_time(chunk_data['ts'] - self.window_size)
        events = loader.load_delta_t(self.window_size + 1) # increment by 1 so the exact timestamp is loaded too
        target = chunk_data['target']

        if self.split == 'train':
            events = random_event_downsample(events, keep_ratio_range=self.downsample_ratio_range)
            if np.random.rand() < self.hflip_p:
                # horizontal flip
                width = self.sensor_size[1]
                events['x'] = width - 1 - events['x']
                if target.shape[0] > 0:
                    target[:, 0] = width - 1 - target[:, 0]  # x_center
                # no change to class_id, y, w, h


        return events, target


    def __len__(self):
        return len(self.chunks)


