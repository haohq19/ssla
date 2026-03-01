from typing import Tuple
import os
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset

from utils.io.h5_events_tools import _load_events_h5


def _xywh_lt_to_xywh_center(target_xywhlt: np.ndarray) -> np.ndarray:
    target = target_xywhlt.astype(np.float32, copy=True)
    target[:, 0] = target[:, 0] + target[:, 2] / 2.0
    target[:, 1] = target[:, 1] + target[:, 3] / 2.0
    return target


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def _crop_events_np(events: np.ndarray, left_xy: np.ndarray, right_xy: np.ndarray) -> np.ndarray:
    if events.shape[0] == 0:
        return events
    lx, ly = int(left_xy[0]), int(left_xy[1])
    rx, ry = int(right_xy[0]), int(right_xy[1])
    m = (events["x"] >= lx) & (events["x"] <= rx) & (events["y"] >= ly) & (events["y"] <= ry)
    return events[m]


def _crop_bbox_xywh_lt(bbox: np.ndarray, left_xy: np.ndarray, right_xy: np.ndarray) -> np.ndarray:
    """
    bbox: (N,5) [x_left, y_top, w, h, class_id]
    按 DAGr _crop_bbox 的语义：转成 xyxy clamp，再转回 xywh
    """
    if bbox.shape[0] == 0:
        return bbox
    lx, ly = float(left_xy[0]), float(left_xy[1])
    rx, ry = float(right_xy[0]), float(right_xy[1])

    b = bbox.astype(np.float32, copy=True)
    x1 = b[:, 0]
    y1 = b[:, 1]
    x2 = b[:, 0] + b[:, 2]
    y2 = b[:, 1] + b[:, 3]

    x1 = np.clip(x1, lx, rx)
    y1 = np.clip(y1, ly, ry)
    x2 = np.clip(x2, lx, rx)
    y2 = np.clip(y2, ly, ry)

    b[:, 0] = x1
    b[:, 1] = y1
    b[:, 2] = np.maximum(0.0, x2 - x1)
    b[:, 3] = np.maximum(0.0, y2 - y1)
    return b


def _scale_and_clip_ratio(r: float, scale: int) -> int:
    return _clamp_int(int(float(r) * float(scale)), 0, scale - 1)


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


class NCaltech101(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        window_size: float = 1.0,
        num_events: int = 50000,
        aug_on_validation: bool = False,
    ):
        super().__init__()
        self.root = root
        self.split = split
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

        H, W = self.sensor_size

        self.window_size = window_size * 1e6    # in us
        self.num_events = num_events
        self.aug_on_validation = aug_on_validation

        # Random Crop
        self._rcrop_w = _scale_and_clip_ratio(0.75, W)
        self._rcrop_h = _scale_and_clip_ratio(0.75, H)
        self._rcrop_left_max_x = max(0, W - self._rcrop_w)
        self._rcrop_left_max_y = max(0, H - self._rcrop_h)

        # Random Translate
        self._tmax_x = _scale_and_clip_ratio(0.1, W)   # e.g. 24 when 0.1
        self._tmax_y = _scale_and_clip_ratio(0.1, H)  # e.g. 18 when 0.1

        # Crop to H x W
        self._full_left = np.array([0, 0], dtype=np.int16)
        self._full_right = np.array([W - 1, H - 1], dtype=np.int16)

    @property
    def sensor_size(self) -> Tuple[int, int]:
        return (180, 240)

    @property
    def num_classes(self) -> int:
        return len(self.classes)
    

    def load_events(self, f_path: Path) -> np.ndarray:
        ev_dict = _load_events_h5(str(f_path), self.num_events)
        
        t = np.asarray(ev_dict["t"])
        x = np.asarray(ev_dict["x"])
        y = np.asarray(ev_dict["y"])
        p = (np.asarray(ev_dict["p"]) + 1) // 2 

        N = t.shape[0]
        out = np.empty((N,), dtype=[("t", np.int64), ("x", np.int16), ("y", np.int16), ("p", np.int8)])
        out["t"] = t.astype(np.int64, copy=False)
        out["x"] = x.astype(np.int16, copy=False)
        out["y"] = y.astype(np.int16, copy=False)
        out["p"] = p.astype(np.int8, copy=False)
        return out
    

    def load_bboxes(self, raw_file: Path, class_id: int) -> np.ndarray:
        rel_path = str(raw_file.relative_to(self.load_dir))
        rel_path = rel_path.replace("image_", "annotation_").replace(".h5", ".bin")
        annotation_file = (self.load_dir / "../annotations" / rel_path).resolve()

        with annotation_file.open("rb") as fh:
            annotations = np.fromfile(fh, dtype=np.int16)
            annotations = np.array(annotations[2:10])

        x_left = int(annotations[0])
        y_top = int(annotations[1])
        x_right = int(annotations[2])
        y_bottom = int(annotations[5])
        w = int(x_right - x_left)
        h = int(y_bottom - y_top)

        return np.array([[x_left, y_top, w, h, class_id]], dtype=np.float32)

    def _random_crop(self, events: np.ndarray, bbox_lt: np.ndarray, p: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > p:
            return events, bbox_lt

        # left ~ U(0, left_max), right = left + size
        H, W = self.sensor_size
        lx = int(np.random.rand() * self._rcrop_left_max_x)
        ly = int(np.random.rand() * self._rcrop_left_max_y)
        rx = min(W, lx + self._rcrop_w)
        ry = min(H, ly + self._rcrop_h)

        left = np.array([lx, ly], dtype=np.int16)
        right = np.array([rx, ry], dtype=np.int16)

        events = _crop_events_np(events, left, right)
        bbox_lt = _crop_bbox_xywh_lt(bbox_lt, left, right)
        return events, bbox_lt

    def _random_translate(self, events: np.ndarray, bbox_lt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dx = int((np.random.rand() * 2 - 1) * self._tmax_x)
        dy = int((np.random.rand() * 2 - 1) * self._tmax_y)

        if dx == 0 and dy == 0:
            return events, bbox_lt

        if events.shape[0] > 0:
            events = events.copy()
            events["x"] = (events["x"].astype(np.int32) + dx).astype(np.int16)
            events["y"] = (events["y"].astype(np.int32) + dy).astype(np.int16)

        if bbox_lt.shape[0] > 0:
            bbox_lt = bbox_lt.copy()
            bbox_lt[:, 0] += dx
            bbox_lt[:, 1] += dy

        return events, bbox_lt


    def _final_full_crop(self, events: np.ndarray, bbox_lt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        events = _crop_events_np(events, self._full_left, self._full_right)
        bbox_lt = _crop_bbox_xywh_lt(bbox_lt, self._full_left, self._full_right)
        return events, bbox_lt


    def _apply_dagr_train_aug(self, events: np.ndarray, bbox_lt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # events, bbox_lt = self._random_hflip(events, bbox_lt)
        events, bbox_lt = self._random_crop(events, bbox_lt, p=0.2)
        # events, bbox_lt = self._random_zoom(events, bbox_lt)
        events, bbox_lt = self._random_translate(events, bbox_lt)
        events, bbox_lt = self._final_full_crop(events, bbox_lt)
        events = random_event_downsample(events, keep_ratio_range=(0.8, 1.0))
        return events, bbox_lt


    def __getitem__(self, idx: int):
        f_path = self.files[idx]
        class_id = self.classes.index(str(f_path.parent.name))

        events = self.load_events(f_path)                 # [t, x, y, p]
        bbox_lt = self.load_bboxes(f_path, class_id)      # [x_left, y_top, w, h, cls]

        if self.split == "train":
            events, bbox_lt = self._apply_dagr_train_aug(events, bbox_lt)
        elif self.split == "val" and self.aug_on_validation:
            events, bbox_lt = self._apply_dagr_train_aug(events, bbox_lt)
        else:
            events, bbox_lt = self._final_full_crop(events, bbox_lt)

        target = _xywh_lt_to_xywh_center(bbox_lt) 
        return events, target

    def __len__(self) -> int:
        return len(self.files)