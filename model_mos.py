from typing import Dict
import os

import torch
from torchvision.ops import nms

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model_base import BaseModel
from layers.temporal_dropout import TemporalDropout
from layers.sparse_pooling import SparseMaxPool, SparseAvgPool, SparseAdaptiveMaxPool, SparseAdaptiveAvgPool
from layers.mos_attention import MosAttention
from layers.linear_attention import LinearAttention
from layers.mos_lstm import MosLSTM
from layers.yolox_head import YOLOXHead
from utils.farsecnn_utils import truncate_packed_sequence, group_events


def batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold, width, height):
    # adopted from torchvision nms, but faster
    # copied from: https://github.com/uzh-rpg/dagr/blob/master/src/dagr/model/utils.py
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_dim = max([width, height])
    offsets = idxs * float(max_dim + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def get_module(config_module, frame_size, channel_size):
    module_name = config_module['name']
    if module_name == 'SparseMaxPool':
        kernel_size = [config_module["kernel_size"]] * 2
        module = SparseMaxPool(frame_size, kernel_size=kernel_size)
    elif module_name == 'SparseAvgPool':
        kernel_size = [config_module["kernel_size"]] * 2
        module = SparseAvgPool(frame_size, kernel_size=kernel_size)
    elif module_name == 'SparseAdaptiveMaxPool':
        output_size = [config_module["output_size"]] * 2
        module = SparseAdaptiveMaxPool(frame_size, output_size)
    elif module_name == 'SparseAdaptiveAvgPool':
        output_size = [config_module["output_size"]] * 2
        module = SparseAdaptiveAvgPool(frame_size, output_size)
    elif module_name == 'TemporalDropout':
        window_size = config_module['window_size']
        module = TemporalDropout(window_size, frame_size)
    elif module_name == 'MosAttention':
        kernel_size = config_module["kernel_size"]
        kernel_size = [min(kernel_size, frame_size[0]), min(kernel_size, frame_size[1])]
        hidden_size = config_module["hidden_size"]
        scatter_proj = config_module.get('scatter_proj', True)
        gather_proj = config_module.get('gather_proj', True)
        module = MosAttention(
            input_size=channel_size, 
            hidden_size=hidden_size, 
            frame_size=frame_size, 
            kernel_size=kernel_size, 
            scatter_proj=scatter_proj, 
            gather_proj=gather_proj,
        )
    elif module_name == 'MosLSTM':
        kernel_size = config_module["kernel_size"]
        kernel_size = [min(kernel_size, frame_size[0]), min(kernel_size, frame_size[1])]
        hidden_size = config_module["hidden_size"]
        scatter_proj = config_module.get('scatter_proj', True)
        gather_proj = config_module.get('gather_proj', True)
        module = MosLSTM(
            input_size=channel_size, 
            hidden_size=hidden_size, 
            frame_size=frame_size, 
            kernel_size=kernel_size, 
            scatter_proj=scatter_proj, 
            gather_proj=gather_proj,
        )
    elif module_name == 'LinearAttention':
        hidden_size = config_module["hidden_size"]
        module = LinearAttention(
            input_size=channel_size,
            hidden_size=hidden_size,
            frame_size=frame_size, 
        )
    else:
        raise ValueError('Module does not exist: ', module_name)
    return module


def get_modulelist(config_layers, sensor_size, in_channels, expected_strides):
    layers = []
    grid_size = sensor_size
    channel_size = in_channels

    stride = 1
    strides = []
    feature_sizes = []
    grid_sizes = []
    output_indices = []

    for i, layer_config in enumerate(config_layers):
        
        if layer_config['name'] in ['SparseMaxPool', 'SparseAvgPool']:
            if stride in expected_strides:
                strides.append(stride)
                feature_sizes.append(channel_size)
                grid_sizes.append(grid_size)
                output_indices.append(i)
            stride = stride * layer_config["kernel_size"]

        m = get_module(layer_config, grid_size, channel_size)
        layers.append(m)
        grid_size = m.frame_output_size
        channel_size = getattr(m, "hidden_size", channel_size)
        
    if stride in expected_strides and stride not in strides:
        strides.append(stride)
        feature_sizes.append(channel_size)
        grid_sizes.append(grid_size)
        output_indices.append(len(layers)-1)
        
    modulelist = torch.nn.ModuleList(layers)
    return modulelist, strides, feature_sizes, grid_sizes, output_indices


class MosModel(BaseModel):
    def __init__(self,
            model_config: Dict, 
            hot_pixel_thres: int = 50,
            conf_thres: float = 0.001,
            iou_thres: float = 0.65,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config = model_config
        channel_size = 2  # initial input channels for events: <dt, p>
        self.hot_pixel_thres = hot_pixel_thres
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        config_layers = model_config["layers"]
        expected_strides = model_config.get("expected_strides", [8, 16])
        
        self.backbone, strides, feature_sizes, grid_sizes, output_indices = get_modulelist(
            config_layers, self.frame_size, channel_size, expected_strides
        )

        self.strides = strides  # [s1, s2, ...]
        self.grid_sizes = grid_sizes  # [(H1, W1), (H2, W2), ...]
        self.grid_size = grid_sizes[-1]
        self.in_features = feature_sizes
        self.output_indices = output_indices

        self.head = YOLOXHead(num_classes=self.num_classes, strides=strides, in_channels=feature_sizes, width=feature_sizes[-1])

        self.validation_step_outputs = []


    def forward(self, batch):
        events = batch["events"] # xytp format
        lengths = batch["lengths"]
        x = self.preprocess_inputs(events, lengths)

        outputs = []
        for i, l in enumerate(self.backbone):
            x = l(x)
            if i in self.output_indices:
                outputs.append(x)

        output_grids = [self.async_event_to_grid(x, k).float() for k, x in enumerate(outputs)]  # List[Tensor (B, C, H, W)]
        gt_bbox = batch['labels']                   # (B, M, 5)  (x, y, w, h, class_id)
        return self.head(output_grids, gt_bbox)


    @torch.no_grad()
    def preprocess_inputs(self, events: torch.Tensor, lengths: torch.Tensor):
        """
        required event format: <x, y, t, p>
        - t in microseconds
        - p in {-1, 1}

        preprocess events:
        1. <x, y, t, p> -> <x, y, t, t, p>
        2. group events into pixel-wise groups, <t, p> per pixel
        3. convert timestamps to delay (pixel-wise delta t) <dt, p>
        4. remove hot pixel events if needed
        """
        # add duplicate timestamp feature
        repeats = torch.ones(events.shape[-1], device=events.device, dtype=torch.int64)
        repeats[2] = 2
        events = events.repeat_interleave(repeats, dim=-1)

        # group events
        grouped_input = group_events((events, lengths), self.frame_size)

        # convert timestamps to delay (pixel-wise delta t)
        timestamps = grouped_input["events"][..., 0]
        offset_start = grouped_input["lengths"].cumsum(dim=0)
        offset_start = offset_start.roll(1)
        offset_start[0] = 0
        timestamps[1:] = timestamps[1:] - timestamps[:-1]
        timestamps[:] = torch.clamp(timestamps[:] / 1e5, max=1) # max delay is 100ms normalized to 1
        timestamps[offset_start] = 1 # initial event set to max delay
        
        # remove hot pixel events
        if self.hot_pixel_thres > 0:
            grouped_input["events"], grouped_input["lengths"], grouped_input["time"] = truncate_packed_sequence(
                grouped_input["events"], grouped_input["lengths"], threshold=self.hot_pixel_thres, others=grouped_input["time"]
            )

        return grouped_input


    def async_event_to_grid(self, x, k):
        events = x['events']
        lengths = x['lengths']
        last_event_idx = lengths.cumsum(dim=0) - 1
        last_events = events[last_event_idx]
        batch_id = x['batch_id']
        h = x['h']
        w = x['w']
        batch_size = x['batch_size']
        event_ids = (batch_id * self.grid_sizes[k][0] * self.grid_sizes[k][1]) + (h * self.grid_sizes[k][1]) + w
        assert len(event_ids) == len(last_events)

        output_grids = torch.zeros([batch_size * self.grid_sizes[k][0] * self.grid_sizes[k][1], self.in_features[k]],
                                   device=last_events.device, dtype=last_events.dtype)
        
        output_grids = output_grids.index_add_(0, event_ids, last_events).reshape(batch_size, self.grid_sizes[k][0], self.grid_sizes[k][1], self.in_features[k]).permute(0,3,1,2)
        return output_grids


    @torch.no_grad()
    def postprocess_with_nms(
        self,
        preds: torch.Tensor,
        conf_thres: float = 0.001,
        iou_thres: float = 0.65,
        filtering: bool = True,
        max_det: int = 100, 
    ):
        """
        preds: (B, N, 4+1+C), bbox is cxcywh in pixel space, obj/cls are logits
        return: List[dict] each dict: boxes(xyxy), scores, labels
        """
        assert preds.dim() == 3 and preds.size(-1) >= 6
        B, N, D = preds.shape
        device = preds.device
        C = self.num_classes

        # 1) cxcywh -> xyxy
        cxcywh = preds[..., :4]
        cx, cy, w, h = cxcywh.unbind(dim=-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, N, 4)

        # 2) score = sigmoid(obj) * max(sigmoid(cls))
        obj = preds[..., 4].sigmoid()                  # (B, N)
        cls_prob = preds[..., 5:5 + C].sigmoid()       # (B, N, C)
        class_conf, class_pred = cls_prob.max(dim=-1)  # (B, N), (B, N)
        scores = obj * class_conf                      # (B, N)

        out = []
        for b in range(B):
            b_boxes = boxes_xyxy[b]
            b_scores = scores[b]
            b_labels = class_pred[b].to(torch.int64)


            if filtering:
                keep = b_scores >= conf_thres
                if keep.sum().item() == 0:
                    out.append({
                        "boxes": torch.zeros((0, 4), device=device, dtype=preds.dtype),
                        "scores": torch.zeros((0,), device=device, dtype=preds.dtype),
                        "labels": torch.zeros((0,), device=device, dtype=torch.int64),
                    })
                    continue

                b_boxes = b_boxes[keep]
                b_scores = b_scores[keep]
                b_labels = b_labels[keep]

            keep_idx = batched_nms_coordinate_trick(
                b_boxes, b_scores, b_labels, iou_thres,
                width=self.frame_size[1], height=self.frame_size[0]
            )

            if max_det is not None and keep_idx.numel() > max_det:
                keep_idx = keep_idx[:max_det]

            out.append({
                "boxes": b_boxes[keep_idx],
                "scores": b_scores[keep_idx],
                "labels": b_labels[keep_idx],
            })

        return out


    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss, loss_iou, loss_obj, loss_cls, num_fg = output['loss'], output['loss_iou'], output['loss_obj'], output['loss_cls'], output['num_fg']
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train/loss_iou', loss_iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_obj', loss_obj, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_cls', loss_cls, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/num_fg', num_fg, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return output


    def validation_step(self, batch, batch_idx):
        output = self(batch)  # (B, N, 4+1+C), (x, y, w, h)
        loss, loss_iou, loss_obj, loss_cls, num_fg = output['loss'], output['loss_iou'], output['loss_obj'], output['loss_cls'], output['num_fg']
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val/loss_iou', loss_iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/loss_obj', loss_obj, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/loss_cls', loss_cls, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val/num_fg', num_fg, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        preds = output['output']  # (B, N, 4+1+C)
        
        # detections
        # List[ Dict{
        #     "boxes": Tensor (N, 4) xyxy in pixel space,
        #     "scores": Tensor (N,),
        #     "labels": Tensor (N,) }]
        detections = self.postprocess_with_nms(
            preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            filtering=True
        )

        # targets
        # List[ Dict{
        #     "boxes": Tensor (M, 4) xyxy in pixel space,
        #     "labels": Tensor (M,) }]
        targets = []
        labels = batch['labels']  # (B, M, 5) cxcywh + cls
        for b in range(labels.size(0)):
            b_labels = labels[b]
            valid_mask = b_labels[:, 2] > 0  # width > 0
            b_labels = b_labels[valid_mask]
            if b_labels.size(0) == 0:
                targets.append({
                    "boxes": torch.zeros((0, 4), device=labels.device, dtype=labels.dtype),
                    "labels": torch.zeros((0,), device=labels.device, dtype=torch.int64),
                })
                continue
            cxcywh = b_labels[:, :4]
            cx, cy, w, h = cxcywh.unbind(dim=-1)
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (M, 4)
            cls = b_labels[:, 4].to(torch.int64)
            targets.append({
                "boxes": boxes_xyxy,
                "labels": cls,
            })
        output_dict = {
            "detections": detections,      # List[Dict], each dict has keys: boxes, scores, labels, bbox format is xyxy
            "targets": targets,            # List[Dict], each dict has keys: boxes, labels, bbox format is xyxy
            "batch_idx": batch_idx
        }
        self.validation_step_outputs.append(output_dict)
        
        return output_dict
    

    def on_validation_epoch_end(self):
        
        outputs = self.validation_step_outputs
        if not outputs:
            return

        # COCO evaluation
        H, W = int(self.frame_size[0]), int(self.frame_size[1])
        dataset_coco = COCO()
        dataset_coco.dataset = {"images": [], "annotations": [], "categories": []}
        for i in range(self.num_classes):
            dataset_coco.dataset["categories"].append({
                "id": i + 1, 
                "name": str(i)}
            )
        
        coco_preds = []
        global_img_id = 0
        annotation_id = 1
        
        for batch_out in outputs:
            batch_dets = batch_out["detections"]
            batch_targets = batch_out["targets"]
            
            for i, (dets, targets) in enumerate(zip(batch_dets, batch_targets)):
                img_id = global_img_id
                
                dataset_coco.dataset["images"].append({
                    "id": img_id, 
                    "height": H, 
                    "width": W, 
                    "file_name": f"{img_id}.jpg"
                })
                # targets: [M, 5] (x, y, w, h, cls)
                if targets is not None:
                    boxes = targets["boxes"]
                    labels = targets["labels"]
                    
                    for j in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[j].tolist()
                        cls_id = labels[j].item()
                        left = x1
                        top = y1
                        w = x2 - x1
                        h = y2 - y1
                        dataset_coco.dataset["annotations"].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": int(cls_id) + 1,
                            "bbox": [left, top, w, h],
                            "area": w * h,
                            "iscrowd": 0
                        })
                        annotation_id += 1
                # dets: [N, 6] (x, y, w, h, score, cls)
                if dets is not None:
                    boxes = dets["boxes"]
                    scores = dets["scores"]
                    labels = dets["labels"]
                    for j in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[j].tolist()
                        score = scores[j].item()
                        cls_id = labels[j].item()
                        left = x1
                        top = y1
                        w = x2 - x1
                        h = y2 - y1
                        coco_preds.append({
                            "image_id": img_id,
                            "category_id": int(cls_id) + 1,
                            "bbox": [left, top, w, h],
                            "score": float(score)
                        })
                        
                global_img_id += 1

        dataset_coco.createIndex()
        if len(coco_preds) == 0:
            for k in ["mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l", "AR_1", "AR_10", "AR_100", "AR_s", "AR_m", "AR_l"]:
                self.log(f"val/{k}", 0.0, on_step=False, on_epoch=True, prog_bar=(k in ["mAP", "mAP_50"]), logger=True, sync_dist=False, rank_zero_only=True)  
            # print("No detections to evaluate.")
            self.validation_step_outputs.clear()
            return
        
        coco_dt = dataset_coco.loadRes(coco_preds)
        coco_eval = COCOeval(dataset_coco, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        names = [
            "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l", 
            "AR_1", "AR_10", "AR_100", "AR_s", "AR_m", "AR_l"
        ]
        for i, n in enumerate(names):
            self.log(f"val/{n}", stats[i], on_step=False, on_epoch=True, prog_bar=(n in ["mAP", "mAP_50"]), logger=True, sync_dist=False, rank_zero_only=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        self.save_detection_results()
        self.on_validation_epoch_end()
    
    @torch.no_grad()
    def save_detection_results(self):
        """
        Save detection results to default output dir.

        Data format:
            -gt: txt file with lines `[class_id, left, top, width, height]`
            -det: txt file with lines `[class_id, confidence, left, top, width, height]`
        """
        gt_dir = os.path.join(self.trainer.log_dir, "gt")
        det_dir = os.path.join(self.trainer.log_dir, "det")
        print(f"Save groundtruth to [{gt_dir}]")
        print(f"Save detections to [{det_dir}]")
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(det_dir, exist_ok=True)

        outputs = self.validation_step_outputs
        sample_counter = 1
        for batch_out in outputs:
            dets_list = batch_out["detections"]
            gts_list = batch_out["targets"]

            for dets, gts in zip(dets_list, gts_list):
                gt_path = os.path.join(gt_dir, f"{sample_counter}.txt")
                det_path = os.path.join(det_dir, f"{sample_counter}.txt")

                # grount truth
                with open(gt_path, "w") as fgt:
                    if gts is not None and gts["boxes"].numel() > 0:
                        boxes = gts["boxes"]    # (M,4) xyxy
                        labels = gts["labels"]  # (M,)
                        for j in range(boxes.size(0)):
                            x1, y1, x2, y2 = boxes[j].tolist()
                            left = x1
                            top = y1
                            w = x2 - x1
                            h = y2 - y1
                            cls_id = int(labels[j].item())
                            fgt.write(
                                f"{cls_id} {left:.6f} {top:.6f} {w:.6f} {h:.6f}\n"
                            )

                # detections
                with open(det_path, "w") as fdet:
                    if dets is not None and dets["boxes"].numel() > 0:
                        boxes = dets["boxes"]
                        scores = dets["scores"]
                        labels = dets["labels"]
                        for j in range(boxes.size(0)):
                            x1, y1, x2, y2 = boxes[j].tolist()
                            left = x1
                            top = y1
                            w = x2 - x1
                            h = y2 - y1
                            conf = float(scores[j].item())
                            cls_id = int(labels[j].item())
                            fdet.write(
                                f"{cls_id} {conf:.6f} {left:.6f} {top:.6f} {w:.6f} {h:.6f}\n"
                            )

                sample_counter += 1
