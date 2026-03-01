#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
    

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="giou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        width: int = 128,
        strides: List[int] = [8, 16, 32],
        in_channels: List[int] = [128, 256, 512],
    ):
        """
        Args:
            num_classes (int): number of classes
            stride (int): stride of the feature map
            in_channels (int): number of input channels
        """
        super().__init__()

        self.num_classes = num_classes
        self.width = width
        self.decode_in_inference = True  # for deploy, set to False
        self.strides = strides
        self.in_channels = in_channels

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for in_channel in in_channels:
            self.cls_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=width,
                        kernel_size=1,
                    ),
                    nn.BatchNorm2d(width),
                    nn.SiLU(),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_channel,
                        out_channels=width,
                        kernel_size=1,
                    ),
                    nn.BatchNorm2d(width),
                    nn.SiLU(),
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=width,
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=width,
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=width,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.grids = [torch.zeros(1)] * len(strides)

        self.initialize_biases(prior_prob=0.01)

    def initialize_biases(self, prior_prob):
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for obj_pred in self.obj_preds:
            b = obj_pred.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin: List[torch.Tensor], labels: List[torch.Tensor]):
        # xin: List of tensor with different strides (B, C_in, H_in, W_in)
        # labels: (B, n_labels, 5) [x, y, w, h, class_id]

        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, cls_pred, reg_pred, obj_pred, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.cls_preds, self.reg_preds, self.obj_preds, xin)
        ):  
            cls_feat = cls_conv(x)
            reg_feat = reg_conv(x)

            cls_output = cls_pred(cls_feat)
            reg_output = reg_pred(reg_feat)
            obj_output = obj_pred(reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(
                output, k, self.strides[k], x.type()
            )

            outputs.append(output)
            
            x_shift = grid[:, :, 0]
            y_shift = grid[:, :, 1]
            expanded_stride = torch.zeros(1, grid.shape[1]).fill_(self.strides[k]).type_as(x)
            x_shifts.append(x_shift)
            y_shifts.append(y_shift)
            expanded_strides.append(expanded_stride)

        outputs = torch.cat(outputs, 1)  # (B, n_anchors, 5 + n_cls)
        x_shifts = torch.cat(x_shifts, 1)  # (1, n_anchors)
        y_shifts = torch.cat(y_shifts, 1)  # (1, n_anchors)
        expanded_strides = torch.cat(expanded_strides, 1)  # (1, n_anchors)

        loss_dict = self.get_losses(
                        x_shifts,
                        y_shifts,
                        expanded_strides,
                        labels,
                        outputs,
                        dtype=xin[0].dtype,
                    )
        output_dict = {**loss_dict, 'output': outputs}
        return output_dict


    def get_output_and_grid(self, output, k, stride, dtype):
        '''
        Get output tensor and grid tensor
        Args:
            output: (B, C, H, W)
            k: stride level
            stride: int
        '''
        grid = self.grids[k]

        B = output.shape[0]
        C = 5 + self.num_classes
        H, W = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
            grid = torch.stack((xv, yv), 2).reshape(1, H, W, 2).type(dtype)
            self.grids[k] = grid  # (1, 1, H, W, 2), (xv, yv)

        output = output.permute(0, 2, 3, 1).reshape(
            B, H * W, C
        )  # (B, H*W, C)
        grid = grid.view(1, H * W, 2)  # (1, H*W, 2)
        output[..., :2] = (output[..., :2] + grid) * stride      #  ((tx + xv) * stride, (ty + yv) * stride)
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  #  (exp(tw) * stride, exp(th) * stride)
        return output, grid
    

    def get_losses(
        self,
        x_shifts: torch.Tensor,
        y_shifts: torch.Tensor,
        expanded_strides: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        dtype,
    ):
        '''
        Args:
            x_shifts: (1, n_anchors)
            y_shifts: (1, n_anchors)
            expanded_strides: (1, n_anchors)
            labels: (B, n_labels, 5) [x, y, w, h, class_id]
            outputs: (B, n_anchors, 5 + n_cls)
        '''
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        num_anchors = outputs.shape[1]

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((num_anchors, 1))
                fg_mask = outputs.new_zeros(num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, :4]  # (num_gt, 4)
                gt_classes = labels[batch_idx, :num_gt, 4]            # (num_gt,)
                bboxes_preds_per_image = bbox_preds[batch_idx]        # (n_anchors, 4)

                # try:
                (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls
        
        output_dict = {
            'loss': loss,
            'loss_iou': reg_weight * loss_iou,
            'loss_obj': loss_obj,
            'loss_cls': loss_cls,
            'num_fg': num_fg / max(num_gts, 1),
        }

        return output_dict

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        obj_preds,
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        expanded_strides_per_image = expanded_strides[0]
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0)

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides_per_image.unsqueeze(0) * center_radius
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        anchor_filter = is_in_centers.sum(dim=0) > 0
        geometry_relation = is_in_centers[:, anchor_filter]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    
    def compute_flops(self):
        # fuse BatchNorm into Conv, so ignore BN flops
        flops = 0
        # per feature level
        for k in range(len(self.strides)):
            # 1 * 1 cls conv with bias 
            flops += self.width * self.in_channels[k] * 2
            flops += self.width * 5 # SiLU
            # 1 * 1 reg conv with bias
            flops += self.width * self.in_channels[k] * 2
            flops += self.width * 5 # SiLU
            # cls pred
            flops += self.width * self.num_classes * 2
            # reg pred
            flops += self.width * 4 * 2
            # obj pred
            flops += self.width * 1 * 2
        return flops