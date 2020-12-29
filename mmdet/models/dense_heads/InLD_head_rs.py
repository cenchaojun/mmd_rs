import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from ..builder import HEADS
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
import numpy as np
import cv2

@HEADS.register_module()
class InLD_head(nn.Module):
    def __init__(self,
                 in_channels,
                 num_diation,
                 num_classes,
                 mask_type='multi',
                 train_cfg=None,
                 test_cfg=None):
        super(InLD_head, self).__init__()

        self.in_channels = in_channels
        self.num_diation = num_diation
        self.num_classes = num_classes
        self.mask_type = mask_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the head."""
        self.diated_convs = nn.ModuleList()
        feat_c = self.in_channels
        # kernel, stride, padding, dilation
        self.diated_convs.append(nn.Conv2d(feat_c,
                                           feat_c, 3, 1, 2, 2))
        self.diated_convs.append(nn.Conv2d(feat_c,
                                            feat_c,3,1,2,2))
        self.diated_convs.append(nn.Conv2d(feat_c,
                                           feat_c,3,1,2,2))
        self.diated_convs.append(nn.Conv2d(feat_c,
                                           feat_c,3,1,2,2))
        self.diated_convs.append(nn.Conv2d(feat_c,
                                           feat_c,1,1,0))
        n_cls = self.num_classes
        self.conv_mask = nn.Conv2d(feat_c, n_cls, 3, 1, 1)
        self.InLD_convs = nn.Conv2d(feat_c, feat_c, 3, 1, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.diated_convs, std=0.001)
        normal_init(self.conv_mask, std=0.001)
        normal_init(self.InLD_convs, std=0.001)

    def forward_single(self, x):

        for conv in self.diated_convs:
            x = conv(x)
        diated_feats = x
        pred_masks = self.conv_mask(diated_feats)
        InLD_W = self.InLD_convs(diated_feats)
        InLD_feat = x * InLD_W
        return pred_masks, InLD_feat

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_train(self, feats,
                      gt_masks,
                      gt_labels,
                      img_metas=None):
        pred_mask_list, InLD_feats = self(feats)

        losses_mask = self.loss(pred_mask_list, gt_masks, gt_labels)

        return InLD_feats, losses_mask

    def loss_single(self, pred_mask, target_mask):
        """

        :param pred_mask:  B x n_cls x W x H
        :param target_mask: B x  W x H
        :return:
        """
        target_mask = pred_mask.new_tensor(target_mask)
        B, n_cls, W, H = pred_mask.shape
        preds = pred_mask.permute(2, 3, 0, 1).reshape(-1, n_cls)
        targets = target_mask.permute(1, 2, 0).reshape(-1, 1).flatten().long()
        loss_fun = nn.CrossEntropyLoss()
        loss_mask = loss_fun(preds, targets)
        return loss_mask

    def loss(self, pred_mask_list, gt_masks, gt_labels):
        feat_sizes = [pm.shape[-2:] for pm in pred_mask_list]
        target_mask_list = multi_apply(self.generate_target_mask,
                                       gt_masks,
                                       gt_labels,
                                       feat_sizes=feat_sizes)
        mask_losses = map(self.loss_single,
                          pred_mask_list,
                          target_mask_list)
        mask_losses = list(mask_losses)
        return mask_losses


    def generate_target_mask(self, gt_masks, gt_labels, feat_sizes=None):
        target_mask_list = []
        gt_areas = gt_masks.areas
        gt_masks = gt_masks.masks.copy()
        if len(gt_masks) == 0:
            for feat_size in feat_sizes:
                mask = np.zeros(tuple(feat_size))
                target_mask_list.append(mask)

        for feat_size in feat_sizes:
            n_bbox, W, H = gt_masks.shape

            inds = np.argsort(gt_areas)
            mask = np.zeros(tuple(feat_size), dtype=np.long)
            for id in inds:
                gt_mask = gt_masks[id]
                gt_label = int(gt_labels[id])
                r_H, r_W = tuple(feat_size)
                gt_mask = cv2.resize(gt_mask, tuple(feat_size))
                mask[gt_mask > 0] = gt_label
            target_mask_list.append(mask)


            gt_masks = gt_masks[inds].reshape(n_bbox, W, H)
            gt_masks = gt_masks.transpose(1, 2, 0).astype(np.float)

            r_W, r_H = tuple(feat_size)
            gt_masks = cv2.resize(gt_masks, (r_W, r_H)).reshape(r_W, r_H, n_bbox)
            gt_masks = gt_masks.transpose(2, 0, 1)
            mask = np.zeros(tuple(feat_size))
            for gt_m, label in zip(gt_masks, gt_labels):
                mask[gt_m > 0] = int(label.detach().cpu())
            target_mask_list.append(mask.astype(np.int32))
        return target_mask_list

