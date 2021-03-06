_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/DIOR_full.py',
    '../_base_/schedules/schedule_2x_rs.py',
    '../_base_/default_runtime.py'
]
from DOTA_configs._base_.datasets.DIOR_full import num_classes, max_bbox_per_img

# model settings
model = dict(
    bbox_head=dict(
        _delete_=True,
        type='SABLRetinaHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[4],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='BucketingBBoxCoder', num_buckets=14, scale_factor=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.5),
        loss_bbox_reg=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.5)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='ApproxMaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0.0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)

test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=max_bbox_per_img)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
