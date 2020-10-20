_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/DOTA_train_val_hbb.py',
    '../_base_/schedules/schedule_2x_rs.py', '../_base_/default_runtime.py'
]


# model settings
model = dict(
    type='FSAF',
    bbox_head=dict(
        type='FSAFHead',
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        # Only anchor-free branch is implemented. The anchor generator only
        #  generates 1 anchor at each feature point, as a substitute of the
        #  grid of features.
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=1,
            scales_per_octave=1,
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(_delete_=True, type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
            reduction='none'),
        loss_bbox=dict(
            _delete_=True,
            type='IoULoss',
            eps=1e-6,
            loss_weight=1.0,
            reduction='none'),
    ))

# training and testing settings
train_cfg = dict(
    assigner=dict(
        _delete_=True,
        type='CenterRegionAssigner',
        pos_scale=0.2,
        neg_scale=0.2,
        min_pos_iof=0.01),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)

test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=2000)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[16, 22])
total_epochs = 24

