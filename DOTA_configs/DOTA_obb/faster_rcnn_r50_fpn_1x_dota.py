_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/DOTA_train_val_obb.py',
    '../_base_/schedules/schedule_2x_rs.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    roi_head=dict(
        type='RbboxRoIHeadRS',
        bbox_roi_extractor=dict(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2)),
        bbox_head=dict(
            type='Shared2FCRbboxHeadRS',
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHARbboxCoderRS',
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))))
# model training and testing settings
train_cfg = dict(
    rpn_proposal=dict(
        nms_post=2000,
        max_num=2000),
    rcnn=dict(
        sampler=dict(
            type='RandomSamplerRS',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        pos_weight=-1,
        debug=False))
# training and testing settings

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='py_cpu_nms_poly_fast', iou_threshold=0.1),
        max_per_img=2000)
)


optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[8, 11])
total_epochs = 12

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])