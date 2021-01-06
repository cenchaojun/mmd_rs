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
        num_classes=num_classes))
# training and testing settings
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=max_bbox_per_img)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    step=[16, 22])
total_epochs = 24