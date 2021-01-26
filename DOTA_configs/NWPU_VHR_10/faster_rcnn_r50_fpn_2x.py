from DOTA_configs.NWPU_VHR_10.a_base_config import *

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    dataset_config,
    '../_base_/schedules/schedule_2x_rs.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes)))
# training and testing settings

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=max_bbox_per_img))