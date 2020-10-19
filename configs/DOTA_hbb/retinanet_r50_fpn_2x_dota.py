_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/DOTA_train_val_hbb_rs.py',
    '../_base_/schedules/schedule_2x_rs.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    bbox_head=dict(
        num_classes=15))
# training and testing settings
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.5),
    max_per_img=2000)