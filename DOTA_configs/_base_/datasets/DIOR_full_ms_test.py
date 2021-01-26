dataset_type = 'CocoDataset'
data_root = 'data/DIOR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1333, 1333), (1000, 1000),
                   (800, 800), (600, 600)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
cat_name_list = ['ship', 'overpass', 'tenniscourt', 'stadium', 'vehicle',
                 'airplane', 'storagetank', 'dam', 'golffield', 'trainstation',
                 'Expressway-Service-area', 'groundtrackfield', 'Expressway-toll-station',
                 'windmill', 'airport', 'harbor', 'baseballfield',
                 'basketballcourt', 'bridge', 'chimney']
num_classes = len(cat_name_list) # 20
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_annotations/train_val_coco_ann.json',
        img_prefix=data_root + 'JPEGImages-trainval',
        classes=cat_name_list,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_annotations/test_coco_ann.json',
        img_prefix=data_root + 'JPEGImages-test',
        classes=cat_name_list,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'coco_annotations/test_coco_ann.json',
        img_prefix=data_root + 'JPEGImages-test',
        classes=cat_name_list,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
max_bbox_per_img = 600
