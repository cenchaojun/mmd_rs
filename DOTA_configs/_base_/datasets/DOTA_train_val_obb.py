# oriented bounding boxes

dataset_type = 'DOTADataset'
# TODO: build dota path
data_root = 'data/dota1_train_val_1024/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train1024/DOTA_train1024.json',
        img_prefix=data_root + 'train1024/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valtest1024/DOTA_valtest1024.json',
        img_prefix=data_root + 'valtest1024/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'valtest1024/DOTA_valtest1024.json',
        img_prefix=data_root + 'valtest1024/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))


















