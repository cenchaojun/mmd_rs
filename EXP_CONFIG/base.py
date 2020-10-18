

exps = dict(
    retinanet_obb=dict(
        config='./configs/DOTA_hbb/retinanet_r50_fpn_2x_dota.py',
        gpus='4',
        work_dir='./results/retinanet_hbb_tv',
        checkpoint='./results/retinanet_hbb_tv/epoch_24.pth',
        result = './results/retinanet_hbb_tv',
    )
)
