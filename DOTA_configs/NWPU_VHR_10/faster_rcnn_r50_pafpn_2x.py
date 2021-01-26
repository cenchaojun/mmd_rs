from DOTA_configs.NWPU_VHR_10.a_base_config import *
_base_ = './faster_rcnn_r50_fpn_2x.py'

model = dict(
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5))
