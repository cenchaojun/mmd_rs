# python train_dota.py --gpus 6,7 retina_obb
# python train_dota.py --

import os
os.chdir('..')
os.system('CUDA_VISIBLE_DEVICES=2 python train_dota.py '
          './DOTA_configs/DOTA_hbb/retinanet_r50_fpn_2x_dota.py '
          '--gpus 1 '
          '--no-validate '
          '--work-dir ./results/retinanet_hbb_tv_test')