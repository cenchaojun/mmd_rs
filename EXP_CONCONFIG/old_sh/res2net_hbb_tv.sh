cd ..
CUDA_VISIBLE_DEVICES=6,7 python train_dota.py \
./DOTA_configs/DOTA_hbb/faster_rcnn_r2_101_fpn_2x_coco.py \
--gpus 2 \
--no-validate \
--work-dir ./results/res2net_hbb_tv

