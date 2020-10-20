cd ..
CUDA_VISIBLE_DEVICES=8,9 python train_dota.py \
./DOTA_configs/DOTA_hbb/faster_rcnn_r50_fpn_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/faster_rcnn_hbb_tv

