cd ..
CUDA_VISIBLE_DEVICES=4,5 python train_dota.py \
./configs/DOTA_hbb/faster_rcnn_r50_fpn_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/faster_rcnn_hbb_tv

