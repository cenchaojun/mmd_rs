cd ..
CUDA_VISIBLE_DEVICES=2,3 python train_dota.py \
./DOTA_configs/DOTA_hbb/cascade_rcnn_r50_fpn_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/cascade_rcnn_hbb_tv

