
cd ..
CUDA_VISIBLE_DEVICES=4,5 python train_dota.py \
./configs/DOTA_hbb/retinanet_r50_fpn_2x_dota_ad_modified.py \
--gpus 2 \
--no-validate \
--work-dir ./results/retinanet_hbb_tv_mod_ad

