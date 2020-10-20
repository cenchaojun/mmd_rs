cd ..
CUDA_VISIBLE_DEVICES=4,5 python train_dota.py \
./DOTA_configs/DOTA_hbb/fovea_r50_fpn_4x4_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/fovea_hbb_tv

