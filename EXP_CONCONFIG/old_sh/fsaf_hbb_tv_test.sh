cd ..
CUDA_VISIBLE_DEVICES=6,7 python train_dota.py \
./DOTA_configs/DOTA_hbb/fsaf_r50_fpn_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/fsaf_hbb_tv

