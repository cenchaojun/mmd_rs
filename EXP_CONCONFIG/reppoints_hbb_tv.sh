cd ..
CUDA_VISIBLE_DEVICES=0,1 python train_dota.py \
./DOTA_configs/DOTA_hbb/reppoints_moment_r50_fpn_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/reppoints_hbb_tv

