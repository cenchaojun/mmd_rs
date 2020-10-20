cd ..
CUDA_VISIBLE_DEVICES=2,3 python train_dota.py \
./DOTA_configs/DOTA_hbb/atss_r50_fpn_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/atss_hbb_tv

