
cd ..
CUDA_VISIBLE_DEVICES=0,1 python train_dota.py \
./configs/DOTA_hbb/gfl_r50_fpn_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/gfl_hbb_tv

