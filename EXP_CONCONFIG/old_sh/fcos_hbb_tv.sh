
cd ..
CUDA_VISIBLE_DEVICES=2,3 python train_dota.py \
./DOTA_configs/DOTA_hbb/fcos_r50_caffe_fpn_4x4_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/fcos_hbb_tv

