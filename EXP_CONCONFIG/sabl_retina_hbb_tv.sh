cd ..
CUDA_VISIBLE_DEVICES=5,6 python train_dota.py \
./DOTA_configs/DOTA_hbb/sabl_retinanet_r50_fpn_2x_dota.py \
--gpus 2 \
--no-validate \
--work-dir ./results/sabl_retina_hbb_tv

