
# args = ['./configs/DOTA_hbb/retinanet_r50_fpn_2x_dota.py',
#         '--gpus', '2',
#         '--no-validate',
#         '--work-dir', './results/retinanet_hbb_tv'
#         ]

# s="retina_hbb_tv"
# if ! screen -list | grep -q $s; then
#    # run bash script
#    echo "create screen ${s}"
#    screen -S ${s}
#else
 #   echo "screen ${s} exist"
  #  screen -S ${s} -X quit
   # screen -S ${s}
#fi
cd ..
CUDA_VISIBLE_DEVICES=8 python train_dota.py \
./configs/DOTA_hbb/fcos_r50_caffe_fpn_4x4_2x_dota.py \
--gpus 1 \
--no-validate \
--work-dir ./results/fcos_hbb_tv_test

