
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
CUDA_VISIBLE_DEVICES=2,3 python train_dota.py \
./configs/DOTA_hbb/retinanet_r50_fpn_2x_dota_org_fix.py \
--gpus 2 \
--no-validate \
--work-dir ./results/retinanet_hbb_tv_org_fix

