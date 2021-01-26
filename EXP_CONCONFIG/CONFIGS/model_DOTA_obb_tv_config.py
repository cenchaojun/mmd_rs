# os.system('CUDA_VISIBLE_DEVICES=2 python train_dota.py '
#           './DOTA_configs/DOTA_hbb/retinanet_r50_fpn_2x_dota.py '
#           '--gpus 1 '
#           '--no-validate '
#           '--work-dir ./results/retinanet_hbb_tv_test')
from EXP_CONCONFIG.CONFIGS.base import Project_root
DOTA_val_ann_path = Project_root + '/data/dota/val/labelTxt/{:s}.txt'
DOTA_val_set_path = Project_root + '/data/dota/val/val_sets.txt'

def gen_dict(name, config,
             result_root=Project_root + '/results',
             epoch=24,
             result_name='results.pkl',
             gpu_num=None,
             note=''):
    work_dir = result_root + '/' + name

    return dict(
        name=name,
        config=config,
        work_dir=work_dir,
        cp_file=work_dir+'/epoch_%d.pth' % epoch,
        result=work_dir+'/' + result_name,

        val_ann_pth=DOTA_val_ann_path,
        val_set_pth=DOTA_val_set_path,
        Task1_results=work_dir + '/Task1_results_nms',
        Task1_results_split=work_dir + '/Task1_results',
        Task2_results=work_dir + '/Task2_results_nms',
        Task2_results_split=work_dir + '/Task2_results',

        eval_resutls=work_dir + '/eval_results.json',
        dota_eval_results=work_dir + '/dota_eval_results.json',

        bbox_type='OBB',
        data_type='DOTA',
        gpu=gpu_num,
        note=note
    )

obb_root = Project_root + '/DOTA_configs/DOTA_obb'
obb_cfgs = [
    gen_dict('DOTA_retina_obb_tv',
             obb_root + '/' + 'retinanet_obb_r50_fpn_2x_dota.py',
             note='AerialDetection中的原始obb实现'),
    gen_dict('DOTA_faster_rcnn_RoITrans_tv',
             obb_root + '/' + 'faster_rcnn_RoITrans_r50_fpn_1x_dota.py',
             note='AerialDetection中的RoITrans实现'),
    gen_dict('DOTA_faster_obb_tv_1GPU_cv2_no_trick',
             obb_root + '/' + 'faster_rcnn_r50_fpn_1x_dota.py',
             gpu_num=1),
    gen_dict('DOTA_faster_obb_tv_2GPU_cv2_no_trick',
             obb_root + '/' + 'faster_rcnn_r50_fpn_1x_dota.py',
             gpu_num=2),
    gen_dict('DOTA_retina_cv2_mod_obb_tv',
             obb_root + '/' + 'retinanet_r50_fpn_2x_dota_cv2_mod.py'),
    gen_dict('DOTA_faster_rcnn_ad_obb_tv',
             obb_root + '/' + 'faster_rcnn_obb_r50_fpn_1x_dota.py'),
    gen_dict('DOTA_retina_ad_obb_tv',
             obb_root + '/' + 'retinanet_r50_fpn_2x_ad.py',
             note='重新包装的ad计算角度版本的retina obb'),
]
obb_cfgs = {cfg.pop('name'):cfg for cfg in obb_cfgs}

def show_dict(d, n):
    for k,v in d.items():
        print('    ' * n, end='')
        if isinstance(v, dict):
            print(k, ':')
            show_dict(v, n+1)
        else:
            print(k, ':', v)
if __name__ == '__main__':
    show_dict(obb_cfgs, 0)


