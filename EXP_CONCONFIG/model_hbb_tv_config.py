# os.system('CUDA_VISIBLE_DEVICES=2 python train_dota.py '
#           './DOTA_configs/DOTA_hbb/retinanet_r50_fpn_2x_dota.py '
#           '--gpus 1 '
#           '--no-validate '
#           '--work-dir ./results/retinanet_hbb_tv_test')

root = '.'
def gen_dict(name, config,
             result_root='/home/huangziyue/data/mmdet_results',
             epoch=24,
             result_name='results.pkl'):
    work_dir = result_root + '/' + name

    return dict(
        name=name,
        config=config,
        work_dir=  work_dir,
        cp_file=   work_dir+'/epoch_%d.pth' % epoch,
        result=    work_dir+'/' + result_name,
        Task2_results = work_dir + '/Task2_results_nms',
        Task2_results_split = work_dir + '/Task2_results',
        dota_eval_results=work_dir + '/dota_eval_results.json',
        type='HBB'
    )

hbb_root = './DOTA_configs/DOTA_hbb'
obb_root = './DOTA_configs/DOTA_obb'
cfgs = [
    gen_dict('retinanet_hbb_tv_test',
             hbb_root + '/' + 'retinanet_r50_fpn_2x_dota.py'),
    gen_dict('retinanet_hbb_tv',
             hbb_root + '/' + 'retinanet_r50_fpn_2x_dota.py'),
    gen_dict('atss_hbb_tv',
             hbb_root + '/' + 'atss_r50_fpn_2x_dota.py'),
    gen_dict('cascade_rcnn_hbb_tv',
             hbb_root + '/' + 'cascade_rcnn_r50_fpn_dota.py'),
    gen_dict('faster_rcnn_hbb_tv',
             hbb_root + '/' + 'faster_rcnn_r50_fpn_2x_dota.py'),
    gen_dict('fcos_hbb_tv',
             hbb_root + '/' + 'fcos_r50_caffe_fpn_4x4_2x_dota.py'),
    gen_dict('fovea_hbb_tv',
             hbb_root + '/' + 'fovea_r50_fpn_4x4_2x_dota.py'),
    gen_dict('fsaf_hbb_tv',
             hbb_root + '/' + 'fsaf_r50_fpn_2x_dota.py'),
    gen_dict('gfl_hbb_tv',
             hbb_root + '/' + 'gfl_r50_fpn_2x_dota.py'),
    gen_dict('reppoints_hbb_tv',
             hbb_root + '/' + 'reppoints_moment_r50_fpn_2x_dota.py'),
    gen_dict('sabl_faster_rcnn_hbb_tv',
             hbb_root + '/' + 'sabl_faster_rcnn_r50_fpn_2x_dota.py'),
    gen_dict('sabl_retina_hbb_tv',
             hbb_root + '/' + 'sabl_retinanet_r50_fpn_2x_dota.py'),
    gen_dict('res2net_hbb_tv',
             hbb_root + '/' + 'faster_rcnn_r2_101_fpn_2x_dota.py'),
    gen_dict('ssd_300_hbb_tv',
             hbb_root + '/' + 'ssd300_dota.py'),
    gen_dict('ssd_512_hbb_tv',
             hbb_root + '/' + 'ssd512_dota.py'),
    gen_dict('yolov3_d53_320_hbb_tv',
             hbb_root + '/' + 'yolov3_d53_320_273e_coco.py.py'),
    gen_dict('yolov3_d53_mstrain-416_hbb_tvy',
             hbb_root + '/' + 'yolov3_d53_mstrain-416_273e_coco.py'),
    gen_dict('yolov3_d53_mstrain-608_hbb_tv',
             hbb_root + '/' + 'yolov3_d53_mstrain-608_273e_coco.py'),
    gen_dict('libra_faster_rcnn_hbb_tv',
             hbb_root + '/' + 'libra_faster_rcnn_r50_fpn_2x_dota.py'),
    gen_dict('libra_retina_hbb_tv',
             hbb_root + '/' + 'libra_retinanet_r50_fpn_2x_dota.py'),
    gen_dict('pafpn_hbb_tv',
             hbb_root + '/' + 'faster_rcnn_r50_pafpn_2x_dota.py'),
    gen_dict('paa_hbb_tv',
             hbb_root + '/' + 'paa_r50_fpn_2x_dota.py')
]
cfgs = {cfg.pop('name'):cfg for cfg in cfgs}

def show_dict(d, n):
    for k,v in d.items():
        print('    ' * n, end='')
        if isinstance(v, dict):
            print(k, ':')
            show_dict(v, n+1)
        else:
            print(k, ':', v)
if __name__ == '__main__':
    show_dict(cfgs, 0)