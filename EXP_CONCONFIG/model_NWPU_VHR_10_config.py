root = '.'
def gen_dict(name, config,
             result_root='/home/huangziyue/data/mmdet_results',
             epoch=24,
             result_name='results.pkl'):
    ## name修饰
    name = 'NV10_' + name
    work_dir = result_root + '/' + name

    return dict(
        name=name,
        config=config,
        work_dir=  work_dir,
        cp_file=   work_dir+'/epoch_%d.pth' % epoch,
        result=    work_dir+'/' + result_name,
        Task2_results = None,
        Task2_results_split = None,
        dota_eval_results= work_dir + '/eval_results.txt',
        type='HBB'
    )

NV10_root = './DOTA_configs/NWPU_VHR_10'
NV10_cfgs = [
    gen_dict('retinanet',
             NV10_root + '/' + 'retinanet_r50_fpn_2x.py'),
    gen_dict('atss',
             NV10_root + '/' + 'atss_r50_fpn_2x.py'),
    gen_dict('cascade_rcnn',
             NV10_root + '/' + 'cascade_rcnn_r50_fpn.py'),
    gen_dict('faster_rcnn',
             NV10_root + '/' + 'faster_rcnn_r50_fpn_2x.py'),
    gen_dict('fcos',
             NV10_root + '/' + 'fcos_r50_caffe_fpn_4x4_2x.py'),
    gen_dict('fovea',
             NV10_root + '/' + 'fovea_r50_fpn_4x4_2x.py'),
    gen_dict('fsaf',
             NV10_root + '/' + 'fsaf_r50_fpn_2x.py'),
    gen_dict('gfl',
             NV10_root + '/' + 'gfl_r50_fpn_2x.py'),
    gen_dict('reppoints',
             NV10_root + '/' + 'reppoints_moment_r50_fpn_2x.py'),
    gen_dict('sabl_faster_rcnn',
             NV10_root + '/' + 'sabl_faster_rcnn_r50_fpn_2x.py'),
    gen_dict('sabl_retina',
             NV10_root + '/' + 'sabl_retinanet_r50_fpn_2x.py'),
    gen_dict('res2net',
             NV10_root + '/' + 'faster_rcnn_r2_101_fpn_2x.py'),
    gen_dict('ssd_300',
             NV10_root + '/' + 'ssd300.py'),
    gen_dict('ssd_512',
             NV10_root + '/' + 'ssd512.py'),
    gen_dict('yolov3_d53_320',
             NV10_root + '/' + 'yolov3_d53_320_273e_coco.py.py'),
    gen_dict('yolov3_d53_mstrain-416',
             NV10_root + '/' + 'yolov3_d53_mstrain-416_273e_coco.py'),
    gen_dict('yolov3_d53_mstrain-608',
             NV10_root + '/' + 'yolov3_d53_mstrain-608_273e_coco.py'),
    gen_dict('libra_faster_rcnn_',
             NV10_root + '/' + 'libra_faster_rcnn_r50_fpn_2x.py'),
    gen_dict('libra_retina',
             NV10_root + '/' + 'libra_retinanet_r50_fpn_2x.py'),
    gen_dict('pafpn',
             NV10_root + '/' + 'faster_rcnn_r50_pafpn_2x.py'),
    gen_dict('paa',
             NV10_root + '/' + 'paa_r50_fpn_2x.py')
]
NV10_cfgs = {cfg.pop('name'): cfg
             for cfg in NV10_cfgs}