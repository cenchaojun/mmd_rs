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
        Task1_results = work_dir + '/Task1_results_nms',
        Task1_results_split = work_dir + '/Task1_results',
        dota_eval_results=work_dir + '/dota_eval_results.json',
        type='OBB'
    )

obb_root = './DOTA_configs/DOTA_obb'
obb_cfgs = [
    gen_dict('DOTA_retina_obb_tv',
             obb_root + '/' + 'retinanet_obb_r50_fpn_2x_dota.py'),
    gen_dict('DOTA_faster_rcnn_RoITrans_tv',
             obb_root + '/' + 'faster_rcnn_RoITrans_r50_fpn_1x_dota.py')
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