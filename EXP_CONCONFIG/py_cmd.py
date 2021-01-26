# python tst_py_cmd.py --gpus 6,7
# python tst_py_cmd.py --

import os
import argparse


from EXP_CONCONFIG.CONFIGS.model_DOTA_hbb_tv_config import cfgs, show_dict
from EXP_CONCONFIG.CONFIGS.model_DOTA_obb_tv_config import obb_cfgs
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_config import DIOR_cfgs
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_ms_test_config import DIOR_ms_test_cfgs
from EXP_CONCONFIG.CONFIGS.model_NWPU_VHR_10_config import NV10_cfgs
from EXP_CONCONFIG.CONFIGS.model_DIOR_full_voc_test_config import DIOR_voc_cfgs
cfgs.update(obb_cfgs)
cfgs.update(DIOR_cfgs)
cfgs.update(NV10_cfgs)
cfgs.update(DIOR_voc_cfgs)


# print(cfgs, obb_cfgs)

def parse_args():
    parser = argparse.ArgumentParser(
        description='parse exp cmd')
    parser.add_argument('model', help='model name')
    parser.add_argument('-d', help='devices id, 0~9')
    parser.add_argument('-c', help='control, list->list models, state->model的状态')
    parser.add_argument('-resume', help='latest -> latest or int -> epoch')

    parser.add_argument('-load_results',
                        action='store_true',
                        help=' does not inference ,just evaluate, default=True')


    parser.add_argument('-m', help='mode, train or test', default='train')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.c:
        # 显示模型列表
        if args.c == 'list':
            show_dict(cfgs, 0)
        # 显示模型状态
        if args.c == 'state':
            cfg_states = {}
            for model_name, cfg in cfgs.items():
                cfg_state = dict(
                    exist=False,
                    train_state='',
                    test_state=[]
                )
                work_dir = cfg['work_dir']
                # 模型存在
                if os.path.exists(work_dir):
                    cfg_state['exist'] = True
                    # 训练状态
                    work_files = os.listdir(work_dir)
                    if os.path.exists(cfg['cp_file']):
                        cfg_state['train_state'] = 'Done'
                    else:
                        epoch_files = [f for f in work_files if 'epoch' in f]
                        if len(epoch_files) > 0:
                            final_epoch = sorted(epoch_files)[-1]
                            cfg_state['train_state'] = final_epoch
                        else:
                            cfg_state['train_state'] = 'Not Training Yet'
                    # 测试状态
                    if os.path.exists(cfg['result']):
                        cfg_state['test_state'].append('Inference Done')
                    if 'dota_eval_results' in cfg.keys() \
                            and os.path.exists(cfg['dota_eval_results']):
                        cfg_state['test_state'].append('DOTA Evaluate Done')
                    if 'eval_results' in cfg.keys() \
                            and os.path.exists(cfg['eval_results']):
                        cfg_state['test_state'].append('Evaluate Done')
                    cfg_states[model_name] = cfg_state
                # 模型不存在
                else:
                    cfg_state['train_state'] = 'Not exist'
                    cfg_states[model_name] = cfg_state
                    continue
            print('=' * 105)
            print('%-40s\t|%-25s\t|%-20s' %
                  ('NAME', 'TRAIN_STATE', 'TEST_STATE'))

            for model_name, cfg_state in cfg_states.items():
                print('%-40s\t|%-25s\t|%-20s' %
                      (model_name, str(cfg_state['train_state']), str(cfg_state['test_state'])))
            print('=' * 105)




    else:
        model = args.model
        if model not in cfgs.keys():
            raise Exception("%s not in cfg keys: %s" %(model, str(list(cfgs.keys()))
            ))

        cfg = cfgs[model]
        os.chdir('..')
        devs = [int(i) for i in args.d.split(',')]
        if len(devs) == 0:
            raise Exception('no deveices ')
        if args.m == 'train':
            cmd = 'CUDA_VISIBLE_DEVICES=%s python train_dota.py ' \
                  '%s ' \
                  '--gpus %s --no-validate ' \
                  '--work-dir %s ' %\
                  (args.d,
                   cfg['config'],
                   len(devs),
                   cfg['work_dir'])
            if args.resume is not None:
                if args.resume == 'latest':
                    cmd += '--resume-from %s' % \
                           (cfg['work_dir'] + '/latest.pth')
                else:
                    raise Exception
            if 'gpu_num' in cfg.keys() and cfg['gpu_num']:
                assert len(devs) == cfg['gpu_num']

        elif args.m == 'test':
            assert len(devs) == 1
            cmd = 'CUDA_VISIBLE_DEVICES=%s python test_dota.py ' \
                  '%s %s --out %s --eval bbox' % \
                  (args.d,
                   cfg['config'],
                   cfg['cp_file'],
                   cfg['result'])
            cmd += ' --eval-options eval_results_path=\'%s\''%cfg['eval_results']
        elif args.m == 'voc_test':
            assert len(devs) == 1
            cmd = 'CUDA_VISIBLE_DEVICES=%s python test_dota.py ' \
                  '%s %s --out %s --eval mAP' % \
                  (args.d,
                   cfg['config'],
                   cfg['cp_file'],
                   cfg['result'])
            cmd += ' --eval-options eval_results_path=\'%s\''%cfg['eval_results']
        elif args.m == 'dota_test':
            assert len(devs) == 1
            cmd = 'CUDA_VISIBLE_DEVICES=%s python test_dota.py ' \
                  '%s %s --out %s' % \
                  (args.d,
                   cfg['config'],
                   cfg['cp_file'],
                   cfg['result'])
            cmd += ' --eval-options eval_results_path=\'%s\''%cfg['dota_eval_results']
        else:
            cmd = ''
        if args.load_results:
            cmd += ' load_results=True'
        print('##########################################################################')
        print('CMD: ', cmd)
        print('##########################################################################')
        os.system(cmd)

        #### dota 的 evaluate #######

        if args.m == 'dota_test':
            os.chdir('./MY_TOOLS')

            cmd = 'python parse_and_merge_results.py ' \
                  '-config %s -type %s -result %s -out %s' %\
            (cfg['config'],
             cfg['bbox_type'],
             cfg['result'],
             cfg['work_dir'])
            os.system(cmd)

            if cfg['bbox_type'] == 'OBB':
                cmd = 'python dota_evaluation_task1.py ' \
                      '-detpath %s -annopath %s -imagesetfile %s -out %s' %\
                (cfg['result'],
                 cfg['val_ann_pth'],
                 cfg['val_set_pth'],
                 cfg['dota_eval_results'])