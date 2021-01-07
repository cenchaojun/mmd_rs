# python tst_py_cmd.py --gpus 6,7
# python tst_py_cmd.py --

import os
import argparse


from EXP_CONCONFIG.model_DOTA_hbb_tv_config import cfgs, show_dict
from EXP_CONCONFIG.model_DOTA_obb_tv_config import obb_cfgs
from EXP_CONCONFIG.model_DIOR_full_config import DIOR_cfgs
cfgs.update(obb_cfgs)
cfgs.update(DIOR_cfgs)

# print(cfgs, obb_cfgs)

def parse_args():
    parser = argparse.ArgumentParser(
        description='parse exp cmd')
    parser.add_argument('model', help='model name')
    parser.add_argument('-d', help='devices id, 0~9')
    parser.add_argument('-c', help='control, list->list models, state->model的状态')
    parser.add_argument('-resume', help='latest -> latest or int -> epoch')

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
                        if len(epoch_files) > 0 :
                            final_epoch = sorted(epoch_files)[-1]
                            cfg_state['train_state'] = final_epoch
                        else:
                            cfg_state['train_state'] = 'Not Training Yet'
                    # 测试状态
                    if os.path.exists(cfg['result']):
                        cfg_state['test_state'].append('Inference Done')
                    if os.path.exists(cfg['dota_eval_results']):
                        cfg_state['test_state'].append('Evaluate Done')
                    cfg_states[model_name] = cfg_state
                # 模型不存在
                else:
                    cfg_states[model_name] = cfg_state
                    continue

            for model_name, cfg_state in cfg_states.items():
                if not cfg_state['exist']:
                    print('%s not exist' % model_name)
                else:
                    print('%s        \t|\t train_state: %s \t|\t test_state: %s' %
                          (model_name, str(cfg_state['train_state']), str(cfg_state['test_state'])))




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

        elif args.m == 'test':
            assert len(devs) == 1
            cmd = 'CUDA_VISIBLE_DEVICES=%s python test_dota.py ' \
                  '%s %s --out %s --eval bbox' % \
                  (args.d,
                   cfg['config'],
                   cfg['cp_file'],
                   cfg['result'])
        else:
            cmd = ''
        print('##########################################################################')
        print('CMD: ', cmd)
        print('##########################################################################')
        os.system(cmd)