# python tst_py_cmd.py --gpus 6,7
# python tst_py_cmd.py --

import os
import argparse


from EXP_CONCONFIG.model_hbb_tv_config import cfgs, show_dict
from EXP_CONCONFIG.model_obb_tv_config import obb_cfgs
from EXP_CONCONFIG.model_DIOR_full_config import DIOR_cfgs
cfgs.update(obb_cfgs)
cfgs.update(DIOR_cfgs)

# print(cfgs, obb_cfgs)

def parse_args():
    parser = argparse.ArgumentParser(
        description='parse exp cmd')
    parser.add_argument('model', help='model name')
    parser.add_argument('-d', help='devices id, 0~9')
    parser.add_argument('-c', help='control, 1->list models')
    parser.add_argument('-m', help='mode, train or test', default='train')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    if args.c and int(args.c) == 1:
        show_dict(cfgs, 0)
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
                  '--work-dir %s' %\
                  (args.d,
                   cfg['config'],
                   len(devs),
                   cfg['work_dir'])
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