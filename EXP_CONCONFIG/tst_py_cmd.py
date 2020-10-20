# python tst_py_cmd.py --gpus 6,7
# python tst_py_cmd.py --

import os
import argparse
from .model_hbb_tv_config import cfgs, show_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='parse exp cmd')
    parser.add_argument('model', help='model name')
    parser.add_argument('-d', help='devices id, 0~9')
    parser.add_argument('-c', help='control, 1->list models')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    if int(args.c) == 1:
        show_dict(cfgs, 0)
    else:
        model = args.model
        if model not in cfgs.keys():
            assert Exception("%s not in cfg keys: %s" %(model, str(list(cfgs.keys()))
            ))

        cfg = cfgs[model]
        os.chdir('..')
        devs = [int(i) for i in args.d.split(',')]
        cmd = 'CUDA_VISIBLE_DEVICES=%s python train_dota.py ' \
              '%s ' \
              '--gpus %s --no-validate ' \
              '--work-dir %s' %\
              (args.d,
               cfg['config'],
               len(devs),
               cfg['work_dir'])
        print(cmd)
        os.system(cmd)