import argparse
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '8'  # '0,1,2,3'
# os.chdir('../..')
import os.path as osp

from mmcv import Config, DictAction
def show_cfg(cfg, cfg_name):
    print(cfg_name, '->')

    print('#' * 30)
    print(cfg.pretty_text)
    print('#' * 30)


def main():
    root = 'test_data/test_configs/'
    show_cfg(Config.fromfile(root + 'base.py'), 'base')
    show_cfg(Config.fromfile(root + 'new.py'), 'new')
    show_cfg(Config.fromfile(root + 'delete_val.py'), 'delete_val')

if __name__ == '__main__':
    main()
