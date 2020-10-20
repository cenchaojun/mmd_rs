import os
import os.path as osp
import re

def del_ep(root):
    p = 'epoch_\d.pth'
    if not osp.isdir(root):
        return
    for f in os.listdir(root):
        f_p = osp.join(root, f)
        m = re.match(p, f_p)
        if not m:
            continue
        epoch_num = int(re.findall(r'\d+', f)[0])
        if epoch_num != 24:
            print('Delete %s' % f_p)

if __name__ == '__main__':
    del_ep('../data')


