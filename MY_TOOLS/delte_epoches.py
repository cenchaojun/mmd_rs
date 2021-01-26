import os
import os.path as osp
import re

def del_ep(root):
    p = 'epoch_\d+.pth'
    if not osp.isdir(root):
        return
    for f in os.listdir(root):
        # dir
        fp = osp.join(root, f)
        if osp.isdir(fp):
            del_ep(fp)
            continue
        # match
        m = re.match(p, f)
        if not m:
            continue
        epoch_num = int(re.findall(r'\d+', f)[0])
        if epoch_num not in [12, 24, 150]:
            print('Delete %s' % fp)
            os.remove(fp)


if __name__ == '__main__':
    del_ep('../results')


