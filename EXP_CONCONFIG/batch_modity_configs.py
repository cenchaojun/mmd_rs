import os
cfg_root = '../DOTA_configs/DIOR_voc_test'
src = 'from DOTA_configs.DIOR_ms_test.a_base_config import *'.strip()
tgt = 'from DOTA_configs.DIOR_voc_test.a_base_config import *'
count = 0
for file in os.listdir(cfg_root):
    new_lines = []
    fp = cfg_root + '/' + file
    with open(fp, 'r') as f:
        for l in f.readlines():
            if l.strip() == src:
                print('MODIFY %d | %s' % (count, fp))
                count += 1
                new_lines.append(tgt.strip() + '\n')
            else:
                new_lines.append(l)
    with open(fp, 'wt+') as f:
        f.writelines(new_lines)

