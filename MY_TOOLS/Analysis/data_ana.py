from commonlibs.common_tools import *
import numpy as np
from commonlibs.drawing_tools.diagram import simple_plot_mul_compare

COUNT = True

if COUNT:
    data = jsonload('../../data/dota/train/train_coco.json')
    a = 0
    id2name = {i['id']: i['name'] for i in data['categories']}
    count = {name: 0 for name in id2name.values()}
    for ann in data['annotations']:
        if ann['category_id'] in id2name.keys():
            count[id2name[ann['category_id']]] += 1
        else:
            print(id2name)
            print(ann)

    jsonsave(count, './sheets/train_cat_nums.json')

train = jsonload('./sheets/train_cat_nums.json')
val = jsonload('./sheets/val_cat_nums.json')

a = 0

keys = list(train.keys())
t = np.array([train[k] for k in keys])
v = np.array([val[k] for k in keys])
t = t.reshape(1, -1)
v = v.reshape(1, -1)

total = t + v
data = np.concatenate((t, v, total), axis=0)
data = data / np.sum(data, axis=1).reshape(-1, 1)

simple_plot_mul_compare('./num_plot.png',
                        range(len(keys)),
                        data,
                        ['train', 'val', 'total'])
for i, k in enumerate(keys):
    print(i, k)





