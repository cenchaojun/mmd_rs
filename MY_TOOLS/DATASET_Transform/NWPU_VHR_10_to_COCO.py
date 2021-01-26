import os
import numpy as np
import matplotlib as mpl
import cv2
mpl.use('Qt5Agg')
import json
from MY_TOOLS.DATASET_Transform.coco_annotation_template import COCOTmp
from commonlibs.common_tools import *
from tqdm import tqdm


def name2imgid(file_name):
    return int(os.path.splitext(file_name)[0])

def make_coco(data_folder,
              ann_folder, img_folder,
              seg_file, out_ann,
              src_cocotmp=None, savejson=True):
    # 从readme.txt中得到的，原数据集中可以得到id就是从0开始，因此可以直接获得id
    cat_name_list = ['airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
                     'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle']


    coco_res = COCOTmp(src_cocotmp)
    coco_res.auto_generate_categories(cat_name_list)
    coco_res.fill_info('This very-high-resolution (VHR) '
                       'remote sensing image dataset; NWPU_VHR_10 dataset')

    # 划分文件
    with open(seg_file) as f:
        seg_imgs = [l.strip('\n')for l in f.readlines()]

    # img_file的数字作为img的id，各个annotation统一
    img_files = sorted(list(os.listdir(img_folder)))
    for img_id in tqdm(range(len(img_files))):
        img_file = img_files[img_id]

        if img_file not in seg_imgs:
            continue

        img_id = name2imgid(img_file)

        img = cv2.imread(img_folder + '/' + img_file)

        # file_name只是file_name而已
        coco_res.add_image(img_file, img.shape[0], img.shape[1], img.shape[2],
                           img_id)
        ann_file = ann_folder + '/' + os.path.splitext(img_file)[0] + '.txt'
        with open(ann_file, 'r') as f:
            anns = list(f.readlines())
            anns = [ann.strip('\n') for ann in anns]
            anns = [ann for ann in anns if len(ann) > 0]

            # print(anns)
            anns = [eval(ann) for ann in anns]
            for ann in anns:
                (x1, y1) = ann[0]
                (x2, y2) = ann[1]
                cat_id = ann[2]
                h = y2 - y1
                w = x2 - x1
                area = h * w
                others = dict()
                coco_res.add_ann(area, 0, img_id,
                                 cat_id, [x1, y1, w, h], others=others)
    if savejson:
        print('Save Json')
        coco_res.save_ann(out_ann)
    return coco_res


# def make_voc_coco(data_folder,
#                   ann_folder, img_folder,
#                   seg_file, out_ann,
#                   src_cocotmp=None, savejson=True):
data_root = '/home/huangziyue/data/NWPU_VHR_10'
ann_folder = data_root + '/ground truth'
img_folder = data_root + '/images'
# 划分数据集 4: 3 : 3
img_files = sorted(list(os.listdir(img_folder)))
n_imgs = len(img_files)
ids = np.arange(n_imgs)
np.random.shuffle(ids)
train_ids = np.sort(ids[0: int(n_imgs*4/10)])
val_ids = np.sort(ids[int(n_imgs*4/10): int(n_imgs*7/10)])
train_val_ids = np.sort(ids[0: int(n_imgs*7/10)])
test_ids = np.sort(ids[int(n_imgs*7/10): n_imgs])

def save_seg(data_root, seg_file_name, img_files, ids):
    with open(data_root + '/%s' % seg_file_name + '.txt', 'wt+') as f:
        for id in ids:
            f.write('%s\n' % img_files[id])

save_seg(data_root, 'train', img_files, train_ids)
save_seg(data_root, 'val', img_files, val_ids)
save_seg(data_root, 'train_val', img_files, train_val_ids)
save_seg(data_root, 'test', img_files, test_ids)

seg_files = [
    data_root + '/train.txt',
    data_root + '/val.txt',
    data_root + '/train_val.txt',
    data_root + '/test.txt',
]

for seg_file in seg_files:
    seg_type = os.path.splitext(os.path.split(seg_file)[1])[0]
    make_coco(data_root, ann_folder, img_folder, seg_file,
              out_ann=data_root + '/'+ seg_type + '_coco_ann.json')




