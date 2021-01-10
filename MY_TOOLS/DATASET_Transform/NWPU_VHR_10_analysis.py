import os
import matplotlib as mpl
import cv2
mpl.use('Qt5Agg')
from xml.etree import ElementTree as et
import json
from MY_TOOLS.DATASET_Transform.coco_annotation_template import COCOTmp
from commonlibs.common_tools import *
from tqdm import tqdm


def make_voc_coco(data_folder, ann_folder, seg_file, out_ann,
                  src_cocotmp=None, savejson=True):
    # 从DIOR_analysis中得到的
    cat_name_list = ['ship', 'overpass', 'tenniscourt', 'stadium', 'vehicle',
                     'airplane', 'storagetank', 'dam', 'golffield', 'trainstation',
                     'Expressway-Service-area', 'groundtrackfield', 'Expressway-toll-station',
                     'windmill', 'airport', 'harbor', 'baseballfield',
                     'basketballcourt', 'bridge', 'chimney']

    coco_res = COCOTmp(src_cocotmp)
    id2name, name2id = coco_res.auto_generate_categories(cat_name_list)
    coco_res.fill_info('DIOR coco version annotations')
    n_img = 0
    # 提取xml文件，获得bbox、标签
    for img_id in tqdm(range(len(file_list))):


        # file_name只是file_name而已
        coco_res.add_image(file_name, img.shape[0], img.shape[1], img.shape[2], img_id)
        objects = root.findall('object')
        # 获得bbox和label
        for obj in objects:
            bbox = obj.find('bndbox')
            try:
                coco_res.add_ann(area, 0, img_id,
                                 name2id[name], [xmin-1, ymin-1, h, w], others=others)
                if name not in name2id.keys():
                    raise Exception('Wrong Name: %s in file %s' % (name, file_path))
            except ValueError:
                print(('Wrong Value in file %s' % (file_path)))
    if savejson:
        print('Save Json')
        coco_res.save_ann(out_ann)
    return coco_res




data_root = '../../data/DIOR'
coco_ann_folder = data_root + '/coco_annotations'
mkdir(coco_ann_folder)




