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
    with open(seg_file, 'r') as f:
        ids = f.readlines()
    ids = [id.strip('\n') for id in ids]
    file_list = sorted(list(os.listdir(data_folder)))

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
        img_file = file_list[img_id]
        (img_name, extension) = os.path.splitext(img_file)
        if img_name not in ids:
            continue
        # print(img_name)
        xml_file = ann_folder + '/' + img_name + '.xml'
        # 开始提取
        tree = et.parse(xml_file)
        root = tree.getroot()
        # 获得文件路径
        file_name = root.find('filename')
        file_name = file_name.text
        img = cv2.imread(data_folder + '/' + file_name)
        file_path = data_folder + '/' + file_name

        n_img += 1
        # print('%d / %d' % (n_img, len(ids)))

        # file_name只是file_name而已
        coco_res.add_image(file_name, img.shape[0], img.shape[1], img.shape[2], img_id)
        objects = root.findall('object')
        # 获得bbox和label
        for obj in objects:
            bbox = obj.find('bndbox')
            try:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                # difficult = int(obj.find('difficult').text)
                # if difficult:
                #     print('Diff')
                h = xmax - xmin
                w = ymax - ymin
                area = h * w
                name = obj.find('name').text
                others = dict()# dict(difficult=difficult)
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

# 训练验证集

# DIOR_root = 'D:/DataBackup/DIOR'
# data_folder = DIOR_root + '/JPEGImages-trainval'
# ann_folder = DIOR_root + '/Annotations'
# seg_folder = DIOR_root + '/ImageSets/Main'
# seg_file = seg_folder + '/' + 'train.txt'
# ann_file = coco_ann_folder + '/train_coco_ann.json'
#
# make_voc_coco(data_folder, ann_folder, seg_file, ann_file,
#                          src_cocotmp=None, savejson=True)
#
# DIOR_root = 'D:/DataBackup/DIOR'
# data_folder = DIOR_root + '/JPEGImages-trainval'
# ann_folder = DIOR_root + '/Annotations'
# seg_folder = DIOR_root + '/ImageSets/Main'
# seg_file = seg_folder + '/' + 'val.txt'
# ann_file = coco_ann_folder + '/val_coco_ann.json'
#
# make_voc_coco(data_folder, ann_folder, seg_file, ann_file,
#                          src_cocotmp=None, savejson=True)


DIOR_root = '../../data/DIOR'
data_folder = DIOR_root + '/JPEGImages-trainval'
ann_folder = DIOR_root + '/Annotations'
seg_folder = DIOR_root + '/ImageSets/Main'
seg_file = seg_folder + '/' + 'train.txt'
ann_file = coco_ann_folder + '/train_coco_ann.json'

train_coco_ann = make_voc_coco(data_folder, ann_folder, seg_file, ann_file,
                         src_cocotmp=None, savejson=False)

data_folder = DIOR_root + '/JPEGImages-trainval'
ann_folder = DIOR_root + '/Annotations'
seg_folder = DIOR_root + '/ImageSets/Main'
seg_file = seg_folder + '/' + 'val.txt'
ann_file = coco_ann_folder + '/train_val_coco_ann.json'

make_voc_coco(data_folder, ann_folder, seg_file, ann_file,
                         src_cocotmp=train_coco_ann, savejson=True)


# 测试集
DIOR_root = '../../data/DIOR'
data_folder = DIOR_root + '/JPEGImages-test'
ann_folder = DIOR_root + '/Annotations'
seg_folder = DIOR_root + '/ImageSets/Main'
seg_file = seg_folder + '/' + 'test.txt'
ann_file = coco_ann_folder + '/test_coco_ann.json'
make_voc_coco(data_folder, ann_folder, seg_file, ann_file, savejson=True)






