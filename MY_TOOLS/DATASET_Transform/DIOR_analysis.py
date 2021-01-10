import os
import matplotlib as mpl
import cv2
mpl.use('Qt5Agg')
from xml.etree import ElementTree as et
import cv2
import json
from commonlibs.transform_tools.coco_annotation_template import COCOTmp
from commonlibs.common_tools import *


def analysis(data_folder, ann_folder, seg_file):
    with open(seg_file, 'r') as f:
        ids = f.readlines()
    ids = [id.strip('\n') for id in ids]
    file_list = os.listdir(data_folder)

    cat_names = set()

    # 提取xml文件，获得bbox、标签
    for img_id, img_file in enumerate(file_list):
        (img_name, ext) = os.path.splitext(img_file)
        xml_file = ann_folder + '/' + img_name + '.xml'
        # 开始提取
        tree = et.parse(xml_file)
        root = tree.getroot()
        if img_name not in ids:
            continue
        print(img_name)

        objects = root.findall('object')
        # 获得bbox和label
        for obj in objects:
            bbox = obj.find('bndbox')
            try:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                name = obj.find('name').text
                cat_names.add(name)


            except ValueError:
                print(('Wrong Value in file %s' % (str(img_id))))
    print('Total %d cat, names: \n' % len(cat_names))
    print(list(cat_names))

def print_img_size(data_folder):
    imgs = os.listdir(data_folder)
    shapes = set()
    count = 0
    for i in imgs:
        img_path = data_folder + '/' + i
        img = cv2.imread(img_path)
        shapes.add(img.shape)
        print(count, img.shape)
        count += 1
    print(shapes)



data_root = 'D:/DataBackup/DIOR'

DIOR_root = 'D:/DataBackup/DIOR'
data_folder = DIOR_root + '/JPEGImages-trainval'
ann_folder = DIOR_root + '/Annotations'
seg_folder = DIOR_root + '/ImageSets/Main'
seg_file = seg_folder + '/' + 'train.txt'
# analysis(data_folder, ann_folder, seg_file)
print_img_size(data_folder)