import matplotlib as mpl
mpl.use('Qt5Agg')
import cv2
import json
import os
from commonlibs.transform_tools.data_transform import *

data_root = 'D:/DataBackup/DIOR'
img_root = data_root + '/JPEGImages-trainval'
ann_file_path = data_root + '/coco_annotations/val_coco_ann.json'
save_dir = 'D:/DataBackup/DIOR_visualization'
mkdir(save_dir)
save_dir += '/val'
mkdir(save_dir)
img_infos, id2name = coco_transform(jsonload(ann_file_path))


count = 0
for img_info in img_infos.values():
    img_path = img_root + '/' + img_info['file_name']
    img = cv2.imread(img_path)
    print(img_path)
    for ann in img_info['anns']:
        (x, y, w, h) = ann['bbox']
        cat_id = ann['category_id']
        (x1, y1) = (int(x), int(y))
        (x2, y2) = (int(x + w), int(y + h))
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), thickness=1)
        cv2.putText(img, id2name[cat_id],
                    (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 200, 200), 2)
    print(count)
    count += 1
    # print(img.shape)
    # cv2.imshow('sdf', img)
    # cv2.waitKey(1000)
    (filepath, filename) = os.path.split(img_path)
    cv2.imwrite(save_dir + '/' + filename, img)
