from commonlibs.transform_tools.data_transform import coco_transform
from commonlibs.common_tools import *
import os
import shutil
import cv2


def coco_2_dota(img_folder, ann_file_path, result_folder):
    img_infos, id2name = coco_transform(jsonload(ann_file_path))

    label_folder = result_folder + '/labelTxt'
    image_folder = result_folder + '/images'
    mkdir(result_folder)
    mkdir(label_folder)
    mkdir(image_folder)


    for img_count, img_info in enumerate(img_infos.values()):
        img_file_name = img_info['file_name']
        img_path = img_folder + '/' + img_file_name
        new_img_path = image_folder + '/' + \
                       os.path.splitext(img_file_name)[0] + '.png'
        img = cv2.imread(img_path)
        cv2.imwrite(new_img_path, img)

        # shutil.copyfile(img_path, image_folder + '/' + img_file_name)

        ann_file_name = os.path.splitext(img_file_name)[0] + '.txt'
        ann_file_path = label_folder + '/' + ann_file_name
        with open(ann_file_path, 'wt+') as f:
            f.write('imagesource: %s' % ann_file_path + '\n')
            f.write('gsd: %f' % 0.0 + '\n')
            for ann in img_info['anns']:
                x1, y1, w, h = ann['bbox']
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)

                name = id2name[ann['category_id']]
                difficulty = 0
                f.write('%d %d %d %d '
                        '%d %d %d %d %s %d\n' %
                        (x1, y1,
                         x1, y1+h,
                         x1+w, y1+h,
                         x1+w, y1,
                         name, difficulty))
            print('Save ann: %s' % ann_file_path)

        # if img_count > 100:
        #     break
        print('Done %d / %d' % (img_count+1, len(img_infos)))


data_folder = 'D:/DataBackup/DIOR'
ann_folder = data_folder + '/coco_annotations'
result_root= 'D:/DataBackup/DIOR_DOTA_ver'
mkdir(result_root)

ann_file = ann_folder + '/val_coco_ann.json'
img_folder = data_folder + '/JPEGImages-trainval'
coco_2_dota(img_folder, ann_file, result_root + '/val')
#
ann_file = ann_folder + '/train_coco_ann.json'
img_folder = data_folder + '/JPEGImages-trainval'
coco_2_dota(img_folder, ann_file, result_root + '/train')

ann_file = ann_folder + '/test_coco_ann.json'
img_folder = data_folder + '/JPEGImages-test'
coco_2_dota(img_folder, ann_file, result_root + '/test')