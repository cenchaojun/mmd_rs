from commonlibs.common_tools import *

import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
# res_root = './DIOR_test_hists'



def simple_ana(res_root, data_ann, img_root):
    data_ann = jsonload(data_ann)
    if not os.path.exists(res_root):
        os.mkdir(res_root)
    def simple_hist(data, bins, bin_names=None, fig_name='test'):

        assert len(bins) >= 2
        if bin_names:
            assert len(bin_names) == len(bins) - 1
        else:
            bin_names = []
            for i in range(1, len(bins)):
                bin_names.append('[' + str(bins[i-1]) + ' ,'
                                 + str(bins[i]) + ']')
        plt.xticks(range(len(bin_names)), bin_names)
        plt.title(fig_name)
        hist, bins = np.histogram(data, bins=bins)
        plt.bar(range(len(bin_names)), hist)
        plt.savefig(res_root + '/%s.png' % fig_name)
        plt.close()

    # a = [0, 1,2, 2, 3, 3, 3, 3, 4, 4, 5]
    # simple_hist(np.array(a), [0, 2, 4, 6])

    a = 0

    img_sizes = []
    img_ratios = []
    bbox_per_imgs = []
    bbox_as_ratios = []
    bbox_areas = []

    for img_info in tqdm(data_ann['images']):
        h, w = img_info['height'], img_info['width']
        img_sizes.append(np.sqrt(h*w))
        img_ratios.append(w / h)

    bpi_tmp = dict()
    for ann in tqdm(data_ann['annotations']):
        if ann['image_id'] not in bpi_tmp.keys():
            bpi_tmp[ann['image_id']] = 0
        bpi_tmp[ann['image_id']] += 1
        bbox_as_ratios.append(float(ann['bbox'][2]) / max(1, ann['bbox'][3]))
        bbox_areas.append(np.sqrt(ann['bbox'][2]*ann['bbox'][3]))

    bbox_per_imgs = list(bpi_tmp.values())
    n_objects = sum(bbox_per_imgs)

    img_sizes = np.array(img_sizes)
    img_ratios = np.array(img_ratios)
    bbox_per_imgs = np.array(bbox_per_imgs)
    bbox_as_ratios = np.array(bbox_as_ratios)
    bbox_areas = np.array(bbox_areas)

    def form_info(array, array_name):
        array = np.array(array).flatten()
        return '%s : range=[%.3f, %.3f], mean=%.3f, std=%.3f' \
               % (array_name, np.min(array), np.max(array), np.mean(array), np.std(array))
    data_ana_infos = ''
    data_ana_infos += form_info(img_sizes, 'img_sizes') + '\n'
    data_ana_infos += form_info(img_ratios, 'img_ratios') + '\n'
    data_ana_infos += form_info(bbox_per_imgs, 'bbox_per_imgs') + '\n'
    data_ana_infos += form_info(bbox_as_ratios, 'bbox_as_ratios') + '\n'
    data_ana_infos += form_info(bbox_areas, 'bbox_areas') + '\n'
    data_ana_infos += 'num_objects : %d\n' % n_objects
    data_ana_infos += 'num_imgs : %d\n' % len(img_sizes)

    data_ana_infos += 'n_empty_images : %d\n' % (len(img_sizes) - len(bbox_per_imgs))


    with open(res_root + '/data_ana_infos.txt', 'wt+') as f:
        f.write(data_ana_infos)
    print(data_ana_infos)

    simple_hist(img_sizes, [0, 200, 500, 750, 1000, 2000, 5000, 10000], fig_name='img_sizes')
    simple_hist(img_ratios, [0, 0.5, 1, 1.5, 2], fig_name='img_ratios')
    simple_hist(bbox_per_imgs, [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 10000], fig_name='bbox_per_imgs')
    simple_hist(bbox_as_ratios, [0, 0.125, 0.25, 0.5, 1, 2, 4, 8], fig_name='bbox_as_ratios')
    simple_hist(bbox_areas, [0, 32, 96, 1000, 10000], fig_name='bbox_areas')

if __name__ == '__main__':
    # data_ann = jsonload('../../data/DIOR/coco_annotations/train_val_coco_ann.json')
    # img_root = '../../data/DIOR/JPEGImages-trainval'
    # data_ann = jsonload('../../data/DIOR/coco_annotations/test_coco_ann.json')
    # img_root = '../../data/DIOR/JPEGImages-test'
    # data_ann = jsonload('../../data/NWPU_VHR_10/train_val_coco_ann.json')
    # img_root = '../../data/NWPU_VHR_10/images'
    # data_ann = jsonload('../../data/dota1_train_val_1024/valtest1024/DOTA_valtest1024.json')
    # img_root = '../../data/dota1_train_val_1024/valtest1024/images'
    # simple_ana('./DOTA_train_hists',
    #            '../../data/dota1_train_val_1024/train1024/DOTA_train1024.json',
    #            '../../data/dota1_train_val_1024/train1024/images')
    # simple_ana('./DOTA_valtest_hists',
    #            '../../data/dota1_train_val_1024/valtest1024/DOTA_valtest1024.json',
    #            '../../data/dota1_train_val_1024/valtest1024/images')
    simple_ana('./DIOR_train_val_hists',
               '../../data/DIOR/coco_annotations/train_val_coco_ann.json',
               '../../data/DIOR/JPEGImages-trainval')
    simple_ana('./DIOR_test_hists',
               '../../data/DIOR/coco_annotations/test_coco_ann.json',
               '../../data/DIOR/JPEGImages-test')
    # simple_ana('./DOTA_train_org_hists',
    #            '../../data/dota/train/train_coco.json',
    #            '../../data/dota/train/images')
    # simple_ana('./DOTA_val_org_hists',
    #            '../../data/dota/val/val_coco.json',
    #            '../../data/dota/val/images')