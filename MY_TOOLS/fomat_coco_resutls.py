# modified from from inference.py
import mmcv
import cv2
from mmdet.datasets import build_dataloader, build_dataset
import mmcv
import numpy as np

import json
import pickle as pkl
import os
from tqdm import tqdm

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('Make dir: %s' % dir_path)


def pklsave(obj, file_path, msg=True):
    with open(file_path, 'wb+') as f:
        pkl.dump(obj, f)
        if msg:
            print('SAVE OBJ: %s' % file_path)

def jsonsave(obj, file_path, msg=True):
    with open(file_path, 'wt+') as f:
        json.dump(obj, f)
        if msg:
            print('SAVE JSON: %s' % file_path)

def pklload(file_path, msg=True):
    with open(file_path, 'rb') as f:
        if msg:
            print('LOAD OBJ: %s' % file_path)
            return pkl.load(f)

def jsonload(file_path, msg=True):
    with open(file_path, 'r') as f:
        if msg:
            print('LOAD OBJ: %s' % file_path)
        try:
            return json.load(f)
        except EOFError:
            print('EOF Error %s' % file_path)


def get_anns(results, dataset):
    if hasattr(dataset, 'img_infos'):
        img_infos = dataset.img_infos
    else:
        img_infos = dataset.data_infos
    print(len(results), len(dataset))
    assert len(results) == len(dataset)
    cat2label = dataset.cat2label
    # cat2label = {cat: label-1 for cat, label in cat2label.items()}
    label2cat = {label: cat for cat, label in cat2label.items()}
    print(dataset.cat2label)
    result_anns = []
    img_prefix = dataset.img_prefix

    for idx in tqdm(range(len(results))):
        r = results[idx]
        info = img_infos[idx]
        ann = dataset.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        gt_cats = [label2cat[l] for l in gt_labels]


        info['gt_anns'] = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            cats=gt_cats
        )
        info['file_path'] = img_prefix + info['filename']
        l = []
        s = []
        b = []
        cats = []

        for label, dets in enumerate(r):
            if len(dets) == 0:
                continue
            labels = np.ones(dets.shape[0]) * label
            cats.extend([label2cat[label] for i in range(dets.shape[0])])
            scores = dets[:, 4]
            bboxes = dets[:, 0: 4]
            l.append(labels)
            s.append(scores)
            b.append(bboxes)
        if len(l) > 0:
            l = np.hstack(l)
            s = np.hstack(s)
            b = np.vstack(b)
        info['dets'] = dict(
            bboxes=b,
            labels=l,
            scores=s,
            cats=cats
        )
        print(idx)
        result_anns.append(info)
    return result_anns

def get_gts(gt_anns):
    return gt_anns['bboxes'], gt_anns['labels']


def draw_gt(img, bbox, cat, rect_color, text_color=(0, 200, 200),
            rect_thick=2, font_size=0.5):
    (x1, y1, x2, y2) = bbox
    (x1, y1) = (int(x1), int(y1))
    (x2, y2) = (int(x2), int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, thickness=rect_thick)
    cv2.putText(img, str(cat),
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 2)


def draw_dt(img, bbox, cat, score, rect_color, text_color=(200, 0, 0),
            no_text=False):
    (x1, y1, x2, y2) = bbox
    (x1, y1) = (int(x1), int(y1))
    (x2, y2) = (int(x2), int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, thickness=2)
    show_text = '%s%.3f' % (str(cat), score)
    if not no_text:
        cv2.putText(img, show_text,
                    (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.9, text_color,
                    thickness=1)

def show(result_anns, folder='./test'):
    mkdir(folder)
    for idx, ann in enumerate(result_anns):
        org_img = cv2.imread(ann['file_path'])
        gt_anns = ann['gt_anns']
        for b, c in zip(gt_anns['bboxes'], gt_anns['cats']):
            draw_gt(org_img, b, c, (0, 200, 200))
        dt_anns = ann['dets']
        for b, c, s in zip(dt_anns['bboxes'], dt_anns['cats'], dt_anns['scores']):
            if s > 0.3:
                draw_dt(org_img, b, c, s, (255, 255, 0), (255, 255, 0))
        cv2.imwrite(folder + '/' + ann['filename'], org_img)
        if idx > 500:
            break
        print(idx)



if __name__ == '__main__':
    from EXP_CONCONFIG.CONFIGS.model_DIOR_full_config import DIOR_cfgs

    cfg = DIOR_cfgs['DIOR_retinanet_full']
    a = pklload(cfg['result'])

    cfg_file = cfg['config']
    cfg = mmcv.Config.fromfile(cfg_file)
    cfg.data.test.img_prefix = '/home/huangziyue/' + cfg.data.test.img_prefix
    cfg.data.test.ann_file = '/home/huangziyue/' + cfg.data.test.ann_file
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    if hasattr(dataset, 'img_infos'):
        img_infos = dataset.img_infos
    else:
        img_infos = dataset.data_infos
    d = dataset[0]
    c = get_anns(a, dataset)
    pklsave(c, './faster_dota.pkl')
    # show(c, folder='./faster_res')
    b = 0