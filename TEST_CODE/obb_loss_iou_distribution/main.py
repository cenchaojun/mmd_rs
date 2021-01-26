import sys
import codecs
import numpy as np
import os
import re
import cv2
import torch
import math
import matplotlib.pyplot as plt
from TEST_CODE.obb_loss_iou_distribution.utils import generage_data, polyiou, draw_poly
from TEST_CODE.obb_loss_iou_distribution.cv2_loss import \
    cv2_loss_target, \
    cv2_loss_mask_target, \
    cv2_loss_target_mod


if __name__ == '__main__':
    W = 200
    H = 100
    a_r = []
    p1 = np.array([[0, 0]]).T
    p2 = np.array([[1, 0]]).T
    angles = np.linspace(0, np.pi, 500)



    for (W, H) in [(200, 100), (100, 200)]:
        ad_angle = []
        cv_angle = []
        iou_distances = []
        l1s = []
        delta_ws = []
        delta_hs = []
        delta_as = []

        for a in angles:
            anchor_poly, anchor_mask = generage_data(300, 300, W, H, 0.0, mask_H=800, mask_W=800)
            gt_poly, gt_mask = generage_data(300, 300, W, H, a, mask_H=800, mask_W=800)

            mask_img = cv2.cvtColor(np.zeros_like(anchor_mask), cv2.COLOR_GRAY2RGB)
            mask_img[mask_img > 0] = 100
            draw_poly(mask_img, anchor_poly)
            draw_poly(mask_img, gt_poly)

            iou_dis = 1 - polyiou(anchor_poly, gt_poly)
            # target_info = cv2_loss_mask_target(anchor_mask, gt_mask)
            # target_info = cv2_loss_target(anchor_poly, gt_mask)
            target_info = cv2_loss_target_mod(anchor_poly, gt_mask)

            l1_str = target_info['l1_norm_str']
            l1 = target_info['l1_norm']
            [dx, dy, dw, dh, da] = np.abs(np.array(target_info['target']))

            iou_distances.append(iou_dis)
            l1s.append(l1)
            delta_ws.append(dw)
            delta_as.append(da)
            delta_hs.append(dh)

            cv2.putText(mask_img,
                        'W: %d, H:%d, a:%.3f, iou_dis:%.3f' % (W, H, a, iou_dis),
                        (0, 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                        thickness=1)
            cv2.putText(mask_img,
                        'target_l1:%s' % (l1_str),
                        (0, 40),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                        thickness=1)


            cv2.imshow('DD', mask_img)
            cv2.waitKey(5)

        prefix = './figs/cv2_mod_W_%d_H_%d'% (W, H)

        plt.plot(iou_distances, l1s)
        plt.xlabel('1-IOU'),  plt.ylabel('L1_NORM')
        plt.scatter(iou_distances, l1s, c=range(len(l1s)), cmap=plt.cm.coolwarm, edgecolor='none', s=40)
        plt.savefig(prefix + '_iou_l1_norm.png')
        # plt.legend()
        plt.close()

        plt.xlabel('angle(arc)'),  plt.ylabel('value')
        plt.plot(angles, l1s, label='target_L1_norm')
        plt.plot(angles, iou_distances, label='iou_dis')
        plt.plot(angles, delta_hs, label='delta h')
        plt.plot(angles, delta_ws, label='delta w')
        plt.plot(angles, delta_as, label='delta angel')
        plt.plot(angles, l1s, label='target_L1_norm')
        plt.legend()
        plt.savefig(prefix + '_anchor_plot.png')
        plt.close()

