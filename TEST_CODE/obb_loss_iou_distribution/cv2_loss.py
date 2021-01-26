import cv2
import numpy as np
from TEST_CODE.obb_loss_iou_distribution.utils import *

def cv2_mask2rbbox_single(bi_mask):
    """
    cv2的方法获得旋转框角度
    :param bi_mask:
    :return:
    """
    contours, hierarchy = cv2.findContours(bi_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    # 点最多的那个，作为contours
    max_contour = max(contours, key=len)
    [(xc, yc), (w, h), theta] = cv2.minAreaRect(max_contour)
    theta = (theta / 360) * (2 * np.pi)
    return [xc, yc, w, h, theta]

def cv2_loss_target(anchor_poly, gt_mask):
    anchor_poly = np.array(anchor_poly)
    a_xc, a_yc = np.mean(anchor_poly, axis=0)
    max_x, max_y = np.max(anchor_poly, axis=0)
    min_x, min_y = np.min(anchor_poly, axis=0)
    a_w = max_x - min_x
    a_h = max_y - min_y

    g_xc, g_yc, g_w, g_h, g_a = cv2_mask2rbbox_single(gt_mask)
    dx = (g_xc - a_xc) / a_w
    dy = (g_yc - a_yc) / a_h
    dw = np.log(g_w / a_w)
    dh = np.log(g_h / a_h)
    da = g_a - 0

    deltas = np.array([dx, dy, dw, dh, da])
    norms = ['%.2f' % i for i in np.abs(deltas)]

    return dict(target=[dx, dy, dw, dh, da],
                l1_norm_str=str(norms),
                l1_norm=np.sum(np.abs(deltas)))

def cv2_loss_target_mod(anchor_poly, gt_mask):
    anchor_poly = np.array(anchor_poly)
    a_xc, a_yc = np.mean(anchor_poly, axis=0)
    max_x, max_y = np.max(anchor_poly, axis=0)
    min_x, min_y = np.min(anchor_poly, axis=0)
    a_w = max_x - min_x
    a_h = max_y - min_y

    g_xc, g_yc, g_w, g_h, g_a = cv2_mask2rbbox_single(gt_mask)
    print(g_a,  -np.pi / 2 + g_a)
    if g_a < -np.pi / 4:
        g_a = -np.pi / 2 - g_a
        g_w, g_h = g_h, g_w
        print('change')

    dx = (g_xc - a_xc) / a_w
    dy = (g_yc - a_yc) / a_h
    dw = np.log(g_w / a_w)
    dh = np.log(g_h / a_h)
    da = g_a - 0

    deltas = np.array([dx, dy, dw, dh, da])
    norms = ['%.2f' % i for i in np.abs(deltas)]

    return dict(target=[dx, dy, dw, dh, da],
                l1_norm_str=str(norms),
                l1_norm=np.sum(np.abs(deltas)))



def cv2_loss_mask_target(anchor_mask, gt_mask):
    a_xc, a_yc, a_w, a_h, a_a = cv2_mask2rbbox_single(anchor_mask)

    g_xc, g_yc, g_w, g_h, g_a = cv2_mask2rbbox_single(gt_mask)
    dx = (g_xc - a_xc) / a_w
    dy = (g_yc - a_yc) / a_h
    dw = np.log(g_w / a_w)
    dh = np.log(g_h / a_h)
    da = g_a - 0

    deltas = np.array([dx, dy, dw, dh, da])
    norms = ['%.2f' % i for i in np.abs(deltas)]

    return dict(target=[dx, dy, dw, dh, da],
                l1_norm_str=str(norms),
                l1_norm=np.sum(np.abs(deltas)))

