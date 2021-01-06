import sys
import codecs
import numpy as np
import os
import re
import cv2

import math
import matplotlib.pyplot as plt


def rbbox2poly(dboxes):
    """
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    cs = np.cos(dboxes[:, 4])
    ss = np.sin(dboxes[:, 4])
    w = dboxes[:, 2] - 1
    h = dboxes[:, 3] - 1

    ## change the order to be the initial definition
    x_ctr = dboxes[:, 0]
    y_ctr = dboxes[:, 1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    x1 = x1[:, np.newaxis]
    y1 = y1[:, np.newaxis]
    x2 = x2[:, np.newaxis]
    y2 = y2[:, np.newaxis]
    x3 = x3[:, np.newaxis]
    y3 = y3[:, np.newaxis]
    x4 = x4[:, np.newaxis]
    y4 = y4[:, np.newaxis]

    polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)
    return polys

def draw_poly(img, any_poly):
    """

    :param img:  org image
    :param poly: [[x1, y1], [x2, y2], ...]
    :return:
    """
    poly = np.array(any_poly, dtype=np.int32)
    cv2.polylines(img, [poly], 1, (255, 0, 0))

    # line 1 -> 2
    fs_points = np.array(poly[0:2], dtype=np.int32)
    cv2.polylines(img, [fs_points], 1, (0, 255, 0))

    # points label
    for i in range(len(poly)):
        p = tuple(np.array(poly[i], dtype=np.int32).tolist())
        cv2.circle(img, p, radius=2, color=(0, 0, 255))
        cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                    thickness=1)

def generage_data(x_c, y_c, w, h, theta,
                  mask_W=600, mask_H=600):
    """
    :param x_c:
    :param y_c:
    :param w:
    :param h:
    :param theta:  theta=0的时候w是图像的宽, [0~2pi]
    :return: poly:[4x2], mask_img:0->background, 1->object
    """
    rbbox = np.array([x_c, y_c, w, h, theta]).reshape(-1, 5)
    poly = rbbox2poly(rbbox)
    poly = poly[0].reshape([4, 2])

    mask_img = np.zeros([mask_W, mask_H], dtype=np.uint8)
    cv2.fillConvexPoly(mask_img, np.array(poly, dtype=np.int32), 1)

    return poly, mask_img


# 原始输入:  poly or mask_img
# output:  [x_c, y_c, w, h, theta], theta有不同的定义方式。

################################### 方法 #########################################

################################### cv2 #########################################
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

def cv2_mask2rbbox(masks):
    rbboxes = list(map(cv2_mask2rbbox_single, masks))
    return rbboxes

################################### AerialDetection #########################################

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1 = coordinate[0][0]
    y1 = coordinate[0][1]
    x2 = coordinate[1][0]
    y2 = coordinate[1][1]
    x3 = coordinate[2][0]
    y3 = coordinate[2][1]
    x4 = coordinate[3][0]
    y4 = coordinate[3][1]
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) + cal_line_length(combinate[i][1],
                                                                                           dst_coordinate[
                                                                                               1]) + cal_line_length(
            combinate[i][2], dst_coordinate[2]) + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return  combinate[force_flag]

def get_best_begin_point(coordinate_list):
    best_coordinate_list = map(get_best_begin_point_single, coordinate_list)
    best_coordinate_list = np.stack(list(best_coordinate_list))
    return best_coordinate_list

def polygonToRotRectangle_batch(poly):

    poly = np.array( poly, dtype=np.float32)

    # poly = poly.reshape([4,2])
    # v1 = poly[1] - poly[0]
    # v2 = poly[2] - poly[1]
    # order = v1[0] * v2[1] - v2[0] * v1[1]
    # # print(order)
    # if order < 0:
    #     poly = np.array([poly[0], poly[3], poly[2], poly[1]])
    #     print(order)

    poly = np.reshape(poly, newshape=(-1, 2, 4), order='F')


    angle = np.arctan2(-(poly[:, 0, 1] - poly[:, 0, 0]), poly[:, 1, 1] - poly[:, 1, 0])
    center = np.zeros((poly.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += poly[:, 0, i]
        center[:, 1, 0] += poly[:, 1, i]

    center = np.array(center,dtype=np.float32)/4.0

    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)), poly - center)


    xmin = np.min(normalized[:, 0, :], axis=1)
    xmax = np.max(normalized[:, 0, :], axis=1)
    ymin = np.min(normalized[:, 1, :], axis=1)
    ymax = np.max(normalized[:, 1, :], axis=1)

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes


def mask2poly_single(binary_mask):
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=len)
    rect = cv2.minAreaRect(max_contour)
    poly = cv2.boxPoints(rect)
    return poly

def mask2poly(binary_mask_list):
    polys = map(mask2poly_single, binary_mask_list)
    return list(polys)

def AerialDet_mask2rbbox(gt_masks):
    """

    :param gt_masks: list of mask
    :return:
    """
    gt_polys = mask2poly(gt_masks)
    gt_bp_polys = get_best_begin_point(gt_polys)
    gt_obbs = polygonToRotRectangle_batch(gt_bp_polys)

    return gt_obbs

def AerialDet_mask2rbbox_batch(gt_masks_list):
    gt_obbs_list = map(AerialDet_mask2rbbox, gt_masks_list)
    return list(gt_obbs_list)


if __name__ == '__main__':
    W = 200
    H = 100
    a_r = []
    p1 = np.array([[0, 0]]).T
    p2 = np.array([[1, 0]]).T
    angles = np.linspace(0, 2 * np.pi, 1000)



    for (W, H) in [(200, 100), (100, 200)]:
        ad_angle = []
        cv_angle = []
        for a in angles:
            poly, mask = generage_data(300, 300, W, H, a)


            poly_ad = AerialDet_mask2rbbox([mask])[0]
            poly_cv = cv2_mask2rbbox(mask)
            ad_angle.append(poly_ad[-1])
            cv_angle.append(poly_cv[-1])

            gt_polys = mask2poly([mask])[0]
            mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_img[mask_img > 0] = 100
            # draw_poly(mask_img, poly)
            draw_poly(mask_img, gt_polys)
            cv2.putText(mask_img, str('AD:%f CV:%f\n' % (poly_ad[-1] / (2*np.pi),
                                                         poly_cv[-1] / 360)), (200, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                        thickness=1)


            cv2.imshow('DD', mask_img)
            cv2.waitKey(50)
        ad_angle = np.array(ad_angle) / (2*np.pi)
        cv_angle = np.array(cv_angle) / (360)


        plt.plot(angles / np.pi, ad_angle, label='AD')
        plt.plot(angles / np.pi, cv_angle, label='CV')
        plt.legend()
        plt.pause(1)
        plt.savefig('./W_%d_H_%d.png' % (W, H))
        print('Save ./W_%d_H_%d.png' % (W, H))
        plt.close()
        # print(min(a_r), max(a_r),
        #       min(np.abs(np.array(a_r))),
        #       max(np.abs(np.array(a_r)))
        #       )
