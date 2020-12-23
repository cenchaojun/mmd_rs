import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import cv2

import math
import matplotlib.pyplot as plt



def draw_dt_poly(img, points, cat, score, rect_color, text_color=(200, 0, 0),
                 no_text=False):
    # points = np.array([[910, 650], [206, 650], [458, 500], [696, 500]])
    p = np.array(points, dtype=np.int32)
    # img = img.copy()
    cv2.polylines(img, [p], 1, rect_color)


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

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point(coordinate):
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


def Aerial_angle(bbox):
    """

    :param bbox: 4 x 2, poly bbox
    :return:
    """
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    print(bbox[:, 0,0], bbox[:, 1,0], bbox[:, 0,1], bbox[:, 1,1])
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # angle = np.arctan2(bbox[:, 1, 1] - bbox[:, 1, 0], bbox[:, 0, 1] - bbox[:, 0, 0])
    return angle

def poly2rotbox_AD_type(poly):
    """

    :param poly: [x1, y1, x2, y2, ...], poly poly
    :return:[x]
    """
    poly = np.array(poly,dtype=np.float32)

    ####  mask 方法
    # MAX = int(np.max(poly.reshape(-1)))
    #
    #
    # poly = poly.reshape([4, 2])
    # poly_img = np.array(poly, dtype=np.int32)
    #
    # img = np.ones([MAX, MAX, 3], dtype=np.uint8) * 0
    # cv2.fillConvexPoly(img, poly_img, (100, 100, 100))
    # bi_mask = img.copy()
    # bi_mask = cv2.cvtColor(bi_mask, cv2.COLOR_RGB2GRAY)
    # mask_poly = mask2poly(bi_mask)
    #
    # mask_poly = mask_poly.astype(np.float32)
    # poly = mask_poly.reshape([4, 2])
    poly = poly.reshape([4, 2])

    v1 = poly[1] - poly[0]
    v2 = poly[2] - poly[1]
    order = v1[0] * v2[1] - v2[0] * v1[1]
    # print(order)
    if order < 0:
        poly = np.array([poly[0], poly[3], poly[2], poly[1]])
        print(order)

    gt_bp_polys = get_best_begin_point(poly)
    poly = np.array([gt_bp_polys]).reshape(-1, 8)


    poly = np.reshape(poly,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(poly[0,1]-poly[0,0]),poly[1,1]-poly[1,0])
    # print('poly: ', poly)
    angle = np.arctan2(-(poly[:, 0,1]-poly[:, 0,0]),poly[:, 1,1]-poly[:, 1,0])
    # angle = np.arctan2(-(poly[:, 0,1]-poly[:, 0,0]),poly[:, 1,1]-poly[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((poly.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += poly[:, 0,i]
        center[:, 1, 0] += poly[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)),poly-center)


    xmin = np.min(normalized[:, 0, :], axis=1)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[:, 0, :], axis=1)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[:, 1, :], axis=1)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    # TODO: check it
    with_module=False
    if with_module:
        angle = angle[:, np.newaxis] % ( 2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes

def cv2_angle(bbox):
    """

    :param bbox: 4 x 2
    :return:
    """
    (xm_id, ym_id) = np.argmax(bbox, axis=0)
    p1 = bbox[ym_id]
    p2 = bbox[xm_id]

    return None

def mask2poly(binary_mask):
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 点最多的那个，作为contours
    max_contour = max(contours, key=len)
    # rect ：（(xc, yc), (W, H), theta）
    rect = cv2.minAreaRect(max_contour)
    # poly: 4x2 corner points
    poly = cv2.boxPoints(rect)
    return poly

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

def generage_data(x_c, y_c, w, h, theta):
    """

    :param x_c:
    :param y_c:
    :param w:
    :param h:
    :param theta:  theta=0的时候w是图像的宽, [0~2pi]

    :return: poly, bi_mask
    """
    rbbox = [x_c, y_c, w, h, theta]
    poly = rbbox2poly(rbbox)
    poly = poly[0].reshape([4, 2])

    poly_img = np.array(poly, dtype=np.int32)
    img = np.ones([600, 600, 3], dtype=np.uint8) * 0
    cv2.fillConvexPoly(img, poly_img, (100, 100, 100))

    bi_mask = img.copy()
    bi_mask = cv2.cvtColor(bi_mask, cv2.COLOR_RGB2GRAY)



# 原始输入:  poly
# output:  [x_c, y_c, w, h, theta], theta有不同的定义方式。

if __name__ == '__main__':
    W = 200
    H = 100
    a_r = []
    p1 = np.array([[0, 0]]).T
    p2 = np.array([[1, 0]]).T
    angles = np.linspace(0, 2 * np.pi, 1000)

    for (W, H) in [(200, 100), (100, 200)]:
        for a in angles:
            rbbox = np.array([[300, 300, W, H, a]])
            poly = rbbox2poly(rbbox)
            poly = poly[0].reshape([4, 2])
            bi_img = np.zeros([600, 600], dtype=np.uint8)
            cv2.fillConvexPoly(bi_img, np.array(poly, dtype=np.int32), 255)

            # bi_mask = bi_mask.astype(np.bool)
            mask_poly = mask2poly(bi_img)
            img = cv2.cvtColor(bi_img, cv2.COLOR_GRAY2RGB)
            img[img>0] = 100

            draw_poly(img, mask_poly)


            cv2.imshow('DD', img)
            cv2.waitKey(1)