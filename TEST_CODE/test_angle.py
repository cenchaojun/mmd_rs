import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import cv2

import math
import matplotlib.pyplot as plt

p1 = np.array([[0, 0]]).T
p2 = np.array([[1, 0]]).T
angles = np.linspace(0, 2*np.pi, 1000)


def draw_dt_poly(img, points, cat, score, rect_color, text_color=(200, 0, 0),
                 no_text=False):
    # points = np.array([[910, 650], [206, 650], [458, 500], [696, 500]])
    p = np.array(points, dtype=np.int32)
    # img = img.copy()
    cv2.polylines(img, [p], 1, rect_color)


def RotBox2Polys(dboxes):
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


def polygonToRotRectangle(bbox):
    # print('bbox: ', bbox)

    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    # print(bbox[:, 0,0], bbox[:, 1,0], bbox[:, 0,1], bbox[:, 1,1])
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # angle = np.arctan2(bbox[:, 1, 1] - bbox[:, 1, 0], bbox[:, 0, 1] - bbox[:, 0, 0])
    return angle

def gt_mask_bp_obbs(gt_polys):


    return angle

def mask2poly(binary_mask):
    try:
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contour_lens = np.array(list(map(len, contours)))
        # max_id = contour_lens.argmax()
        # max_contour = contours[max_id]
        max_contour = max(contours, key=len)
        rect = cv2.minAreaRect(max_contour)
        poly = cv2.boxPoints(rect)
        # poly = TuplePoly2Poly(poly)
    except:
        import pdb
        pdb.set_trace()
    return poly



for W, H in [(200, 100), (100, 200), (5, 100), (100, 5), (100, 100)]:
    a_r = []
    for a in angles:
        R = np.array([[np.cos(a), -np.sin(a)],
                      [np.sin(a), np.cos(a)]])
        p2_R = np.matmul(R, p2)


        m = np.concatenate([p1, p2_R], axis=1)

        angle = np.arctan2(-(m[0, 1] - m[0, 0]), m[1, 1] - m[1, 0])

        rbbox = np.array([[300, 300, W, H, a]])
        poly = RotBox2Polys(rbbox)
        poly = poly[0].reshape([4, 2])

        v1 = poly[1] - poly[0]
        v2 = poly[2] - poly[1]
        order = v1[0] * v2[1] - v2[0] * v1[1]
        if order < 0:
            poly = np.array([poly[0], poly[3], poly[2], poly[1]])
            print(order)

        # a = poly[3].copy()
        # poly[3] = poly[1]
        # poly[1] = a

        img = np.ones([600, 600, 3], dtype=np.uint8) * 255
        p = np.array(poly, dtype=np.int32)

        # img = img.copy()
        cv2.polylines(img, [p], 1, (255, 255, 0))


        gt_bp_polys = get_best_begin_point(poly)
        # gt_bp_polys = poly


        p = np.array(gt_bp_polys[0:2], dtype=np.int32)
        cv2.polylines(img, [p], 1, (255, 0, 255))

        # 1
        p = tuple(np.array(gt_bp_polys[0], dtype=np.int32).tolist())
        cv2.circle(img, p, radius=2, color=(0, 0, 255))
        cv2.putText(img, '1',p,  cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                    thickness=1)
        # 2

        p = tuple(np.array(gt_bp_polys[1], dtype=np.int32).tolist())

        cv2.circle(img, p, radius=2, color=(0, 255, 0))
        cv2.putText(img, '2',p,  cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                    thickness=1)
        # 3

        p = tuple(np.array(gt_bp_polys[2], dtype=np.int32).tolist())

        cv2.circle(img, p, radius=2, color=(0, 255, 0))
        cv2.putText(img, '3',p,  cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                    thickness=1)
        # 4
        p = tuple(np.array(gt_bp_polys[3], dtype=np.int32).tolist())

        cv2.circle(img, p, radius=2, color=(0, 255, 0))
        cv2.putText(img, '4',p,  cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                    thickness=1)


        angle = polygonToRotRectangle(np.array([gt_bp_polys]).reshape(-1, 8)) / np.pi

        a_r.append(angle)

        # print(a / np.pi, angle)

        cv2.putText(img, str(angle),(200, 100),  cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                    thickness=1)


        cv2.imshow('DD', img)
        cv2.waitKey(1)


        # cv2.polylines(poly[0])


        #
        # print(p2_R, angle)
        # a_r.append(angle)
    plt.plot(angles / np.pi, a_r)
    plt.pause(1)
    plt.savefig('./W_%d_H_%d.png' %(W, H))
    print('Save ./W_%d_H_%d.png' %(W, H))
    print(min(a_r), max(a_r), min(np.abs(np.array(a_r))), max(np.abs(np.array(a_r)))
          )






# 角度计算方式：以[0, 1]向量为标准，
# 如果x在右侧（x>0），则夹角为-angle(x,[0, 1]) -> [-pi, 0]
# 如果x在左侧（x<0），则夹角为angle(x,[0, 1]) -> [0, pi]

#
# import cv2
# cv2.rotatedRectangleIntersection()