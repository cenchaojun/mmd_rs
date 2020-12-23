import sys
import codecs
import numpy as np
import shapely.geometry as shgeo
import os
import re
import cv2

import math
import matplotlib.pyplot as plt

def draw_poly(poly):
    poly = np.array(poly, dtype=np.float)
    poly = poly.reshape(4, 2)
    min_x, min_y = np.min(poly,axis=0)
    max_x, max_y = np.max(poly,axis=0)
    c_x = (max_x + min_x) / 2
    c_y = (max_y + min_y) / 2

    r = 400 / (2 * np.max([max_x - min_x, max_y - min_y, 1]))
    poly[:, 0] = (poly[:, 0] - c_x) * r + 200

    poly[:, 1] = (poly[:, 1] - c_y) * r + 200


    print(poly)


    # MAX = max(int(np.max(max_y - min_y)) + 10, 200)

    img = np.ones([400, 400, 3], dtype=np.uint8) * 255

    p = np.array(poly, dtype=np.int32)
    cv2.polylines(img, [p], 1, (255, 0, 255))

    # 1
    p = tuple(np.array(poly[0], dtype=np.int32).tolist())
    cv2.circle(img, p, radius=2, color=(0, 0, 255))
    cv2.putText(img, '1', p, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                thickness=1)
    # 2

    p = tuple(np.array(poly[1], dtype=np.int32).tolist())

    cv2.circle(img, p, radius=2, color=(0, 255, 0))
    cv2.putText(img, '2', p, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                thickness=1)
    # 3

    p = tuple(np.array(poly[2], dtype=np.int32).tolist())

    cv2.circle(img, p, radius=2, color=(0, 255, 0))
    cv2.putText(img, '3', p, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                thickness=1)
    # 4
    p = tuple(np.array(poly[3], dtype=np.int32).tolist())

    cv2.circle(img, p, radius=2, color=(0, 255, 0))
    cv2.putText(img, '4', p, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0),
                thickness=1)
    
    return img


from commonlibs.common_tools import *

out_angle_bbox = pklload('../dota_analysis/out_anble_bbox.pkl')
for a_b in out_angle_bbox:
    print(a_b['angle'] / np.pi, a_b['bbox'])
    poly = a_b['bbox']
    img = draw_poly(poly)
    cv2.imshow('DDD', img)
    cv2.waitKey(1000)