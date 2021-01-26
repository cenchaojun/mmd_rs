
import cv2
import numpy as np
from shapely.geometry import Polygon

# def polygon_area(points):
#     """返回多边形面积
#
#     """
#     area = 0
#     q = points[-1]
#     for p in points:
#         area += p[0] * q[1] - p[1] * q[0]
#         q = p
#     return area / 2

def Polybon2points(poly):
    assert type(poly) == Polygon
    poly = list(poly.exterior.coords)
    if len(poly) >2:
        poly = poly[0:len(poly)-1]
    return poly

def polyiou(poly1, poly2):
    """

    :param poly1: list(tuple:(x,y))
    :param poly2:
    :return: iou
    """
    if len(poly1) <= 2 or len(poly2) <= 2:
        return 0
    poly1 = Polygon(poly1)
    poly2 = Polygon(poly2)
    intersection = poly1.intersection(poly2)
    int_area = intersection.area
    iou = int_area / (poly1.area + poly2.area - int_area)
    return iou

######################################################
def rbbox2poly(dboxes):
    """cv2.boxPoints
    :param dboxes: (x_ctr, y_ctr, w, h, angle)
        (numboxes, 5)
    :return: quadranlges:
        (numboxes, 8)
    """
    # cv2.boxPoints
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


if __name__ == '__main__':
    # poly1 = [[0, 0], [0, 1], [1, 1], [1, 0]]
    # poly2 = [[0, 0], [-1/2, -1/2], [0, 1], [1/2,1/2]]
    #
    # # poly2 = [[10, 10], [10, 11], [11, 11], [11, 10]]
    #
    # print(polyiou(poly1, poly2))

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
            anchor_poly, anchor_mask = generage_data(300, 300, W, H, 0)
            gt_poly, gt_mask = generage_data(300, 300, W, H, a)

            mask_img = cv2.cvtColor(np.zeros_like(anchor_mask), cv2.COLOR_GRAY2RGB)
            mask_img[mask_img > 0] = 100
            draw_poly(mask_img, anchor_poly)
            draw_poly(mask_img, gt_poly)

            iou = polyiou(anchor_poly, gt_poly)

            cv2.putText(mask_img,
                        'W: %d, H:%d, a:%.3f, iou:%.3f' % (W, H, a, iou),
                        (20, 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                        thickness=1)

            cv2.imshow('DD', mask_img)
            cv2.waitKey(20)