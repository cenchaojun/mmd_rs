import cv2
import numpy as np
import os
import imageio
from color_dict import bgr_colors
colors = bgr_colors
# DIRECTIONS是对矩阵的直观感受，对于里头的元素（i，j）。向下：i增加，向右：j增加
DIRECTIONS = ['up', 'down', 'right', 'left']
DIR_Y = 'up'
DIR_X = 'right'
# 设计理念：
# canvas：画布，基础背景
# grid：网格
# fig：要绘制的图形

def convert_fig():
    pass
    # 转换图像到

def put_text(img, text, point):
    cv2.putText(img, str(text), point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                colors['White'],
                thickness=2)


def draw_axis(img, img_H, img_W,
              zero_point=(20, 20),
              r_axis = 4/5,
              r_tip = 1/30,
              dir_x='right',
              dir_y='down'):
    """

    :param img:
    :param img_H:
    :param img_W:
    :param zero_point:
    :param r_axis:  len(axis) / img_w
    :param r_tip:   len(arrow tip) / len(axis)
    :param dir_x:
    :param dir_y:
    :return:
    """
    z_x, z_y = zero_point
    axis_thick = int(5/800 * np.sqrt(img_H*img_W))
    # 基础刻度轴
    cv2.arrowedLine(img,
                    (0, z_y), (int(img_H*r_axis), z_y),
                    colors['Chocolate'],
                    thickness=axis_thick,
                    tipLength=r_tip)
    cv2.arrowedLine(img,
                    (z_x, 0), (z_x, int(img_H*r_axis)),
                    colors['Chocolate'],
                    thickness=axis_thick,
                    tipLength=r_tip)

    # 刻度轴标签
    put_text(img, 'X', (int(img_H*r_axis)-20, z_y-20))
    put_text(img, 'Y', (z_x-20, int(img_H*r_axis)-20))


    return img
def draw_poly(img, any_poly, thick_ness=1):
    """

    :param img:  org image
    :param poly: [[x1, y1], [x2, y2], ...]
    :return:
    """
    # outlines
    poly = np.array(any_poly, dtype=np.int32)
    cv2.polylines(img, [poly], 1,
                  colors['DeepSkyBlue'], thickness=thick_ness)

    # line 1 -> 2
    fs_points = np.array(poly[0:2], dtype=np.int32)
    cv2.polylines(img, [fs_points], 1,
                  colors['OliveDrab1'], thickness=thick_ness)

    # points label
    for i in range(len(poly)):
        p = tuple(np.array(poly[i], dtype=np.int32).tolist())
        cv2.circle(img, p, radius=3,
                   color=colors['Purple'],thickness=3)
        put_text(img, str(i), p)

    # 绘制最下边的一条边
    max_y = np.max(poly, axis=0)[1]
    cv2.line(img, (0, max_y), (1000, max_y),
             colors['OliveDrab1'], thickness=thick_ness)

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
    # x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    # x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    # x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    # x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)
    #
    # y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    # y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    # y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    # y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

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

def draw_rect(img, xc, yc, w, h, theta, zero_point=(0, 0)):
    xc += zero_point[0]
    yc += zero_point[1]
    rbbox = np.array([xc, yc, w, h, theta]).reshape(-1, 5)
    poly = rbbox2poly(rbbox)[0].reshape([4, 2])
    # 绘制多边形
    draw_poly(img, poly, thick_ness=2)

def rotate_point(p, theta):
    cs = np.cos(theta)
    ss = np.sin(theta)
    return [cs * p[0] - ss * p[1], ss*p[0]+cs*p[1]]

def grid(img, interval=50):
    pass

def show_img(img, delay=1000, flipCode=-1):
    # flipCode = 0 -> up down flip
    #          = 1 -> right left flip
    img = img.copy()
    if not flipCode == -1:
        img = cv2.flip(img, flipCode=0)
    cv2.imshow('DDD', img)
    cv2.waitKey(delay)


def cv2_angel(img_w, img_h,
              x_c, y_c, w, h, theta,
              zero_point=(0, 0)):
    x_c += zero_point[0]
    y_c += zero_point[1]
    rbbox = np.array([x_c, y_c, w, h, theta]).reshape(-1, 5)
    poly = rbbox2poly(rbbox)
    poly = poly[0].reshape([4, 2])

    mask_img = np.zeros([img_w, img_h], dtype=np.uint8)
    cv2.fillConvexPoly(mask_img, np.array(poly, dtype=np.int32), 1)
    # cv2.imshow('www', mask_img * 100)
    # cv2.waitKey(1)
    contours, hierarchy = cv2.findContours(mask_img,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
    # 点最多的那个，作为contours
    max_contour = max(contours, key=len)
    [(xc, yc), (w_new, h_new), theta_new] = cv2.minAreaRect(max_contour)
    print(x_c, y_c, w, h, theta, theta_new)
    theta_new = theta_new / 360 * (2 * np.pi)
    return w_new, h_new, theta_new

def create_gif(image_list, gif_name, duration=0.01):
    imageio.mimsave(gif_name, image_list, 'GIF', duration=duration)
    return

if __name__ =='__main__':
    img = np.zeros([800, 800, 3], dtype=np.uint8)
    zp = (300, 300)
    draw_axis(img, 800, 800, zero_point=zp)
    imgs = []

    for a in np.linspace(0, np.pi, 30):
        rect_img = img.copy()
        xc, yc, w, h, theta = (50, 100, 100, 200, a)
        p0 = np.array([xc - w/2, yc - h/2])
        xc, yc = p0 + np.array(rotate_point([w/2, h/2], theta))
        draw_rect(rect_img, xc, yc, w, h, theta, zero_point=zp)

        ct = cv2_angel(800, 800, xc, yc, w, h, theta,
                              zero_point=zp)
        cv2.putText(rect_img,
                    'org(w,h,angle): (%.2f, %.2f, %.2f)'
                    % (w, h, theta),
                    (20, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                    thickness=1)
        cv2.putText(rect_img,
                    'cv2(w,h,angle):(%.2f, %.2f, %.2f)'
                    % (ct[0], ct[1], ct[2]),
                    (20, 40),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                    thickness=1)


        show_img(rect_img, delay=1, flipCode=-1)
        imgs.append(rect_img)

    create_gif(imgs, './cv2_rotate_rect.gif', 0.35)



