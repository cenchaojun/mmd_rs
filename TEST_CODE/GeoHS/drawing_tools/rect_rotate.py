import cv2
import numpy as np
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

def draw_axis(img, img_H, img_W,
              zero_point=(20, 20),
              r_axis = 4/5,
              r_tip = 1/30,
              dir_x='right', dir_y='down'):
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
    cv2.arrowedLine(img, (0, z_y), (int(img_H*r_axis), z_y), (0, 0, 0),
                    thickness=axis_thick,
                    tipLength=r_tip)
    cv2.arrowedLine(img, (z_x, 0), (z_x, int(img_H*r_axis)), (0, 0, 0),
                    thickness=axis_thick,
                    tipLength=r_tip)

    # 刻度轴标签

    return img

def grid(img, interval=50):

def show_img(img):
    cv2.imshow('DDD', img)
    cv2.waitKey(3000)

if __name__ =='__main__':
    img = np.ones([800, 800, 3], dtype=np.uint8) * 255
    draw_axis(img, 800, 800)
    show_img(img)

