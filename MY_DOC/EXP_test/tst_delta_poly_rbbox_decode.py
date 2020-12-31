# import sys
# import codecs
# import numpy as np
# import os
# import re
# import cv2
# import torch
# import math
# import matplotlib.pyplot as plt
#
#
# def rbbox2poly(dboxes):
#     """cv2.boxPoints
#     :param dboxes: (x_ctr, y_ctr, w, h, angle)
#         (numboxes, 5)
#     :return: quadranlges:
#         (numboxes, 8)
#     """
#     # cv2.boxPoints
#     cs = np.cos(dboxes[:, 4])
#     ss = np.sin(dboxes[:, 4])
#     w = dboxes[:, 2] - 1
#     h = dboxes[:, 3] - 1
#
#     ## change the order to be the initial definition
#     x_ctr = dboxes[:, 0]
#     y_ctr = dboxes[:, 1]
#     x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
#     x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
#     x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
#     x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)
#
#     y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
#     y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
#     y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
#     y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)
#
#     x1 = x1[:, np.newaxis]
#     y1 = y1[:, np.newaxis]
#     x2 = x2[:, np.newaxis]
#     y2 = y2[:, np.newaxis]
#     x3 = x3[:, np.newaxis]
#     y3 = y3[:, np.newaxis]
#     x4 = x4[:, np.newaxis]
#     y4 = y4[:, np.newaxis]
#
#     polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)
#     return polys
#
# def generage_data(x_c, y_c, w, h, theta,
#                   mask_W=600, mask_H=600):
#     """
#     :param x_c:
#     :param y_c:
#     :param w:
#     :param h:
#     :param theta:  theta=0的时候w是图像的宽, [0~2pi]
#     :return: poly:[4x2], mask_img:0->background, 1->object
#     """
#     rbbox = np.array([x_c, y_c, w, h, theta]).reshape(-1, 5)
#     poly = rbbox2poly(rbbox)
#     poly = poly[0].reshape([4, 2])
#
#     mask_img = np.zeros([mask_W, mask_H], dtype=np.uint8)
#     cv2.fillConvexPoly(mask_img, np.array(poly, dtype=np.int32), 1)
#
#     return poly, mask_img
#
# def cv2_mask2rbbox_single(bi_mask):
#     """
#     cv2的方法获得旋转框角度
#     :param bi_mask:
#     :return:
#     """
#     contours, hierarchy = cv2.findContours(bi_mask,
#                                            cv2.RETR_EXTERNAL,
#                                            cv2.CHAIN_APPROX_NONE)
#     # 点最多的那个，作为contours
#     max_contour = max(contours, key=len)
#     [(xc, yc), (w, h), theta] = cv2.minAreaRect(max_contour)
#     theta = theta / 360
#     return [xc, yc, w, h, theta]
#
# def cv2_mask2rbbox(masks):
#     rbboxes = list(map(cv2_mask2rbbox_single, masks))
#     return rbboxes
#
#
# def rbbox2delta(proposals, gt, means=(0., 0., 0., 0., 0.), stds=(1., 1., 1., 1., 1.)):
#     """
#
#     :param proposals: anchor, [x, y, h, w, a]
#     :param gt:                [x, y, h, w, a]
#     :param means:
#     :param stds:
#     :return:
#     """
#     assert proposals.size() == gt.size()
#
#
#     dx = (gt[..., 0] - proposals[..., 0]) / proposals[..., 2]
#     dy = (gt[..., 1] - proposals[..., 1]) / proposals[..., 3]
#     dw = torch.log(gt[..., 2] / proposals[..., 2])
#     dh = torch.log(gt[..., 3] / proposals[..., 3])
#     da = gt[..., 4] - proposals[..., 4]
#     deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
#
#     means = deltas.new_tensor(means).unsqueeze(0)
#     stds = deltas.new_tensor(stds).unsqueeze(0)
#     deltas = deltas.sub_(means).div_(stds)
#
#     return deltas
#
#
# def delta2rbbox(rois,
#                deltas,
#                means=(0., 0., 0., 0.),
#                stds=(1., 1., 1., 1.),
#                max_shape=None,
#                wh_ratio_clip=16 / 1000):
#     """
#     :param rois:
#     (N, 5). N = num_anchors * W * H
#     :param deltas:
#     (N, 5 * num_classes).N = num_anchors * W * H
#     :param means:
#     :param stds:
#     :param max_shape:      Maximum bounds for boxes. specifies (H, W)
#     :param wh_ratio_clip:  Maximum aspect ratio for boxes.
#     :return:
#     """
#     means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
#     stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
#     denorm_deltas = deltas * stds + means
#     dx = denorm_deltas[:, 0::5]
#     dy = denorm_deltas[:, 1::5]
#     dw = denorm_deltas[:, 2::5]
#     dh = denorm_deltas[:, 3::5]
#     da = denorm_deltas[:, 5::5]
#
#     max_ratio = np.abs(np.log(wh_ratio_clip))
#     dw = dw.clamp(min=-max_ratio, max=max_ratio)
#     dh = dh.clamp(min=-max_ratio, max=max_ratio)
#
#     px = rois[:, 0].unsqueeze(1).expand_as(dx)
#     py = rois[:, 1].unsqueeze(1).expand_as(dy)
#     pw = rois[:, 2].unsqueeze(1).expand_as(dw)
#     ph = rois[:, 3].unsqueeze(1).expand_as(dh)
#     pa = rois[:, 4].unsqueeze(1).expand_as(da)
#
#     gx = px + pw * dx
#     gy = py + ph * dy
#     gw = pw * dw.exp()
#     gh = ph * dh.exp()
#     ga = da + pa
#
#     # gx: N x num_classes
#     # torch.stack([gx, gy, gw, gh, ga], dim=-1): N x num_classes x 5
#     # view(N x num_classes, 5): [[gx1, gy1, gw1, gh1, ga1, gx2, gy2, ...], ...]
#     bboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view(deltas.size())
#     return bboxes
#
# def transform_to_rbbox(bboxes):
#     # https://editor.csdn.net/md/?articleId=108725272
#     bbox_dim =  bboxes.size(-1)
#     assert bbox_dim in [4, 5, 8]
#     if bbox_dim == 4:
#         x_c = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
#         y_c = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
#         w = bboxes[..., 2] - bboxes[..., 0] + 1.0
#         h = bboxes[..., 3] - bboxes[..., 1] + 1.0
#         a = bboxes.new_zeros(len(bboxes))
#         return torch.stack([x_c, y_c, w, h, a], dim=-1)
#     else:
#         return bboxes
#
#
# def mask2poly_single(binary_mask):
#     contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     max_contour = max(contours, key=len)
#     rect = cv2.minAreaRect(max_contour)
#     poly = cv2.boxPoints(rect)
#     return poly
#
# def mask2poly(binary_mask_list):
#     polys = map(mask2poly_single, binary_mask_list)
#     return list(polys)
#
#
# def RotBox2Polys(dboxes):
#     """
#     :param dboxes: (x_ctr, y_ctr, w, h, angle)
#         (numboxes, 5)
#     :return: quadranlges:
#         (numboxes, 8)
#     """
#     cs = np.cos(dboxes[:, 4])
#     ss = np.sin(dboxes[:, 4])
#     w = dboxes[:, 2] - 1
#     h = dboxes[:, 3] - 1
#
#     ## change the order to be the initial definition
#     x_ctr = dboxes[:, 0]
#     y_ctr = dboxes[:, 1]
#     x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
#     x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
#     x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
#     x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)
#
#     y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
#     y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
#     y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
#     y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)
#
#     x1 = x1[:, np.newaxis]
#     y1 = y1[:, np.newaxis]
#     x2 = x2[:, np.newaxis]
#     y2 = y2[:, np.newaxis]
#     x3 = x3[:, np.newaxis]
#     y3 = y3[:, np.newaxis]
#     x4 = x4[:, np.newaxis]
#     y4 = y4[:, np.newaxis]
#
#     polys = np.concatenate((x1, y1, x2, y2, x3, y3, x4, y4), axis=1)
#     return polys
#
# def draw_poly(img, any_poly):
#     """
#
#     :param img:  org image
#     :param poly: [[x1, y1], [x2, y2], ...]
#     :return:
#     """
#     poly = np.array(any_poly, dtype=np.int32)
#     cv2.polylines(img, [poly], 1, (255, 0, 0))
#
#     # line 1 -> 2
#     fs_points = np.array(poly[0:2], dtype=np.int32)
#     cv2.polylines(img, [fs_points], 1, (0, 255, 0))
#
#     # points label
#     for i in range(len(poly)):
#         p = tuple(np.array(poly[i], dtype=np.int32).tolist())
#         cv2.circle(img, p, radius=2, color=(0, 0, 255))
#         cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
#                     thickness=1)
#
# def generate_roi(poly):
#
#
#
# if __name__ == '__main__':
#     # generage_data -> gt_poly, gt_mask
#     # gt_poly -> random -> rois : [x1, y1, x2, y2]
#     # gt_mask -> cv2_mask2rbbox -> cv2_rbbox: [x, y, h, w, a]
#     # rois -> transform_to_rbbox -> roi_rbboxes
#     # roi_rbboxes + cv2_rbbox -> rbbox2delta -> delta: [dx, dy, dw, dh, da]
#     # delta + roi_rbboxes -> delta2rbbox -> pred_rbboxes: [x, y, h, w, a]
#     # pred_rbboxes -> rbbox2poly -> pred_poly
#
#     # pred_poly -> pred_mask
#     # 比较pred_mask和gt_mask之间的差异
#     W = 200
#     H = 100
#     a_r = []
#     p1 = np.array([[0, 0]]).T
#     p2 = np.array([[1, 0]]).T
#     angles = np.linspace(0, 2 * np.pi, 1000)
#
#
#
#     for (W, H) in [(200, 100), (100, 200)]:
#         ad_angle = []
#         cv_angle = []
#         for a in angles:
#             gt_poly, gt_mask = generage_data(300, 300, W, H, a)
#
#
#             cv2_rbbox = cv2_mask2rbbox(mask)
#             cv_angle.append(poly_cv[-1])
#
#             gt_polys = mask2poly([mask])[0]
#             mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
#             mask_img[mask_img > 0] = 100
#             draw_poly(mask_img, poly)
#             draw_poly(mask_img, gt_polys)
#             cv2.putText(mask_img, str('CV:%f\n' % (poly_cv[-1] / 360)), (200, 100),
#                         cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
#                         thickness=1)
#
#
#             cv2.imshow('DD', mask_img)
#             cv2.waitKey(50)
#         ad_angle = np.array(ad_angle) / (2*np.pi)
#         cv_angle = np.array(cv_angle) / (360)
#
#
#         plt.plot(angles / np.pi, ad_angle, label='AD')
#         plt.plot(angles / np.pi, cv_angle, label='CV')
#         plt.legend()
#         plt.pause(1)
#         plt.savefig('./W_%d_H_%d.png' % (W, H))
#         print('Save ./W_%d_H_%d.png' % (W, H))
#         plt.close()
