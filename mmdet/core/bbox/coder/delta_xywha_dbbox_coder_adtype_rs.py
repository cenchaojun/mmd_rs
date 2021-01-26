import numpy as np
import torch
import math
import cv2
import copy

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYWHARbboxCoderADTypeRS(BaseBBoxCoder):
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.),
                 use_mod=False):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.use_mod = use_mod

    def encode(self, rbboxes, gt_rbboxes):
        """

        :param rbboxes:      anchor, N x 5(xc, yc, w, h, angle)
        :param gt_rbboxes:   gt,     N x 5(xc, yc, w, h, angle)
        :return:
        """
        assert rbboxes.size(0) == gt_rbboxes.size(0)
        assert rbboxes.size(-1) == 5  \
               and gt_rbboxes.size(-1) == 5
        encoded_bboxes = rbbox2delta(rbboxes, gt_rbboxes,
                                     self.means, self.stds)
        return encoded_bboxes

    def decode(self,
               rbboxes,
               deltas,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """

        :param rbboxes:  rotated anchor, N x 5(xc, yc, w, h, angle)
        :param deltas:   predict deltas, N x 5(dxc, dyc, dw, dh, dangle)
        :param max_shape:
        :param wh_ratio_clip:
        :return:
        """

        assert deltas.size(0) == rbboxes.size(0)
        assert rbboxes.size(-1) == 5  \
               and deltas.size(-1) == 5

        decoded_bboxes = delta2rbbox(rbboxes, deltas,
                                     self.means, self.stds,
                                     max_shape, wh_ratio_clip)

        return decoded_bboxes

    def to_obb(self, org_bboxes, type='hbb'):
        if type == 'hbb':
            return hbb2obb(org_bboxes)
        if type == 'mask':
            return gt_mask_bp_obbs(org_bboxes)
        if type == 'poly':
            raise Exception('Not finished yet ')
            return polygonToRotRectangle_batch(org_bboxes)


def hbb2obb(hbboxes):
    """
    fix a bug
    :param hbboxes: shape (n, 4) (xmin, ymin, xmax, ymax)
    :return: dbboxes: shape (n, 5) (x_ctr, y_ctr, w, h, angle)
    """
    num_boxes = hbboxes.size(0)
    ex_heights = hbboxes[..., 2] - hbboxes[..., 0] + 1.0
    ex_widths = hbboxes[..., 3] - hbboxes[..., 1] + 1.0
    ex_ctr_x = hbboxes[..., 0] + 0.5 * (ex_heights - 1.0)
    ex_ctr_y = hbboxes[..., 1] + 0.5 * (ex_widths - 1.0)
    c_bboxes = torch.cat(
        (ex_ctr_x.unsqueeze(1), ex_ctr_y.unsqueeze(1),
         ex_widths.unsqueeze(1), ex_heights.unsqueeze(1)), 1)
    initial_angles = -c_bboxes.new_ones((num_boxes, 1)) * np.pi / 2
    # initial_angles = -torch.ones((num_boxes, 1)) * np.pi/2
    dbboxes = torch.cat((c_bboxes, initial_angles), 1)

    return dbboxes


def TuplePoly2Poly(poly):
    outpoly = [poly[0][0], poly[0][1],
                       poly[1][0], poly[1][1],
                       poly[2][0], poly[2][1],
                       poly[3][0], poly[3][1]
                       ]
    return outpoly

def Tuplelist2Polylist(tuple_poly_list):
    polys = map(TuplePoly2Poly, tuple_poly_list)

    return list(polys)

def mask2poly_single(binary_mask):
    """

    :param binary_mask:
    :return:
    """
    try:
        contours, hierarchy = cv2.findContours(binary_mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=len)
        rect = cv2.minAreaRect(max_contour)
        poly = cv2.boxPoints(rect)
    except:
        import pdb
        pdb.set_trace()
    return poly

def mask2poly(binary_mask_list):
    polys = map(mask2poly_single, binary_mask_list)
    # polys = np.stack(polys
    return list(polys)

def gt_mask_bp_obbs(gt_masks, with_module=True):

    # trans gt_masks to gt_obbs
    if len(gt_masks) >= 1:
        gt_polys = mask2poly(gt_masks)
        gt_bp_polys = get_best_begin_point(gt_polys)
        gt_obbs = polygonToRotRectangle_batch(gt_bp_polys, with_module)
    else:
        gt_obbs = []
        print('gt_masks shape: %s, no mask' % str(gt_masks.shape))

    return gt_obbs

# def gt_mask_bp_obbs_list(gt_masks_list):
#
#     gt_obbs_list = map(gt_mask_bp_obbs, gt_masks_list)
#
#     return list(gt_obbs_list)

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
    return combinate[force_flag]

def get_best_begin_point_warp_single(coordinate):

    return TuplePoly2Poly(get_best_begin_point_single(coordinate))

def get_best_begin_point(coordinate_list):
    best_coordinate_list = map(get_best_begin_point_warp_single,
                               coordinate_list)
    best_coordinate_list = np.stack(list(best_coordinate_list))
    return best_coordinate_list


def polygonToRotRectangle_batch(bbox, with_module=True):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]
    """
    # print('bbox: ', bbox)
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(-1, 2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox: ', bbox)
    angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((bbox.shape[0], 2, 1))
    for i in range(4):
        center[:, 0, 0] += bbox[:, 0,i]
        center[:, 1, 0] += bbox[:, 1,i]

    center = np.array(center,dtype=np.float32)/4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((2, 1, 0)),bbox-center)


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
    if with_module:
        angle = angle[:, np.newaxis] % ( 2 * np.pi)
    else:
        angle = angle[:, np.newaxis]
    dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    return dboxes



def rbbox2delta(proposals, gt, means = [0, 0, 0, 0, 0], stds=[1, 1, 1, 1, 1]):
    """
    :param proposals: (x_ctr, y_ctr, w, h, angle)
            shape (n, 5)
    :param gt: (x_ctr, y_ctr, w, h, angle)
    :param means:
    :param stds:
    :return: encoded targets: shape (n, 5)
    """
    proposals = proposals.float()
    gt = gt.float()
    gt_widths = gt[..., 2]
    gt_heights = gt[..., 3]
    gt_angle = gt[..., 4]

    proposals_widths = proposals[..., 2]
    proposals_heights = proposals[..., 3]
    proposals_angle = proposals[..., 4]

    coord = gt[..., 0:2] - proposals[..., 0:2]
    dx = (torch.cos(proposals[..., 4]) * coord[..., 0] +
          torch.sin(proposals[..., 4]) * coord[..., 1]) / proposals_widths
    dy = (-torch.sin(proposals[..., 4]) * coord[..., 0] +
          torch.cos(proposals[..., 4]) * coord[..., 1]) / proposals_heights
    dw = torch.log(gt_widths / proposals_widths)
    dh = torch.log(gt_heights / proposals_heights)
    dangle = (gt_angle - proposals_angle) % (2 * math.pi) / (2 * math.pi)
    deltas = torch.stack((dx, dy, dw, dh, dangle), -1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    # TODO: expand bbox regression
    return deltas



def delta2rbbox(Rrois,
                deltas,
                means=[0, 0, 0, 0, 0],
                stds=[1, 1, 1, 1, 1],
                max_shape=None,
                wh_ratio_clip=16 / 1000):
    """

    :param Rrois: (cx, cy, w, h, theta)
    :param deltas: (dx, dy, dw, dh, dtheta)
    :param means:
    :param stds:
    :param max_shape:
    :param wh_ratio_clip:
    :return:
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    dangle = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    Rroi_x = (Rrois[:, 0]).unsqueeze(1).expand_as(dx)
    Rroi_y = (Rrois[:, 1]).unsqueeze(1).expand_as(dy)
    Rroi_w = (Rrois[:, 2]).unsqueeze(1).expand_as(dw)
    Rroi_h = (Rrois[:, 3]).unsqueeze(1).expand_as(dh)
    Rroi_angle = (Rrois[:, 4]).unsqueeze(1).expand_as(dangle)
    # import pdb
    # pdb.set_trace()
    gx = dx * Rroi_w * torch.cos(Rroi_angle) \
         - dy * Rroi_h * torch.sin(Rroi_angle) + Rroi_x
    gy = dx * Rroi_w * torch.sin(Rroi_angle) \
         + dy * Rroi_h * torch.cos(Rroi_angle) + Rroi_y
    gw = Rroi_w * dw.exp()
    gh = Rroi_h * dh.exp()

    # TODO: check the hard code
    gangle = (2 * np.pi) * dangle + Rroi_angle
    gangle = gangle % ( 2 * np.pi)

    if max_shape is not None:
        pass

    bboxes = torch.stack([gx, gy, gw, gh, gangle], dim=-1).view_as(deltas)
    return bboxes
