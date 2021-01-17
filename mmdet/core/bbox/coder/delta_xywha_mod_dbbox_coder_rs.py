import numpy as np
import torch

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder


@BBOX_CODERS.register_module()
class DeltaXYWHARbboxCoderRS(BaseBBoxCoder):
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.),
                 use_mod=False):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.use_mod = use_mod

    def encode(self, bboxes, gt_rbboxes):
        """
        :param bboxes:     anchor,
        n x 4 or n x 5, [x1, y1, x2, y2] or [x, y, w, h, a]
        :param gt_rbboxes:
        n x 5, [x, y, w, h, a]
        :return:
        """
        assert bboxes.size(0) == gt_rbboxes.size(0)
        assert bboxes.size(-1) in [4, 5] \
               and gt_rbboxes.size(-1) == 5
        rbboxes = formulate_rbbox(bboxes)
        encoded_bboxes = rbbox2delta(rbboxes, gt_rbboxes, self.means, self.stds, self.use_mod)
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_rbboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            boxes (torch.Tensor): Basic boxes.
            pred_rbboxes (torch.Tensor): Encoded boxes with shape
            max_shape (tuple[int], optional): Maximum shape of boxes.
                Defaults to None.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_rbboxes.size(0) == bboxes.size(0)
        rbboxes = formulate_rbbox(bboxes)

        decoded_bboxes = delta2rbbox(rbboxes, pred_rbboxes, self.means, self.stds,
                                     max_shape, wh_ratio_clip,
                                     self.use_mod)

        return decoded_bboxes

def formulate_rbbox(bboxes):
    # https://editor.csdn.net/md/?articleId=108725272
    bbox_dim =  bboxes.size(-1)
    assert bbox_dim in [4, 5, 8]
    if bbox_dim == 4:
        x_c = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        y_c = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        w = bboxes[..., 2] - bboxes[..., 0] + 1.0
        h = bboxes[..., 3] - bboxes[..., 1] + 1.0
        a = bboxes.new_zeros(len(bboxes))
        return torch.stack([x_c, y_c, w, h, a], dim=-1)
    else:
        return bboxes




def rbbox2delta(proposals, gt,
                means=(0., 0., 0., 0., 0.),
                stds=(1., 1., 1., 1., 1.),
                use_mod=False):
    """

    :param proposals: anchor, [x, y, h, w, a]
    :param gt:                [x, y, h, w, a], a：弧度制
    :param means:
    :param stds:
    :return:
    """
    assert proposals.size() == gt.size()

    if use_mod:
        inds = gt[..., 4] < -np.pi/4
        gt[inds, 2], gt[inds, 3] = gt[inds, 3], gt[inds, 2]
        gt[inds, 4] = -np.pi/2 - gt[inds, 4]

    dx = (gt[..., 0] - proposals[..., 0]) / proposals[..., 2]
    dy = (gt[..., 1] - proposals[..., 1]) / proposals[..., 3]
    dw = torch.log(gt[..., 2] / proposals[..., 2])
    dh = torch.log(gt[..., 3] / proposals[..., 3])
    da = gt[..., 4] - proposals[..., 4]
    deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
    ####################################################

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2rbbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
                use_mod=False):
    """
    :param rois:
    (N, 5). N = num_anchors * W * H
    :param deltas:
    (N, 5 * num_classes).N = num_anchors * W * H
    :param means:
    :param stds:
    :param max_shape:      Maximum bounds for boxes. specifies (H, W)
    :param wh_ratio_clip:  Maximum aspect ratio for boxes.
    :return:
    """
    means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
    stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::5]
    dy = denorm_deltas[:, 1::5]
    dw = denorm_deltas[:, 2::5]
    dh = denorm_deltas[:, 3::5]
    da = denorm_deltas[:, 4::5]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px = rois[:, 0].unsqueeze(1).expand_as(dx)
    py = rois[:, 1].unsqueeze(1).expand_as(dy)
    pw = rois[:, 2].unsqueeze(1).expand_as(dw)
    ph = rois[:, 3].unsqueeze(1).expand_as(dh)
    pa = rois[:, 4].unsqueeze(1).expand_as(da)

    gx = px + pw * dx
    gy = py + ph * dy
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    if use_mod:
        pass
        # ga = (da + pa) % (np.pi / 2)
        # ga[ga < -np.pi / 4
        # gt[inds, 2], gt[inds, 3] = gt[inds, 3], gt[inds, 2]
        # gt[inds, 4] = -np.pi / 2 - gt[inds, 4]
    else:
        ga = (da + pa) % (2 * np.pi)

    # gx: N x num_classes
    # torch.stack([gx, gy, gw, gh, ga], dim=-1): N x num_classes x 5
    # view(N x num_classes, 5): [[gx1, gy1, gw1, gh1, ga1, gx2, gy2, ...], ...]
    bboxes = torch.stack([gx, gy, gw, gh, ga], dim=-1).view(deltas.size())
    return bboxes
