# This file is added for back-compatibility. Thus, downstream codebase
# could still use and import mmdet.ops.

# yapf: disable
from mmcv.ops import (Conv2d, ConvTranspose2d,
                      CornerPool, RoIAlign, RoIPool, SAConv2d,
                      SigmoidFocalLoss, SimpleRoIAlign, batched_nms,deform_conv,
                      get_compiler_version,
                      get_compiling_cuda_version, modulated_deform_conv,
                      nms_match, point_sample, rel_roi_point_to_rel_img_point,
                      roi_align, roi_pool, sigmoid_focal_loss, soft_nms)
from mmcv.ops.nms import nms

from mmcv.cnn import (ContextBlock, Conv2d,  GeneralizedAttention, Linear,
                      build_plugin_layer, conv_ws_2d, MaxPool2d)



# yapf: enable

__all__ = [
    'nms', 'soft_nms', 'RoIAlign', 'roi_align', 'RoIPool', 'roi_pool',
    'deform_conv', 'modulated_deform_conv',
    'SigmoidFocalLoss', 'sigmoid_focal_loss',
    'ContextBlock', 'GeneralizedAttention',
    'get_compiler_version', 'get_compiling_cuda_version',
    'conv_ws_2d', 'build_plugin_layer', 'batched_nms', 'Conv2d',
    'ConvTranspose2d', 'MaxPool2d', 'Linear', 'nms_match', 'CornerPool',
    'point_sample', 'rel_roi_point_to_rel_img_point', 'SimpleRoIAlign',
    'SAConv2d'
]

####################################################################
from .sigmoid_focal_loss_old import SigmoidFocalLossOld, sigmoid_focal_loss_old

from .roi_align_rotated import RoIAlignRotated
from .psroi_align_rotated import PSRoIAlignRotated


__all__.extend(['SigmoidFocalLossOld',
                'sigmoid_focal_loss_old',
                'RoIAlignRotated',
                'PSRoIAlignRotated'
                ])



