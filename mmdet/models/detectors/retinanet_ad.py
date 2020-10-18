from .single_stage_ad import SingleStageDetectorAD
from ..builder import DETECTORS


@DETECTORS.register_module
class RetinaNetAD(SingleStageDetectorAD):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetAD, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
