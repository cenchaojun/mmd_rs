from ..builder import DETECTORS
from .single_stage_rs import SingleStageDetectorRS


@DETECTORS.register_module()
class RetinaNetRS(SingleStageDetectorRS):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNetRS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
