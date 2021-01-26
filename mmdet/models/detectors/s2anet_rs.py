from .single_stage_poly_rs import SingleStageDetectorPolyRS
from ..builder import DETECTORS


@DETECTORS.register_module
class S2ANetDetector(SingleStageDetectorPolyRS):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(S2ANetDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                             test_cfg, pretrained)
