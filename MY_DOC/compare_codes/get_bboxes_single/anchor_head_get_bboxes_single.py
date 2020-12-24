def _get_bboxes_single(self,
                       cls_score_list,
                       bbox_pred_list,
                       mlvl_anchors,
                       img_shape,
                       scale_factor,
                       cfg,
                       rescale=False,
                       with_nms=True):
    """Transform outputs for a single batch item into bbox predictions.

    Args:
        cls_score_list (list[Tensor]): Box scores for a single scale level
            Has shape (num_anchors * num_classes, H, W).
        bbox_pred_list (list[Tensor]): Box energies / deltas for a single
            scale level with shape (num_anchors * 4, H, W).
        mlvl_anchors (list[Tensor]): Box reference for a single scale level
            with shape (num_total_anchors, 4).
        img_shape (tuple[int]): Shape of the input image,
            (height, width, 3).
        scale_factor (ndarray): Scale factor of the image arange as
            (w_scale, h_scale, w_scale, h_scale).
        cfg (mmcv.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
            are bounding box positions (tl_x, tl_y, br_x, br_y) and the
            5-th column is a score between 0 and 1.
    """
    cfg = self.test_cfg if cfg is None else cfg
    assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
    mlvl_bboxes = []
    mlvl_scores = []
    for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                             bbox_pred_list, mlvl_anchors):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        cls_score = cls_score.permute(1, 2,
                                      0).reshape(-1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            scores = cls_score.sigmoid()
        else:
            scores = cls_score.softmax(-1)
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            # Get maximum scores for foreground classes.
            if self.use_sigmoid_cls:
                max_scores, _ = scores.max(dim=1)
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                max_scores, _ = scores[:, :-1].max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            anchors = anchors[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
        bboxes = self.bbox_coder.decode(
            anchors, bbox_pred, max_shape=img_shape)
        mlvl_bboxes.append(bboxes)
        mlvl_scores.append(scores)
    mlvl_bboxes = torch.cat(mlvl_bboxes)
    if rescale:
        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
    mlvl_scores = torch.cat(mlvl_scores)
    if self.use_sigmoid_cls:
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

    if with_nms:
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
    else:
        return mlvl_bboxes, mlvl_scores