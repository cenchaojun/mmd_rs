def _get_bboxes_single(self,
                       cls_scores,
                       bbox_preds,
                       mlvl_anchors,
                       img_shape,
                       scale_factor,
                       cfg,
                       rescale=False):
    """Transform outputs for a single batch item into bbox predictions.

    Args:
        cls_scores (list[Tensor]): Box scores for each scale level
            Has shape (num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each scale
            level with shape (num_anchors * 4, H, W).
        mlvl_anchors (list[Tensor]): Box reference for each scale level
            with shape (num_total_anchors, 4).
        img_shape (tuple[int]): Shape of the input image,
            (height, width, 3).
        scale_factor (ndarray): Scale factor of the image arange as
            (w_scale, h_scale, w_scale, h_scale).
        cfg (mmcv.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.

    Returns:
        Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
            are bounding box positions (tl_x, tl_y, br_x, br_y) and the
            5-th column is a score between 0 and 1.
    """
    cfg = self.test_cfg if cfg is None else cfg
    # bboxes from different level should be independent during NMS,
    # level_ids are used as labels for batched NMS to separate them
    level_ids = []
    mlvl_scores = []
    mlvl_bbox_preds = []
    mlvl_valid_anchors = []
    for idx in range(len(cls_scores)):
        rpn_cls_score = cls_scores[idx]
        rpn_bbox_pred = bbox_preds[idx]
        assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
        rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
        if self.use_sigmoid_cls:
            rpn_cls_score = rpn_cls_score.reshape(-1)
            scores = rpn_cls_score.sigmoid()
        else:
            rpn_cls_score = rpn_cls_score.reshape(-1, 2)
            # we set FG labels to [0, num_class-1] and BG label to
            # num_class in other heads since mmdet v2.0, However we
            # keep BG label as 0 and FG label as 1 in rpn head
            scores = rpn_cls_score.softmax(dim=1)[:, 1]
        rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
        anchors = mlvl_anchors[idx]
        if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
            # sort is faster than topk
            # _, topk_inds = scores.topk(cfg.nms_pre)
            ranked_scores, rank_inds = scores.sort(descending=True)
            topk_inds = rank_inds[:cfg.nms_pre]
            scores = ranked_scores[:cfg.nms_pre]
            rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
            anchors = anchors[topk_inds, :]
        mlvl_scores.append(scores)
        mlvl_bbox_preds.append(rpn_bbox_pred)
        mlvl_valid_anchors.append(anchors)
        level_ids.append(
            scores.new_full((scores.size(0),), idx, dtype=torch.long))

    scores = torch.cat(mlvl_scores)
    anchors = torch.cat(mlvl_valid_anchors)
    rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
    proposals = self.bbox_coder.decode(
        anchors, rpn_bbox_pred, max_shape=img_shape)
    ids = torch.cat(level_ids)

    if cfg.min_bbox_size > 0:
        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]
        valid_inds = torch.nonzero(
            (w >= cfg.min_bbox_size)
            & (h >= cfg.min_bbox_size),
            as_tuple=False).squeeze()
        if valid_inds.sum().item() != len(proposals):
            proposals = proposals[valid_inds, :]
            scores = scores[valid_inds]
            ids = ids[valid_inds]

    # TODO: remove the hard coded nms type
    nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
    dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
    return dets[:cfg.nms_post]
