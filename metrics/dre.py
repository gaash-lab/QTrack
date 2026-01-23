from metrics.iou import iou


def drift_rate_error(pred_boxes, gt_boxes, iou_threshold=0.9):
    """
    Drift-Rate Error (DRE)

    pred_boxes: list of predicted boxes per frame (or None if absent)
    gt_boxes: list of ground-truth boxes per frame
    iou_threshold: IoU threshold below which prediction is considered drift

    Returns:
        DRE value in [0, 1]
    """
    assert len(pred_boxes) == len(gt_boxes)

    drift_frames = 0
    valid_frames = 0

    for pred, gt in zip(pred_boxes, gt_boxes):
        if pred is None:
            continue  # model says object absent → not drift
        valid_frames += 1
        iou_val = iou(pred, gt)
        if 0 < iou_val < iou_threshold:
            drift_frames += 1

    if valid_frames == 0:
        return 0.0

    return drift_frames / valid_frames


def dre_vot(pred_boxes, gt_boxes, iou_threshold=0.0):
    T = F = M = 0

    for p, g in zip(pred_boxes, gt_boxes):
        if g is None:
            continue  # GT absent → ignored

        if p is None:
            M += 1
        else:
            if iou(p, g) > iou_threshold:
                T += 1
            else:
                F += 1

    denom = T + F + M
    return F / denom if denom > 0 else 0.0
