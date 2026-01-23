from metrics.iou import iou


def robustness(pred_boxes, gt_boxes):
    assert len(pred_boxes) == len(gt_boxes)

    success = 0
    total = len(gt_boxes)

    for p, g in zip(pred_boxes, gt_boxes):
        if p is not None and iou(p, g) > 0:
            success += 1

    return success / total if total > 0 else 0.0


def robustness_vot(pred_boxes, gt_boxes, iou_threshold=0.0):
    T = F = M = 0

    for p, g in zip(pred_boxes, gt_boxes):
        if g is None:
            continue

        if p is None:
            M += 1
        else:
            if iou(p, g) > iou_threshold:
                T += 1
            else:
                F += 1

    denom = T + F + M
    return T / denom if denom > 0 else 0.0
