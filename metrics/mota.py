import numpy as np
from metrics.iou import iou


def compute_mota_motp(pred_boxes, gt_boxes, iou_threshold=0.8):
    assert len(pred_boxes) == len(gt_boxes)

    FN = FP = IDSW = 0
    total_gt = 0
    matched_ious = []

    for p, g in zip(pred_boxes, gt_boxes):
        total_gt += 1

        if p is None:
            FN += 1
            continue

        iou_val = iou(p, g)
        if iou_val >= iou_threshold:
            matched_ious.append(iou_val)
        else:
            FN += 1
            FP += 1

    MOTA = 0.0 if total_gt == 0 else 1.0 - (FN + FP + IDSW) / total_gt
    MOTP = float(np.mean(matched_ious)) if matched_ious else 0.0

    return MOTA, MOTP
