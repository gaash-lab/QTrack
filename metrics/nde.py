import numpy as np
from metrics.iou import box_center, box_size


def normalized_displacement_error(pred_boxes, gt_boxes):
    assert len(pred_boxes) == len(gt_boxes)

    vals = []
    for p, g in zip(pred_boxes, gt_boxes):
        if p is None:
            continue
        d = np.linalg.norm(box_center(p) - box_center(g))
        w, h = box_size(g)
        diag = np.sqrt(w**2 + h**2)
        if diag > 0:
            vals.append(d / diag)

    return float(np.mean(vals)) if vals else 0.0
