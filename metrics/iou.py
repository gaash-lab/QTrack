import numpy as np


def iou(box_a, box_b):
    """
    box format: [x, y, w, h]
    """
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    xa2, ya2 = xa + wa, ya + ha
    xb2, yb2 = xb + wb, yb + hb

    inter_x1 = max(xa, xb)
    inter_y1 = max(ya, yb)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = wa * ha
    area_b = wb * hb

    union = area_a + area_b - inter_area
    return 0.0 if union == 0 else inter_area / union


def box_center(box):
    x, y, w, h = box
    return np.array([x + w / 2.0, y + h / 2.0])


def box_size(box):
    _, _, w, h = box
    return w, h
