import numpy as np


# -------------------------
# Utilities
# -------------------------

def bbox_center_xywh(bbox):
    """
    bbox: [x, y, w, h]
    returns: [cx, cy]
    """
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)


def get_centers(bboxes):
    """
    bboxes: list of [x, y, w, h]
    returns: Nx2 array of centers
    """
    return np.array([bbox_center_xywh(b) for b in bboxes], dtype=np.float32)


# -------------------------
# 1. Center Location Error (CLE)
# -------------------------

def center_location_error(pr_boxes, gt_boxes, diag):
    pr_c = get_centers(pr_boxes)
    gt_c = get_centers(gt_boxes)

    errors = np.linalg.norm(pr_c - gt_c, axis=1)   # in pixels
    cle_pixels = float(np.mean(errors))

    cle_normalized = cle_pixels 
    return cle_normalized



# -------------------------
# 2. Normalized Center Error (NCE)
# -------------------------

def normalized_center_error(pr_boxes, gt_boxes, mode="diag"):
    """
    mode:
      - 'diag': normalize by sqrt(w^2 + h^2)
      - 'max' : normalize by max(w, h)
    """
    pr_c = get_centers(pr_boxes)
    gt_c = get_centers(gt_boxes)

    errors = np.linalg.norm(pr_c - gt_c, axis=1)

    gt_boxes = np.array(gt_boxes, dtype=np.float32)
    w = gt_boxes[:, 2]
    h = gt_boxes[:, 3]

    if mode == "diag":
        norm = np.sqrt(w ** 2 + h ** 2)
    elif mode == "max":
        norm = np.maximum(w, h)
    else:
        raise ValueError("mode must be 'diag' or 'max'")

    return float(np.mean(errors / (norm + 1e-6)))


# -------------------------
# 3. Velocity Error (VE)
# -------------------------

def velocity_error(pr_boxes, gt_boxes):
    """
    Measures how different the motion vectors are.
    Penalizes static predictions strongly.
    """
    pr_c = get_centers(pr_boxes)
    gt_c = get_centers(gt_boxes)

    if len(pr_c) < 2:
        return 0.0

    pr_v = pr_c[1:] - pr_c[:-1]
    gt_v = gt_c[1:] - gt_c[:-1]

    ve = np.linalg.norm(pr_v - gt_v, axis=1)
    return float(np.mean(ve))


# -------------------------
# 4. Motion Recall
# -------------------------

def motion_recall(pr_boxes, gt_boxes, delta=1.0):
    """
    Measures whether motion is detected when GT object is moving.
    delta: minimum pixel motion to be considered as movement.
    """
    pr_c = get_centers(pr_boxes)
    gt_c = get_centers(gt_boxes)

    if len(pr_c) < 2:
        return 0.0

    gt_motion = np.linalg.norm(gt_c[1:] - gt_c[:-1], axis=1)
    pr_motion = np.linalg.norm(pr_c[1:] - pr_c[:-1], axis=1)

    gt_moving = gt_motion > delta
    pr_moving = pr_motion > delta

    if np.sum(gt_moving) == 0:
        # No motion in GT → perfect recall by definition
        return 1.0

    recall = np.sum(gt_moving & pr_moving) / np.sum(gt_moving)
    return float(recall)


# -------------------------
# 5. Trajectory Deviation Error (TDE)
# -------------------------

def trajectory_deviation_error(pr_boxes, gt_boxes):
    """
    Mean distance between predicted and GT trajectory points.
    """
    pr_c = get_centers(pr_boxes)
    gt_c = get_centers(gt_boxes)

    errors = np.linalg.norm(pr_c - gt_c, axis=1)
    return float(np.mean(errors))
