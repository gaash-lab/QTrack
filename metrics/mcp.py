import numpy as np


# def compute_mcp(gt_bboxes, pred_bboxes):
#     gt_bboxes = np.asarray(gt_bboxes, dtype=float)
#     pred_bboxes = np.asarray(pred_bboxes, dtype=float)

#     if gt_bboxes.shape[0] < 2:
#         return None

#     mcp_vals = []

#     for t in range(1, gt_bboxes.shape[0]):
#         dg = gt_bboxes[t, :2] - gt_bboxes[t - 1, :2]
#         dp = pred_bboxes[t, :2] - pred_bboxes[t - 1, :2]

#         ng, np_ = np.linalg.norm(dg), np.linalg.norm(dp)
#         if ng == 0 or np_ == 0:
#             continue

#         direction = max(0.0, np.dot(dp, dg) / (ng * np_))
#         speed = np.exp(-abs(np_ - ng) / ng)
#         mcp_vals.append(direction * speed)

#     return float(np.mean(mcp_vals)) if mcp_vals else 0.0, direction, speed


def compute_mcp(gt_bboxes, pred_bboxes, alpha=0.9, eps=1e-6):
    """
    Compute Motion Consistency Precision (LMCP) between GT and predicted bounding boxes.
    
    Parameters:
        gt_bboxes: list/array of shape (T, 4) [x, y, w, h]
        pred_bboxes: list/array of shape (T, 4) [x, y, w, h]
        alpha: looseness factor for speed consistency (default 0.5)
        beta: small motion stabilization constant (default 3.0 pixels)
        eps: small constant to avoid divide by zero
    
    Returns:
        LMCP score in [0,1]
    """
    gt_bboxes = np.asarray(gt_bboxes, dtype=float)
    pred_bboxes = np.asarray(pred_bboxes, dtype=float)

    if gt_bboxes.shape[0] < 2:
        return None, None, None

    lmcp_vals = []
    direction, speed = 0, 0

    for t in range(1, gt_bboxes.shape[0]):
        # motion vectors
        dg = gt_bboxes[t, :2] - gt_bboxes[t - 1, :2]
        dp = pred_bboxes[t, :2] - pred_bboxes[t - 1, :2]

        ng, np_ = np.linalg.norm(dg), np.linalg.norm(dp)
        if ng < eps:
            continue  # skip frames with negligible GT motion

        # Soft direction consistency: scale [-1,1] to [0,1]
        cos_theta = np.dot(dp, dg) / (np_ * ng + eps)
        direction = (1 + cos_theta) / 2

        # Gaussian-style speed consistency
        speed = np.exp(-((np_ - ng)**2) / (2 * (alpha * ng)**2 + eps))

        # Small motion weight to stabilize
        # weight = np.tanh(ng / beta)

        lmcp_vals.append(direction * speed)

    return float(np.mean(lmcp_vals)) if lmcp_vals else 0.0, direction, speed
