import numpy as np
from metrics.mcp import compute_mcp
from metrics.dre import drift_rate_error, dre_vot
from metrics.rob import robustness, robustness_vot
from metrics.nde import normalized_displacement_error
from metrics.mota import compute_mota_motp
from metrics.motion import (
    center_location_error,
    normalized_center_error,
    velocity_error,
    motion_recall,
    trajectory_deviation_error,
)



def convert_xywh_to_center_format(records_by_frame, tid):
    frames, bboxes = [], []
    for f in sorted(records_by_frame.keys()):
        for rec in records_by_frame[f]:
            if rec[0] == tid:
                _, x, y, w, h = rec
                cx = x + w / 2.0
                cy = y + h / 2.0
                frames.append(f)
                bboxes.append([cx, cy, w, h])
    return np.array(frames), np.array(bboxes)


def extract_xywh_bboxes(records_by_frame, tid):
    frames, bboxes = [], []
    for f in sorted(records_by_frame.keys()):
        for rec in records_by_frame[f]:
            if rec[0] == tid:
                _, x, y, w, h = rec
                frames.append(f)
                bboxes.append([x, y, w, h])
    return np.array(frames), bboxes


def compute_all_metrics(gt_by_frame, pr_by_frame, tid, diag):
    gt_frames, gt_center = convert_xywh_to_center_format(gt_by_frame, tid)
    pr_frames, pr_center = convert_xywh_to_center_format(pr_by_frame, tid)

    _, gt_boxes = extract_xywh_bboxes(gt_by_frame, tid)
    _, pr_boxes = extract_xywh_bboxes(pr_by_frame, tid)

    MCP = DRE = Rob = NDE = MOTA = MOTP = direction = speed = None
    CLE  = VE = MR  = None

    if len(gt_boxes) > 0 and len(pr_boxes) > 0:
        if len(gt_boxes) == len(pr_boxes):

            # Your existing metrics
            MCP, direction, speed  = compute_mcp(gt_center, pr_center)
            DRE = dre_vot(pr_boxes, gt_boxes)
            Rob = robustness_vot(pr_boxes, gt_boxes)
            NDE = normalized_displacement_error(pr_boxes, gt_boxes)
            MOTA, MOTP = compute_mota_motp(pr_boxes, gt_boxes)

            # Motion-aware metrics (new)
            CLE = center_location_error(pr_boxes, gt_boxes, diag)
            VE = velocity_error(pr_boxes, gt_boxes)
            MR = motion_recall(pr_boxes, gt_boxes, delta=1.0)

    return {
        'MCP': MCP,
        'DRE': DRE,
        'Robustness': Rob,
        'NDE': NDE,
        'MOTA': MOTA,
        'MOTP': MOTP,
        'A_t': direction,
        'S_t': speed,

        # New metrics that punish static predictions
        'CLE': CLE,
        'VelocityError': VE,
        'MotionRecall': MR,
    }
