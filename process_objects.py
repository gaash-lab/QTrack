import os
import numpy as np
from collections import defaultdict
from compute_all_metrics import compute_all_metrics


def write_mot_txt(records, out_path):
    """Optional: writes per-sequence results for inspection."""
    with open(out_path, 'w') as f:
        for (frame, tid, x, y, w, h, score) in sorted(records, key=lambda r: (r[0], r[1])):
            f.write(f"{frame},{tid},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.3f},-1,-1,-1\n")


def process_single_object(item, out_dir, diag):
    """Process single-object tracking format. Input bbox format: (x, y, w, h)"""
    seq_id = item["id"]
    tid = int(item.get("track_id", 1))

    gt_by_frame, pr_by_frame = defaultdict(list), defaultdict(list)
    gt_recs, pr_recs = [], []

    for e in item.get("expected_output", []):
        f = int(e["frame"])
        x, y, w, h = e["bbox"]
        gt_recs.append((f, tid, x, y, w, h, 1.0))
        gt_by_frame[f].append((tid, x, y, w, h))

    for p in item.get("predicted_output", []):
        f = int(p["frame"])
        if len(p["bbox"]) != 4:
            continue
        x, y, w, h = p["bbox"]
        s = float(p.get("score", 1.0))
        pr_recs.append((f, tid, x, y, w, h, s))
        pr_by_frame[f].append((tid, x, y, w, h))

    write_mot_txt(gt_recs, os.path.join(out_dir, f"{seq_id}_gt.txt"))
    write_mot_txt(pr_recs, os.path.join(out_dir, f"{seq_id}_pred.txt"))

    metrics = compute_all_metrics(gt_by_frame, pr_by_frame, tid, diag)

    return {
        'seq_id': seq_id,
        'track_id': tid,
        **metrics
    }


def process_multi_object(item, out_dir, diag):
    """Process multi-object tracking format. Input bbox format: (x, y, w, h)"""
    seq_id = item["id"]
    results = []

    objects_info = item.get("objects", [])
    expected_output = item.get("expected_output", {})
    predicted_output = item.get("predicted_output", {})

    for obj_info in objects_info:
        obj_id = obj_info["object_id"]
        tid = int(obj_info["track_id"])
        obj_key = f"object_{obj_id}"

        gt_by_frame, pr_by_frame = defaultdict(list), defaultdict(list)
        gt_recs, pr_recs = [], []

        if obj_key in expected_output:
            for e in expected_output[obj_key].get("trajectory", []):
                f = int(e["frame"])
                x, y, w, h = e["bbox"]
                gt_recs.append((f, tid, x, y, w, h, 1.0))
                gt_by_frame[f].append((tid, x, y, w, h))

        if obj_key in predicted_output:
            for p in predicted_output[obj_key].get("trajectory", []):
                f = int(p["frame"])
                if len(p["bbox"]) != 4:
                    continue
                x, y, w, h = p["bbox"]
                s = float(p.get("score", 1.0))
                pr_recs.append((f, tid, x, y, w, h, s))
                pr_by_frame[f].append((tid, x, y, w, h))

        write_mot_txt(gt_recs, os.path.join(out_dir, f"{seq_id}_obj{obj_id}_gt.txt"))
        write_mot_txt(pr_recs, os.path.join(out_dir, f"{seq_id}_obj{obj_id}_pred.txt"))

        metrics = compute_all_metrics(gt_by_frame, pr_by_frame, tid, diag)

        results.append({
            'seq_id': f"{seq_id}_obj{obj_id}",
            'track_id': tid,
            'object_id': obj_id,
            **metrics
        })

    return results
