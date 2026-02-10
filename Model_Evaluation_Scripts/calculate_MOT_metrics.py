#!/usr/bin/env python3
"""
Evaluate motion consistency metrics between ground truth and predicted tracks.
Handles both single-object and multi-object tracking formats.

Input: JSON array where each element can be:
  Single-object format:
    - id, track_id, expected_output, predicted_output
  Multi-object format:
    - id, objects[], expected_output{object_1, object_2, ...}, predicted_output{object_1, object_2, ...}

Input bbox format: (x1, y1, x2, y2) - corners format

Output:
  - MCP, DRE, Robustness, NDE, MOTA, MOTP
  - Motion-aware metrics:
        CLE, NCE, VelocityError, MotionRecall, TDE
  - Separate averages for single-object, multi-object, and overall
"""

import json
import argparse
import os
import numpy as np
import cv2
import math
from process_objects import process_single_object, process_multi_object


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def evaluate(json_path,dataset_root, out_dir="results"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    ensure_dir(out_dir)

    single_results = []
    multi_results = []

    # Check if data is wrapped in "results" key
    items = data.get("results", data) if isinstance(data, dict) else data

    for item in items:
        # Read one image to get H, W
        image_rel_path = item["input_images"][0]
        image_path = os.path.join(dataset_root, image_rel_path)

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        H, W = img.shape[:2]
        diag = math.sqrt(W * W + H * H)  # normalization factor
        item_type = item.get("type", "single")

        if item_type == "single" or "track_id" in item:
            # Single-object format
            result = process_single_object(item, out_dir, diag)
            single_results.append(result)
        else:
            # Multi-object format
            results = process_multi_object(item, out_dir, diag)
            multi_results.extend(results)

    # ---------------- SINGLE OBJECT ---------------- #
    if single_results:
        mcp_s, dre_s, rob_s, nde_s, mota_s, motp_s, direction_s, speed_s = [], [], [], [], [], [], [], []
        cle_s, nce_s, ve_s, mr_s, tde_s = [], [], [], [], []

        for res in single_results:
            if res.get('MCP') is not None: mcp_s.append(res['MCP'])
            if res.get('DRE') is not None: dre_s.append(res['DRE'])
            if res.get('Robustness') is not None: rob_s.append(res['Robustness'])
            if res.get('NDE') is not None: nde_s.append(res['NDE'])
            if res.get('MOTA') is not None: mota_s.append(res['MOTA'])
            if res.get('MOTP') is not None: motp_s.append(res['MOTP'])
            if res["A_t"] is not None: direction_s.append(res['A_t'])
            if res["S_t"] is not None: speed_s.append(res["S_t"])

            # Motion-aware
            if res.get('CLE') is not None: cle_s.append(res['CLE'])
            if res.get('NCE') is not None: nce_s.append(res['NCE'])
            if res.get('VelocityError') is not None: ve_s.append(res['VelocityError'])
            if res.get('MotionRecall') is not None: mr_s.append(res['MotionRecall'])
            if res.get('TDE') is not None: tde_s.append(res['TDE'])

        print("\n" + "-" * 70)
        print("SINGLE-OBJECT AVERAGES:")
        print(f"  Avg MCP: {np.mean(mcp_s):.4f}" if mcp_s else "  Avg MCP: N/A")
        print(f"  Avg DRE: {np.mean(dre_s):.4f}" if dre_s else "  Avg DRE: N/A")
        print(f"  Avg Robustness: {np.mean(rob_s):.4f}" if rob_s else "  Avg Robustness: N/A")
        print(f"  Avg NDE: {np.mean(nde_s):.4f}" if nde_s else "  Avg NDE: N/A")
        print(f"  Avg MOTA: {np.mean(mota_s):.4f}" if mota_s else "  Avg MOTA: N/A")
        print(f"  Avg MOTP: {np.mean(motp_s):.4f}" if motp_s else "  Avg MOTP: N/A")
        print(f"  Avg Direction (A_t): {np.mean(direction_s):.4f}" if direction_s else "  Avg Direction (A_t): N/A")
        print(f"  Avg Speed (S_t): {np.mean(speed_s):.4f}" if speed_s else "  Avg Speed (S_t): N/A")

        print("  --- Motion-aware metrics ---")
        print(f"  Avg CLE: {np.mean(cle_s):.4f}" if cle_s else "  Avg CLE: N/A")
        print(f"  Avg NCE: {np.mean(nce_s):.4f}" if nce_s else "  Avg NCE: N/A")
        print(f"  Avg Velocity Error: {np.mean(ve_s):.4f}" if ve_s else "  Avg Velocity Error: N/A")
        print(f"  Avg Motion Recall: {np.mean(mr_s):.4f}" if mr_s else "  Avg Motion Recall: N/A")
        print(f"  Avg TDE: {np.mean(tde_s):.4f}" if tde_s else "  Avg TDE: N/A")
    else:
        print("\nNo single-object tracking samples found.")

    # ---------------- MULTI OBJECT ---------------- #
    if multi_results:
        mcp_m, dre_m, rob_m, nde_m, mota_m, motp_m, direction_m, speed_m = [], [], [], [], [], [], [], []
        cle_m, nce_m, ve_m, mr_m, tde_m = [], [], [], [], []

        for res in multi_results:
            if res.get('MCP') is not None: mcp_m.append(res['MCP'])
            if res.get('DRE') is not None: dre_m.append(res['DRE'])
            if res.get('Robustness') is not None: rob_m.append(res['Robustness'])
            if res.get('NDE') is not None: nde_m.append(res['NDE'])
            if res.get('MOTA') is not None: mota_m.append(res['MOTA'])
            if res.get('MOTP') is not None: motp_m.append(res['MOTP'])
            if res["A_t"] is not None: direction_m.append(res['A_t'])
            if res["S_t"] is not None: speed_m.append(res["S_t"])

            # Motion-aware
            if res.get('CLE') is not None: cle_m.append(res['CLE'])
            if res.get('NCE') is not None: nce_m.append(res['NCE'])
            if res.get('VelocityError') is not None: ve_m.append(res['VelocityError'])
            if res.get('MotionRecall') is not None: mr_m.append(res['MotionRecall'])
            if res.get('TDE') is not None: tde_m.append(res['TDE'])

        print("\n" + "-" * 70)
        print("MULTI-OBJECT AVERAGES:")
        print(f"  Avg MCP: {np.mean(mcp_m):.4f}" if mcp_m else "  Avg MCP: N/A")
        print(f"  Avg DRE: {np.mean(dre_m):.4f}" if dre_m else "  Avg DRE: N/A")
        print(f"  Avg Robustness: {np.mean(rob_m):.4f}" if rob_m else "  Avg Robustness: N/A")
        print(f"  Avg NDE: {np.mean(nde_m):.4f}" if nde_m else "  Avg NDE: N/A")
        print(f"  Avg MOTA: {np.mean(mota_m):.4f}" if mota_m else "  Avg MOTA: N/A")
        print(f"  Avg MOTP: {np.mean(motp_m):.4f}" if motp_m else "  Avg MOTP: N/A")
        print(f"  Avg Direction (A_t): {np.mean(direction_m):.4f}" if direction_m else "  Avg Direction (A_t): N/A")
        print(f"  Avg Speed (S_t): {np.mean(speed_m):.4f}" if speed_m else "  Avg Speed (S_t): N/A")

        print("  --- Motion-aware metrics ---")
        print(f"  Avg CLE: {np.mean(cle_m):.4f}" if cle_m else "  Avg CLE: N/A")
        print(f"  Avg NCE: {np.mean(nce_m):.4f}" if nce_m else "  Avg NCE: N/A")
        print(f"  Avg Velocity Error: {np.mean(ve_m):.4f}" if ve_m else "  Avg Velocity Error: N/A")
        print(f"  Avg Motion Recall: {np.mean(mr_m):.4f}" if mr_m else "  Avg Motion Recall: N/A")
        print(f"  Avg TDE: {np.mean(tde_m):.4f}" if tde_m else "  Avg TDE: N/A")
    else:
        print("\nNo multi-object tracking samples found.")

    # ---------------- OVERALL ---------------- #
    print("\n" + "=" * 70)
    print("OVERALL AVERAGES (ALL TRACKS)")
    print("=" * 70)

    all_mcp = (mcp_s if single_results else []) + (mcp_m if multi_results else [])
    all_dre = (dre_s if single_results else []) + (dre_m if multi_results else [])
    all_rob = (rob_s if single_results else []) + (rob_m if multi_results else [])
    all_nde = (nde_s if single_results else []) + (nde_m if multi_results else [])
    all_mota = (mota_s if single_results else []) + (mota_m if multi_results else [])
    all_motp = (motp_s if single_results else []) + (motp_m if multi_results else [])
    all_direction = (direction_s if single_results else []) + (direction_m if multi_results else [])
    all_speed = (speed_s if single_results else []) + (speed_m if multi_results else [])

    all_cle = (cle_s if single_results else []) + (cle_m if multi_results else [])
    all_nce = (nce_s if single_results else []) + (nce_m if multi_results else [])
    all_ve = (ve_s if single_results else []) + (ve_m if multi_results else [])
    all_mr = (mr_s if single_results else []) + (mr_m if multi_results else [])
    all_tde = (tde_s if single_results else []) + (tde_m if multi_results else [])

    print(f"  Avg MCP: {np.mean(all_mcp):.4f}" if all_mcp else "  Avg MCP: N/A")
    print(f"  Avg DRE: {np.mean(all_dre):.4f}" if all_dre else "  Avg DRE: N/A")
    print(f"  Avg Robustness: {np.mean(all_rob):.4f}" if all_rob else "  Avg Robustness: N/A")
    print(f"  Avg NDE: {np.mean(all_nde):.4f}" if all_nde else "  Avg NDE: N/A")
    print(f"  Avg MOTA: {np.mean(all_mota):.4f}" if all_mota else "  Avg MOTA: N/A")
    print(f"  Avg MOTP: {np.mean(all_motp):.4f}" if all_motp else "  Avg MOTP: N/A")
    print(f"  Avg Direction (A_t): {np.mean(all_direction):.4f}" if all_direction else "  Avg Direction (A_t): N/A")
    print(f"  Avg Speed (S_t): {np.mean(all_speed):.4f}" if all_speed else "  Avg Speed (S_t): N/A")

    print("  --- Motion-aware metrics ---")
    print(f"  Avg CLE: {np.mean(all_cle):.4f}" if all_cle else "  Avg CLE: N/A")
    print(f"  Avg NCE: {np.mean(all_nce):.4f}" if all_nce else "  Avg NCE: N/A")
    print(f"  Avg Velocity Error: {np.mean(all_ve):.4f}" if all_ve else "  Avg Velocity Error: N/A")
    print(f"  Avg Motion Recall: {np.mean(all_mr):.4f}" if all_mr else "  Avg Motion Recall: N/A")
    print(f"  Avg TDE: {np.mean(all_tde):.4f}" if all_tde else "  Avg TDE: N/A")

    print("\n" + "=" * 70)
    print(f"Total single-object samples: {len(single_results)}")
    print(f"Total multi-object samples: {len(multi_results)}")
    print(f"Total tracks evaluated: {len(single_results) + len(multi_results)}")
    print(f"\nResults directory: {os.path.abspath(out_dir)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        default="/home/gaash/Tawheed/Reasoning/Output_json/visionreasoner_bbox_eval_after_train.json",
        help="JSON file with expected_output and predicted_output"
    )
    parser.add_argument(
        "--dataset-root",
        default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/test",
        help="Output folder"
    )
    parser.add_argument(
        "--out",
        default="/home/gaash/Tawheed/Reasoning/Ealuation_logs/visionreasoner_bbox_eval_after_train",
        help="Output folder"
    )

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    evaluate(args.input_json, args.dataset_root, out_dir=args.out)
