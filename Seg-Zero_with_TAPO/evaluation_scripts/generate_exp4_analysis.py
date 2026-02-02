#!/usr/bin/env python3
"""
generate_exp4_analysis.py (Fixed Index Error)
- Added safety check for empty prediction lists.
- Generates all plots: Loss, Robustness, Success Curve, Waterfall.
"""
import argparse
import os
import json
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Professional Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk", font_scale=1.0)
COLORS = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"] 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_json", required=True)
    parser.add_argument("--exp4_json", required=True)
    parser.add_argument("--log_files", nargs='+', required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--last_n_epochs", type=int, default=10)
    return parser.parse_args()

# ---------------------------------------------------------
# 1. 4-WAY LOG PARSING
# ---------------------------------------------------------
def parse_log(entry, last_n):
    if ":" not in entry: return None
    name, path_pattern = entry.split(":", 1)
    
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime)
    if not files: return None
    path = files[-1]

    epochs, losses = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = re.search(r"(?:Avg Loss:|mean_loss=)\s*([\d\.]+)", line)
            if m:
                losses.append(float(m.group(1)))
                epochs.append(len(losses))

    if not losses: return None

    if len(losses) > last_n:
        return {"name": name, "x": list(range(1, last_n+1)), "y": losses[-last_n:]}
    return {"name": name, "x": list(range(1, len(losses)+1)), "y": losses}

def plot_training_loss(log_args, last_n, out_dir):
    data = []
    for entry in log_args:
        res = parse_log(entry, last_n)
        if res: data.append(res)
    
    if not data: return

    plt.figure(figsize=(14, 8))
    finals = []
    
    for i, d in enumerate(data):
        c = COLORS[i % len(COLORS)]
        plt.plot(d['x'], d['y'], marker='o', markersize=8, linewidth=3, label=d['name'], color=c, alpha=0.9)
        finals.append((d['name'], d['y'][-1], d['x'][-1]))

    if len(data) >= 2:
        try:
            x_grid = data[0]['x']
            y_arrays = [np.array(d['y']) for d in data]
            y_min = np.min(y_arrays, axis=0)
            y_max = np.max(y_arrays, axis=0)
            plt.fill_between(x_grid, y_min, y_max, color='gray', alpha=0.1, label='Optimization Gap')
        except: pass

    plt.title("Training Convergence", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Average Loss", fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, frameon=True, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "viz_training_loss_4way.png"), dpi=300)
    print("Saved Training Loss Plot.")

# ---------------------------------------------------------
# 2. ROBUSTNESS METRICS
# ---------------------------------------------------------
def get_metrics(pred, gt):
    if not pred or not gt: return 1.0, 0.0
    cx_p, cy_p = (pred[0]+pred[2])/2, (pred[1]+pred[3])/2
    cx_g, cy_g = (gt[0]+gt[2])/2, (gt[1]+gt[3])/2
    dist = np.sqrt((cx_p-cx_g)**2 + (cy_p-cy_g)**2)
    area_p = max(0, pred[2]-pred[0]) * max(0, pred[3]-pred[1])
    area_g = max(0, gt[2]-gt[0]) * max(0, gt[3]-gt[1])
    ratio = area_p / area_g if area_g > 0 else 0.0
    return dist, ratio

def plot_robustness(df, out_dir):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["Base Dist"], color=COLORS[0], label="Baseline", fill=True, alpha=0.1, linewidth=2)
    sns.kdeplot(df["Exp4 Dist"], color=COLORS[3], label="Exp4 (Augment)", fill=True, alpha=0.1, linewidth=2)
    plt.xlim(0, 0.5)
    plt.title("Center Precision (Lower is Better)")
    plt.xlabel("Center Distance Error")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "viz_center_error.png"), dpi=300)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["Base Area"], color=COLORS[0], label="Baseline", fill=True, alpha=0.1, linewidth=2)
    sns.kdeplot(df["Exp4 Area"], color=COLORS[3], label="Exp4 (Augment)", fill=True, alpha=0.1, linewidth=2)
    plt.axvline(1.0, color='black', linestyle='--', label="Perfect Scale")
    plt.xlim(0, 3.0)
    plt.title("Scale Consistency")
    plt.xlabel("Predicted Area / GT Area")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "viz_area_ratio.png"), dpi=300)

# ---------------------------------------------------------
# 3. CREATIVE PLOTS
# ---------------------------------------------------------
def run_eval_plots(base_path, exp4_path, out_dir):
    with open(base_path) as f: base_data = json.load(f)
    with open(exp4_path) as f: exp4_data = json.load(f)
    
    base_map = {str(x.get('image_id')): x for x in base_data}
    records = []
    
    for item in exp4_data:
        img_id = str(item.get('image_id'))
        if img_id not in base_map: continue
        
        b_item = base_map[img_id]
        gt = b_item.get("gt_bbox") or item.get("gt_bbox")
        
        # --- FIX: SAFE PREDICTION ACCESS ---
        # Baseline
        b_preds = b_item.get("pred_bboxes", [])
        b_box = b_item.get("head_pred_bbox") or (b_preds[0] if b_preds else None)
        
        # Exp4
        e_preds = item.get("pred_bboxes", [])
        e_box = item.get("head_pred_bbox") or (e_preds[0] if e_preds else None)
        # -----------------------------------
        
        b_dist, b_area = get_metrics(b_box, gt)
        e_dist, e_area = get_metrics(e_box, gt)
        
        records.append({
            "Base IoU": b_item.get("bbox_iou", 0.0),
            "Exp4 IoU": item.get("bbox_iou", 0.0),
            "Base Dist": b_dist, "Exp4 Dist": e_dist,
            "Base Area": b_area, "Exp4 Area": e_area
        })
    
    df = pd.DataFrame(records)
    
    plot_robustness(df, out_dir)
    
    # Success Curve
    thresholds = np.linspace(0, 1, 100)
    b_pass = [np.mean(df["Base IoU"] >= t) for t in thresholds]
    e_pass = [np.mean(df["Exp4 IoU"] >= t) for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, b_pass, label='Supervised Pretraining (Exp1)', color=COLORS[0], linewidth=3, alpha=0.6)
    plt.plot(thresholds, e_pass, label='Data Augmentation (Exp4)', color=COLORS[3], linewidth=3)
    plt.fill_between(thresholds, b_pass, e_pass, where=(np.array(e_pass)>np.array(b_pass)), 
                     color=COLORS[3], alpha=0.1, label='Robustness Gain')
    plt.xlabel("IoU Threshold"); plt.ylabel("Success Rate")
    plt.title("Robustness Success Curve")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.savefig(os.path.join(out_dir, "viz_success_curve.png"), dpi=300)
    
    # Waterfall
    df["Delta"] = df["Exp4 IoU"] - df["Base IoU"]
    sorted_deltas = df.sort_values("Delta").reset_index()
    bar_colors = [COLORS[0] if x < -0.05 else COLORS[3] if x > 0.05 else '#bdc3c7' for x in sorted_deltas["Delta"]]
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(df)), sorted_deltas["Delta"], color=bar_colors, width=1.0)
    plt.axhline(0, color='black')
    plt.title("Net Impact per Image (Waterfall)")
    plt.ylabel("IoU Change")
    plt.ylim(-1, 1)
    plt.savefig(os.path.join(out_dir, "viz_waterfall.png"), dpi=300)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_training_loss(args.log_files, args.last_n_epochs, args.output_dir)
    run_eval_plots(args.baseline_json, args.exp4_json, args.output_dir)
    print(f"\n✅ Done. Plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()