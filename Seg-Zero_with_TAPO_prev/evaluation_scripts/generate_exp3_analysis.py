#!/usr/bin/env python3
"""
analyze_exp3_charts.py
Generates numerical charts comparing Baseline vs. Exp3.
1. IoU Distribution (Histogram/KDE)
2. Scatter Plot (Baseline vs. Exp3)
3. Performance by Object Size
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_json", required=True)
    parser.add_argument("--exp3_json", required=True)
    parser.add_argument("--output_dir", default="exp3_charts")
    return parser.parse_args()

def calculate_area(bbox):
    # bbox: [x, y, x2, y2]
    if not bbox: return 0
    w = max(0, bbox[2] - bbox[0])
    h = max(0, bbox[3] - bbox[1])
    return w * h

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    print("Loading data...")
    with open(args.baseline_json) as f: base_data = json.load(f)
    with open(args.exp3_json) as f: exp3_data = json.load(f)
    
    # Create DataFrame
    # Map by Image ID to ensure alignment
    base_map = {str(x.get('image_id')): x for x in base_data}
    exp3_map = {str(x.get('image_id')): x for x in exp3_data}
    
    records = []
    
    for img_id, e_item in exp3_map.items():
        if img_id not in base_map: continue
        b_item = base_map[img_id]
        
        # Get Metrics
        b_iou = b_item.get("bbox_iou", 0.0)
        e_iou = e_item.get("bbox_iou", 0.0)
        
        # Get Object Size (Area)
        gt = e_item.get("gt_bbox") or b_item.get("gt_bbox")
        area = calculate_area(gt)
        
        records.append({
            "Image ID": img_id,
            "Baseline IoU": b_iou,
            "Exp3 IoU": e_iou,
            "Delta": e_iou - b_iou,
            "Object Area": area
        })
        
    df = pd.DataFrame(records)
    print(f"Aligned {len(df)} samples.")
    
    # Bin object sizes
    if not df.empty and df["Object Area"].max() > 0:
        df["Size Bin"] = pd.qcut(df["Object Area"], q=3, labels=["Small", "Medium", "Large"])
    else:
        df["Size Bin"] = "Unknown"

    # --- PLOT 1: IoU Distribution (Histogram) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Baseline IoU"], color="red", alpha=0.3, label="Baseline", kde=True, bins=20)
    sns.histplot(df["Exp3 IoU"], color="blue", alpha=0.3, label="Exp3 (Cross-Attn)", kde=True, bins=20)
    plt.title("IoU Distribution Shift: Baseline vs. Cross-Attention")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    plt.legend()
    save_p1 = os.path.join(args.output_dir, "iou_distribution.png")
    plt.savefig(save_p1)
    print(f"Saved {save_p1}")
    plt.close()

    # --- PLOT 2: Head-to-Head Scatter ---
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x="Baseline IoU", y="Exp3 IoU", alpha=0.5, hue="Size Bin")
    # Draw diagonal line (y=x)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="No Change")
    plt.title("Head-to-Head: Did Exp3 Improve Individual Samples?")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    save_p2 = os.path.join(args.output_dir, "iou_scatter_comparison.png")
    plt.savefig(save_p2)
    print(f"Saved {save_p2}")
    plt.close()

    # --- PLOT 3: Improvement by Object Size ---
    plt.figure(figsize=(10, 6))
    # Melt for boxplot
    melted = df.melt(id_vars=["Image ID", "Size Bin"], value_vars=["Baseline IoU", "Exp3 IoU"], 
                     var_name="Model", value_name="IoU")
    sns.barplot(data=melted, x="Size Bin", y="IoU", hue="Model", palette={"Baseline IoU": "salmon", "Exp3 IoU": "skyblue"})
    plt.title("Performance by Object Size")
    plt.ylim(0, 1.0)
    save_p3 = os.path.join(args.output_dir, "iou_by_size.png")
    plt.savefig(save_p3)
    print(f"Saved {save_p3}")
    plt.close()
    
    # Save CSV for numerical inspection
    csv_path = os.path.join(args.output_dir, "chart_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw data to {csv_path}")

if __name__ == "__main__":
    main()