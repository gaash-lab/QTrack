import os
import json
import glob
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="folder path of output files")
    return parser.parse_args()

def calculate_metrics(output_dir):
    # get all output files
    output_files = sorted(glob.glob(os.path.join(output_dir, "output_*.json")))
    
    if not output_files:
        print(f"cannot find output files in {output_dir}")
        return
    
    # for accumulating all data
    all_metrics = [] # Renamed for clarity
    
    # read and process all files
    for file_path in output_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        # process all items in each file
        for item in results:
            intersection = item['intersection']
            union = item['union']
            
            # calculate Mask IoU of each item
            mask_iou = intersection / union if union > 0 else 0
            
            # --- START MODIFICATION ---
            # Get the BBox IoU (which is now a float from the other script)
            bbox_iou = item.get('bbox_iou', 0.0)
            
            all_metrics.append({
                'image_id': item['image_id'],
                'mask_iou': mask_iou,
                'bbox_iou': bbox_iou
            })
            # --- END MODIFICATION ---
            
    
    # --- START MODIFICATION ---
    # calculate gIoU (average of mask IoU)
    gIoU = np.mean([item['mask_iou'] for item in all_metrics])
    
    # NEW: calculate average BBox IoU (the "precision" metric)
    avg_bbox_iou = np.mean([item['bbox_iou'] for item in all_metrics])
    
    # NEW: calculate the "Pass Rate" metric (the original one)
    pass_rate_count = sum(1 for item in all_metrics if item['bbox_iou'] > 0.5)
    pass_rate_percent = (pass_rate_count / len(all_metrics)) * 100
    
    # print the results
    print(f"--- Evaluation Results ---")
    print(f"Total Images: {len(all_metrics)}")
    print(f"gIoU (Average Mask IoU): {gIoU:.4f}")
    print(f"Average BBox IoU (Precision): {avg_bbox_iou:.4f}  <-- This is the new metric for fair comparison")
    print(f"BBox Pass Rate (> 0.5 IoU): {pass_rate_percent:.2f}% ({pass_rate_count}/{len(all_metrics)})")
    # --- END MODIFICATION ---
    

if __name__ == "__main__":
    args = parse_args()
    calculate_metrics(args.output_dir)