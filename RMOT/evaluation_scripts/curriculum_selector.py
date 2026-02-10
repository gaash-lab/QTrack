#!/usr/bin/env python3
"""
Create curriculum splits (Easy / Medium / Hard) from evaluation JSON (bbox_iou),
aligned to a HuggingFace Arrow dataset on disk.

- Easy   : bbox_iou >= easy_min_iou
- Hard   : bbox_iou <= hard_max_iou
- Medium : everything else

Notes
-----
* We DO NOT decode images; we cast image column with decode=False to avoid PIL/libstdc++ issues.
* Output:
    - split_indices.json: lists of dataset indices for each split
    - summary.json: counts & thresholds
    - (optional) hf_splits/: HuggingFace DatasetDict with three subsets

Usage (example)
---------------
python3 evaluation_scripts/curriculum_selector.py \
  --base_json "./reasonseg_eval_results/exp7/STS/output_0.json" \
  --dataset_path "/home/gaash/Wasif/Seg-Zero/data/STS" \
  --out_dir "/home/gaash/Wasif/Seg-Zero/curriculum/exp2_splits" \
  --easy_min_iou 0.30 \
  --hard_max_iou 0.10 \
  --save_subsets True
"""

import argparse
import json
import math
import os
from typing import Dict, Tuple, List

# Import DatasetDict to check types
from datasets import load_from_disk, DatasetDict
from datasets import Image as HFImage  # for decode=False cast


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_json", type=str, required=True,
                    help="Path to evaluation output JSON (e.g., output_0.json) containing bbox_iou.")
    ap.add_argument("--dataset_path", type=str, required=True,
                    help="Path to HF dataset (load_from_disk).")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory to save indices / summary (and optionally subsets).")
    
    # Thresholds
    ap.add_argument("--easy_min_iou", type=float, default=0.5,
                    help="Samples with bbox_iou >= this go to Easy.")
    ap.add_argument("--hard_max_iou", type=float, default=0.2,
                    help="Samples with bbox_iou <= this go to Hard.")
    
    ap.add_argument("--save_subsets", type=lambda x: str(x).lower() in {"1", "true", "yes"},
                    default=False, help="If True, saves HF subsets to out_dir/hf_splits")
    return ap.parse_args()


def safe_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def load_iou_map(json_path: str) -> Dict[Tuple[str, int], float]:
    """Return map: (image_id, ann_id) -> bbox_iou."""
    print(f"[selector] Reading JSON from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    iou_map: Dict[Tuple[str, int], float] = {}
    for rec in data:
        # The scoring script likely saved "image_id"
        image_id = rec.get("image_id")
        ann_id = rec.get("ann_id")
        
        try:
            ann_id = int(ann_id) if ann_id is not None else 0
        except Exception:
            ann_id = 0
            
        if image_id is None:
            continue
            
        iou = safe_float(rec.get("bbox_iou", 0.0))
        
        # Key is (str(image_id), int(ann_id))
        iou_map[(str(image_id), ann_id)] = iou
        
    return iou_map


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load dataset (without decoding images)
    print(f"[selector] Loading dataset from {args.dataset_path}...")
    ds_raw = load_from_disk(args.dataset_path)

    # Handle DatasetDict
    if isinstance(ds_raw, dict) or isinstance(ds_raw, DatasetDict):
        if "train" in ds_raw:
            ds = ds_raw["train"]
        else:
            first_key = list(ds_raw.keys())[0]
            ds = ds_raw[first_key]
    else:
        ds = ds_raw

    # 2. Avoid PIL import / decoding
    if "image" in ds.features:
        print("[selector] Casting 'image' column to decode=False...")
        ds = ds.cast_column("image", HFImage(decode=False))

    # 3. Build IoU lookup
    iou_map = load_iou_map(args.base_json)

    easy_idx: List[int] = []
    med_idx: List[int] = []
    hard_idx: List[int] = []
    
    missing_count = 0

    # 4. Iterate dataset and assign splits
    print(f"[selector] Processing {len(ds)} samples...")
    for i in range(len(ds)):
        row = ds[i]
        
        # FIX: Check 'id' if 'image_id' is missing (handles your STS dataset format)
        image_id = row.get("image_id")
        if image_id is None:
            image_id = row.get("id")
        
        image_id = str(image_id)
        
        ann_id = row.get("ann_id")
        try:
            ann_id = int(ann_id) if ann_id is not None else 0
        except Exception:
            ann_id = 0
            
        # Look up IoU
        iou = iou_map.get((image_id, ann_id), None)
        
        if iou is None:
            # If exact match failed, try checking just image_id with ann_id=0
            # (Some datasets rely only on image_id)
            iou = iou_map.get((image_id, 0), None)

        if iou is None:
            missing_count += 1
            # Strategy: Skip missing items to avoid training on unscored data
            continue

        if iou >= args.easy_min_iou:
            easy_idx.append(i)
        elif iou <= args.hard_max_iou:
            hard_idx.append(i)
        else:
            med_idx.append(i)

    # 5. Save indices
    indices_path = os.path.join(args.out_dir, "split_indices.json")
    with open(indices_path, "w") as f:
        json.dump(
            {"easy": easy_idx, "medium": med_idx, "hard": hard_idx},
            f, indent=2
        )

    # 6. Save summary
    total_matched = len(easy_idx) + len(med_idx) + len(hard_idx)
    summary = {
        "dataset_path": args.dataset_path,
        "eval_json": args.base_json,
        "easy_min_iou": args.easy_min_iou,
        "hard_max_iou": args.hard_max_iou,
        "counts": {
            "easy": len(easy_idx),
            "medium": len(med_idx),
            "hard": len(hard_idx),
            "total_in_ds": len(ds),
            "matched": total_matched,
            "missing_pairs": missing_count
        }
    }
    
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Curriculum Split Summary ===")
    print(json.dumps(summary, indent=2))

    if total_matched == 0:
        print("[selector] ERROR: No samples matched! Check image_id format in dataset vs JSON.")
        exit(1)

    # 7. Optional: save HF subsets
    if args.save_subsets:
        subsets_dir = os.path.join(args.out_dir, "hf_splits")
        os.makedirs(subsets_dir, exist_ok=True)
        
        splits_to_save = {}
        if len(easy_idx) > 0: splits_to_save["easy"] = ds.select(easy_idx)
        if len(med_idx) > 0: splits_to_save["medium"] = ds.select(med_idx)
        if len(hard_idx) > 0: splits_to_save["hard"] = ds.select(hard_idx)
        
        dsd = DatasetDict(splits_to_save)
        
        print(f"[selector] Saving HF subsets to {subsets_dir}...")
        dsd.save_to_disk(subsets_dir)
        print(f"[selector] Saved.")

    print("[selector] Done.")

if __name__ == "__main__":
    main()