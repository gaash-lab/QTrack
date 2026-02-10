#!/usr/bin/env python
# coding: utf-8

# ##############################################################################
# MODIFIED SCRIPT FOR SEG-ZERO DATA PREPARATION
# This version bypasses the SAM2 model loading and filtering step entirely.
# It generates the training annotation list directly from the ground-truth
# data in the RefCOCOg dataset.
# ##############################################################################

import os
import sys
import json
from tqdm import tqdm
import numpy as np
from pycocotools import mask

# Ensure the parent directory is in the path to find refer_seg_dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refer_seg_dataset import ReferSegDataset

print("--- Starting Data Preparation (No SAM2 Filter) ---")

# --- Step 1: Load ReferSegDataset ---
print("Loading ReferSeg Dataset...")
dataset_name = "refcocog"
# IMPORTANT: Verify this base_image_dir path is correct for your system
base_image_dir = "/home/gaash/Wasif/datasets"

try:
    dataset = ReferSegDataset(base_image_dir=base_image_dir, refer_seg_data=dataset_name, data_split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print(f"Please ensure the path '{base_image_dir}' and the dataset files within it are correct.")
    sys.exit(1)

refer_seg_ds = dataset.refer_seg_data[dataset_name]
images = refer_seg_ds["images"]
annotations = refer_seg_ds["annotations"]
img2refs = refer_seg_ds["img2refs"]
print("Dataset loaded successfully.")

# --- Step 2: Generate annotation list directly from ground truth ---
print("Generating annotation list...")

seg_zero_annotation_list = []

for image_info in tqdm(images, desc="Processing images"):
    image_path = image_info["file_name"]
    image_id = image_info["id"]

    if image_id not in img2refs:
        continue

    refs = img2refs[image_id]
    
    texts = []
    bboxes = []
    points = []
    ann_ids = []
    
    for ref in refs:
        ann_id = ref["ann_id"]
        
        # Skip if there are no sentence annotations
        if not ref["sentences"]:
            continue
        # We'll just take the first sentence for the prompt
        text = ref["sentences"][0]["raw"].strip().strip(".?!").lower()
        
        ann = annotations[ann_id]
        if len(ann["segmentation"]) == 0:
            continue

        # Decode the ground-truth mask
        if type(ann["segmentation"][0]) == list:  # polygon format
            rle = mask.frPyObjects(
                ann["segmentation"], image_info["height"], image_info["width"]
            )
        else:  # RLE format
            rle = ann["segmentation"]
        
        m = mask.decode(rle)
        if m.ndim > 2:
            m = np.sum(m, axis=2) # Combine multiple segments if they exist
        m = m.astype(np.uint8)
        
        # Find coordinates of the mask to derive the bounding box
        y_indices, x_indices = np.where(m == 1)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue
        
        # Calculate bounding box [x1, y1, x2, y2] from ground truth mask
        left = int(x_indices.min())
        top = int(y_indices.min())
        right = int(x_indices.max())
        bottom = int(y_indices.max())
        box = [left, top, right, bottom]
        
        # Use the center of the bounding box as the representative point
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        point = [center_x, center_y]
        
        # Append the ground-truth derived data to our lists
        bboxes.append(box)
        points.append(point)
        texts.append(text)
        ann_ids.append(str(ann_id))
    
    if len(bboxes) == 0:
        continue
    
    # Create the final annotation dictionary for this image
    seg_zero_annotation_list.append({
        "id": f"{dataset_name}_" + "_".join(ann_ids[:3]),
        "image_id": image_id,
        "image_path": image_path,
        "problem": "'" + "' and '".join(texts) + "'",
        "bboxes": bboxes,
        "center_points": points
    })
            
print(f"Total annotations generated: {len(seg_zero_annotation_list)}")

# --- Step 3: Save the list to a JSON file ---
output_filename = f'seg_zero_{dataset_name}_annotation_list_no_filter.json'
print(f"Saving annotations to {output_filename}...")
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(seg_zero_annotation_list, f, ensure_ascii=False, indent=4)

print("--- Data Preparation Job Complete ---")
print(f"Output file: {output_filename}")