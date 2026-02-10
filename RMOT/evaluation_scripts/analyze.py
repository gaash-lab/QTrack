import os
import cv2
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

# --- Configuration ---
DATASET_PATH   = "/home/gaash/Wasif/datasets/TestData/Combined_1016"
OUTPUT_DIR     = "/home/gaash/Wasif/Seg-Zero/analysis_cases/GT"

def main():
    # 1. Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # 2. Load the Arrow Dataset
    print(f"Loading Arrow dataset from: {DATASET_PATH}")
    try:
        dataset = load_from_disk(DATASET_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"Dataset columns found: {dataset.column_names}")

    # 3. Identify the GT Column
    # We look for standard names for bounding boxes
    possible_gt_cols = ['ground_truth', 'bbox', 'gt_bbox', 'label']
    gt_col = next((col for col in possible_gt_cols if col in dataset.column_names), None)

    if not gt_col:
        print(f"Error: Could not find a recognized bounding box column in {dataset.column_names}")
        return
    
    print(f"Using column '{gt_col}' for Bounding Box.")

    # 4. Iterate and Visualize
    count = 0
    for idx, item in enumerate(tqdm(dataset, desc="Visualizing GT")):
        
        # --- Handle Image ---
        if 'image' not in item:
            continue
            
        pil_image = item['image']
        image_np = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 2:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # --- Handle GT BBox ---
        # Expecting a list/array like [x, y, w, h]
        bbox = item[gt_col]

        if bbox and len(bbox) == 4:
            # Unpack COCO format: x_min, y_min, width, height
            x, y, w, h = map(int, bbox)
            
            # Calculate bottom-right corner for OpenCV
            x2 = x + w
            y2 = y + h
            
            # Draw Green Box
            cv2.rectangle(image_cv, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_cv, 'GT', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Construct a filename
            img_id = item.get('image_id', f"idx_{idx}")
            filename = f"gt_{img_id}.jpg"
            
            save_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(save_path, image_cv)
            count += 1

    print(f"\nProcessing complete.")
    print(f"Saved {count} images to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()