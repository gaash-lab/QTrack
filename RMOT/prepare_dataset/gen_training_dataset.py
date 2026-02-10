import argparse
import json
import os
import cv2
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Image, Features, Value
from PIL import Image as PILImage

# --- Helper Functions ---

def scale_box_coordinates(bbox_2d, x_factor, y_factor):
    """
    Scales bounding box coordinates from [x1, y1, x2, y2] format.
    """
    scaled_bbox = [
        int(bbox_2d[0] * x_factor + 0.5),  # x1
        int(bbox_2d[1] * y_factor + 0.5),  # y1
        int(bbox_2d[2] * x_factor + 0.5),  # x2
        int(bbox_2d[3] * y_factor + 0.5)   # y2
    ]
    return scaled_bbox

def scale_point_coordinates(point_2d, x_factor, y_factor):
    """
    Scales center point coordinates.
    """
    scaled_point = [
        int(point_2d[0] * x_factor + 0.5),  # x
        int(point_2d[1] * y_factor + 0.5)   # y
    ]
    return scaled_point

def create_local_dataset(train_data, output_dir, image_resize):
    """
    Creates and saves the dataset locally in Hugging Face format.
    """
    def process_split(split_data, image_resize):
        processed_data = split_data.copy()
        images = []
        print(f"Resizing images to {image_resize}x{image_resize}...")
        for img_path in tqdm(split_data['image'], desc="Processing images"):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image at path: {img_path}. Skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
            images.append(img)
        
        processed_data['image'] = images
        return processed_data
    
    dataset = DatasetDict({
        'train': Dataset.from_dict(
            process_split(train_data, image_resize),
            features=Features({
                'id': Value('string'),
                'problem': Value('string'),
                'solution': Value('string'),
                'image': Image(),
                'img_height': Value('int64'),
                'img_width': Value('int64')
            })
        )
    })
    
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Correctly formatted dataset saved to: {output_dir}")
    return dataset

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- ADDED: Argument parsing for flexibility ---
    parser = argparse.ArgumentParser(description="Process surgical data annotations to create a Hugging Face dataset.")
    parser.add_argument("--input_json_path", type=str, required=True, help="Full path to the intermediate annotation file (e.g., output.json).")
    parser.add_argument("--output_dataset_path", type=str, required=True, help="Full path to save the final processed dataset.")
    args = parser.parse_args()

    print(f"Loading intermediate data from: {args.input_json_path}")
    with open(args.input_json_path, 'r') as f:
        data = json.load(f)

    # --- ADDED: Robust path handling ---
    # The directory of the input JSON is used as the base path for finding images.
    base_image_dir = os.path.dirname(args.input_json_path)
    print(f"Base directory for images determined as: {base_image_dir}")

    image_resize = 840
    
    # Initialize lists to hold the processed data
    id_list, problem_list, solution_list, image_list, img_height_list, img_width_list = [], [], [], [], [], []

    print("Processing annotations and correcting bounding box format...")
    for item in tqdm(data, desc="Converting annotations"):
        
        # --- ADDED: Intelligent image path construction ---
        # Handles both absolute and relative paths from the JSON file
        relative_image_path = item['image_path'].lstrip('/') 
        full_image_path = os.path.join(base_image_dir, relative_image_path)
        
        if not os.path.exists(full_image_path):
            print(f"Warning: Image path not found: {full_image_path}. Skipping item ID {item['id']}.")
            continue
        
        id_list.append(str(item['id'])) # Ensure ID is a string for dataset compatibility
        problem_list.append(item['problem'])
        image_list.append(full_image_path)
        
        # Get original image dimensions for accurate scaling
        with PILImage.open(full_image_path) as img:
            width, height = img.size
        
        img_height_list.append(height)
        img_width_list.append(width)
        
        x_factor = image_resize / width
        y_factor = image_resize / height
        
        solution = []
        
        # This script now correctly handles single-object annotation format
        raw_box = item['bboxes']
        center_point = item['center_points']

        # --- THIS IS THE CRITICAL BUG FIX ---
        # 1. Assume the input box from your JSON is [x, y, w, h]
        x, y, w, h = raw_box
        # 2. Convert it to the correct [x_min, y_min, x_max, y_max] format
        bbox_xyxy = [x, y, x + w, y + h]
        # --- END OF FIX ---

        solution.append({
            # Now, pass the CORRECTED box to the scaling function
            "bbox_2d": scale_box_coordinates(bbox_xyxy, x_factor, y_factor), 
            "point_2d": scale_point_coordinates(center_point, x_factor, y_factor)
        })
        
        solution_list.append(json.dumps(solution))

    # Assemble the data into a dictionary
    train_data = {
        'id': id_list,
        'problem': problem_list,
        'solution': solution_list,
        'image': image_list,
        'img_height': img_height_list,
        'img_width': img_width_list
    }
    
    # Create the final dataset
    dataset = create_local_dataset(
        train_data=train_data,
        output_dir=args.output_dataset_path,
        image_resize=image_resize
    )
    
    print("\n--- All Done! ---")