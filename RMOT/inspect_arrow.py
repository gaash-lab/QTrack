import json
from datasets import load_from_disk
import os

# --- Configuration ---
# This should be the same directory where your gen_training_dataset.py saved the data
dataset_dir = "data/new_val"
num_examples_to_print = 5
# ---------------------

print(f"--- Loading dataset from: {dataset_dir} ---")

# Check if the dataset directory exists
if not os.path.exists(dataset_dir):
    print(f"\nError: Directory not found at '{dataset_dir}'")
    print("Please make sure you have run the 'gen_training_dataset.py' script successfully.")
else:
    try:
        # Load the dataset from the Arrow files on disk
        dataset = load_from_disk(dataset_dir)
        
        # Access the 'train' split
        train_dataset = dataset['train']
        
        print(f"\n--- Displaying first {num_examples_to_print} samples ---")
        
        # Iterate and print the first few examples
        for i in range(min(num_examples_to_print, len(train_dataset))):
            example = train_dataset[i]
            
            # The 'solution' is stored as a JSON string, so we need to parse it
            solution_data = json.loads(example['solution'])
            
            print(f"\n--- Sample #{i+1} ---")
            print(f"  ID: {example['id']}")
            print(f"  Original Dimensions: {example['img_width']}w x {example['img_height']}h")
            print(f"  Problem: {example['problem']}")
            
            # Print each bounding box within the solution
            for j, bbox_info in enumerate(solution_data):
                bbox = bbox_info.get('bbox_2d', 'N/A')
                print(f"    - Bounding Box #{j+1} (Corrected & Scaled): {bbox}")
            
    except Exception as e:
        print(f"\nAn error occurred while trying to load or read the dataset: {e}")
        print("Please ensure the dataset was created correctly and the directory path is correct.")