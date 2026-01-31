from datasets import load_from_disk

dataset_path = "/home/gaash/Wasif/Tawheed/MOT_grounding_Dataset/hf_dataset_mcp"

ds = load_from_disk(dataset_path)

# Access train split
train_ds = ds["train"]

# Read first 5 examples
for i in range(5):
    example = train_ds[i]
    print(example.keys())
    print(example)
