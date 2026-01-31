import json
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image
from datasets import Dataset, DatasetDict
import numpy as np

DATASET_ROOT = Path("/home/gaash/Wasif/Tawheed/MOT_grounding_Dataset")
TRAIN_DIR = DATASET_ROOT / "train"
JSONL_PATH = TRAIN_DIR / "train_split_1.jsonl"
OUTPUT_PATH = DATASET_ROOT / "hf_dataset"
# =========================================


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((512, 512)),   # GPU-accelerated after tensor conversion
        T.ToTensor()
    ])

    records = []

    with open(JSONL_PATH, "r") as f:
        for line in tqdm(f, desc="Processing MOT dataset"):
            item = json.loads(line)

            pil_images_out = []

            for img_rel_path in item["images"]:
                img_path = TRAIN_DIR / img_rel_path

                # --- Load image (CPU) ---
                image = Image.open(img_path).convert("RGB")

                # --- Move to GPU ---
                img_tensor = transform(image).unsqueeze(0).to(device)  # [1,3,H,W]

                # (Optional GPU ops here)
                img_tensor = img_tensor.clamp(0, 1)

                # --- Back to CPU & PIL ---
                img_cpu = (img_tensor.squeeze(0).cpu() * 255).byte()
                img_pil = Image.fromarray(
                    img_cpu.permute(1, 2, 0).numpy(), mode="RGB"
                )

                pil_images_out.append(img_pil)

            records.append({
                "id": item["id"],
                "images": pil_images_out,       # LIST[PIL.Image]
                "problem": item["prompt"],
                "solution": item["answer"],
            })

    dataset = Dataset.from_list(records)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(OUTPUT_PATH)

    print(f"Saved HuggingFace dataset (GPU-processed PIL images) to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
