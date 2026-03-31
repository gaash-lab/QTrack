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
JSONL_PATH = TRAIN_DIR / "output_qtrack.jsonl"
OUTPUT_PATH = DATASET_ROOT / "hf_dataset_qtrack"  


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor()
    ])

    with open(JSONL_PATH, "r") as f:
        all_items = [json.loads(line) for line in f]

    grouped = {}
    for item in all_items:
        obj_id = "_".join(item["id"].split("_")[:-1])  
        grouped.setdefault(obj_id, []).append(item)

    records = []

    for obj_id, entries in grouped.items():
        valid_entries = []
        for entry in entries:
            try:
                ans = json.loads(entry["answer"])
                if ans and "frame" in ans[0] and "bbox" in ans[0]:
                    valid_entries.append(entry)
            except Exception:
                continue  

        entries = sorted(valid_entries, key=lambda e: json.loads(e["answer"])[0]["frame"])

        prev_bbox = None
        prev_image = None

        for entry in entries:
            try:
                curr_bbox_json = json.loads(entry["answer"])
                if not curr_bbox_json:
                    continue 

                curr_bbox = curr_bbox_json[0]["bbox"]

                ref_image_path = TRAIN_DIR / entry["images"][0]
                ref_image = Image.open(ref_image_path).convert("RGB")
                img_tensor = transform(ref_image).unsqueeze(0).to(device)
                img_tensor = img_tensor.clamp(0, 1)
                img_cpu = (img_tensor.squeeze(0).cpu() * 255).byte()
                ref_img_pil = Image.fromarray(img_cpu.permute(1, 2, 0).numpy(), mode="RGB")


                curr_image_path = TRAIN_DIR / entry["images"][1]
                image = Image.open(curr_image_path).convert("RGB")
                img_tensor = transform(image).unsqueeze(0).to(device)
                img_tensor = img_tensor.clamp(0, 1)
                img_cpu = (img_tensor.squeeze(0).cpu() * 255).byte()
                img_pil = Image.fromarray(img_cpu.permute(1, 2, 0).numpy(), mode="RGB")

                if prev_bbox is not None:
                    records.append({
                        "id": f"{obj_id}_frame{curr_bbox_json[0]['frame']}",
                        "images": [ref_img_pil, img_pil],
                        "problem": entry["prompt"],
                        "solution": {
                            "prev": [{"bbox_2d": prev_bbox}],
                            "curr": [{"bbox_2d": curr_bbox}]
                        }
                    })

                prev_bbox = curr_bbox
                prev_image = img_pil

            except Exception as e:
                print(f"Skipping entry {entry['id']} due to error: {e}")
                continue

    dataset = Dataset.from_list(records)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(OUTPUT_PATH)
    print(f"Saved HuggingFace MCP dataset to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
