import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image
from datasets import Dataset, DatasetDict


def parse_args():
    parser = argparse.ArgumentParser(description="Build QTrack dataset from raw annotations.json")

    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--json", type=str, default="annotations.json")
    parser.add_argument("--output_dir", type=str, default="hf_dataset_qtrack")
    parser.add_argument("--image_size", type=int, default=512)

    return parser.parse_args()



def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def parse_expected_output(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except:
            return None

    # Single object: list of trajectory
    if isinstance(raw, list):
        return raw

    # Multi-object: dict
    elif isinstance(raw, dict):
        all_traj = []
        for obj in raw.values():
            if "trajectory" in obj:
                all_traj.extend(obj["trajectory"])
        return sorted(all_traj, key=lambda x: x["frame"])

    return None


def load_image(img_path, transform, device):
    try:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        img_tensor = img_tensor.clamp(0, 1)

        img_cpu = (img_tensor.squeeze(0).cpu() * 255).byte()
        img_pil = Image.fromarray(
            img_cpu.permute(1, 2, 0).numpy(), mode="RGB"
        )
        return img_pil
    except Exception as e:
        print(f"Image load failed: {img_path} | {e}")
        return None



def main():
    args = parse_args()

    DATASET_ROOT = Path(args.dataset_root)
    TRAIN_DIR = DATASET_ROOT / args.train_dir
    JSON_PATH = TRAIN_DIR / args.json
    OUTPUT_PATH = DATASET_ROOT / args.output_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor()
    ])

    data = load_json(JSON_PATH)
    print(f"Total raw items: {len(data)}")

    records = []

    for item in tqdm(data, desc="Processing items"):

        item_id = item.get("id", "unknown")

        image_list = item.get("input_images") or item.get("images")
        if not image_list or len(image_list) < 2:
            continue

        prompt = item.get("question", "")

        raw_answer = item.get("expected_output") or item.get("answer")
        trajectory = parse_expected_output(raw_answer)

        if not trajectory or len(trajectory) < 2:
            continue

        frame_to_bbox = {}
        for t in trajectory:
            if "frame" in t and "bbox" in t:
                frame_to_bbox[t["frame"]] = t["bbox"]

        def get_frame(path):
            try:
                return int(Path(path).stem)
            except:
                return None

        frames = [get_frame(p) for p in image_list]

        ref_image_path = TRAIN_DIR / image_list[0]

        for i in range(1, len(image_list)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            if prev_frame not in frame_to_bbox or curr_frame not in frame_to_bbox:
                continue

            prev_bbox = frame_to_bbox[prev_frame]
            curr_bbox = frame_to_bbox[curr_frame]

            curr_img_path = TRAIN_DIR / image_list[i]

            img_pil = load_image(curr_img_path, transform, device)

            ref_img_pil = load_image(ref_image_path, transform, device)
            if img_pil is None or ref_img_pil is None:
                continue

            records.append({
                "id": f"{item_id}_frame{curr_frame}",
                "images": [ref_img_pil, img_pil],
                "problem": prompt,
                "solution": {
                    "prev": [{"bbox_2d": prev_bbox}],
                    "curr": [{"bbox_2d": curr_bbox}]
                }
            })

    print(f"Final records: {len(records)}")

    if len(records) == 0:
        raise ValueError("No valid records found. Check dataset format.")

    dataset = Dataset.from_list(records)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.save_to_disk(OUTPUT_PATH)

    print(f"Saved dataset to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()