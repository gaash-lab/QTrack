import os
import json
import re
import numpy as np
from tqdm import tqdm

from vllm import LLM
from vllm.sampling_params import SamplingParams


# ================= IoU =================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    return inter / (boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter + 1e-6)


# ================= Parse Output =================
def parse_bbox(text):
    try:
        data = json.loads(text)
        if isinstance(data, list) and "bbox" in data[0]:
            return data[0]["bbox"]
    except:
        pass

    m = re.search(r'\[(\-?\d+\.?\d*),\s*(\-?\d+\.?\d*),\s*(\-?\d+\.?\d*),\s*(\-?\d+\.?\d*)\]', text)
    if m:
        return [float(m.group(i)) for i in range(1, 5)]

    return [0, 0, 0, 0]


# ================= Detect =================
def detect_format(item):
    return "multi" if "objects" in item else "single"


# ================= Pixtral Call =================
def pixtral_infer(llm, prompt, ref_img, frame_img):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"file://{ref_img}"}},
            {"type": "image_url", "image_url": {"url": f"file://{frame_img}"}},
            {"type": "text", "text": prompt},
        ],
    }]

    out = llm.chat(messages)
    return out[0].outputs[0].text.strip()


# ================= Single =================
def eval_single(item, root, llm):
    ref = os.path.join(root, item["ref_image"])
    frames = [os.path.join(root, f) for f in item["input_images"]]

    gt = json.loads(item["expected_output"]) if isinstance(item["expected_output"], str) else item["expected_output"]

    preds, ious = [], []

    for img, gt_f in zip(frames, gt):
        prompt = (
            f"{item['question']}\n"
            f"Initial bbox: {item.get('initial_bbox')}\n"
            "Return ONLY JSON: [{\"frame\": N, \"bbox\": [x,y,w,h]}]"
        )

        text = pixtral_infer(llm, prompt, ref, img)
        bbox = parse_bbox(text)

        preds.append({"frame": gt_f["frame"], "bbox": bbox})
        ious.append(compute_iou(gt_f["bbox"], bbox))

    return float(np.mean(ious)), preds


# ================= Multi =================
def eval_multi(item, root, llm):
    ref = os.path.join(root, item["ref_image"])
    frames = [os.path.join(root, f) for f in item["input_images"]]

    gt_all = json.loads(item["expected_output"]) if isinstance(item["expected_output"], str) else item["expected_output"]
    object_ious = []

    for obj in item["objects"]:
        traj = gt_all[f"object_{obj['object_id']}"]["trajectory"]
        ious = []

        for img, gt_f in zip(frames, traj):
            prompt = (
                f"{item['question']}\n"
                f"Object {obj['object_id']} initial bbox: {obj['initial_bbox']}\n"
                "Return ONLY JSON."
            )

            text = pixtral_infer(llm, prompt, ref, img)
            bbox = parse_bbox(text)
            ious.append(compute_iou(gt_f["bbox"], bbox))

        object_ious.append(np.mean(ious))

    return float(np.mean(object_ious))


# ================= Main =================
def evaluate(json_path, root, out_json):
    data = json.load(open(json_path))

    llm = LLM(
        model="mistralai/Pixtral-12B-2409",
        tokenizer_mode="mistral"
    )

    scores = []

    for item in tqdm(data):
        if detect_format(item) == "single":
            score, _ = eval_single(item, root, llm)
        else:
            score = eval_multi(item, root, llm)
        scores.append(score)

    json.dump({"average_iou": float(np.mean(scores))}, open(out_json, "w"), indent=2)
    print("Avg IoU:", np.mean(scores))


# ================= CLI =================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate object tracking model on single or multi-object datasets")
    parser.add_argument("--json", default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/annotations_real.json", help="Path to the JSON file")
    parser.add_argument("--dataset_root", default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset", help="Path to dataset root")
    parser.add_argument("--json_log", default="/home/gaash/Tawheed/Reasoning/Ealuation_logs/evaluation_pixtral.json", help="Path to save JSON results file")
    args = parser.parse_args()

    evaluate(args.json, args.dataset_root, args.json_log)