#!/usr/bin/env python3
"""
Micro evaluation to sanity-check the full pipeline end-to-end on a small
subset (default 32 samples). Safe to run interactively or via SLURM — it
will NOT request FlashAttention2 on CPU.
"""
import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage
import torch

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# -------------------------
# Helper: extract bboxes/points/think from generated text
# (kept local to avoid external dependency issues)
# -------------------------
import re

def extract_bbox_points_think(output_text: str, x_factor: float, y_factor: float):
    """
    Parse <answer> JSON and <think> content from model output.
    Returns: (pred_bboxes: List[List[int]], pred_points: List[List[int]], think_text: str)
    """
    pred_bboxes = []
    pred_points = []
    think_text = ""

    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if json_match:
        raw = json_match.group(1).strip()
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for item in data:
                    # Defensive extraction
                    bb = item.get("bbox_2d") if isinstance(item, dict) else None
                    pt = item.get("point_2d") if isinstance(item, dict) else None
                    if bb and len(bb) == 4:
                        pred_bboxes.append([
                            int(bb[0] * x_factor + 0.5),
                            int(bb[1] * y_factor + 0.5),
                            int(bb[2] * x_factor + 0.5),
                            int(bb[3] * y_factor + 0.5),
                        ])
                    if pt and len(pt) == 2:
                        pred_points.append([
                            int(pt[0] * x_factor + 0.5),
                            int(pt[1] * y_factor + 0.5),
                        ])
        except Exception:
            # JSON parse failed — leave lists empty
            pass

    think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
    if think_match:
        think_text = think_match.group(1).strip()

    return pred_bboxes, pred_points, think_text

# -------------------------
# IoU helpers
# -------------------------
def compute_bbox_iou(bbox1: List[int], bbox2: List[int]) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0, (bbox1[2] - bbox1[0])) * max(0, (bbox1[3] - bbox1[1]))
    area2 = max(0, (bbox2[2] - bbox2[0])) * max(0, (bbox2[3] - bbox2[1]))
    union = area1 + area2 - inter
    return (inter / union) if union > 0 else 0.0

# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reasoning_model_path", type=str,
                   default="/home/gaash/Wasif/Seg-Zero/checkpoints_STS/bbox_pretrain/final")
    p.add_argument("--test_data_path", type=str,
                   default="/home/gaash/Wasif/datasets/TestData/Combined_1016")
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--resize_size", type=int, default=840)
    p.add_argument("--batch_size", type=int, default=1)  # generation is easiest with batch_size=1
    return p.parse_args()

def main():
    args = parse_args()

    # Make sure you run this on a GPU node. If CUDA isn't available we fall back
    # to CPU and disable flash_attention_2 to avoid the earlier error.
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    # Choose attn implementation only if CUDA is available
    load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if use_cuda:
        load_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"[micro_eval] Loading reasoning model from: {args.reasoning_model_path} (device={device})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.reasoning_model_path, **load_kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    # load dataset (disk or HF) — prefer load_from_disk because your dataset is local
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(args.test_data_path)
    except Exception:
        from datasets import load_dataset
        dataset = load_dataset(args.test_data_path)

    total = len(dataset)
    n = min(args.num_samples, total)
    subset = dataset.select(range(n))

    resize_size = args.resize_size

    iou_list = []
    pass_count = 0
    processed = 0

    for item in tqdm(subset, desc="micro-eval"):
        try:
            image = item["image"].convert("RGB")
        except Exception:
            # if dataset stores images differently, skip
            print("[micro_eval] skipping item due to image load error")
            continue

        # build the message: use same template as your evaluation script
        QUESTION_TEMPLATE = (
            "Please find \"{Question}\" with bboxs and points."
            "Compare the difference between tool(s) and find the most closely matched tool."
            "Output the thinking process in <think> </think> (the output must be in english) and final answer in <answer> </answer> tags."
            "Output the bbox(es) and point(s) inside the interested tool in JSON format."
            "i.e., <think> thinking process here </think>"
            "<answer>{Answer}</answer>"
        )
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)},
                {"type": "text", "text": QUESTION_TEMPLATE.format(
                    Question=item.get("text", "").lower().strip(".\"?!"),
                    Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}]"
                )}
            ]
        }]

        # apply chat template (same as in eval pipeline)
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        # process vision inputs using the same helper
        image_inputs, video_inputs = process_vision_info([message])
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        # move to device
        inputs = inputs.to(device)

        # generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

        # trim input tokens to obtain generated portion
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        batch_output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # parse the model output (first element)
        out_text = batch_output_text[0]
        x_factor = item.get("img_width", resize_size) / float(resize_size)
        y_factor = item.get("img_height", resize_size) / float(resize_size)

        bboxes, points, think = extract_bbox_points_think(out_text, x_factor=x_factor, y_factor=y_factor)

        best_iou = 0.0
        if "bbox" in item and item["bbox"] is not None and len(bboxes) > 0:
            gt = item["bbox"]
            for pb in bboxes:
                best_iou = max(best_iou, compute_bbox_iou(pb, gt))

        iou_list.append(best_iou)
        if best_iou > 0.5:
            pass_count += 1

        processed += 1

        # free some GPU memory (safety)
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    import math
    avg_iou = float(np.mean(iou_list)) if len(iou_list) > 0 else 0.0
    pass_rate = (pass_count / processed) * 100 if processed > 0 else 0.0

    print("\n=== MICRO-EVAL RESULTS ===")
    print(f"Samples processed: {processed}")
    print(f"Avg BBox IoU: {avg_iou:.4f}")
    print(f"Pass Rate (>0.5): {pass_rate:.2f}%")

if __name__ == "__main__":
    main()
