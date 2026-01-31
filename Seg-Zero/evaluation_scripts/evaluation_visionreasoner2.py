#!/usr/bin/env python3
import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2 # ADDED: For local checkpoints
from qwen_vl_utils import process_vision_info
import torch
import torch.nn as nn
import json
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import pdb
import os
import re
import numpy as np

# ---------------------------------------------------------
# ARCHITECTURES: Support Both Simple (Exp1/4) and Cross-Attn (Exp3)
# ---------------------------------------------------------
class SimpleBboxHead(nn.Module):
    def __init__(self, hidden_dim, mid=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, 4),
        )
    def forward(self, pooled, full_seq=None):
        return self.net(pooled)

class CrossAttnBboxHead(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, pooled, full_seq):
        query = pooled.unsqueeze(1)
        attn_out, _ = self.cross_attn(query, full_seq, full_seq)
        x = self.norm(query + attn_out).squeeze(1)
        return self.mlp(x)

# ---------------------------------------------------------
# Existing Logic
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    # ADDED: Config argument for local checkpoints
    parser.add_argument("--sam_config", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml", 
                       help="Config file for SAM2 (only used if loading local checkpoint)")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    return parser.parse_args()

def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    pred_bboxes = []
    pred_points = []
    
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if isinstance(data, list):
                for item in data:
                    if 'bbox_2d' in item:
                        pred_bboxes.append([
                            int(item['bbox_2d'][0] * x_factor + 0.5),
                            int(item['bbox_2d'][1] * y_factor + 0.5),
                            int(item['bbox_2d'][2] * x_factor + 0.5),
                            int(item['bbox_2d'][3] * y_factor + 0.5)
                        ])
                    if 'point_2d' in item:
                        pred_points.append([
                            int(item['point_2d'][0] * x_factor + 0.5),
                            int(item['point_2d'][1] * y_factor + 0.5)
                        ])
        except Exception:
            pass
    
    think_pattern = r'<think>([^<]+)</think>'
    think_text = ""
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1).strip()
    
    return pred_bboxes, pred_points, think_text

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection, union

def compute_bbox_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    return intersection / union

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Load Reasoning Model
    print(f"[Eval] Loading reasoning model: {args.reasoning_model_path}")
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    reasoning_model.eval()
    
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    # --- AUTO-DETECT HEAD TYPE ---
    bbox_head = None
    path_cross = os.path.join(args.reasoning_model_path, "bbox_head_crossattn.pt")
    path_simple = os.path.join(args.reasoning_model_path, "bbox_head.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Try Cross-Attention (Exp3)
    if os.path.exists(path_cross):
        print(f"[Eval] Found Cross-Attention Head: {path_cross}")
        sd = torch.load(path_cross, map_location="cpu")
        dim = sd["mlp.0.weight"].shape[1] if "mlp.0.weight" in sd else list(sd.values())[0].shape[1]
        bbox_head = CrossAttnBboxHead(hidden_dim=dim).to(device).float()
        bbox_head.load_state_dict(sd)
    
    # 2. Try Simple MLP (Exp1 / Exp4)
    elif os.path.exists(path_simple):
        print(f"[Eval] Found Simple MLP Head: {path_simple}")
        sd = torch.load(path_simple, map_location="cpu")
        key = "net.0.weight" if "net.0.weight" in sd else "0.weight"
        mid, hidden = sd[key].shape
        bbox_head = SimpleBboxHead(hidden, mid).to(device).float()
        bbox_head.load_state_dict(sd)
        
    if bbox_head:
        bbox_head.eval()
        print("[Eval] Head loaded successfully.")
    else:
        print("[Eval] No custom head found. Running text-only eval.")

    # Load SAM2 (Modified to support Local + HF)
    print(f"[Eval] Loading SAM2: {args.segmentation_model_path}")
    if os.path.exists(args.segmentation_model_path) and os.path.isfile(args.segmentation_model_path):
        print(f"Loading local SAM2 checkpoint from: {args.segmentation_model_path}")
        print(f"Using config: {args.sam_config}")
        sam_model = build_sam2(args.sam_config, args.segmentation_model_path)
        segmentation_model = SAM2ImagePredictor(sam_model)
    else:
        print(f"Loading SAM2 from HuggingFace Hub: {args.segmentation_model_path}")
        segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    resize_size = 840
    # Dataset loading logic (unchanged)
    try:
        dataset = load_from_disk(args.test_data_path)
    except:
        dataset = load_dataset(args.test_data_path, split='test')
        
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    dataset = dataset.select(range(start_idx, end_idx))
    
    if len(dataset) > 0 and 'bbox' in dataset[0]:
        has_bbox = True
    else:
        has_bbox = False

    # YOUR ORIGINAL PROMPT TEMPLATE (Preserved)
    QUESTION_TEMPLATE = \
    "Please find \"{Question}\" with bboxs and points." \
    "Compare the difference between tool(s) and find the most closely matched tool." \
    "Output the thinking process in <think> </think> (the output must be in english) and final answer in <answer> </answer> tags." \
    "Output the bbox(es) and point(s) inside the interested tool in JSON format." \
    "i.e., <think> thinking process here </think>" \
    "<answer>{Answer}</answer>"

    messages = []
    id_list = []
    
    for item in dataset:
        image = item["image"]
        if not hasattr(image, "convert"): # Handle non-PIL
             image = PILImage.open(image) if isinstance(image, str) else PILImage.fromarray(np.array(image))
        
        image = image.convert("RGB")
        
        # Handle 'text' vs 'problem' field
        q_text = item.get("text", "") or item.get("problem", "")
        
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
                },
                { 
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(
                        Question=q_text.lower().strip(".\"?!"),
                        Answer='[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}, {"bbox_2d": [225,296,706,786], "point_2d": [302,410]}]'
                    ) 
                }
            ]
        }]
        messages.append(message)
        id_list.append({
            "image_id": item.get("image_id", item.get("id")),
            "ann_id": item.get("ann_id", 0),
            "image": image,
            "mask": item.get("mask"),
            "img_height": item.get("img_height", image.height),
            "img_width": item.get("img_width", image.width),
            "bbox": item.get("bbox") if has_bbox else None
        })

    all_outputs = []
    
    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch_messages = messages[i:i + args.batch_size]
        batch_id_list = id_list[i:i + args.batch_size]

        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 1. Inference: Generation (Text)
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 2. Inference: BBox Head
        head_preds = [None] * len(batch_messages)
        if bbox_head is not None:
            with torch.no_grad():
                out = reasoning_model(**inputs, output_hidden_states=True)
                last_hidden = out.hidden_states[-1] # [B, S, D]
                pooled = last_hidden.mean(dim=1).float()
                
                # Forward pass depends on head type
                if isinstance(bbox_head, CrossAttnBboxHead):
                    seq = last_hidden.float()
                    raw_pred = bbox_head(pooled, seq)
                else:
                    # Simple MLP ignores seq
                    raw_pred = bbox_head(pooled)
                
                norm_pred = torch.sigmoid(raw_pred).cpu().numpy()
                head_preds = norm_pred

        # Processing Results
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                
                # --- HEAD PREDICTION PROCESSING ---
                head_bbox_iou = 0.0
                head_bbox_px = None
                
                if head_preds[id_idx] is not None:
                    w_orig = batch_id_list[id_idx]["img_width"]
                    h_orig = batch_id_list[id_idx]["img_height"]
                    p = head_preds[id_idx]
                    head_bbox_px = [
                        int(p[0] * w_orig), int(p[1] * h_orig),
                        int(p[2] * w_orig), int(p[3] * h_orig)
                    ]
                    if has_bbox and batch_id_list[id_idx]["bbox"]:
                        head_bbox_iou = compute_bbox_iou(head_bbox_px, batch_id_list[id_idx]["bbox"])

                # --- EXISTING REASONING LOGIC ---
                try:
                    bboxes, points, think = extract_bbox_points_think(
                        batch_output_text[id_idx], 
                        batch_id_list[id_idx]["img_width"]/resize_size, 
                        batch_id_list[id_idx]["img_height"]/resize_size
                    )
                except Exception as e:
                    print("Reasoning error: ", e, batch_id_list[id_idx]["image_id"])
                    think = ""
                    bboxes, points = [], []
                
                # SAM2 Segmentation
                intersection = 0
                union = 0
                if batch_id_list[id_idx]["mask"] is not None:
                    union = np.array(batch_id_list[id_idx]["mask"]).sum() 
                
                try:
                    segmentation_model.set_image(batch_id_list[id_idx]["image"])
                    mask_all = np.zeros((batch_id_list[id_idx]["img_height"], batch_id_list[id_idx]["img_width"]), dtype=bool)
                    
                    for bbox, point in zip(bboxes, points):
                        masks, scores, _ = segmentation_model.predict(
                            point_coords=[point],
                            point_labels=[1],
                            box=bbox
                        )
                        sorted_ind = np.argsort(scores)[::-1]
                        mask = masks[sorted_ind][0].astype(bool)
                        mask_resized = np.array(
                            PILImage.fromarray(mask.astype(np.uint8)).resize(
                                (batch_id_list[id_idx]["img_width"], batch_id_list[id_idx]["img_height"]),
                                resample=PILImage.NEAREST
                            )
                        ).astype(bool)
                        mask_all = np.logical_or(mask_all, mask_resized)

                    if batch_id_list[id_idx]["mask"] is not None:
                        gt_mask = np.array(batch_id_list[id_idx]["mask"])
                        if gt_mask.shape != mask_all.shape:
                            gt_mask = np.array(
                                PILImage.fromarray(gt_mask.astype(np.uint8)).resize(
                                    (mask_all.shape[1], mask_all.shape[0]),
                                    resample=PILImage.NEAREST
                                )
                            ).astype(bool)
                        intersection, union = compute_iou(mask_all, gt_mask)
                except Exception:
                    pass # Skip SAM errors gracefully

                # Reasoning BBox IoU
                bbox_iou = 0.0
                if has_bbox and batch_id_list[id_idx]["bbox"]:
                    try:     
                        gt_bbox = batch_id_list[id_idx]["bbox"]
                        for pred_bbox in bboxes:
                            cur_iou = compute_bbox_iou(pred_bbox, gt_bbox)
                            if cur_iou > bbox_iou:
                                bbox_iou = cur_iou
                    except Exception:
                        bbox_iou = 0.0

                all_outputs.append({
                    "image_id": batch_id_list[id_idx]["image_id"],
                    "ann_id": batch_id_list[id_idx]["ann_id"],
                    "think": think,
                    "intersection": int(intersection),
                    "union": int(union),
                    "bbox_iou": bbox_iou,          # Reasoning IoU
                    "head_bbox_iou": float(head_bbox_iou), # NEW: Head IoU
                    "head_pred_bbox": head_bbox_px # NEW: Head Prediction
                })

        print(f"Processed batch {i//args.batch_size + 1}")
        
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()