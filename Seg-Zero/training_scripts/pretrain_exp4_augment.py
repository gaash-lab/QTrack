#!/usr/bin/env python3
"""
pretrain_exp4_augment.py (Fixed Head Loading)
- Architecture: Simple Linear Head.
- Augmentation: Rotation, Scale, Translation.
- Fix: Automatically skips loading old weights if architecture doesn't match.
"""
import argparse
import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW
from PIL import Image as PILImage
import torchvision.transforms.functional as TF

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--resize_size", type=int, default=840)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--freeze_backbone", type=str, default="partial")
    
    # Augmentation Params
    p.add_argument("--aug_prob", type=float, default=0.5)
    p.add_argument("--max_rotate", type=float, default=15.0)
    return p.parse_args()

# -----------------------
# Augmentation Logic
# -----------------------
def apply_augmentation(image, bbox, max_rotate=15.0):
    w, h = image.size
    angle = random.uniform(-max_rotate, max_rotate)
    scale = random.uniform(0.9, 1.1)
    tx = random.uniform(-0.05 * w, 0.05 * w)
    ty = random.uniform(-0.05 * h, 0.05 * h)
    
    aug_img = TF.affine(image, angle, (tx, ty), scale, 0, fill=0)
    
    if bbox is None: return aug_img, None

    x1, y1, x2, y2 = bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    cx, cy = w / 2, h / 2
    rad = -np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    
    new_corners = []
    for x, y in corners:
        x -= cx; y -= cy
        x *= scale; y *= scale
        nx = x * cos_a - y * sin_a
        ny = x * sin_a + y * cos_a
        nx += cx + tx; ny += cy + ty
        new_corners.append((nx, ny))
        
    xs = [p[0] for p in new_corners]
    ys = [p[1] for p in new_corners]
    
    nx1, ny1 = max(0, min(xs)), max(0, min(ys))
    nx2, ny2 = min(w, max(xs)), min(h, max(ys))
    
    if nx2 <= nx1 or ny2 <= ny1: return image, bbox
    return aug_img, [nx1/w, ny1/h, nx2/w, ny2/h]

# -----------------------
# Dataset
# -----------------------
class AugmentedBoxDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, processor, resize_size=840, aug_prob=0.5):
        self.ds = hf_split
        self.processor = processor
        self.resize_size = resize_size
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.ds)

    def _extract_first_bbox(self, solution_field):
        if not solution_field: return None
        try:
            data = json.loads(solution_field) if isinstance(solution_field, str) else solution_field
            return data[0]["bbox_2d"] if data and "bbox_2d" in data[0] else None
        except: return None

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if not hasattr(image, "convert"):
             from PIL import Image as PILImage
             if isinstance(image, dict) and "bytes" in image:
                 import io; image = PILImage.open(io.BytesIO(image["bytes"]))
             else: image = PILImage.open(image).convert("RGB")
        
        image = image.convert("RGB")
        text = f"Locate the object: {item.get('problem') or item.get('text') or 'Locate object.'}"
        bbox = self._extract_first_bbox(item.get("solution", None))
        
        norm_bbox = None
        if bbox:
            w, h = item.get("img_width", image.width), item.get("img_height", image.height)
            norm_bbox = [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]

        if random.random() < self.aug_prob:
            image, norm_bbox = apply_augmentation(image, norm_bbox)

        image = image.resize((self.resize_size, self.resize_size))
        return {"image": image, "text": text, "bbox": norm_bbox}

def collate_fn(batch, processor):
    messages = [[{"role": "user", "content": [{"type": "image", "image": b["image"]}, {"type": "text", "text": b["text"]}]}] for b in batch]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = processor(text=texts, images=[m[0]["content"][0]["image"] for m in messages], padding=True, return_tensors="pt")
    
    b_targets = []
    for b in batch:
        tgt = b["bbox"]
        if tgt is None: b_targets.append([0.0]*4)
        else: b_targets.append(np.clip(np.nan_to_num(tgt, nan=0.0), 0.0, 1.0).tolist())
    
    inputs["bbox_targets"] = torch.tensor(b_targets, dtype=torch.float32)
    return inputs

# -----------------------
# Model (Simple MLP)
# -----------------------
class SimpleBboxHead(nn.Module):
    def __init__(self, hidden_dim, mid=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, mid), nn.ReLU(), nn.Linear(mid, 4))
    def forward(self, x): return self.net(x)

def bbox_iou(pred, target, eps=1e-7):
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = (pred[:, 2]-pred[:, 0])*(pred[:, 3]-pred[:, 1]) + (target[:, 2]-target[:, 0])*(target[:, 3]-target[:, 1]) - inter
    return inter / (union + eps)

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    print(f"[Exp4] Loading model: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float32, device_map="auto"
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.freeze_backbone == "partial":
        for p in model.parameters(): p.requires_grad = False
        for name, p in model.named_parameters():
            if "vision" in name and any(f"blocks.{i}" in name for i in range(20, 40)): p.requires_grad = True
    
    try: dataset = load_from_disk(args.dataset_path)
    except: dataset = load_dataset(args.dataset_path, split="train")
    if hasattr(dataset, "keys") and not hasattr(dataset, "select"): 
        dataset = dataset["train" if "train" in dataset else list(dataset.keys())[0]]

    ds = AugmentedBoxDataset(dataset, processor, resize_size=args.resize_size, aug_prob=args.aug_prob)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, collate_fn=lambda b: collate_fn(b, processor))

    bbox_head = None
    optim = None

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        loop = tqdm(dl, ncols=120, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            vision_kwargs = {k:v.to(device) for k,v in batch.items() if k in ["pixel_values", "image_grid_thw"]}
            bbox_targets = batch["bbox_targets"].to(device, torch.float32)

            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, **vision_kwargs)
            
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
            pooled = last_hidden.to(torch.float32).mean(dim=1)

            if bbox_head is None:
                hidden_dim = pooled.size(-1)
                bbox_head = SimpleBboxHead(hidden_dim).to(device, dtype=torch.float32)
                
                # --- FIXED LOAD LOGIC ---
                head_path = os.path.join(args.model_name_or_path, "bbox_head.pt")
                if os.path.exists(head_path):
                    print(f"[Exp4] Found existing head at {head_path}")
                    try:
                        # Attempt to load
                        bbox_head.load_state_dict(torch.load(head_path))
                        print("[Exp4] Success: Loaded existing SimpleBboxHead weights.")
                    except RuntimeError:
                        # Fallback for architecture mismatch
                        print("[Exp4] Warning: Architecture mismatch (likely Complex->Simple).")
                        print("[Exp4] Action: Initializing FRESH SimpleHead. Backbone is preserved.")
                        # No load = fresh random init
                # ------------------------
                
                params = list(bbox_head.parameters()) + [p for p in model.parameters() if p.requires_grad]
                optim = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

            pred = torch.sigmoid(bbox_head(pooled))
            
            valid_mask = (bbox_targets.sum(dim=1) > 0).float()
            if valid_mask.sum() > 0:
                l1 = F.l1_loss(pred, bbox_targets, reduction='none').mean(dim=1)
                iou = bbox_iou(pred, bbox_targets)
                loss = ((l1 + (1.0 - iou)) * valid_mask).sum() / valid_mask.sum()
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss / max(1, len(dl)):.5f}")

    print(f"[Exp4] Saving to {final_dir}")
    model.save_pretrained(final_dir)
    torch.save(bbox_head.state_dict(), os.path.join(final_dir, "bbox_head.pt"))
    try: processor.save_pretrained(final_dir)
    except: pass

if __name__ == "__main__":
    main()