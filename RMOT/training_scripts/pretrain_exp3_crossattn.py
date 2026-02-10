#!/usr/bin/env python3
"""
pretrain_exp3_crossattn.py
Experiment 3: Cross-Attention BBox Head.
- Uses "problem" and "solution" fields from STS dataset.
- Freezes backbone by default (trains only the new head) for memory safety.
- Saves tokenizer/processor for seamless evaluation.
"""
import argparse
import os
import json
import re
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--resize_size", type=int, default=840)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--attn_heads", type=int, default=8)
    p.add_argument("--attn_dropout", type=float, default=0.1)
    return p.parse_args()

# -----------------------
# Helpers
# -----------------------
def giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred.clone()
    target = target.clone()
    px1, px2 = torch.min(pred[:,0], pred[:,2]), torch.max(pred[:,0], pred[:,2])
    py1, py2 = torch.min(pred[:,1], pred[:,3]), torch.max(pred[:,1], pred[:,3])
    pred = torch.stack([px1, py1, px2, py2], dim=1).clamp(0.0, 1.0)
    
    tx1, tx2 = torch.min(target[:,0], target[:,2]), torch.max(target[:,0], target[:,2])
    ty1, ty2 = torch.min(target[:,1], target[:,3]), torch.max(target[:,1], target[:,3])
    target = torch.stack([tx1, ty1, tx2, ty2], dim=1).clamp(0.0, 1.0)
    
    inter_w = (torch.min(pred[:,2], target[:,2]) - torch.max(pred[:,0], target[:,0])).clamp(min=0)
    inter_h = (torch.min(pred[:,3], target[:,3]) - torch.max(pred[:,1], target[:,1])).clamp(min=0)
    inter = inter_w * inter_h
    
    area_p = (pred[:,2]-pred[:,0]) * (pred[:,3]-pred[:,1])
    area_t = (target[:,2]-target[:,0]) * (target[:,3]-target[:,1])
    union = area_p + area_t - inter + eps
    iou = inter / union
    
    cw = torch.max(pred[:,2], target[:,2]) - torch.min(pred[:,0], target[:,0])
    ch = torch.max(pred[:,3], target[:,3]) - torch.min(pred[:,1], target[:,1])
    c_area = cw.clamp(min=0) * ch.clamp(min=0) + eps
    
    return 1.0 - (iou - (c_area - union) / c_area)

class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, processor, resize_size=840):
        self.ds = hf_split
        self.processor = processor
        self.resize_size = resize_size

    def __len__(self):
        return len(self.ds)

    def _extract_first_bbox(self, solution_field):
        if solution_field is None: return None
        try:
            if isinstance(solution_field, str): data = json.loads(solution_field)
            elif isinstance(solution_field, list): data = solution_field
            else: return None
            if not data or not isinstance(data[0], dict) or "bbox_2d" not in data[0]: return None
            return data[0]["bbox_2d"]
        except: return None

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if not hasattr(image, "convert"):
             from PIL import Image as PILImage
             if isinstance(image, dict) and "bytes" in image:
                 import io; image = PILImage.open(io.BytesIO(image["bytes"]))
             else: image = PILImage.open(image).convert("RGB")
        
        image = image.convert("RGB").resize((self.resize_size, self.resize_size))
        # Handle STS fields
        text_raw = item.get("problem") or item.get("text") or "Locate object."
        text = f"Locate the object: {text_raw}"
        bbox = self._extract_first_bbox(item.get("solution", None))
        width = item.get("img_width", self.resize_size)
        height = item.get("img_height", self.resize_size)
        return {"image": image, "text": text, "bbox": bbox, "width": width, "height": height}

def collate_fn(batch, processor, resize_size=840):
    messages = []
    targets = []
    for b in batch:
        messages.append([{"role": "user", "content": [{"type": "image", "image": b["image"]}, {"type": "text", "text": b["text"]}]}])
        targets.append(b["bbox"])
    
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = processor(text=texts, images=[m[0]["content"][0]["image"] for m in messages], padding=True, return_tensors="pt")
    
    b_targets = []
    for tgt, b in zip(targets, batch):
        w, h = b["width"], b["height"]
        if tgt is None:
            b_targets.append([0.0, 0.0, 0.0, 0.0])
            continue
        arr = np.array(tgt, dtype=float)
        if arr.size != 4:
            b_targets.append([0.0, 0.0, 0.0, 0.0])
            continue
        arr[0] /= float(w); arr[2] /= float(w)
        arr[1] /= float(h); arr[3] /= float(h)
        arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        b_targets.append(arr.tolist())
    
    inputs["bbox_targets"] = torch.tensor(b_targets, dtype=torch.float32)
    return inputs

# -----------------------
# Cross-Attention Architecture
# -----------------------
class CrossAttnBboxHead(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, pooled_query, full_sequence):
        # pooled_query: [B, D] -> [B, 1, D]
        query = pooled_query.unsqueeze(1)
        # Attention
        attn_out, _ = self.cross_attn(query, full_sequence, full_sequence)
        # Residual + Norm
        x = self.norm(query + attn_out)
        x = x.squeeze(1)
        return self.mlp(x)

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    print(f"[Exp3] Loading model: {args.model_name_or_path}")
    try:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, padding_side="left")
    except:
        # Fallback
        print("[Exp3] Warning: processor load failed, falling back to default.")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", padding_side="left")
        
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float32, device_map="auto"
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Freeze Backbone
    print("[Exp3] Freezing backbone. Training CrossAttn Head only.")
    for p in model.parameters(): p.requires_grad = False

    print(f"[Exp3] Loading dataset: {args.dataset_path}")
    try:
        dataset = load_from_disk(args.dataset_path)
    except:
        dataset = load_dataset(args.dataset_path, split="train")
    
    # Handle DatasetDict (like in your curriculum script)
    if hasattr(dataset, "keys") and not hasattr(dataset, "select"): 
        if "train" in dataset: dataset = dataset["train"]
        else: dataset = dataset[list(dataset.keys())[0]]

    ds = BoxDataset(dataset, processor, resize_size=args.resize_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, collate_fn=lambda b: collate_fn(b, processor, args.resize_size))

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
            bbox_targets = torch.nan_to_num(batch["bbox_targets"].to(device, torch.float32), nan=0.0)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, **vision_kwargs)
            
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
            last_hidden = torch.nan_to_num(last_hidden.to(torch.float32), nan=0.0)
            pooled = last_hidden.mean(dim=1)

            if bbox_head is None:
                hidden_dim = pooled.size(-1)
                print(f"[Exp3] Init CrossAttnHead dim={hidden_dim}")
                bbox_head = CrossAttnBboxHead(hidden_dim, num_heads=args.attn_heads, dropout=args.attn_dropout).to(device, dtype=torch.float32)
                optim = AdamW(bbox_head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)

            raw_coords = bbox_head(pooled_query=pooled, full_sequence=last_hidden)
            pred = torch.sigmoid(raw_coords)
            
            valid_mask = (bbox_targets.sum(dim=1) > 0).float()
            if valid_mask.sum() > 0:
                l1 = F.l1_loss(pred, bbox_targets, reduction='none').mean(dim=1)
                giou = giou_loss(pred, bbox_targets)
                loss = ((l1 + 2.0*giou) * valid_mask).sum() / valid_mask.sum()
                
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bbox_head.parameters(), args.grad_clip)
                optim.step()
                
                epoch_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss / max(1, len(dl)):.5f}")

    print(f"[Exp3] Saving to {final_dir}")
    # Save Head
    torch.save(bbox_head.state_dict(), os.path.join(final_dir, "bbox_head.pt"))
    # Save base model & processor
    try:
        model.save_pretrained(final_dir)
        # Robust processor copy (matches your stable script logic)
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model_name_or_path)
            proc = AutoProcessor.from_pretrained(args.model_name_or_path, padding_side="left")
            tok.save_pretrained(final_dir)
            proc.save_pretrained(final_dir)
            print("[Exp3] Tokenizer/Processor saved successfully.")
        except Exception as e:
            # Fallback: try just saving the processor if we loaded it
            if processor: processor.save_pretrained(final_dir)
            print(f"[Exp3] Warning on processor save: {e}")
    except: pass

if __name__ == "__main__":
    main()