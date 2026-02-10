#!/usr/bin/env python3
"""
pretrain_curriculum.py (Fixed: Removed buggy dummy forward)
Matches pretrain_bbox_supervise.py logic (CoarseFine head + Refiner)
but loads specific curriculum splits from disk.
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
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--base_model_path", type=str, required=True, help="Fallback for processor")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--split", type=str, required=True, help="easy, medium, or hard")
    p.add_argument("--resize_size", type=int, default=840)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=5e-4)
    p.add_argument("--lr_refiner", type=float, default=5e-4)
    p.add_argument("--backbone_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--freeze_backbone", type=str, default="partial")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--unfreeze_last_n_vision", type=int, default=1)
    p.add_argument("--unfreeze_last_m_lm", type=int, default=1)
    return p.parse_args()

# -----------------------
# Robust GIoU
# -----------------------
def giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    pred = pred.clone()
    target = target.clone()
    
    px1 = torch.min(pred[:, 0], pred[:, 2])
    px2 = torch.max(pred[:, 0], pred[:, 2])
    py1 = torch.min(pred[:, 1], pred[:, 3])
    py2 = torch.max(pred[:, 1], pred[:, 3])
    pred = torch.stack([px1, py1, px2, py2], dim=1).clamp(0.0, 1.0)
    
    tx1 = torch.min(target[:, 0], target[:, 2])
    tx2 = torch.max(target[:, 0], target[:, 2])
    ty1 = torch.min(target[:, 1], target[:, 3])
    ty2 = torch.max(target[:, 1], target[:, 3])
    target = torch.stack([tx1, ty1, tx2, ty2], dim=1).clamp(0.0, 1.0)
    
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

    xA = torch.max(pred[:,0], target[:,0])
    yA = torch.max(pred[:,1], target[:,1])
    xB = torch.min(pred[:,2], target[:,2])
    yB = torch.min(pred[:,3], target[:,3])

    inter_w = (xB - xA).clamp(min=0.0)
    inter_h = (yB - yA).clamp(min=0.0)
    inter = inter_w * inter_h

    area_p = ((pred[:,2] - pred[:,0]).clamp(min=0.0)) * ((pred[:,3] - pred[:,1]).clamp(min=0.0))
    area_t = ((target[:,2] - target[:,0]).clamp(min=0.0)) * ((target[:,3] - target[:,1]).clamp(min=0.0))
    union = area_p + area_t - inter
    iou = inter / (union + eps)
    
    xC = torch.min(pred[:,0], target[:,0])
    yC = torch.min(pred[:,1], target[:,1])
    xD = torch.max(pred[:,2], target[:,2])
    yD = torch.max(pred[:,3], target[:,3])
    enc_w = (xD - xC).clamp(min=0.0)
    enc_h = (yD - yC).clamp(min=0.0)
    enc_area = enc_w * enc_h + eps

    giou = iou - (enc_area - union) / enc_area
    return 1.0 - giou

def bbox_iou(pred, target, eps=1e-7):
    return 1.0 - giou_loss(pred, target, eps)

# -----------------------
# Dataset
# -----------------------
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
            if isinstance(solution_field, str):
                data = json.loads(solution_field)
            elif isinstance(solution_field, list):
                data = solution_field
            else:
                return None
            if not data or not isinstance(data[0], dict) or "bbox_2d" not in data[0]:
                return None
            return data[0]["bbox_2d"]
        except Exception:
            return None

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if not hasattr(image, "convert"):
             from PIL import Image as PILImage
             if isinstance(image, dict) and "bytes" in image:
                 import io
                 image = PILImage.open(io.BytesIO(image["bytes"]))
             else:
                 image = PILImage.open(image).convert("RGB")
        
        image = image.convert("RGB").resize((self.resize_size, self.resize_size))
        text = "Locate the object(s) in the image and output bounding box coordinates."
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
# Model Components (MATCHING STABLE SCRIPT)
# -----------------------
class CoarseFineBboxHead(nn.Module):
    def __init__(self, hidden_dim, num_bins=256, mid=512):
        super().__init__()
        self.num_bins = num_bins
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.ReLU(),
            nn.Linear(mid, mid), nn.ReLU(),
        )
        self.bin_heads = nn.ModuleList([nn.Linear(mid, num_bins) for _ in range(4)])
        self.residual_heads = nn.ModuleList([nn.Linear(mid, 1) for _ in range(4)])

    def forward(self, hidden_state):
        x = self.shared(hidden_state)
        bin_logits = [head(x) for head in self.bin_heads]
        residuals = [head(x).squeeze(-1) for head in self.residual_heads]
        bin_probs = [F.softmax(b, dim=-1) for b in bin_logits]
        device = hidden_state.device
        centers = torch.arange(0, self.num_bins, device=device).float() / float(self.num_bins - 1)
        coarse = [ (p * centers).sum(dim=-1) for p in bin_probs ]
        coarse = torch.stack(coarse, dim=1)
        residuals = torch.stack(residuals, dim=1)
        residuals = torch.tanh(residuals) * (1.0 / self.num_bins)
        coords = (coarse + residuals).clamp(0.0, 1.0)
        return {"bin_logits": bin_logits, "coords": coords}

class BboxRefiner(nn.Module):
    def __init__(self, visual_feat_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_feat_dim + 4, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )
    def forward(self, visual_feat, init_bbox):
        x = torch.cat([visual_feat, init_bbox], dim=-1)
        delta = self.net(x)
        refined = (init_bbox + 0.2 * torch.tanh(delta)).clamp(0.0, 1.0)
        return refined

def compute_start_idxs(input_ids: torch.Tensor, pad_token_id: int):
    lens = (input_ids != pad_token_id).long().sum(dim=1)
    start_idxs = (lens - 1).clamp(min=0)
    return start_idxs

def apply_partial_unfreeze(model, unfreeze_last_n_vision=1, unfreeze_last_m_lm=1):
    for p in model.parameters(): p.requires_grad = False
    # (Reuse existing logic from stable script if needed, simpler block here for brevity)
    # Assuming standard Qwen2.5-VL structure
    for name, p in model.named_parameters():
        if "vision" in name and any(f"blocks.{i}" in name for i in range(20, 40)): # Heuristic
             p.requires_grad = True
        if "model.layers" in name and any(f"layers.{i}" in name for i in range(20, 40)):
             p.requires_grad = True
    print("[Curriculum] Applied partial unfreeze.")

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    print(f"[Curriculum] Loading split '{args.split}' from {args.dataset_path}")
    ds_dict = load_from_disk(args.dataset_path)
    hf_split = ds_dict[args.split]
    
    print(f"[Curriculum] Loading model: {args.model_name_or_path}")
    
    try:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path, padding_side="left")
    except Exception as e:
        print(f"[Curriculum] Warning: Failed to load processor ({e}). Fallback: {args.base_model_path}")
        processor = AutoProcessor.from_pretrained(args.base_model_path, padding_side="left")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float32, device_map="auto"
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Freeze logic
    if args.freeze_backbone.lower() == "true":
        for p in model.parameters(): p.requires_grad = False
    elif args.freeze_backbone.lower() == "partial":
        apply_partial_unfreeze(model, args.unfreeze_last_n_vision, args.unfreeze_last_m_lm)
    
    ds = BoxDataset(hf_split, processor, resize_size=args.resize_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=True, collate_fn=lambda b: collate_fn(b, processor, args.resize_size))

    # BBox Head Logic (COARSE + REFINER)
    bbox_head = None
    refiner = None
    head_path = os.path.join(args.model_name_or_path, "bbox_head.pt")
    refiner_path = os.path.join(args.model_name_or_path, "bbox_refiner.pt")
    
    # Initialize heads
    # We need hidden_dim. We can assume standard Qwen size or do a safe lookup.
    # Instead of dummy forward, we will lazy init on first batch.
    
    optim = None 
    pad_token_id = processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else 0

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        loop = tqdm(dl, ncols=120, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            vision_kwargs = {k:v.to(device) for k,v in batch.items() if k in ["pixel_values", "image_grid_thw"]}
            bbox_targets = torch.nan_to_num(batch["bbox_targets"].to(device, torch.float32), nan=0.0)

            with torch.set_grad_enabled(optim is not None):
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, **vision_kwargs)
            
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
            last_hidden = torch.nan_to_num(last_hidden.to(torch.float32), nan=0.0)
            
            start_idxs = compute_start_idxs(input_ids, pad_token_id).to(device)
            batch_idx = torch.arange(last_hidden.size(0), device=device)
            pooled = last_hidden[batch_idx, start_idxs, :]

            # Lazy Init Head/Refiner
            if bbox_head is None:
                hidden_dim = pooled.size(-1)
                print(f"[Curriculum] Initializing CoarseFineBboxHead (dim={hidden_dim})")
                bbox_head = CoarseFineBboxHead(hidden_dim, num_bins=args.num_bins, mid=max(64, hidden_dim // 2)).to(device, dtype=torch.float32)
                refiner = BboxRefiner(visual_feat_dim=512, hidden=256).to(device, dtype=torch.float32)

                if os.path.exists(head_path):
                    print(f"[Curriculum] Loading bbox_head from {head_path}")
                    bbox_head.load_state_dict(torch.load(head_path, map_location="cpu"))
                if os.path.exists(refiner_path):
                    print(f"[Curriculum] Loading refiner from {refiner_path}")
                    refiner.load_state_dict(torch.load(refiner_path, map_location="cpu"))
                
                # Init Optimizer now that params exist
                head_params = list(bbox_head.parameters())
                refiner_params = list(refiner.parameters())
                backbone_params = [p for n,p in model.named_parameters() if p.requires_grad]
                
                param_groups = [
                    {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
                    {"params": refiner_params, "lr": args.lr_refiner, "weight_decay": args.weight_decay},
                ]
                if backbone_params:
                    param_groups.append({"params": backbone_params, "lr": args.backbone_lr, "weight_decay": 1e-6})
                
                optim = AdamW(param_groups)

            # Forward Head
            head_out = bbox_head(pooled)
            coords = head_out["coords"]
            
            # Visual feats for refiner
            visual_feats = torch.zeros((pooled.size(0), 512), device=device, dtype=torch.float32)
            if hasattr(out, "image_embeds") and out.image_embeds is not None:
                 visual_feats = torch.nan_to_num(out.image_embeds.to(torch.float32), nan=0.0)
            
            refined = refiner(visual_feats, coords)

            # Loss
            valid_mask = (bbox_targets.sum(dim=1) > 0).float()
            if valid_mask.sum() > 0:
                bin_ce = 0.0
                for i in range(4):
                    gt_coord = bbox_targets[:, i]
                    gt_bin = (gt_coord * (args.num_bins - 1)).round().long().clamp(0, args.num_bins - 1)
                    bin_ce += F.cross_entropy(head_out["bin_logits"][i], gt_bin)
                bin_ce /= 4.0
                
                res_l1 = F.l1_loss(coords, bbox_targets)
                g_loss = giou_loss(refined, bbox_targets).mean()
                
                loss = 3.0 * g_loss + 1.0 * bin_ce + 0.5 * res_l1
                
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(param_groups[0]["params"] + param_groups[1]["params"] + backbone_params, args.grad_clip)
                optim.step()
                
                epoch_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
        
        print(f"Epoch {epoch+1} done. Avg Loss: {epoch_loss / max(1, len(dl)):.5f}")

    print(f"[Curriculum] Saving to {final_dir}")
    model.save_pretrained(final_dir)
    torch.save(bbox_head.state_dict(), os.path.join(final_dir, "bbox_head.pt"))
    torch.save(refiner.state_dict(), os.path.join(final_dir, "bbox_refiner.pt"))
    
    try:
        processor.save_pretrained(final_dir)
    except Exception:
        pass

if __name__ == "__main__":
    main()