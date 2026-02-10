#!/usr/bin/env python3
"""
pretrain_exp5_alignment.py
Experiment 5: Reasoning-BBox Alignment Loss.
- Architecture: Cross-Attention (Same as Exp3).
- Loss: Standard BBox Loss + 'Attention Alignment Loss'.
- Goal: Force the model's attention weights to concentrate INSIDE the Ground Truth BBox.
"""
import argparse
import os
import json
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
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--grad_clip", type=float, default=1.0)
    
    # Alignment Param
    p.add_argument("--attn_loss_weight", type=float, default=1.0, help="Weight for alignment loss")
    return p.parse_args()

# ---------------------------------------------------------
# HELPER: Attention Alignment Loss
# ---------------------------------------------------------
def compute_alignment_loss(attn_map, gt_bbox, grid_size=None):
    """
    attn_map: [B, 1, SeqLen] (We assume last N tokens are image)
    gt_bbox: [B, 4] (Normalized 0-1)
    """
    B, _, SeqLen = attn_map.shape
    
    # 1. Identify Image Tokens
    # Heuristic: Image features are the largest square at the end of sequence
    # Qwen2-VL: 24x24, 32x32, etc.
    if grid_size is None:
        grid_size = int(np.sqrt(SeqLen)) # Approximation
    
    img_len = grid_size * grid_size
    if img_len > SeqLen: img_len = SeqLen # Safety
    
    # Extract only image attention [B, H*W]
    img_attn = attn_map[:, 0, -img_len:] 
    
    # Reshape to 2D spatial grid [B, H, W]
    img_attn = img_attn.view(B, grid_size, grid_size)
    
    # 2. Create GT Masks
    masks = torch.zeros_like(img_attn)
    for i in range(B):
        x1, y1, x2, y2 = gt_bbox[i]
        # Scale normalized bbox to grid coordinates
        c1 = int(x1 * grid_size)
        r1 = int(y1 * grid_size)
        c2 = int(x2 * grid_size)
        r2 = int(y2 * grid_size)
        # Fill mask (valid region = 1.0)
        masks[i, r1:r2+1, c1:c2+1] = 1.0
        
    # 3. Compute Loss: Maximize attention inside mask
    # We want sum(attn * mask) to be close to 1.0 (since attn sums to 1)
    # Loss = 1.0 - (Mass Inside Box)
    
    # Normalize attn to sum to 1 over the image part (it might have leaked to text)
    img_attn_norm = img_attn / (img_attn.sum(dim=(1,2), keepdim=True) + 1e-6)
    
    mass_inside = (img_attn_norm * masks).sum(dim=(1,2))
    loss = 1.0 - mass_inside.mean()
    
    return loss

# ---------------------------------------------------------
# Architecture (Modified to return Weights)
# ---------------------------------------------------------
class CrossAttnBboxHead(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4))

    def forward(self, pooled, full_seq):
        query = pooled.unsqueeze(1)
        # Return weights
        attn_out, attn_weights = self.cross_attn(query, full_seq, full_seq, need_weights=True)
        x = self.norm(query + attn_out).squeeze(1)
        return self.mlp(x), attn_weights

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[Exp5] Loading from Exp3 Base: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, padding_side="left")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float32, device_map="auto"
    )
    # Freeze backbone (We only train the head to align)
    for p in model.parameters(): p.requires_grad = False
    
    # Load Dataset
    try: dataset = load_from_disk(args.dataset_path)
    except: dataset = load_dataset(args.dataset_path, split="train")
    if hasattr(dataset, "keys"): dataset = dataset["train" if "train" in dataset else list(dataset.keys())[0]]

    # Helper class from prev script
    class BoxDataset(torch.utils.data.Dataset):
        def __init__(self, ds, proc, size): self.ds, self.proc, self.size = ds, proc, size
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            item = self.ds[i]
            img = item["image"].convert("RGB").resize((self.size, self.size)) if hasattr(item["image"], "convert") else item["image"]
            txt = f"Locate: {item.get('problem') or item.get('text')}"
            try: bbox = json.loads(item["solution"])[0]["bbox_2d"]
            except: bbox = None
            return {"image": img, "text": txt, "bbox": bbox, "w": self.size, "h": self.size}

    def collate(batch):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": b["image"]}, {"type": "text", "text": b["text"]}]}] for b in batch]
        txts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inp = processor(text=txts, images=[b["image"] for b in batch], padding=True, return_tensors="pt")
        tgts = []
        for b in batch:
            if b["bbox"]: 
                arr = np.array(b["bbox"], dtype=float)
                arr[0]/=b["w"]; arr[2]/=b["w"]; arr[1]/=b["h"]; arr[3]/=b["h"]
                tgts.append(np.clip(arr,0,1).tolist())
            else: tgts.append([0.0]*4)
        inp["bbox_targets"] = torch.tensor(tgts, dtype=torch.float32)
        return inp

    dl = DataLoader(BoxDataset(dataset, processor, args.resize_size), batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    # Init Head (Start from Exp3 weights)
    bbox_head = None
    head_path = os.path.join(args.model_name_or_path, "bbox_head_crossattn.pt")
    
    if os.path.exists(head_path):
        print(f"[Exp5] Loading Exp3 Head: {head_path}")
        sd = torch.load(head_path)
        dim = sd["mlp.0.weight"].shape[1]
        bbox_head = CrossAttnBboxHead(dim).to(args.device, dtype=torch.float32)
        bbox_head.load_state_dict(sd)
    else:
        print("Error: Exp3 head not found. Cannot perform Exp5.")
        return

    optim = AdamW(bbox_head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
    
    model.train()
    for epoch in range(args.epochs):
        ep_loss = 0
        loop = tqdm(dl)
        for batch in loop:
            input_ids = batch["input_ids"].to(args.device)
            bbox_targets = batch["bbox_targets"].to(args.device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=batch["attention_mask"].to(args.device), 
                           pixel_values=batch["pixel_values"].to(args.device), 
                           image_grid_thw=batch["image_grid_thw"].to(args.device),
                           output_hidden_states=True)
                last = out.hidden_states[-1].to(torch.float32)
                pooled = last.mean(dim=1)

            # Forward Head (Returns preds AND weights)
            preds, attn_weights = bbox_head(pooled, last)
            preds = torch.sigmoid(preds)
            
            valid = (bbox_targets.sum(dim=1)>0).float()
            if valid.sum() > 0:
                l1 = F.l1_loss(preds, bbox_targets, reduction='none').mean(dim=1)
                
                # ALIGNMENT LOSS
                l_align = compute_alignment_loss(attn_weights, bbox_targets)
                
                loss = ((l1 + args.attn_loss_weight * l_align) * valid).sum() / valid.sum()
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                ep_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}", align=f"{l_align.item():.4f}")
        
        print(f"Epoch {epoch+1} Avg Loss: {ep_loss/len(dl):.4f}")

    print(f"Saving to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    torch.save(bbox_head.state_dict(), os.path.join(args.output_dir, "bbox_head_crossattn.pt"))
    try: processor.save_pretrained(args.output_dir)
    except: pass

if __name__ == "__main__":
    main()