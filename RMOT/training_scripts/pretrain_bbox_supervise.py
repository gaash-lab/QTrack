#!/usr/bin/env python3
"""
Robust pretrain_bbox_supervise.py (single-file) — FINAL

Changes:
- Model is loaded in FP32 (torch_dtype=torch.float32) to avoid fp16 overflow/NaN issues.
- Coarse-to-fine bbox head (bins + residual)
- Visual refiner (MLP)
- Start-token pooling (instead of mean pooling)
- Targeted partial-unfreeze (unfreeze last N vision blocks + last M LM layers)
- Robust GIoU implementation + NaN/Inf protections
- Sanitization of tensors before loss
- Gradient clipping
- Separate optimizer param groups for head/refiner/backbone
- Safer default LRs for head/refiner to reduce initial instability

Usage: same CLI as before; run with a small debug job first:
  python training_scripts/pretrain_bbox_supervise.py --model_name_or_path <path> --output_dir <out> --dataset_name <ds> --epochs 1 --batch_size 2
"""
import argparse
import os
import json
import re
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from torch.optim import AdamW
import torch.nn.functional as F

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--resize_size", type=int, default=840)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lr_head", type=float, default=5e-4, help="LR for bbox head (safer default)")
    p.add_argument("--lr_refiner", type=float, default=5e-4, help="LR for refiner (safer default)")
    p.add_argument("--backbone_lr", type=float, default=5e-6, help="LR for partially-unfrozen backbone params")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--freeze_backbone",
        type=str,
        default="partial",
        help="true (freeze all), partial (targeted), false (train all)",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--num_bins", type=int, default=256)
    p.add_argument("--unfreeze_last_n_vision", type=int, default=1)
    p.add_argument("--unfreeze_last_m_lm", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    return p.parse_args()


# -----------------------
# Robust GIoU & helpers
# -----------------------
def bbox_area(box: torch.Tensor):
    w = (box[:,2] - box[:,0]).clamp(min=0.0)
    h = (box[:,3] - box[:,1]).clamp(min=0.0)
    return w * h


def giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Robust GIoU loss:
    - reorder coords so x1<=x2, y1<=y2
    - clamp to [0,1]
    - safe nan handling
    """
    pred = pred.clone()
    target = target.clone()

    # reorder pred
    px1 = torch.min(pred[:, 0], pred[:, 2])
    px2 = torch.max(pred[:, 0], pred[:, 2])
    py1 = torch.min(pred[:, 1], pred[:, 3])
    py2 = torch.max(pred[:, 1], pred[:, 3])
    pred = torch.stack([px1, py1, px2, py2], dim=1)

    # reorder target
    tx1 = torch.min(target[:, 0], target[:, 2])
    tx2 = torch.max(target[:, 0], target[:, 2])
    ty1 = torch.min(target[:, 1], target[:, 3])
    ty2 = torch.max(target[:, 1], target[:, 3])
    target = torch.stack([tx1, ty1, tx2, ty2], dim=1)

    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
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
    iou = torch.nan_to_num(iou, nan=0.0, posinf=1.0, neginf=0.0)

    xC = torch.min(pred[:,0], target[:,0])
    yC = torch.min(pred[:,1], target[:,1])
    xD = torch.max(pred[:,2], target[:,2])
    yD = torch.max(pred[:,3], target[:,3])
    enc_w = (xD - xC).clamp(min=0.0)
    enc_h = (yD - yC).clamp(min=0.0)
    enc_area = enc_w * enc_h + eps

    giou = iou - (enc_area - union) / enc_area
    giou = torch.nan_to_num(giou, nan=0.0, posinf=1.0, neginf=-1.0)
    return 1.0 - giou


# -----------------------
# Dataset wrapper
# -----------------------
class BoxDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, processor, resize_size: int = 840):
        self.ds = hf_split
        self.processor = processor
        self.resize_size = resize_size

    def __len__(self):
        return len(self.ds)

    def _extract_first_bbox(self, solution_field):
        if solution_field is None:
            return None
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.ds[idx]
        image = item["image"].convert("RGB").resize((self.resize_size, self.resize_size))
        text = "Locate the object(s) in the image and output bounding box coordinates."
        bbox = self._extract_first_bbox(item.get("solution", None))
        width = item.get("img_width", self.resize_size)
        height = item.get("img_height", self.resize_size)
        return {"image": image, "text": text, "bbox": bbox, "width": width, "height": height}


def collate_fn(batch: List[Dict[str, Any]], processor, resize_size: int = 840):
    messages, targets, widths, heights = [], [], [], []
    for b in batch:
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": b["image"]},
                    {"type": "text", "text": b["text"]},
                ],
            },
        ])
        targets.append(b["bbox"])
        widths.append(b["width"])
        heights.append(b["height"])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    images = [m[0]["content"][0]["image"] for m in messages]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")

    b_targets = []
    for tgt, w, h in zip(targets, widths, heights):
        if tgt is None:
            b_targets.append([0.0, 0.0, 0.0, 0.0])
            continue
        arr = np.array(tgt, dtype=float)
        if arr.size != 4:
            b_targets.append([0.0, 0.0, 0.0, 0.0])
            continue
        arr[0] /= float(w)
        arr[2] /= float(w)
        arr[1] /= float(h)
        arr[3] /= float(h)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        b_targets.append(arr.tolist())

    inputs["bbox_targets"] = torch.tensor(b_targets, dtype=torch.float32)
    return inputs


# -----------------------
# Model components (coarse-to-fine head + refiner)
# -----------------------
class CoarseFineBboxHead(nn.Module):
    def __init__(self, hidden_dim, num_bins=256, mid=512):
        super().__init__()
        self.num_bins = num_bins
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, mid),
            nn.ReLU(),
        )
        self.bin_heads = nn.ModuleList([nn.Linear(mid, num_bins) for _ in range(4)])
        self.residual_heads = nn.ModuleList([nn.Linear(mid, 1) for _ in range(4)])

    def forward(self, hidden_state):
        x = self.shared(hidden_state)  # [B, mid]
        bin_logits = [head(x) for head in self.bin_heads]  # 4 x [B, num_bins]
        residuals = [head(x).squeeze(-1) for head in self.residual_heads]  # 4 x [B]
        bin_probs = [F.softmax(b, dim=-1) for b in bin_logits]  # 4 x [B, num_bins]
        device = hidden_state.device
        centers = torch.arange(0, self.num_bins, device=device).float() / float(self.num_bins - 1)
        coarse = [ (p * centers).sum(dim=-1) for p in bin_probs ]
        coarse = torch.stack(coarse, dim=1)  # [B,4]
        residuals = torch.stack(residuals, dim=1)  # [B,4]
        residuals = torch.tanh(residuals) * (1.0 / self.num_bins)
        coords = (coarse + residuals).clamp(0.0, 1.0)
        return {
            "bin_logits": bin_logits,
            "bin_probs": bin_probs,
            "coarse": coarse,
            "residuals": residuals,
            "coords": coords,
        }


class BboxRefiner(nn.Module):
    def __init__(self, visual_feat_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_feat_dim + 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, visual_feat, init_bbox):
        x = torch.cat([visual_feat, init_bbox], dim=-1)
        delta = self.net(x)
        refined = (init_bbox + 0.2 * torch.tanh(delta)).clamp(0.0, 1.0)
        return refined


# -----------------------
# compute start idx
# -----------------------
def compute_start_idxs(input_ids: torch.Tensor, pad_token_id: int):
    lens = (input_ids != pad_token_id).long().sum(dim=1)
    start_idxs = (lens - 1).clamp(min=0)
    return start_idxs


# -----------------------
# targeted partial-unfreeze
# -----------------------
def apply_partial_unfreeze(model, unfreeze_last_n_vision=1, unfreeze_last_m_lm=1):
    for p in model.parameters():
        p.requires_grad = False

    vision_param_names = [name for name, _ in model.named_parameters() if ("vision" in name or "image" in name or "patch_embed" in name)]
    if len(vision_param_names) > 0:
        idxs = set()
        for n in vision_param_names:
            found = re.findall(r"\.(\d+)\.", n)
            if found:
                for f in found:
                    idxs.add(int(f))
        if idxs:
            max_idx = max(idxs)
            unfreeze_idxs = set(range(max(0, max_idx - unfreeze_last_n_vision + 1), max_idx + 1))
            for name, p in model.named_parameters():
                for ui in unfreeze_idxs:
                    if f".{ui}." in name:
                        p.requires_grad = True
        else:
            for name, p in model.named_parameters():
                if ("vision" in name or "image" in name):
                    p.requires_grad = True

    lm_param_names = [name for name, _ in model.named_parameters() if ("decoder" in name or "transformer" in name or "layers" in name)]
    if len(lm_param_names) > 0:
        idxs = set()
        for n in lm_param_names:
            found = re.findall(r"\.(\d+)\.", n)
            if found:
                for f in found:
                    idxs.add(int(f))
        if idxs:
            max_idx = max(idxs)
            unfreeze_idxs = set(range(max(0, max_idx - unfreeze_last_m_lm + 1), max_idx + 1))
            for name, p in model.named_parameters():
                for ui in unfreeze_idxs:
                    if f".{ui}." in name:
                        p.requires_grad = True
        else:
            for name, p in model.named_parameters():
                if ("lm_head" in name or "lm" in name):
                    p.requires_grad = True

    print("[partial-freeze] Trainable params after targeted unfreeze:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("  TRAINABLE:", name)


# -----------------------
# main
# -----------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    print(f"[bbox-pretrain] Loading dataset: {args.dataset_name} (split={args.split})")
    hf_ds = load_dataset(args.dataset_name)
    split = args.split if args.split in hf_ds else "train"
    hf_split = hf_ds[split]

    print(f"[bbox-pretrain] Loading processor & model from {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, padding_side="left")
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    except Exception:
        pass

    # --- LOAD MODEL IN FP32 to avoid fp16 overflow/NaN ---
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.float32, device_map="auto"
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # freeze/unfreeze policy
    if args.freeze_backbone.lower() == "true":
        print("[bbox-pretrain] Freezing entire backbone.")
        for p in model.parameters():
            p.requires_grad = False
    elif args.freeze_backbone.lower() == "partial":
        print("[bbox-pretrain] Applying targeted partial unfreeze.")
        apply_partial_unfreeze(model, unfreeze_last_n_vision=args.unfreeze_last_n_vision, unfreeze_last_m_lm=args.unfreeze_last_m_lm)
    else:
        print("[bbox-pretrain] Training full model + bbox head + refiner.")

    ds = BoxDataset(hf_split, processor, resize_size=args.resize_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, processor, args.resize_size),
    )

    bbox_head = None
    refiner = None
    optim = None
    global_step = 0

    best_loss = float("inf")
    best_epoch = -1
    patience_ctr = 0

    pad_token_id = processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_batches = 0
        total_valid_boxes = 0
        print(f"[bbox-pretrain] Epoch {epoch + 1}/{args.epochs}")
        loop = tqdm(dl, ncols=120)

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch.get("position_ids", None)
            if position_ids is not None:
                position_ids = position_ids.to(device)

            vision_kwargs = {}
            if "pixel_values" in batch:
                vision_kwargs["pixel_values"] = batch["pixel_values"].to(device)
            if "image_grid_thw" in batch:
                vision_kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)

            bbox_targets = batch["bbox_targets"].to(device=device, dtype=torch.float32)
            bbox_targets = torch.nan_to_num(bbox_targets, nan=0.0, posinf=1.0, neginf=0.0)

            # forward through model: compute in fp32 (weights are fp32)
            with torch.set_grad_enabled(any(p.requires_grad for p in model.parameters())):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    **vision_kwargs,
                )

            # get last hidden; cast to float32 and sanitize NaNs/Infs
            last_hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None else out.hidden_states[-1]
            last_hidden = last_hidden.to(torch.float32)
            if not torch.isfinite(last_hidden).all():
                print("[bbox-pretrain][WARN] last_hidden contains NaN/Inf; replacing with zeros for this batch.")
                try:
                    print(" last_hidden stats min/max/mean:", float(torch.nanmin(last_hidden)), float(torch.nanmax(last_hidden)), float(torch.nanmean(last_hidden)))
                except Exception:
                    pass
                last_hidden = torch.nan_to_num(last_hidden, nan=0.0, posinf=0.0, neginf=0.0)

            # start-token pooling
            start_idxs = compute_start_idxs(input_ids, pad_token_id).to(device)
            batch_idx = torch.arange(last_hidden.size(0), device=device)
            pooled = last_hidden[batch_idx, start_idxs, :].to(torch.float32)

            # lazy init of head/refiner
            if bbox_head is None:
                hidden_dim = pooled.size(-1)
                print(f"[bbox-pretrain] creating bbox head (hidden_dim={hidden_dim}, num_bins={args.num_bins})")
                bbox_head = CoarseFineBboxHead(hidden_dim, num_bins=args.num_bins, mid=max(64, hidden_dim // 2)).to(device=device, dtype=torch.float32)
                refiner = BboxRefiner(visual_feat_dim=512, hidden=256).to(device=device, dtype=torch.float32)

                # build optimizer param groups
                head_params = list(bbox_head.parameters())
                refiner_params = list(refiner.parameters())
                backbone_params = [p for n, p in model.named_parameters() if p.requires_grad]

                param_groups = [
                    {"params": head_params, "lr": args.lr_head, "weight_decay": args.weight_decay},
                    {"params": refiner_params, "lr": args.lr_refiner, "weight_decay": args.weight_decay},
                ]
                if len(backbone_params) > 0:
                    param_groups.append({"params": backbone_params, "lr": args.backbone_lr, "weight_decay": 1e-6})

                optim = AdamW(param_groups)

            # head forward (safe)
            head_out = bbox_head(pooled)
            # sanitize bin_logits
            sanitized_bin_logits = []
            for bl in head_out["bin_logits"]:
                bl = torch.nan_to_num(bl, nan=0.0, posinf=1e6, neginf=-1e6)
                sanitized_bin_logits.append(bl)
            head_out["bin_logits"] = sanitized_bin_logits

            coords = head_out["coords"]
            coords = torch.nan_to_num(coords, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            # visual features: try to use model-provided image_embeds if available, otherwise zeros
            if hasattr(out, "image_embeds") and out.image_embeds is not None:
                visual_feats = out.image_embeds.to(torch.float32)
                if not torch.isfinite(visual_feats).all():
                    print("[bbox-pretrain][WARN] visual_feats NaN/Inf; replacing with zeros.")
                    visual_feats = torch.nan_to_num(visual_feats, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                visual_feats = torch.zeros((pooled.size(0), 512), device=device, dtype=torch.float32)

            refined = refiner(visual_feats, coords)
            refined = torch.nan_to_num(refined, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            valid_mask = (bbox_targets.sum(dim=1) > 0).float()
            valid_count = int(valid_mask.sum().item())
            total_valid_boxes += valid_count
            if valid_count == 0:
                loop.set_postfix(loss="nan", valid=0)
                continue

            # compute losses: bin CE, residual L1, giou on refined
            bin_ce = 0.0
            for i in range(4):
                gt_coord = bbox_targets[:, i]
                gt_bin = (gt_coord * (args.num_bins - 1)).round().long().clamp(0, args.num_bins - 1)
                bin_logits = head_out["bin_logits"][i]
                bin_logits = torch.nan_to_num(bin_logits, nan=0.0, posinf=1e6, neginf=-1e6)
                bin_ce += F.cross_entropy(bin_logits, gt_bin, reduction="mean")
            bin_ce = bin_ce / 4.0

            residual_l1 = F.l1_loss(coords, bbox_targets, reduction="mean")
            giou_l = giou_loss(refined, bbox_targets).mean()

            # you can tune CE weight if needed (defaults: giou*3 + bin_ce*1 + residual*0.5)
            loss = 3.0 * giou_l + 1.0 * bin_ce + 0.5 * residual_l1

            if not torch.isfinite(loss):
                print("[bbox-pretrain][ERROR] Non-finite loss detected. Diagnostics:")
                try:
                    print(" coords finite:", torch.isfinite(coords).all().item(), " refined finite:", torch.isfinite(refined).all().item())
                    for i, bl in enumerate(head_out["bin_logits"]):
                        print(f" bin_logits[{i}] finite:", torch.isfinite(bl).all().item(), " min/max:", float(torch.nanmin(bl)), float(torch.nanmax(bl)))
                    print(" bbox_targets min/max:", float(torch.nanmin(bbox_targets)), float(torch.nanmax(bbox_targets)))
                except Exception as e:
                    print("  Could not print diagnostics fully:", e)
                raise RuntimeError("Non-finite loss encountered; check diagnostics above.")

            optim.zero_grad()
            loss.backward()

            # gradient clipping across head/refiner/backbone trainable params
            params_to_clip = list(bbox_head.parameters()) + list(refiner.parameters()) + [p for n,p in model.named_parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(params_to_clip, args.grad_clip)

            optim.step()

            global_step += 1
            epoch_loss += float(loss.item())
            epoch_batches += 1
            loop.set_postfix(loss=f"{loss.item():.5f}", avg=f"{(epoch_loss / max(1, epoch_batches)):.5f}", valid=valid_count)

        if epoch_batches == 0:
            print("[bbox-pretrain] No batches processed this epoch, stopping.")
            break

        epoch_mean_loss = epoch_loss / epoch_batches
        print(f"[bbox-pretrain] Epoch {epoch+1} mean_loss={epoch_mean_loss:.6f} valid_boxes={total_valid_boxes}")

        # early stopping & saving best
        if epoch_mean_loss + args.min_delta < best_loss:
            best_loss = epoch_mean_loss
            best_epoch = epoch + 1
            patience_ctr = 0
            print(f"[bbox-pretrain] ✅ New best mean loss {best_loss:.5f} at epoch {best_epoch}. Saving to {final_dir}")
            try:
                model.save_pretrained(final_dir)
            except Exception:
                pass
            if bbox_head is not None:
                torch.save(bbox_head.state_dict(), os.path.join(final_dir, "bbox_head.pt"))
            if refiner is not None:
                torch.save(refiner.state_dict(), os.path.join(final_dir, "bbox_refiner.pt"))
        else:
            patience_ctr += 1
            print(f"[bbox-pretrain] No significant improvement. patience {patience_ctr}/{args.patience}")
            if patience_ctr >= args.patience:
                print("[bbox-pretrain] Early stopping triggered.")
                break

    # save tokenizer/processor if available
    print("[bbox-pretrain] Attempting to save tokenizer/processor...")
    try:
        if tokenizer is not None:
            tokenizer.save_pretrained(final_dir)
        processor.save_pretrained(final_dir)
    except Exception as e:
        print(f"[bbox-pretrain] Warning saving tokenizer/processor: {e}")

    print("=====================================")
    print(f"[bbox-pretrain] Finished. Best epoch: {best_epoch}, best mean_loss={best_loss:.5f}")
    print(f"Final models (head/refiner) in: {final_dir}")
    print("=====================================")


if __name__ == "__main__":
    main()
