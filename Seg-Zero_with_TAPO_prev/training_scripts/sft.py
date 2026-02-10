#!/usr/bin/env python3
"""
sft_stage1.py (Fixed Sequence Length)
- Fix: Increased max_length to 8192 (Images take ~3-5k tokens).
- Fix: Enabled input_require_grads for gradient checkpointing.
"""
import argparse
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_from_disk, load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torch.optim import AdamW

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    return p.parse_args()

# --- SEG-ZERO PROMPT TEMPLATE ---
SYSTEM_PROMPT = (
    "Please find '{Question}' with bboxs and points.\n"
    "Compare the difference between tool(s) and find the most closely matched tool.\n"
    "Output the thinking process in <think> </think> (the output must be in english) and final answer in <answer> </answer> tags.\n"
    "Output the bbox(es) and point(s) inside the interested tool in JSON format.\n"
    "i.e., <think> thinking process here </think>\n"
    "<answer>{Answer}</answer>"
)

# Dummy example for the prompt (showing the model what "Answer" looks like)
EXAMPLE_JSON = "{'bbox_2d': [100, 200, 300, 400], 'point_2d': [150, 250]}"

class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(self, ds, processor):
        self.ds = ds
        self.processor = processor
    
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item["image"]
        if not hasattr(image, "convert"): 
            from PIL import Image as PILImage
            import io
            image = PILImage.open(io.BytesIO(image["bytes"])).convert("RGB")
        
        # 1. User Prompt
        tool_name = item.get('problem') or item.get('text') or "object"
        user_text = SYSTEM_PROMPT.format(Question=tool_name, Answer=EXAMPLE_JSON)
        
        # 2. Assistant Target
        try:
            sol_raw = item["solution"]
            if isinstance(sol_raw, str): sol_data = json.loads(sol_raw)
            else: sol_data = sol_raw
            target_obj = sol_data[0] 
            
            final_json = {}
            if "bbox_2d" in target_obj: final_json["bbox_2d"] = target_obj["bbox_2d"]
            if "point_2d" in target_obj: final_json["point_2d"] = target_obj["point_2d"]
            
            target_json_str = json.dumps(final_json)
        except:
            target_json_str = "{'bbox_2d': [0,0,0,0]}"

        # Bridge: Placeholder Thought
        assistant_text = f"<think>Locating the {tool_name} in the image based on visual features.</think>\n<answer>{target_json_str}</answer>"
        
        # 3. Tokenize
        conversation = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]
        
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        
        # FIX: Increased max_length to 8192 to fit image tokens (~3600) + text
        inputs = self.processor(
            text=[text], images=[image], videos=None, 
            padding="max_length", max_length=8192, truncation=True, 
            return_tensors="pt"
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in inputs.items()}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[Stage 1] Loading: {args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, min_pixels=256*28*28, max_pixels=1280*28*28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # 1. Enable Checkpointing
    model.gradient_checkpointing_enable()
    # 2. Fix "None of the inputs have requires_grad" warning
    model.enable_input_require_grads()
    
    try: dataset = load_from_disk(args.dataset_path)
    except: dataset = load_dataset(args.dataset_path, split="train")
    if hasattr(dataset, "keys") and not hasattr(dataset, "select"): dataset = dataset["train"]
    
    ds = Stage1Dataset(dataset, processor)
    # Reduced batch size slightly just in case 8192 tokens pushes memory
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    for param in model.visual.parameters():
        param.requires_grad = False
        
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    model.train()
    for epoch in range(args.epochs):
        loop = tqdm(dl, desc=f"Epoch {epoch+1}")
        epoch_loss = 0
        for batch in loop:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss
            optim.zero_grad(); loss.backward(); optim.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(dl):.4f}")
        
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

    print(f"Done. Final model at: {args.output_dir}/final")
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

if __name__ == "__main__":
    main()