import os
import json
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from qwen_vl_utils import process_vision_info
from functions import  detect_format, evaluate_single_object, evaluate_multi_object


def evaluate_model(json_path, dataset_root, output_json_path="evaluation_results.json"):
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Load model and processor
    model_id = "google/gemma-3-27b-it"
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_iou = 0
    total_samples = 0
    results = []
    
    single_count = 0
    multi_count = 0

    for item in tqdm(data, desc="Evaluating"):
        format_type = detect_format(item)
        
        try:
            if format_type == "single":
                single_count += 1
                avg_iou, result_entry = evaluate_single_object(item, dataset_root, model, processor, device)
            else:
                multi_count += 1
                avg_iou, result_entry = evaluate_multi_object(item, dataset_root, model, processor, device)
            
            total_iou += avg_iou
            total_samples += 1
            results.append(result_entry)
            
        except Exception as e:
            print(f"\nError processing {item.get('id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final Results
    overall_score = total_iou / total_samples if total_samples > 0 else 0.0
    summary = {
        "total_samples": total_samples,
        "single_object_samples": single_count,
        "multi_object_samples": multi_count,
        "average_iou": overall_score
    }

    # Write to JSON
    output_data = {
        "results": results,
        "summary": summary
    }

    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nEvaluation results saved to {output_json_path}")
    print(f"Total Samples: {total_samples}")
    print(f"  Single-object: {single_count}")
    print(f"  Multi-object: {multi_count}")
    print(f"Average IoU Score: {overall_score:.4f}")

    return overall_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate object tracking model on single or multi-object datasets")
    parser.add_argument("--json", default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/annotations_real.json", help="Path to the JSON file")
    parser.add_argument("--dataset_root", default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset", help="Path to dataset root")
    parser.add_argument("--json_log", default="/home/gaash/Tawheed/Reasoning/Ealuation_logs/evaluation_gemma3.json", help="Path to save JSON results file")
    args = parser.parse_args()

    evaluate_model(args.json, args.dataset_root, args.json_log)