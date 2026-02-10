import argparse
import json
import os
import re
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

def parse_args():
    parser = argparse.ArgumentParser(description="Run an improved evaluation with advanced prompting and TTA to maximize gIoU.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model's huggingface directory.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the custom evaluation dataset.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the detailed results JSON.")
    parser.add_argument("--num_text_analysis", type=int, default=10, help="Number of worst-performing samples to print for analysis.")
    return parser.parse_args()

def extract_bbox_and_thought(output_text):
    thought_match = re.search(r'(?s)<think>(.*?)</think>', output_text)
    thought_process = thought_match.group(1).strip() if thought_match else "No <think> tag found."
    answer_match = re.search(r'(?s)<answer>(.*?)</answer>', output_text)
    if not answer_match:
        return None, thought_process, "no_answer_tag"
    try:
        data = json.loads(answer_match.group(1))
        if isinstance(data, list) and data:
            item = data[0]
            if 'bbox_2d' in item and len(item['bbox_2d']) == 4:
                bbox = item['bbox_2d']
                return bbox, thought_process, "valid_bbox"
        return None, thought_process, "hallucinated_bbox"
    except (json.JSONDecodeError, TypeError):
        return None, thought_process, "malformed_json"
    except Exception:
        return None, thought_process, "unknown_error"

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0

def run_inference(model, processor, image, prompt_text, resize_size):
    message = [{"role": "user", "content": [
        {"type": "image", "image": image.resize((resize_size, resize_size), Image.BILINEAR)},
        {"type": "text", "text": prompt_text}
    ]}]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image.resize((resize_size, resize_size), Image.BILINEAR)],
        return_tensors="pt", padding=True
    ).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    return processor.decode(generated_ids[0, len(inputs.input_ids[0]):], skip_special_tokens=True)

def main():
    args = parse_args()
    print("Loading model and processor...")
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    reasoning_model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)['train']
    resize_size = 840

    prompt_header = (
        "Your task is to identify a specific surgical tool in the provided image based on the user's question. "
        "First, in the <think> tag, reason step-by-step about what the user is asking for and what visual cues in the image match that description. "
        "Then, in the <answer> tag, provide the bounding box for the single, most relevant object you identified. "
        "The answer must be a JSON list containing a single dictionary.\n\n"
    )
    prompt_example = (
        "--- EXAMPLE ---\n"
        "User Question: 'The primary knife is being used to make an incision in the cornea.'\n"
        "<think>The user is asking to identify the 'primary knife'. I will look for a sharp, blade-like instrument actively touching the cornea. I see such an instrument in the upper right quadrant. This matches the description.</think>\n"
        "<answer>[{{\"bbox_2d\": [450, 110, 550, 210], \"point_2d\": [500, 160]}}]</answer>\n"
        "--- END EXAMPLE ---\n\n"
    )
    prompt_task = (
        "--- CURRENT TASK ---\n"
        "User Question: '{Question}'"
    )
    QUESTION_TEMPLATE = prompt_header + prompt_example + prompt_task

    all_results = []
    failure_counts = {"no_answer_tag": 0, "malformed_json": 0, "hallucinated_bbox": 0, "unknown_error": 0}

    for item in tqdm(dataset, desc="Evaluating Samples"):
        image = item["image"].convert("RGB")
        prompt_text = QUESTION_TEMPLATE.format(Question=item["problem"])
        
        output_text_orig = run_inference(reasoning_model, processor, image, prompt_text, resize_size)
        pred_box_orig, thought_process, status = extract_bbox_and_thought(output_text_orig)
        if status != "valid_bbox": failure_counts.setdefault(status, 0); failure_counts[status] += 1

        image_flipped = ImageOps.mirror(image)
        output_text_flipped = run_inference(reasoning_model, processor, image_flipped, prompt_text, resize_size)
        pred_box_flipped, _, _ = extract_bbox_and_thought(output_text_flipped)

        if pred_box_flipped:
            x1, y1, x2, y2 = pred_box_flipped
            pred_box_flipped = [resize_size - x2, y1, resize_size - x1, y2]

        final_pred_box = None
        if pred_box_orig and pred_box_flipped:
            final_pred_box = np.mean([pred_box_orig, pred_box_flipped], axis=0).astype(int).tolist()
        elif pred_box_orig:
            final_pred_box = pred_box_orig

        iou_score = 0.0
        gt_box = None 
        try:
            gt_solution = json.loads(item["solution"])
            gt_box = gt_solution[0]['bbox_2d']
            if final_pred_box:
                iou_score = compute_iou(final_pred_box, gt_box)
        except Exception:
            pass
        
        all_results.append({
            "id": item["id"], "problem": item["problem"], "prediction_raw": output_text_orig,
            "iou": iou_score, "thought": thought_process, "predicted_box": final_pred_box,
            "ground_truth_box": gt_box
        })

    mean_iou = np.mean([res['iou'] for res in all_results])

    print("\n" + "="*50)
    print("--- EVALUATION COMPLETE ---")
    print("="*50)
    print(f"Mean IoU (with TTA and Improved Prompting): {mean_iou:.4f}")
    if args.num_text_analysis > 0:
        print("\n" + "="*50)
        print(f"--- Analysis of {args.num_text_analysis} Worst Performing Samples ---")
        print("="*50)
        worst_samples = sorted(all_results, key=lambda x: x['iou'])[:args.num_text_analysis]
        for i, sample in enumerate(worst_samples):
            print(f"\n--- Worst Sample #{i+1} ---")
            print(f" ID: {sample['id']} | IoU Score: {sample['iou']:.4f}")
            print(f" Problem: \"{sample['problem']}\"")
            print(f" Model Reasoning: \"{sample['thought']}\"")
            print(f" Ground Truth Box: {sample['ground_truth_box']}")
            print(f" Final Predicted Box (TTA): {sample['predicted_box']}")
            
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "detailed_evaluation_results_tta.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull detailed results saved to {output_file}")

if __name__ == "__main__":
    main()