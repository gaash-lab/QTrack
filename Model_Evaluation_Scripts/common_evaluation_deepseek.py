import os
import json
import torch
import re
import numpy as np
from tqdm import tqdm
from PIL import Image

# DeepSeek VL2 imports
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

from functions import detect_format

def evaluate_multi_object(item, dataset_root, model, processor, device):
    """
    DeepSeek-VL2 version of multi-object evaluation.
    Each object is processed independently, all frames passed at once.
    """

    qid = item["id"]
    ref_image = os.path.join(dataset_root, item["ref_image"])
    input_images = [os.path.join(dataset_root, img) for img in item["input_images"]]

    # Parse expected output
    gt_outputs = item["expected_output"]
    if isinstance(gt_outputs, str):
        gt_outputs = json.loads(gt_outputs)

    objects = item.get("objects", [])

    all_object_ious = []
    predicted_objects = {}
    expected_objects = {}

    # Process each object separately
    for obj in objects:
        obj_id = obj["object_id"]
        initial_bbox = obj["initial_bbox"]
        track_id = obj.get("track_id")

        obj_key = f"object_{obj_id}"
        if obj_key not in gt_outputs:
            print(f"Warning: {obj_key} not found in ground truth")
            continue

        gt_trajectory = gt_outputs[obj_key]["trajectory"]

        # Build question
        question = (
            item["question"]
            + f"\nInitially, object {obj_id} is present at {initial_bbox} in the reference image."
            + "\nThe following images are consecutive frames of a video."
            + "\nPlease provide the bounding box for this object in EACH frame, "
              "in the format:"
              "\n[{\"frame\": N, \"bbox\": [x, y, w, h]}, ...]."
              "\nDo not write any extra text."
        )

        # Build DeepSeek conversation
        content = "Reference image:\n<image>\n"
        for i in range(len(input_images)):
            content += f"Frame {i+1}:\n<image>\n"
        content += question

        conversation = [
            {
                "role": "<|User|>",
                "content": content,
                "images": [ref_image] + input_images
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        # Load images
        pil_images = load_pil_images(conversation)

        # Prepare inputs
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(device)

        # Prepare embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        # Inference (once per object)
        with torch.no_grad():
            outputs = model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

        # Decode
        pred_text = processor.tokenizer.decode(
            outputs[0].cpu().tolist(),
            skip_special_tokens=False
        ).strip()

        print(f"\nObject {obj_id} Prediction (DeepSeek):\n{pred_text}")

        # -------------------------
        # Parse Predictions
        # -------------------------
        pred_bboxes = []
        try:
            frame_bboxes = parse_bboxes_from_text(pred_text)
            print(f"Parsed {len(frame_bboxes)} bboxes for object {obj_id}")
            print(frame_bboxes)

            for i, gt in enumerate(gt_trajectory):
                frame_id = gt["frame"]

                if i < len(frame_bboxes):
                    fb = frame_bboxes[i]
                    bbox = fb.get("bbox") if isinstance(fb, dict) else None

                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        try:
                            bbox = [float(x) for x in bbox]
                        except Exception:
                            bbox = None
                else:
                    bbox = None

                if bbox is None:
                    bbox = [0.0, 0.0, 0.0, 0.0]

                pred_bboxes.append({"frame": frame_id, "bbox": bbox})

        except Exception as e:
            print(f"Warning: failed to parse predictions for object {obj_id}: {e}")
            for gt in gt_trajectory:
                pred_bboxes.append({
                    "frame": gt["frame"],
                    "bbox": [0.0, 0.0, 0.0, 0.0]
                })

        # -------------------------
        # Compute IoU
        # -------------------------
        frame_ious = []
        for gt in gt_trajectory:
            frame = gt["frame"]
            gt_box = gt["bbox"]
            pred_box = next(
                (p["bbox"] for p in pred_bboxes if p["frame"] == frame),
                None
            )

            if pred_box:
                frame_ious.append(compute_iou(gt_box, pred_box))
            else:
                frame_ious.append(0.0)

        obj_avg_iou = float(np.mean(frame_ious)) if frame_ious else 0.0
        all_object_ious.append(obj_avg_iou)

        # -------------------------
        # Store results
        # -------------------------
        predicted_objects[obj_key] = {
            "object_id": obj_id,
            "track_id": track_id,
            "initial_bbox": initial_bbox,
            "trajectory": pred_bboxes,
            "frame_ious": frame_ious,
            "average_iou": obj_avg_iou,
        }

        expected_objects[obj_key] = {
            "object_id": obj_id,
            "track_id": track_id,
            "initial_bbox": initial_bbox,
            "trajectory": gt_trajectory,
        }

    # -------------------------
    # Final average IoU
    # -------------------------
    overall_iou = float(np.mean(all_object_ious)) if all_object_ious else 0.0

    result_entry = {
        "id": qid,
        "type": "multi",
        "ref_image": item["ref_image"],
        "input_images": item["input_images"],
        "num_input_images": len(item["input_images"]),
        "question": item["question"],
        "num_objects": len(objects),
        "objects": item.get("objects", []),
        "predicted_output": predicted_objects,
        "expected_output": expected_objects,
        "average_iou": overall_iou,
    }

    return overall_iou, result_entry


def evaluate_single_object(item, dataset_root, model, processor, device):
    """
    DeepSeek-VL2 version of single object evaluation.
    All frames are passed at once using DeepSeek conversation format.
    """

    qid = item["id"]
    ref_image = os.path.join(dataset_root, item["ref_image"])
    input_images = [os.path.join(dataset_root, img) for img in item["input_images"]]

    # Parse expected output
    gt_outputs = item["expected_output"]
    if isinstance(gt_outputs, str):
        gt_outputs = json.loads(gt_outputs)

    initial_bbox = item.get("initial_bbox", [0, 0, 0, 0])

    # Build question
    question = (
        item["question"]
        + f"\nInitially, the object is present at {initial_bbox} in the reference image."
        + "\nThe following images are consecutive frames of a video."
        + "\nPlease provide the bounding box for this object in EACH frame, "
          "in the format:"
          "\n[{\"frame\": N, \"bbox\": [x, y, w, h]}, ...]."
          "\nDo not write any extra text."
    )

    # Build DeepSeek conversation
    # Each <image> token corresponds to one image in the images list
    content = "Reference image:\n<image>\n"
    for i in range(len(input_images)):
        content += f"Frame {i+1}:\n<image>\n"
    content += question

    conversation = [
        {
            "role": "<|User|>",
            "content": content,
            "images": [ref_image] + input_images
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

    # Load images as PIL
    pil_images = load_pil_images(conversation)

    # Prepare inputs
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(device)

    # Prepare embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # Inference
    with torch.no_grad():
        outputs = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=processor.tokenizer.eos_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

    # Decode output
    pred_text = processor.tokenizer.decode(
        outputs[0].cpu().tolist(),
        skip_special_tokens=False
    ).strip()

    print(f"\nSingle Object Prediction (DeepSeek):\n{pred_text}")

    # -------------------------
    # Parse Predictions
    # -------------------------
    pred_bboxes = []

    try:
        frame_bboxes = parse_bboxes_from_text(pred_text)
        print("Parsed frame bboxes:", frame_bboxes)

        for i, gt in enumerate(gt_outputs):
            frame_id = gt["frame"]

            if i < len(frame_bboxes):
                fb = frame_bboxes[i]
                bbox = fb.get("bbox") if isinstance(fb, dict) else None

                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    try:
                        bbox = [float(x) for x in bbox]
                    except Exception:
                        bbox = None
            else:
                bbox = None

            if bbox is None:
                bbox = [0.0, 0.0, 0.0, 0.0]

            pred_bboxes.append({"frame": frame_id, "bbox": bbox})

    except Exception as e:
        print(f"Warning: failed to parse predictions: {e}")
        for gt in gt_outputs:
            pred_bboxes.append({
                "frame": gt["frame"],
                "bbox": [0.0, 0.0, 0.0, 0.0]
            })

    # -------------------------
    # Compute IoU per frame
    # -------------------------
    frame_ious = []
    for gt in gt_outputs:
        frame = gt["frame"]
        gt_box = gt["bbox"]
        pred_box = next(
            (p["bbox"] for p in pred_bboxes if p["frame"] == frame),
            None
        )

        if pred_box:
            frame_ious.append(compute_iou(gt_box, pred_box))
        else:
            frame_ious.append(0.0)

    avg_iou = float(np.mean(frame_ious)) if frame_ious else 0.0

    # -------------------------
    # Result entry
    # -------------------------
    result_entry = {
        "id": qid,
        "type": "single",
        "ref_image": item["ref_image"],
        "input_images": item["input_images"],
        "num_input_images": len(item["input_images"]),
        "question": item["question"],
        "track_id": item.get("track_id", None),
        "initial_bbox": initial_bbox,
        "predicted_output": pred_bboxes,
        "expected_output": gt_outputs,
        "frame_ious": frame_ious,
        "average_iou": avg_iou,
    }

    return avg_iou, result_entry


# ---------- Load DeepSeek VL2 Model ----------
def load_deepseek_model(model_path="deepseek-ai/deepseek-vl2-tiny"):
    print(f"Loading DeepSeek model: {model_path}")

    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    return model, processor, tokenizer, device


# ---------- Helper function to run DeepSeek inference ----------
def deepseek_generate(model, processor, tokenizer, device, conversation, images):
    """
    conversation: list of dicts in DeepSeek format:
        [
            {"role":"<|User|>","content":"... <image> ...","images":[...paths...]},
            {"role":"<|Assistant|>","content":""}
        ]
    images: list of PIL images (already loaded)
    """
    prepare_inputs = processor(
        conversations=conversation,
        images=images,
        force_batchify=True,
        system_prompt=""
    ).to(device)

    # Prepare embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # Generate
    outputs = model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
    return answer, prepare_inputs


# ---------- Main Evaluation ----------
def evaluate_model(json_path, dataset_root, output_json_path="evaluation_results.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Load DeepSeek instead of LLaMA
    model, processor, tokenizer, device = load_deepseek_model(
        model_path="deepseek-ai/deepseek-vl2-tiny"
    )

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
                avg_iou, result_entry = evaluate_single_object(
                    item,
                    dataset_root,
                    model,
                    processor,
                    device,
                    model_type="deepseek"   # add a flag so your function knows how to call the model
                )
            else:
                multi_count += 1
                avg_iou, result_entry = evaluate_multi_object(
                    item,
                    dataset_root,
                    model,
                    processor,
                    device,
                    model_type="deepseek"
                )

            total_iou += avg_iou
            total_samples += 1
            results.append(result_entry)

        except Exception as e:
            print(f"\nError processing {item.get('id','unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue

    overall_score = total_iou / total_samples if total_samples > 0 else 0.0
    summary = {
        "total_samples": total_samples,
        "single_object_samples": single_count,
        "multi_object_samples": multi_count,
        "average_iou": overall_score
    }

    output_data = {"results": results, "summary": summary}
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"\nEvaluation results saved to {output_json_path}")
    print(
        f"Total Samples: {total_samples}, "
        f"Single: {single_count}, "
        f"Multi: {multi_count}, "
        f"Average IoU: {overall_score:.4f}"
    )
    return overall_score


# ---------- Example of how evaluate_single_object / evaluate_multi_object must call DeepSeek ----------
"""
Inside your functions.py you will need something like:

def run_deepseek(model, processor, tokenizer, device, image_paths, prompt):
    conversation = [
        {
            "role": "<|User|>",
            "content": f"{prompt}\n<image>",
            "images": image_paths,
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

    pil_images = load_pil_images(conversation)

    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    outputs = model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False
    )

    answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    return answer
"""


# ---------- Entry Point ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/test/annotations_real.json"
    )
    parser.add_argument(
        "--dataset_root",
        default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/test"
    )
    parser.add_argument(
        "--json_log",
        default="evaluation_deepseek.json"
    )
    args = parser.parse_args()

    evaluate_model(args.json, args.dataset_root, args.json_log)
