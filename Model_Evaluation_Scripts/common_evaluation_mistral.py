import os
import json
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from pathlib import Path


def compute_iou(boxA, boxB):
    """Compute IoU between two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


# ---------- Helper to parse model output ----------
def parse_bboxes_from_text(text):
    """
    Extracts list of dicts like [{"frame": N, "bbox": [x, y, w, h]}] from model output.
    Handles both JSON and semi-structured responses.
    """
    try:
        # Try direct JSON
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    # Try regex fallback
    pattern = r'\{.*?"frame":\s*(\d+).*?"bbox":\s*\[([^\]]+)\].*?\}'
    matches = re.findall(pattern, text)
    parsed = []
    for m in matches:
        frame = int(m[0])
        bbox = [float(x) for x in m[1].split(",")]
        parsed.append({"frame": frame, "bbox": bbox})
    return parsed


# ---------- Detect data format ----------
def detect_format(item):
    """
    Detect if the item is single-object or multi-object tracking format.
    
    Single-object format has:
    - "track_id" (integer)
    - "expected_output" (list of dicts with frame and bbox)
    
    Multi-object format has:
    - "objects" (list of objects)
    - "expected_output" (JSON string with object_N keys)
    """
    if "track_id" in item and isinstance(item.get("expected_output"), (list, str)):
        # Check if expected_output is a simple list
        try:
            exp_out = item["expected_output"]
            if isinstance(exp_out, str):
                exp_out = json.loads(exp_out)
            if isinstance(exp_out, list) and len(exp_out) > 0:
                if "frame" in exp_out[0] and "bbox" in exp_out[0]:
                    return "single"
        except:
            pass
    
    if "objects" in item or "num_objects" in item:
        return "multi"
    
    # Default fallback: try to parse expected_output
    try:
        exp_out = item["expected_output"]
        if isinstance(exp_out, str):
            exp_out = json.loads(exp_out)
        if isinstance(exp_out, dict) and any(k.startswith("object_") for k in exp_out.keys()):
            return "multi"
    except:
        pass
    
    return "single"  # Default

def _to_file_url(path):
    return Path(path).absolute().as_uri()



def evaluate_single_object(
    item,
    dataset_root,
    model,
    tokenizer,      # 🔴 CHANGED: processor -> tokenizer (MistralCommonBackend)
    device,
):
    """
    Evaluate single-object tracking with all frames passed at once
    using Mistral / Ministral-3 multimodal models.
    """

    qid = item["id"]
    ref_image = os.path.join(dataset_root, item["ref_image"])
    input_images = [os.path.join(dataset_root, img) for img in item["input_images"]]

    # ------------------------------------------------------------
    # Parse expected output
    # ------------------------------------------------------------
    gt_outputs = item["expected_output"]
    if isinstance(gt_outputs, str):
        gt_outputs = json.loads(gt_outputs)

    initial_bbox = item.get("initial_bbox", [0, 0, 0, 0])

    # ------------------------------------------------------------
    # QUESTION (UNCHANGED LOGIC)
    # ------------------------------------------------------------
    question = (
        item["question"]
        + f"\nInitially, the object is present at {initial_bbox} in the reference image."
        + "\nThe following images are consecutive frames of a video."
        + "\nPlease provide the bounding box for this object in EACH frame, "
          "in the format:"
          "\n[{\"frame\": N, \"bbox\": [x, y, w, h]}, ...]."
          "\nDo not write any extra text."
    )

    # ------------------------------------------------------------
    # BUILD MESSAGE (🔴 CHANGED FORMAT FOR MISTRAL)
    # ------------------------------------------------------------
    messages = [
        {
            "role": "user",
            "content": (
                # Reference image
                [{
                    "type": "image_url",
                    "image_url": {"url": _to_file_url(ref_image)}
                }]
                # Video frames
                + [{
                    "type": "image_url",
                    "image_url": {"url": _to_file_url(img)}
                } for img in input_images]
                # Question text
                + [{
                    "type": "text",
                    "text": question
                }]
            ),
        }
    ]

    # ------------------------------------------------------------
    # TOKENIZATION (🔴 CHANGED: NO processor, NO process_vision_info)
    # ------------------------------------------------------------
    tokenized = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    # 🔴 REQUIRED FOR MISTRAL VISION
    tokenized["pixel_values"] = tokenized["pixel_values"].to(torch.bfloat16)

    # 🔴 REQUIRED EXPLICITLY
    num_images = 1 + len(input_images)  # ref image + frames
    image_sizes = [
        tokenized["pixel_values"].shape[-2:]
    ] * num_images
    # ------------------------------------------------------------
    # MODEL INFERENCE (ONCE)
    # ------------------------------------------------------------
    with torch.no_grad():
        output_ids = model.generate(
            **tokenized,
            image_sizes=image_sizes,
            max_new_tokens=256,
        )

    # ------------------------------------------------------------
    # DECODE OUTPUT (🔴 CHANGED)
    # ------------------------------------------------------------
    pred_text = tokenizer.decode(
        output_ids[0][len(tokenized["input_ids"][0]):],
        skip_special_tokens=True,
    ).strip()

    print(f"\nSingle Object Prediction:\n{pred_text}")

    # ------------------------------------------------------------
    # PARSE PREDICTIONS (UNCHANGED LOGIC)
    # ------------------------------------------------------------
    pred_bboxes = []

    try:
        frame_bboxes = parse_bboxes_from_text(pred_text)
        print(frame_bboxes)

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
                "bbox": [0.0, 0.0, 0.0, 0.0],
            })

    # ------------------------------------------------------------
    # COMPUTE IoU PER FRAME (UNCHANGED)
    # ------------------------------------------------------------
    frame_ious = []
    for gt in gt_outputs:
        frame = gt["frame"]
        gt_box = gt["bbox"]
        pred_box = next(
            (p["bbox"] for p in pred_bboxes if p["frame"] == frame),
            None,
        )
        frame_ious.append(compute_iou(gt_box, pred_box) if pred_box else 0.0)

    avg_iou = float(np.mean(frame_ious)) if frame_ious else 0.0

    # ------------------------------------------------------------
    # STORE RESULTS (UNCHANGED)
    # ------------------------------------------------------------
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


def evaluate_multi_object(
    item,
    dataset_root,
    model,
    tokenizer,          # 🔴 CHANGED: processor -> tokenizer
    device,
):
    """
    Evaluate multi-object tracking with all frames passed at once per object
    using Mistral / Ministral-3 multimodal models.
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

    # ------------------------------------------------------------
    # PROCESS EACH OBJECT SEPARATELY
    # ------------------------------------------------------------
    for obj in objects:
        obj_id = obj["object_id"]
        initial_bbox = obj["initial_bbox"]
        track_id = obj.get("track_id")

        obj_key = f"object_{obj_id}"
        if obj_key not in gt_outputs:
            print(f"Warning: {obj_key} not found in ground truth")
            continue

        gt_trajectory = gt_outputs[obj_key]["trajectory"]

        # --------------------------------------------------------
        # QUESTION (UNCHANGED LOGIC, FORMATTING KEPT)
        # --------------------------------------------------------
        question = (
            item["question"]
            + f"\nInitially, object {obj_id} is present at {initial_bbox} in the reference image."
            + "\nThe following images are consecutive frames of a video."
            + "\nPlease provide the bounding box for this object in EACH frame, "
              "in the format:"
              "\n[{\"frame\": N, \"bbox\": [x, y, w, h]}, ...]."
              "\nDo not write any extra text."
        )

        # --------------------------------------------------------
        # BUILD MESSAGE (🔴 CHANGED FORMAT FOR MISTRAL)
        # --------------------------------------------------------
        messages = [
            {
                "role": "user",
                "content": (
                    # Reference image
                    [{
                        "type": "image_url",
                        "image_url": {"url": _to_file_url(ref_image)}
                    }]
                    # Video frames
                    + [{
                        "type": "image_url",
                        "image_url": {"url": _to_file_url(img)}
                    } for img in input_images]
                    # Question text
                    + [{
                        "type": "text",
                        "text": question
                    }]
                ),
            }
        ]

        # --------------------------------------------------------
        # TOKENIZATION (🔴 CHANGED: NO processor, NO process_vision_info)
        # --------------------------------------------------------
        tokenized = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        # 🔴 REQUIRED FOR MISTRAL VISION
        tokenized["pixel_values"] = tokenized["pixel_values"].to(torch.bfloat16)

        # 🔴 REQUIRED EXPLICITLY
        num_images = 1 + len(input_images)  # ref image + frames
        image_sizes = [
            tokenized["pixel_values"].shape[-2:]
        ] * num_images


        # --------------------------------------------------------
        # MODEL INFERENCE (UNCHANGED LOGIC, CHANGED INPUTS)
        # --------------------------------------------------------
        with torch.no_grad():
            output_ids = model.generate(
                **tokenized,
                image_sizes=image_sizes,
                max_new_tokens=256,
            )

        # --------------------------------------------------------
        # DECODE OUTPUT (🔴 CHANGED)
        # --------------------------------------------------------
        pred_text = tokenizer.decode(
            output_ids[0][len(tokenized["input_ids"][0]):],
            skip_special_tokens=True,
        ).strip()

        print(f"\nObject {obj_id} Prediction:\n{pred_text}")

        # --------------------------------------------------------
        # PARSE PREDICTIONS (UNCHANGED)
        # --------------------------------------------------------
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
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                })

        # --------------------------------------------------------
        # COMPUTE IoU (UNCHANGED)
        # --------------------------------------------------------
        frame_ious = []
        for gt in gt_trajectory:
            frame = gt["frame"]
            gt_box = gt["bbox"]
            pred_box = next(
                (p["bbox"] for p in pred_bboxes if p["frame"] == frame),
                None,
            )
            frame_ious.append(compute_iou(gt_box, pred_box) if pred_box else 0.0)

        obj_avg_iou = float(np.mean(frame_ious)) if frame_ious else 0.0
        all_object_ious.append(obj_avg_iou)

        # --------------------------------------------------------
        # STORE RESULTS (UNCHANGED)
        # --------------------------------------------------------
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

    # ------------------------------------------------------------
    # FINAL AVERAGE IoU
    # ------------------------------------------------------------
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



def evaluate_model(json_path, dataset_root, output_json_path="evaluation_results.json"):
    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Load model and processor
    model_name =  "mistralai/Ministral-3-8B-Instruct-2512"
    print(f"Loading model: {model_name}")
    tokenizer = MistralCommonBackend.from_pretrained(model_name)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_name, device_map="cuda"
    )

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
                avg_iou, result_entry = evaluate_single_object(item, dataset_root, model, tokenizer, device)
            else:
                multi_count += 1
                avg_iou, result_entry = evaluate_multi_object(item, dataset_root, model, tokenizer, device)
            
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
    parser.add_argument("--json", default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/test/annotations_real.json", help="Path to the JSON file")
    parser.add_argument("--dataset_root", default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/test", help="Path to dataset root")
    parser.add_argument("--json_log", default="/home/gaash/Tawheed/Reasoning/Ealuation_logs/evaluation_mistral_instruct.json", help="Path to save JSON results file")
    args = parser.parse_args()

    evaluate_model(args.json, args.dataset_root, args.json_log)