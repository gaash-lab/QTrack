import os
import json
import torch
import re
import numpy as np
from tqdm import tqdm
from qwen_vl_utils import process_vision_info


# ---------- IoU Function ----------
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


# ---------- Single Object Evaluation ----------
# def evaluate_single_object(item, dataset_root, model, processor, device):
#     """Evaluate single object tracking task."""
#     qid = item["id"]
#     ref_image = os.path.join(dataset_root, item["ref_image"])
#     input_images = [os.path.join(dataset_root, img) for img in item["input_images"]]
    
#     # Parse expected output
#     gt_outputs = item["expected_output"]
#     if isinstance(gt_outputs, str):
#         gt_outputs = json.loads(gt_outputs)
    
#     initial_bbox = item.get("initial_bbox", [0, 0, 0, 0])
    
#     question = (
#         item["question"] + 
#         f"\nInitially, the object is present at {initial_bbox} in the reference image."
#         + "\nPlease provide the bounding box for this frame only, "
#           "in the format: [{\"frame\": N, \"bbox\": [x, y, w, h]}]. "
#           "Do not write any extra text."
#     )

#     pred_bboxes = []

#     # Process each input image independently
#     for img_path, gt_entry in zip(input_images, gt_outputs):
#         frame_id = gt_entry["frame"]

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": ref_image},
#                     {"type": "image", "image": img_path},
#                     {"type": "text", "text": question},
#                 ],
#             }
#         ]

#         # Prepare chat input
#         text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         image_inputs, video_inputs = process_vision_info(messages)
#         inputs = processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         ).to(device)

#         # Run model inference
#         with torch.no_grad():
#             generated_ids = model.generate(**inputs, max_new_tokens=64)
#         generated_ids_trimmed = [
#             out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)
#         ]
#         output_texts = processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
#         pred_text = output_texts[0].strip()
#         print(f"\nFrame {frame_id} Prediction:\n{pred_text}")

#         # Parse bbox from model output
#         try:
#             frame_bboxes = parse_bboxes_from_text(pred_text)
#             chosen_bbox = None

#             if frame_bboxes:
#                 for fb in frame_bboxes:
#                     try:
#                         fb_frame = int(fb.get("frame", frame_id))
#                     except Exception:
#                         fb_frame = None
#                     fb_bbox = fb.get("bbox") if isinstance(fb, dict) else None
#                     if fb_bbox and isinstance(fb_bbox, (list, tuple)) and len(fb_bbox) == 4:
#                         try:
#                             fb_bbox = [float(x) for x in fb_bbox]
#                         except Exception:
#                             continue
#                         if fb_frame == frame_id:
#                             chosen_bbox = fb_bbox
#                             break
#                         if chosen_bbox is None:
#                             chosen_bbox = fb_bbox

#             # Regex fallback
#             if chosen_bbox is None:
#                 m = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', pred_text)
#                 if m:
#                     chosen_bbox = [float(m.group(i)) for i in range(1, 5)]

#             if chosen_bbox is None:
#                 raise ValueError("No valid bbox found in model output")

#             pred_bboxes.append({"frame": frame_id, "bbox": chosen_bbox})

#         except Exception as e:
#             print(f"Warning: failed to parse bbox for frame {frame_id}: {e}\nRaw output: {pred_text}")
#             pred_bboxes.append({"frame": frame_id, "bbox": [0.0, 0.0, 0.0, 0.0]})

#     # Compute IoU per frame
#     frame_ious = []
#     for gt in gt_outputs:
#         frame = gt["frame"]
#         gt_box = gt["bbox"]
#         pred_box = next((p["bbox"] for p in pred_bboxes if p["frame"] == frame), None)
#         if pred_box:
#             iou = compute_iou(gt_box, pred_box)
#             frame_ious.append(iou)
#         else:
#             frame_ious.append(0.0)

#     avg_iou = float(np.mean(frame_ious)) if frame_ious else 0.0

#     result_entry = {
#         "id": qid,
#         "type": "single",
#         "ref_image": item["ref_image"],
#         "input_images": item["input_images"],
#         "num_input_images": len(item["input_images"]),
#         "question": item["question"],
#         "track_id": item.get("track_id", None),
#         "initial_bbox": initial_bbox,
#         "predicted_output": pred_bboxes,
#         "expected_output": gt_outputs,
#         "frame_ious": frame_ious,
#         "average_iou": avg_iou
#     }

#     return avg_iou, result_entry


def evaluate_single_object(item, dataset_root, model, processor, device):
    """Evaluate single object tracking task with all frames passed at once."""

    qid = item["id"]
    ref_image = os.path.join(dataset_root, item["ref_image"])
    input_images = [os.path.join(dataset_root, img) for img in item["input_images"]]

    # Parse expected output
    gt_outputs = item["expected_output"]
    if isinstance(gt_outputs, str):
        gt_outputs = json.loads(gt_outputs)

    initial_bbox = item.get("initial_bbox", [0, 0, 0, 0])

    # UPDATED QUESTION (asks for all frames)
    question = (
        item["question"]
        + f"\nInitially, the object is present at {initial_bbox} in the reference image."
        + "\nThe following images are consecutive frames of a video."
        + "\nPlease provide the bounding box for this object in EACH frame, "
          "in the format:"
          "\n[{\"frame\": N, \"bbox\": [x, y, w, h]}, ...]."
          "\nDo not write any extra text."
    )

    # BUILD MESSAGE WITH ALL IMAGES AT ONCE
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": ref_image}] +
                [{"type": "image", "image": img} for img in input_images] +
                [{"type": "text", "text": question}]
            ),
        }
    ]

    # PREPARE INPUTS
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # MODEL INFERENCE (ONCE)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    pred_text = output_texts[0].strip()
    print(f"\nSingle Object Prediction:\n{pred_text}")

    # PARSE PREDICTIONS
    pred_bboxes = []
    # try:
    #     frame_bboxes = parse_bboxes_from_text(pred_text)

    #     for gt in gt_outputs:
    #         frame_id = gt["frame"]
    #         chosen_bbox = None

    #         for fb in frame_bboxes:
    #             if (
    #                 isinstance(fb, dict)
    #                 and fb.get("frame") == frame_id
    #                 and isinstance(fb.get("bbox"), (list, tuple))
    #                 and len(fb["bbox"]) == 4
    #             ):
    #                 chosen_bbox = [float(x) for x in fb["bbox"]]
    #                 break

    #         if chosen_bbox is None:
    #             chosen_bbox = [0.0, 0.0, 0.0, 0.0]

    #         pred_bboxes.append({"frame": frame_id, "bbox": chosen_bbox})

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
                "bbox": [0.0, 0.0, 0.0, 0.0]
            })

    # COMPUTE IoU PER FRAME
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



# ---------- Multi Object Evaluation ----------
# def evaluate_multi_object(item, dataset_root, model, processor, device):
#     """Evaluate multi-object tracking task."""
#     qid = item["id"]
#     ref_image = os.path.join(dataset_root, item["ref_image"])
#     input_images = [os.path.join(dataset_root, img) for img in item["input_images"]]
    
#     # Parse expected output
#     gt_outputs = item["expected_output"]
#     if isinstance(gt_outputs, str):
#         gt_outputs = json.loads(gt_outputs)
    
#     objects = item.get("objects", [])
    
#     all_object_ious = []
#     predicted_objects = {}
#     expected_objects = {}

#     # Process each object separately
#     for obj in objects:
#         obj_id = obj["object_id"]
#         initial_bbox = obj["initial_bbox"]
#         track_id = obj.get("track_id")
        
#         # Get ground truth trajectory for this object
#         obj_key = f"object_{obj_id}"
#         if obj_key not in gt_outputs:
#             print(f"Warning: {obj_key} not found in ground truth")
#             continue
        
#         gt_trajectory = gt_outputs[obj_key]["trajectory"]
        
#         question = (
#             item["question"] + 
#             f"\nInitially, object {obj_id} is present at {initial_bbox} in the reference image."
#             + "\nPlease provide the bounding box for this object in this frame only, "
#               "in the format: [{\"frame\": N, \"bbox\": [x, y, w, h]}]. "
#               "Do not write any extra text."
#         )

#         pred_bboxes = []

#         # Process each input image for this object
#         for img_path, gt_entry in zip(input_images, gt_trajectory):
#             frame_id = gt_entry["frame"]

#             messages = [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "image", "image": ref_image},
#                         {"type": "image", "image": img_path},
#                         {"type": "text", "text": question},
#                     ],
#                 }
#             ]

#             text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#             image_inputs, video_inputs = process_vision_info(messages)
#             inputs = processor(
#                 text=[text],
#                 images=image_inputs,
#                 videos=video_inputs,
#                 padding=True,
#                 return_tensors="pt",
#             ).to(device)

#             with torch.no_grad():
#                 generated_ids = model.generate(**inputs, max_new_tokens=64)
#             generated_ids_trimmed = [
#                 out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)
#             ]
#             output_texts = processor.batch_decode(
#                 generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#             )
#             pred_text = output_texts[0].strip()
#             print(f"\nObject {obj_id}, Frame {frame_id} Prediction:\n{pred_text}")

#             # Parse bbox
#             try:
#                 frame_bboxes = parse_bboxes_from_text(pred_text)
#                 chosen_bbox = None

#                 if frame_bboxes:
#                     for fb in frame_bboxes:
#                         try:
#                             fb_frame = int(fb.get("frame", frame_id))
#                         except Exception:
#                             fb_frame = None
#                         fb_bbox = fb.get("bbox") if isinstance(fb, dict) else None
#                         if fb_bbox and isinstance(fb_bbox, (list, tuple)) and len(fb_bbox) == 4:
#                             try:
#                                 fb_bbox = [float(x) for x in fb_bbox]
#                             except Exception:
#                                 continue
#                             if fb_frame == frame_id:
#                                 chosen_bbox = fb_bbox
#                                 break
#                             if chosen_bbox is None:
#                                 chosen_bbox = fb_bbox

#                 if chosen_bbox is None:
#                     m = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', pred_text)
#                     if m:
#                         chosen_bbox = [float(m.group(i)) for i in range(1, 5)]

#                 if chosen_bbox is None:
#                     raise ValueError("No valid bbox found")

#                 pred_bboxes.append({"frame": frame_id, "bbox": chosen_bbox})

#             except Exception as e:
#                 print(f"Warning: failed to parse bbox for object {obj_id}, frame {frame_id}: {e}")
#                 pred_bboxes.append({"frame": frame_id, "bbox": [0.0, 0.0, 0.0, 0.0]})

#         # Compute IoU for this object
#         frame_ious = []
#         for gt in gt_trajectory:
#             frame = gt["frame"]
#             gt_box = gt["bbox"]
#             pred_box = next((p["bbox"] for p in pred_bboxes if p["frame"] == frame), None)
#             if pred_box:
#                 iou = compute_iou(gt_box, pred_box)
#                 frame_ious.append(iou)
#             else:
#                 frame_ious.append(0.0)

#         obj_avg_iou = float(np.mean(frame_ious)) if frame_ious else 0.0
#         all_object_ious.append(obj_avg_iou)

#         # Store predictions and ground truth for this object
#         predicted_objects[obj_key] = {
#             "object_id": obj_id,
#             "track_id": track_id,
#             "initial_bbox": initial_bbox,
#             "trajectory": pred_bboxes,
#             "frame_ious": frame_ious,
#             "average_iou": obj_avg_iou
#         }
        
#         expected_objects[obj_key] = {
#             "object_id": obj_id,
#             "track_id": track_id,
#             "initial_bbox": initial_bbox,
#             "trajectory": gt_trajectory
#         }

#     # Average IoU across all objects
#     overall_iou = float(np.mean(all_object_ious)) if all_object_ious else 0.0

#     result_entry = {
#         "id": qid,
#         "type": "multi",
#         "ref_image": item["ref_image"],
#         "input_images": item["input_images"],
#         "num_input_images": len(item["input_images"]),
#         "question": item["question"],
#         "num_objects": len(objects),
#         "objects": item.get("objects", []),
#         "predicted_output": predicted_objects,
#         "expected_output": expected_objects,
#         "average_iou": overall_iou
#     }

#     return overall_iou, result_entry


def evaluate_multi_object(item, dataset_root, model, processor, device):
    """Evaluate multi-object tracking task with all frames passed at once per object."""

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

        # UPDATED QUESTION (asks for all frames)
        question = (
            item["question"]
            + f"\nInitially, object {obj_id} is present at {initial_bbox} in the reference image."
            + "\nThe following images are consecutive frames of a video."
            + "\nPlease provide the bounding box for this object in EACH frame, "
              "in the format:"
              "\n[{\"frame\": N, \"bbox\": [x, y, w, h]}, ...]."
              "\nDo not write any extra text."
        )

        # BUILD MESSAGE WITH ALL IMAGES AT ONCE (FIXED)
        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": ref_image}] +
                    [{"type": "image", "image": img} for img in input_images] +
                    [{"type": "text", "text": question}]
                ),
            }
        ]

        # PREPARE INPUTS
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # MODEL INFERENCE (ONCE PER OBJECT)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [
            out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)
        ]

        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        pred_text = output_texts[0].strip()
        print(f"\nObject {obj_id} Prediction:\n{pred_text}")

        # PARSE PREDICTIONS
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

        # COMPUTE IoU
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

        # STORE RESULTS
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

    # FINAL AVERAGE IoU
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

