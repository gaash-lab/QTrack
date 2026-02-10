import os
import json
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration
from PIL import Image

# ---------- IoU Function ----------
def compute_iou(boxA, boxB):
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

# ---------- Parse bboxes from model output ----------
def parse_bboxes_from_text(text):
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except:
        pass
    # Regex fallback
    pattern = r'\{.*?"frame":\s*(\d+).*?"bbox":\s*\[([^\]]+)\].*?\}'
    matches = re.findall(pattern, text)
    parsed = []
    for m in matches:
        frame = int(m[0])
        bbox = [float(x) for x in m[1].split(",")]
        parsed.append({"frame": frame, "bbox": bbox})
    return parsed

# ---------- Detect single/multi-object ----------
def detect_format(item):
    if "track_id" in item and isinstance(item.get("expected_output"), (list, str)):
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
    try:
        exp_out = item["expected_output"]
        if isinstance(exp_out, str):
            exp_out = json.loads(exp_out)
        if isinstance(exp_out, dict) and any(k.startswith("object_") for k in exp_out.keys()):
            return "multi"
    except:
        pass
    return "single"

# ---------- Single Object Evaluation ----------
def evaluate_single_object(item, dataset_root, model, processor, device):
    qid = item["id"]
    ref_image_obj = Image.open(os.path.join(dataset_root, item["ref_image"])).convert("RGB")
    input_image_objs = [Image.open(os.path.join(dataset_root, img)).convert("RGB") for img in item["input_images"]]

    gt_outputs = item["expected_output"]
    if isinstance(gt_outputs, str):
        gt_outputs = json.loads(gt_outputs)

    initial_bbox = item.get("initial_bbox", [0,0,0,0])

    question = (
        item["question"]
        + f"\nInitially, the object is present at {initial_bbox} in the reference image."
        + "\nProvide the bounding box for this frame only in JSON format: [{\"frame\": N, \"bbox\": [x, y, w, h]}]."
    )
    prompt_text = "You are a vision-language model performing object tracking.\nAnswer strictly in JSON format.\n\n" + question

    pred_bboxes = []
    for input_image_obj, gt_entry in zip(input_image_objs, gt_outputs):
        frame_id = gt_entry["frame"]
        images = [{"type":"image","image":ref_image_obj},{"type":"image","image":input_image_obj}]
        inputs = processor(text=[prompt_text], images=images, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64)

        generated_ids_trimmed = [out[len(in_ids):] for in_ids,out in zip(inputs.input_ids, generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred_text = output_texts[0].strip()
        print(f"\nFrame {frame_id} Prediction:\n{pred_text}")

        try:
            frame_bboxes = parse_bboxes_from_text(pred_text)
            chosen_bbox = None
            if frame_bboxes:
                for fb in frame_bboxes:
                    fb_frame = fb.get("frame", frame_id)
                    fb_bbox = fb.get("bbox")
                    if isinstance(fb_bbox,(list,tuple)) and len(fb_bbox)==4:
                        fb_bbox = [float(x) for x in fb_bbox]
                        if fb_frame==frame_id:
                            chosen_bbox = fb_bbox
                            break
                        if chosen_bbox is None:
                            chosen_bbox = fb_bbox
            if chosen_bbox is None:
                m = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]',pred_text)
                if m: chosen_bbox = [float(m.group(i)) for i in range(1,5)]
            if chosen_bbox is None: raise ValueError("No valid bbox found")
            pred_bboxes.append({"frame": frame_id, "bbox": chosen_bbox})
        except Exception as e:
            print(f"Warning: failed to parse bbox for frame {frame_id}: {e}")
            pred_bboxes.append({"frame": frame_id, "bbox":[0.0,0.0,0.0,0.0]})

    frame_ious = []
    for gt in gt_outputs:
        frame = gt["frame"]
        gt_box = gt["bbox"]
        pred_box = next((p["bbox"] for p in pred_bboxes if p["frame"]==frame), None)
        frame_ious.append(compute_iou(gt_box,pred_box) if pred_box else 0.0)

    avg_iou = float(np.mean(frame_ious)) if frame_ious else 0.0

    result_entry = {
        "id": qid,
        "type":"single",
        "ref_image":item["ref_image"],
        "input_images":item["input_images"],
        "num_input_images":len(item["input_images"]),
        "question":item["question"],
        "track_id":item.get("track_id"),
        "initial_bbox":initial_bbox,
        "predicted_output":pred_bboxes,
        "expected_output":gt_outputs,
        "frame_ious":frame_ious,
        "average_iou":avg_iou
    }
    return avg_iou, result_entry

# ---------- Multi Object Evaluation ----------
def evaluate_multi_object(item, dataset_root, model, processor, device):
    qid = item["id"]
    ref_image_obj = Image.open(os.path.join(dataset_root, item["ref_image"])).convert("RGB")
    input_image_objs = [Image.open(os.path.join(dataset_root, img)).convert("RGB") for img in item["input_images"]]

    gt_outputs = item["expected_output"]
    if isinstance(gt_outputs, str): gt_outputs = json.loads(gt_outputs)
    objects = item.get("objects",[])

    all_object_ious = []
    predicted_objects = {}
    expected_objects = {}

    def build_prompt(question):
        return "You are a vision-language model performing object tracking.\nAnswer strictly in JSON format.\n\n" + question

    for obj in objects:
        obj_id = obj["object_id"]
        initial_bbox = obj["initial_bbox"]
        track_id = obj.get("track_id")
        obj_key = f"object_{obj_id}"
        if obj_key not in gt_outputs: 
            print(f"Warning: {obj_key} not found in ground truth")
            continue
        gt_trajectory = gt_outputs[obj_key]["trajectory"]

        question = (
            item["question"]
            + f"\nInitially, object {obj_id} is present at {initial_bbox} in the reference image."
            + "\nProvide the bounding box for this frame only in JSON format."
        )
        prompt_text = build_prompt(question)
        pred_bboxes = []

        for input_image_obj, gt_entry in zip(input_image_objs, gt_trajectory):
            frame_id = gt_entry["frame"]

            # Images: reference + current frame (as a list of PIL Images)
            images = [ref_image_obj, input_image_obj]  # <- pass PIL.Image list directly

            inputs = processor(
                text=[prompt_text],
                images=images,
                padding=True,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=64)

            generated_ids_trimmed = [
                out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, generated_ids)
            ]

            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            pred_text = output_texts[0].strip()
            print(f"\nObject {obj_id}, Frame {frame_id} Prediction:\n{pred_text}")

            try:
                frame_bboxes = parse_bboxes_from_text(pred_text)
                chosen_bbox = None
                if frame_bboxes:
                    for fb in frame_bboxes:
                        fb_frame = fb.get("frame", frame_id)
                        fb_bbox = fb.get("bbox")
                        if isinstance(fb_bbox,(list,tuple)) and len(fb_bbox)==4:
                            fb_bbox = [float(x) for x in fb_bbox]
                            if fb_frame==frame_id: chosen_bbox=fb_bbox; break
                            if chosen_bbox is None: chosen_bbox=fb_bbox
                if chosen_bbox is None:
                    m = re.search(r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]', pred_text)
                    if m: chosen_bbox = [float(m.group(i)) for i in range(1,5)]
                if chosen_bbox is None: raise ValueError("No valid bbox found")
                pred_bboxes.append({"frame": frame_id, "bbox": chosen_bbox})
            except Exception as e:
                print(f"Warning: failed to parse bbox for object {obj_id}, frame {frame_id}: {e}")
                pred_bboxes.append({"frame": frame_id, "bbox":[0.0,0.0,0.0,0.0]})

        frame_ious = []
        for gt in gt_trajectory:
            frame = gt["frame"]
            gt_box = gt["bbox"]
            pred_box = next((p["bbox"] for p in pred_bboxes if p["frame"]==frame), None)
            frame_ious.append(compute_iou(gt_box,pred_box) if pred_box else 0.0)

        obj_avg_iou = float(np.mean(frame_ious)) if frame_ious else 0.0
        all_object_ious.append(obj_avg_iou)
        predicted_objects[obj_key] = {
            "object_id": obj_id,
            "track_id": track_id,
            "initial_bbox": initial_bbox,
            "trajectory": pred_bboxes,
            "frame_ious": frame_ious,
            "average_iou": obj_avg_iou
        }
        expected_objects[obj_key] = {
            "object_id": obj_id,
            "track_id": track_id,
            "initial_bbox": initial_bbox,
            "trajectory": gt_trajectory
        }

    overall_iou = float(np.mean(all_object_ious)) if all_object_ious else 0.0
    result_entry = {
        "id": qid,
        "type":"multi",
        "ref_image":item["ref_image"],
        "input_images":item["input_images"],
        "num_input_images":len(item["input_images"]),
        "question":item["question"],
        "num_objects":len(objects),
        "objects":objects,
        "predicted_output":predicted_objects,
        "expected_output":expected_objects,
        "average_iou":overall_iou
    }
    return overall_iou, result_entry

# ---------- Main Evaluation ----------
def evaluate_model(json_path, dataset_root, output_json_path="evaluation_results.json"):
    with open(json_path,"r") as f: data=json.load(f)
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    print(f"Loading model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_iou = 0
    total_samples = 0
    results=[]
    single_count=0
    multi_count=0

    for item in tqdm(data, desc="Evaluating"):
        format_type = detect_format(item)
        try:
            if format_type=="single":
                single_count+=1
                avg_iou,result_entry = evaluate_single_object(item,dataset_root,model,processor,device)
            else:
                multi_count+=1
                avg_iou,result_entry = evaluate_multi_object(item,dataset_root,model,processor,device)
            total_iou+=avg_iou
            total_samples+=1
            results.append(result_entry)
        except Exception as e:
            print(f"\nError processing {item.get('id','unknown')}: {e}")
            import traceback
            traceback.print_exc()
            continue

    overall_score = total_iou/total_samples if total_samples>0 else 0.0
    summary = {
        "total_samples":total_samples,
        "single_object_samples":single_count,
        "multi_object_samples":multi_count,
        "average_iou":overall_score
    }
    output_data={"results":results,"summary":summary}
    with open(output_json_path,"w") as f:
        json.dump(output_data,f,indent=4)

    print(f"\nEvaluation results saved to {output_json_path}")
    print(f"Total Samples: {total_samples}, Single: {single_count}, Multi: {multi_count}, Average IoU: {overall_score:.4f}")
    return overall_score

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--json",default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/test/annotations_real.json")
    parser.add_argument("--dataset_root",default="/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/test")
    parser.add_argument("--json_log",default="evaluation_llama.json")
    args=parser.parse_args()
    evaluate_model(args.json,args.dataset_root,args.json_log)
