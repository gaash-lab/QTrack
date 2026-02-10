import re
import json
from scipy.optimize import linear_sum_assignment
import numpy as np

def vision_reasoner_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0 
    
    def segmentation_format(predict_str: str) -> float:
        segmentation_format_reward = 0.0
        try:
            json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
            if not json_match:
                return segmentation_format_reward
            data = json.loads(json_match.group(1))
            
            data_cnt = len(data)
            
            for item in data:
                cur_reward = 0.0

                if 'bbox_2d' in item:
                    bbox_2d = item['bbox_2d']
                    if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        cur_reward += 1.0
                    
                if 'point_2d' in item:
                    point_2d = item['point_2d']
                    if isinstance(point_2d, list) and len(point_2d) == 2:
                        cur_reward += 1.0
                
                segmentation_format_reward += cur_reward / data_cnt
        except Exception:
            pass
        return segmentation_format_reward
        
    segmentation_format_reward = segmentation_format(predict_str)
    
    return thinking_format_reward + segmentation_format_reward

def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120  # 设置上限
    
    try:
        gt_data = json.loads(ground_truth)
        gt_bboxes = [item['bbox_2d'] for item in gt_data]
        gt_points = [item['point_2d'] for item in gt_data]
            
        #json_match = re.search(r'```json\s*(.*?)\s*```', predict_str, re.DOTALL)
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            pred_bboxes = [item['bbox_2d'] for item in data]
            pred_points = [item['point_2d'] for item in data]
            
            # 只有当预测或真实值超过上限时才截断
            if len(pred_bboxes) > MAX_OBJECTS:
                pred_bboxes = pred_bboxes[:MAX_OBJECTS]
                pred_points = pred_points[:MAX_OBJECTS]
            
            if len(gt_bboxes) > MAX_OBJECTS:
                gt_bboxes = gt_bboxes[:MAX_OBJECTS]
                gt_points = gt_points[:MAX_OBJECTS]
            
            # 预处理数据为numpy数组
            pred_bboxes = np.array(pred_bboxes)  # (M,4)
            pred_points = np.array(pred_points)  # (M,2)
            gt_bboxes = np.array(gt_bboxes)    # (N,4)
            gt_points = np.array(gt_points)     # (N,2)
            
            # 并行计算所有指标
            iou_matrix = batch_iou(pred_bboxes, gt_bboxes)  # (M,N)
            l1_matrix = batch_l1_distance(pred_bboxes, gt_bboxes)  # (M,N)
            points_dist_matrix = batch_points_distance(pred_points, gt_points)  # (M,N)
            points_in_box = batch_points_in_box(pred_points, pred_bboxes)  # (M,)
            
            # 计算reward矩阵
            iou_reward = (iou_matrix > 0.5).astype(float)
            bbox_l1_reward = (l1_matrix < 10).astype(float)
            point_reward = ((points_dist_matrix < 30) & points_in_box[:,np.newaxis]).astype(float)
            
            # 构建最终的cost矩阵
            cost_matrix = 3.0 - (iou_reward + bbox_l1_reward + point_reward)
            
            # 使用匈牙利算法找最优匹配
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # 直接从cost_matrix计算总reward
            total_reward = len(row_indices) * 3.0 - cost_matrix[row_indices, col_indices].sum()
            
            # 计算平均reward
            max_length = max(len(pred_bboxes), len(gt_bboxes))
            max_accuracy_reward = total_reward / max_length
            
    except Exception:
        pass
    return max_accuracy_reward

def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0  # 初始满分
    try:
        sentences = predict_str.split('.')
        
        # 移除空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 检查重复
        seen = set()
        repeats = 0
        
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >=2:
                non_repeat_reward = 0
                break
            seen.add(sentence)
            
    except Exception:
        pass
    
    return non_repeat_reward

def vision_reasoner_mcp_reward(gt_prev_bbox, gt_curr_bbox, pred_curr_bbox, alpha=0.9, eps=1e-6):
    """
    Motion Consistency Precision (MCP) reward for next-frame prediction.

    Parameters:
        gt_prev_bbox: [x, y, w, h] at time t-1
        gt_curr_bbox: [x, y, w, h] at time t
        pred_curr_bbox: [x, y, w, h] predicted for time t
        alpha: looseness factor for speed consistency
        eps: small constant for numerical stability

    Returns:
        reward in [0,1]
    """
    gt_prev_bbox = np.asarray(gt_prev_bbox, dtype=float)
    gt_curr_bbox = np.asarray(gt_curr_bbox, dtype=float)
    pred_curr_bbox = np.asarray(pred_curr_bbox, dtype=float)

    # Ground-truth motion vector (using top-left corner x,y)
    dg = gt_curr_bbox[:2] - gt_prev_bbox[:2]
    # Predicted motion vector
    dp = pred_curr_bbox[:2] - gt_prev_bbox[:2]

    ng = np.linalg.norm(dg)
    np_ = np.linalg.norm(dp)

    # If GT object does not move, we do not enforce motion
    if ng < eps:
        return 1.0

    # ------------------------
    # 1. Direction consistency
    # ------------------------
    cos_theta = np.dot(dp, dg) / (np_ * ng + eps)
    direction_reward = (1.0 + cos_theta) / 2.0   # scale [-1,1] → [0,1]

    # ------------------------
    # 2. Speed consistency
    # ------------------------
    speed_reward = np.exp(-((np_ - ng) ** 2) / (2 * (alpha * ng) ** 2 + eps))

    # ------------------------
    # 3. Anti-static penalty
    # ------------------------
    # If model predicts almost no motion while GT moves, penalize strongly
    if np_ < 0.1 * ng:
        static_penalty = 0.2
    else:
        static_penalty = 1.0

    # ------------------------
    # Final MCP reward
    # ------------------------
    mcp_reward = direction_reward * speed_reward * static_penalty

    # Clamp for safety
    mcp_reward = float(np.clip(mcp_reward, 0.0, 1.0))

    return mcp_reward

def extract_predicted_bbox_xywh(predict_str: str):
    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return None

        data = json.loads(json_match.group(1))
        if not data or not isinstance(data, list):
            return None

        obj = data[0]  # assuming single object

        if "bbox" in obj:
            return np.asarray(obj["bbox"], dtype=float)
        elif "bbox_2d" in obj:
            return np.asarray(obj["bbox_2d"], dtype=float)
        else:
            return None

    except Exception:
        return None



def extract_gt_bbox_xywh(gt_list: str):
    if not gt_list:
        return None
    # return np.asarray(json.loads(gt_list)[0]["bbox"], dtype=float)  # assuming single object
    return np.asarray(gt_list[0]["bbox_2d"], dtype=float)

def vision_reasoner_compute_score(predict_str: str, ground_truth: dict) -> float:
    format_reward = vision_reasoner_format_reward(predict_str)
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, json.dumps(ground_truth["curr"]))

    # Extract predicted bbox
    pred_bbox = extract_predicted_bbox_xywh(predict_str)
    gt_prev_bbox = extract_gt_bbox_xywh(ground_truth["prev"])
    gt_curr_bbox = extract_gt_bbox_xywh(ground_truth["curr"])
    with open("/home/gaash/Wasif/Tawheed/Rmot/debug.txt", "a") as f:
        f.write(f"PRED BBOX: {pred_bbox}\n")
        f.write(f"GT PREV BBOX: {gt_prev_bbox}\n")
        f.write(f"GT CURR BBOX: {gt_curr_bbox}\n")
    
    ############### Uncomment the below code to enable MCP reward calculation ###############
    if pred_bbox is None or gt_prev_bbox is None or gt_curr_bbox is None:
        print("Failed to extract bbox for MCP reward.")
        mcp_reward = 0.0
    else:
        mcp_reward = vision_reasoner_mcp_reward(
            gt_prev_bbox,
            gt_curr_bbox,
            pred_bbox
        )

    ############## Comment the below line to enable MCP reward calculation ##############
    # mcp_reward = 0.0
    
    reward = format_reward + accuracy_reward + mcp_reward
    return reward

def batch_iou(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    # 广播机制自动扩展维度
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # (M,1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # (N,1)
    
    xA = np.maximum(x11, np.transpose(x21))  # (M,N)
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)  # (M,1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)  # (N,1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / unionArea  # (M,N)
    return iou

def batch_l1_distance(boxes1, boxes2):
    # boxes1: (M,4), boxes2: (N,4)
    boxes1 = boxes1[:, np.newaxis, :]  # (M,1,4)
    boxes2 = boxes2[np.newaxis, :, :]  # (1,N,4)
    return np.mean(np.abs(boxes1 - boxes2), axis=2)  # (M,N)

def batch_points_distance(points1, points2):
    # points1: (M,2), points2: (N,2)
    points1 = points1[:, np.newaxis, :]  # (M,1,2)
    points2 = points2[np.newaxis, :, :]  # (1,N,2)
    
    # 计算欧氏距离
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))  # (M,N)
    return dist

def batch_points_in_box(points, boxes):
    """
    检查每个点是否在对应的框内
    points: (M,2) - M个点的坐标
    boxes: (M,4) - M个框的坐标 [x1,y1,x2,y2]
    返回: (M,) 布尔数组
    """
    x_check = (points[:,0] >= boxes[:,0]) & (points[:,0] <= boxes[:,2])
    y_check = (points[:,1] >= boxes[:,1]) & (points[:,1] <= boxes[:,3])
    return x_check & y_check

if __name__ == "__main__":
    predict_str = """
<answer>
[{"bbox": [10, 100, 398, 423]}]
</answer>
"""
    ground_truth = {'curr': [{'bbox_2d': [1465.0, 420.0, 74.0, 176.0]}], 'prev': [{'bbox_2d': [1471.0, 419.0, 74.0, 176.0]}]}
    print(predict_str)
    print(ground_truth)
    print("GT JSON:", json.dumps(ground_truth["curr"]))
    print("PRED STR:", predict_str)
    print(vision_reasoner_compute_score(predict_str, ground_truth))
    
