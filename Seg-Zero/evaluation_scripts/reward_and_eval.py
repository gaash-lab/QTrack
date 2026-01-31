#!/usr/bin/env python3
"""
Simple parser + unit test for extracting predicted bboxes/points and `<think>` text
from model-generated strings. This version is more forgiving to common formatting issues.

Usage:
  python evaluation_scripts/reward_and_eval.py --unit_test
  python evaluation_scripts/reward_and_eval.py --input_file some_output.json   # future use
"""

import re
import json
import argparse
from typing import Tuple, List, Dict, Any

def _find_first_bracketed_json(s: str):
    """
    Find the first [...] or {...} substring in s and return it, else None.
    """
    m = re.search(r"(\[.*?\])", s, re.DOTALL)
    if m:
        return m.group(1)
    m2 = re.search(r"(\{.*?\})", s, re.DOTALL)
    if m2:
        return m2.group(1)
    return None

def extract_bbox_points_think(output_text: str, x_factor: float = 1.0, y_factor: float = 1.0
                              ) -> Tuple[List[List[int]], List[List[int]], str]:
    """
    Parses an output string for:
      - <think> ... </think>  (string)
      - <answer> JSON </answer> where JSON is a list of objects containing bbox_2d and point_2d

    Behavior changes (for robustness):
      - If no <answer> block is found, tries to locate the first bracketed JSON anywhere in the text.
      - If <answer> block is present but empty, attempts bracketed JSON recovery.
      - If parsed JSON is an object (dict), we wrap it into a list.
      - On unrecoverable parse errors we return empty lists (instead of raising) so evaluation can continue.
    """
    pred_bboxes: List[List[int]] = []
    pred_points: List[List[int]] = []
    think_text = ""

    # 1) extract <think> ... </think> if present
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", output_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_text = think_match.group(1).strip()

    # 2) extract <answer> ... </answer> if present
    answer_re = re.search(r"<answer>\s*(.*?)\s*</answer>", output_text, re.DOTALL | re.IGNORECASE)
    answer_text = None
    if answer_re:
        answer_text = answer_re.group(1).strip()
        # if empty string, fallback to bracket find
        if answer_text == "":
            answer_text = None

    # 3) fallback: find first bracketed JSON anywhere
    if answer_text is None:
        bracket = _find_first_bracketed_json(output_text)
        if bracket is not None:
            answer_text = bracket

    if answer_text is None:
        # No answer-like content found. Return empty results (caller will treat as no prediction).
        return pred_bboxes, pred_points, think_text

    # 4) attempt to clean common issues and parse JSON
    cleaned = answer_text.replace("'", "\"")                  # single -> double quotes
    cleaned = re.sub(r",\s*}", "}", cleaned)                 # trailing commas before }
    cleaned = re.sub(r",\s*]", "]", cleaned)                 # trailing commas before ]
    cleaned = cleaned.strip()

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        # Try to extract the bracketed portion from cleaned text and parse
        bracket = _find_first_bracketed_json(cleaned)
        if bracket:
            try:
                data = json.loads(bracket.replace("'", "\""))
            except Exception:
                data = None
        else:
            data = None

    if data is None:
        # Unrecoverable parse: return empty predictions instead of raising.
        return pred_bboxes, pred_points, think_text

    # If data is a dict, wrap into list (we expect a list of objects)
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        # Not a list even after recovery - treat as no prediction
        return pred_bboxes, pred_points, think_text

    for idx, item in enumerate(data):
        try:
            if not isinstance(item, dict):
                continue
            if "bbox_2d" not in item or "point_2d" not in item:
                continue
            bbox = item["bbox_2d"]
            point = item["point_2d"]
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            if not (isinstance(point, (list, tuple)) and len(point) == 2):
                continue

            bx = int(bbox[0] * x_factor + 0.5)
            by = int(bbox[1] * y_factor + 0.5)
            bx2 = int(bbox[2] * x_factor + 0.5)
            by2 = int(bbox[3] * y_factor + 0.5)
            px = int(point[0] * x_factor + 0.5)
            py = int(point[1] * y_factor + 0.5)

            pred_bboxes.append([bx, by, bx2, by2])
            pred_points.append([px, py])
        except Exception:
            # skip malformed item
            continue

    return pred_bboxes, pred_points, think_text


# -----------------------
# unit test harness
# -----------------------
def run_unit_tests(verbose: bool = True) -> Dict[str, Any]:
    """
    Runs a set of synthetic/edge-case strings through the parser and reports failure rate
    and example failures.
    """
    samples = [
        # 1. clean well-formed output
        '<think>checked</think><answer>[{"bbox_2d": [10,20,110,120], "point_2d": [50,60]}]</answer>',
        # 2. single quotes instead of double
        "<think>mine</think><answer>[{'bbox_2d': [5,5,55,55], 'point_2d': [30,30]}]</answer>",
        # 3. missing answer
        "<think>no answer here</think>",
        # 4. malformed JSON (trailing comma)
        "<answer>[{'bbox_2d':[1,2,3,4],'point_2d':[2,3],},]</answer>",
        # 5. empty answer
        "<answer>   </answer>",
        # 6. answer not list (dict)
        "<answer>{'bbox_2d':[1,2,3,4],'point_2d':[2,3]}</answer>",
        # 7. answer with multiple objects
        '<answer>[{"bbox_2d":[0,0,10,10],"point_2d":[2,3]},{"bbox_2d":[20,20,40,40],"point_2d":[30,30]}]</answer>',
        # 8. answer with floats (normalized coords)
        '<answer>[{"bbox_2d":[0.1,0.2,0.3,0.4],"point_2d":[0.15,0.25]}]</answer>',
        # 9. noisy text surrounding blocks
        'some text <think>analysis</think> random <answer>[{"bbox_2d":[1,1,2,2],"point_2d":[1,1]}]</answer> end',
        # 10. badly nested tags
        '<answer>[{"bbox_2d":[1,2,3,4],"point_2d":[5,6]}]</answer><think>ok</think>',
    ]

    results = []
    failures = []
    for i, s in enumerate(samples):
        try:
            b, p, t = extract_bbox_points_think(s, x_factor=1.0, y_factor=1.0)
            ok = (len(b) > 0)
            if not ok:
                failures.append({"idx": i, "sample": s, "error": "No bboxes extracted"})
            results.append({"ok": ok, "bboxes": b, "points": p, "think": t})
        except Exception as e:
            failures.append({"idx": i, "sample": s, "error": str(e)})
            results.append({"ok": False, "error": str(e)})

    parse_fail_rate = len(failures) / max(1, len(samples))
    report = {
        "total": len(samples),
        "failures": failures,
        "parse_fail_rate": parse_fail_rate,
        "results": results
    }

    if verbose:
        print(f"Unit test samples: {len(samples)}")
        print(f"Failures: {len(failures)}")
        print(f"parse_fail_rate: {parse_fail_rate:.3f}")
        if failures:
            print("\nExample failures (up to 5):")
            for f in failures[:5]:
                print(f"- idx={f['idx']} error={f['error']} sample={f['sample'][:200]}")

    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--unit_test", action="store_true", help="Run internal parser unit tests and exit.")
    p.add_argument("--input_file", type=str, help="(future) parse a file of generated texts")
    return p.parse_args()

def main():
    args = parse_args()
    if args.unit_test:
        report = run_unit_tests(verbose=True)
        # also save a small json report for inspection
        with open("evaluation_scripts/parser_unit_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Saved report to evaluation_scripts/parser_unit_test_report.json")
        return

    print("No mode supplied. Use --unit_test for now.")

if __name__ == "__main__":
    main()
