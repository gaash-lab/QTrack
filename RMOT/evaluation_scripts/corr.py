#!/usr/bin/env python3
"""
Correlation analysis between reasoning and BBox IoU
- Loads two evaluation JSONs (baseline vs exp5)
- Computes per-sample reasoning metrics (think length, #pred boxes)
- Computes IoU stats, pass rate, Pearson/Spearman correlations
- Saves plots (scatter, overlay, histograms), CSVs, and a summary JSON

CPU-only. No GPUs or extra deps beyond numpy/matplotlib.
"""

import os
import json
import re
import math
import csv
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr  # if missing: pip install scipy

# =======================
# EDIT THESE PATHS/SETTINGS
# =======================
BASELINE_JSON = "/home/gaash/Wasif/Seg-Zero/reasonseg_eval_results/STS(baseline)/Combined_1016/output_0.json"
EXP5_JSON     = "/home/gaash/Wasif/Seg-Zero/reasonseg_eval_results/exp5/Combined_1016/output_0.json"

OUT_DIR = "./corr_compare_outputs"   # results will be saved here
PASS_THRESH = 0.5                    # IoU threshold for pass rate
REASONING_LEN_CUTOFF = 3             # (Optional) treat len<thres as "very short" reasoning

# =======================


@dataclass
class SampleRow:
    image_id: str
    ann_id: Any
    bbox_iou: float
    think: str
    pred_bboxes_count: int
    reasoning_len_chars: int
    reasoning_len_words: int
    reasoning_fail: bool


def _safe_len_words(text: str) -> int:
    if not text:
        return 0
    return len(text.strip().split())


def _detect_reasoning_fail(item: Dict[str, Any]) -> bool:
    """
    Heuristic:
      - if 'pred_bboxes' present and empty -> fail
      - else if missing, fall back to: bbox_iou == 0 and think is empty/whitespace -> likely fail
    """
    if "pred_bboxes" in item:
        try:
            return len(item.get("pred_bboxes") or []) == 0
        except Exception:
            return True
    # fallback: rely on text + zero IoU
    think = (item.get("think") or "").strip()
    iou = float(item.get("bbox_iou") or 0.0)
    return (len(think) == 0) and (iou == 0.0)


def _pred_bbox_count(item: Dict[str, Any]) -> int:
    try:
        if "pred_bboxes" in item and isinstance(item["pred_bboxes"], list):
            return len(item["pred_bboxes"])
    except Exception:
        pass
    return 0


def _to_rows(items: List[Dict[str, Any]]) -> List[SampleRow]:
    rows = []
    for it in items:
        image_id = str(it.get("image_id", ""))
        ann_id = it.get("ann_id", None)
        iou = float(it.get("bbox_iou", 0.0))
        think = it.get("think", "") or ""
        rows.append(
            SampleRow(
                image_id=image_id,
                ann_id=ann_id,
                bbox_iou=iou,
                think=think,
                pred_bboxes_count=_pred_bbox_count(it),
                reasoning_len_chars=len(think),
                reasoning_len_words=_safe_len_words(think),
                reasoning_fail=_detect_reasoning_fail(it),
            )
        )
    return rows


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stats_and_correlations(rows: List[SampleRow]) -> Dict[str, Any]:
    if not rows:
        return {
            "n": 0, "mean_iou": None, "pass_rate": None,
            "pearson_char": None, "spearman_char": None,
            "pearson_word": None, "spearman_word": None,
            "reasoning_fail_count": 0
        }
    ious = np.array([r.bbox_iou for r in rows], dtype=float)
    char_len = np.array([r.reasoning_len_chars for r in rows], dtype=float)
    word_len = np.array([r.reasoning_len_words for r in rows], dtype=float)
    pass_rate = float(np.mean(ious > PASS_THRESH)) if len(ious) else 0.0

    def _safe_corr(a, b):
        try:
            if np.all(np.isnan(a)) or np.all(np.isnan(b)) or np.std(a) == 0 or np.std(b) == 0:
                return None
            r, p = pearsonr(a, b)
            return {"r": float(r), "p": float(p)}
        except Exception:
            return None

    def _safe_spearman(a, b):
        try:
            if np.all(np.isnan(a)) or np.all(np.isnan(b)):
                return None
            r, p = spearmanr(a, b)
            if math.isnan(r) or math.isnan(p):
                return None
            return {"r": float(r), "p": float(p)}
        except Exception:
            return None

    return {
        "n": int(len(rows)),
        "mean_iou": float(np.mean(ious)) if len(ious) else None,
        "median_iou": float(np.median(ious)) if len(ious) else None,
        "pass_rate": float(pass_rate),
        "reasoning_fail_count": int(sum(r.reasoning_fail for r in rows)),
        "pearson_char": _safe_corr(char_len, ious),
        "spearman_char": _safe_spearman(char_len, ious),
        "pearson_word": _safe_corr(word_len, ious),
        "spearman_word": _safe_spearman(word_len, ious),
    }


def _write_csv(path: str, rows: List[SampleRow]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "image_id", "ann_id", "bbox_iou", "reasoning_fail",
            "reasoning_len_chars", "reasoning_len_words", "pred_bboxes_count"
        ])
        for r in rows:
            w.writerow([
                r.image_id, r.ann_id, f"{r.bbox_iou:.6f}", int(r.reasoning_fail),
                r.reasoning_len_chars, r.reasoning_len_words, r.pred_bboxes_count
            ])


def _scatter(x, y, title, outpath, xlabel, ylabel):
    plt.figure()
    plt.scatter(x, y, s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def _hist(values_a, values_b, labels, title, outpath, bins=30):
    plt.figure()
    plt.hist(values_a, bins=bins, alpha=0.5, label=labels[0])
    plt.hist(values_b, bins=bins, alpha=0.5, label=labels[1])
    plt.title(title)
    plt.xlabel("BBox IoU")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # load
    base_items = _load_json(BASELINE_JSON)
    exp5_items = _load_json(EXP5_JSON)

    base_rows = _to_rows(base_items)
    exp5_rows = _to_rows(exp5_items)

    # stats
    base_stats = _stats_and_correlations(base_rows)
    exp5_stats = _stats_and_correlations(exp5_rows)

    # save CSVs
    _write_csv(os.path.join(OUT_DIR, "baseline_rows.csv"), base_rows)
    _write_csv(os.path.join(OUT_DIR, "exp5_rows.csv"), exp5_rows)

    # scatter plots: reasoning length vs IoU
    b_char = np.array([r.reasoning_len_chars for r in base_rows], float)
    b_iou  = np.array([r.bbox_iou for r in base_rows], float)
    e_char = np.array([r.reasoning_len_chars for r in exp5_rows], float)
    e_iou  = np.array([r.bbox_iou for r in exp5_rows], float)

    _scatter(
        b_char, b_iou,
        "Baseline: Reasoning length (chars) vs BBox IoU",
        os.path.join(OUT_DIR, "scatter_baseline_chars_vs_iou.png"),
        "think length (chars)", "bbox IoU"
    )
    _scatter(
        e_char, e_iou,
        "Exp5: Reasoning length (chars) vs BBox IoU",
        os.path.join(OUT_DIR, "scatter_exp5_chars_vs_iou.png"),
        "think length (chars)", "bbox IoU"
    )

    # overlay (downsample if huge)
    def _downsample(x, y, n=5000):
        if len(x) <= n:
            return x, y
        idx = np.random.choice(len(x), n, replace=False)
        return x[idx], y[idx]

    b_xo, b_yo = _downsample(b_char, b_iou)
    e_xo, e_yo = _downsample(e_char, e_iou)

    plt.figure()
    plt.scatter(b_xo, b_yo, s=8, label="baseline")
    plt.scatter(e_xo, e_yo, s=8, label="exp5")
    plt.title("Reasoning length vs IoU (overlay)")
    plt.xlabel("think length (chars)")
    plt.ylabel("bbox IoU")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "scatter_overlay_chars_vs_iou.png"), dpi=160)
    plt.close()

    # histogram of IoU
    _hist(b_iou, e_iou, ["baseline", "exp5"], "IoU distribution", os.path.join(OUT_DIR, "hist_iou.png"))

    # summary json
    summary = {
        "paths": {
            "baseline": BASELINE_JSON,
            "exp5": EXP5_JSON
        },
        "settings": {
            "pass_thresh": PASS_THRESH,
            "reasoning_len_cutoff": REASONING_LEN_CUTOFF
        },
        "baseline_stats": base_stats,
        "exp5_stats": exp5_stats,
        "deltas": {
            "mean_iou": (exp5_stats["mean_iou"] or 0) - (base_stats["mean_iou"] or 0) if (exp5_stats["mean_iou"] and base_stats["mean_iou"]) else None,
            "pass_rate": (exp5_stats["pass_rate"] or 0) - (base_stats["pass_rate"] or 0) if (exp5_stats["pass_rate"] is not None and base_stats["pass_rate"] is not None) else None,
            "reasoning_fail_count": (base_stats["reasoning_fail_count"] - exp5_stats["reasoning_fail_count"]) if (base_stats["reasoning_fail_count"] is not None and exp5_stats["reasoning_fail_count"] is not None) else None
        }
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # friendly print
    def fmt_corr(c):
        if not c: return "n/a"
        return f"r={c['r']:.3f}, p={c['p']:.3g}"

    print("\n=== CORRELATION SUMMARY ===")
    print(f"Baseline:   n={base_stats['n']}, meanIoU={base_stats['mean_iou']:.4f}  pass={base_stats['pass_rate']*100:.2f}%  fails={base_stats['reasoning_fail_count']}")
    print(f"  Pearson(chars vs IoU):   {fmt_corr(base_stats['pearson_char'])}")
    print(f"  Spearman(chars vs IoU):  {fmt_corr(base_stats['spearman_char'])}")
    print(f"Exp5:       n={exp5_stats['n']}, meanIoU={exp5_stats['mean_iou']:.4f}  pass={exp5_stats['pass_rate']*100:.2f}%  fails={exp5_stats['reasoning_fail_count']}")
    print(f"  Pearson(chars vs IoU):   {fmt_corr(exp5_stats['pearson_char'])}")
    print(f"  Spearman(chars vs IoU):  {fmt_corr(exp5_stats['spearman_word'])}")  # word-based optional

    print(f"\nSaved outputs -> {os.path.abspath(OUT_DIR)}")
    print("Files:")
    for fn in [
        "baseline_rows.csv", "exp5_rows.csv",
        "scatter_baseline_chars_vs_iou.png",
        "scatter_exp5_chars_vs_iou.png",
        "scatter_overlay_chars_vs_iou.png",
        "hist_iou.png",
        "summary.json",
    ]:
        print(" -", fn)


if __name__ == "__main__":
    main()
