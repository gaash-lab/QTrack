import matplotlib.pyplot as plt
from collections import defaultdict

# ----------------------------
# Read MOT txt and extract bbox centers
# ----------------------------
def read_mot(path):
    tracks = defaultdict(list)
    with open(path) as f:
        for line in f:
            frame, tid, x, y, w, h, *_ = map(float, line.strip().split(","))
            cx = x + w / 2
            cy = y + h / 2
            tracks[int(tid)].append((int(frame), cx, cy))
    for k in tracks:
        tracks[k] = sorted(tracks[k], key=lambda x: x[0])
    return tracks


# ----------------------------
# Plot frame-wise center (ONLY ONE POINT PER FRAME)
# Using cy as the center coordinate
# ----------------------------
def plot_tracks_framewise(gt_tracks, pred_tracks):
    plt.figure(figsize=(12, 8))

    # ---- Ground Truth (Green) ----
    for tid, pts in gt_tracks.items():
        frames = [p[0] for p in pts]
        centers = [p[2] for p in pts]   # <-- ONLY cy

        # Big faint dots
        plt.scatter(frames, centers, s=2500, color="green", alpha=0.25)

        # Small bold dots
        plt.scatter(frames, centers, s=50, color="green", edgecolor="black",
                    label="GT" if tid == list(gt_tracks.keys())[0] else "")

    # ---- Predictions (Red) ----
    for tid, pts in pred_tracks.items():
        frames = [p[0] for p in pts]
        centers = [p[2] for p in pts]   # <-- ONLY cy

        # Big faint dots
        plt.scatter(frames, centers, s=2500, color="red", alpha=0.25)

        # Small bold dots
        plt.scatter(frames, centers, s=50, color="red", edgecolor="black",
                    label="Prediction" if tid == list(pred_tracks.keys())[0] else "")

    plt.xlabel("Frame Number")
    plt.ylabel("BBox Center Y Coordinate (cy)")
    plt.title("Frame-wise BBox Center Position\nBig dot: low opacity | Small dot: true center")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_framewise_single_center.png", dpi=300)
    plt.show()


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    gt_file = "/home/gaash/Tawheed/Reasoning/Ealuation_logs/evaluation_qwen_2.5_prev/dancetrack0002_000051_gt.txt"
    pred_file = "/home/gaash/Tawheed/Reasoning/Ealuation_logs/evaluation_qwen_2.5_prev/dancetrack0002_000051_pred.txt"

    gt_tracks = read_mot(gt_file)
    pred_tracks = read_mot(pred_file)

    plot_tracks_framewise(gt_tracks, pred_tracks)
