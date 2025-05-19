import os
import shutil
import cv2
from pathlib import Path
import motmetrics as mm
from tqdm import tqdm
import pandas as pd

# === PATHS ===
runs_dir = Path("raw_ds_results")
output_root = Path("deepsort_results")
test_data_root = Path("data/test")
metrics_csv_path = output_root / "metrics_summary.csv"

# === MOT-format conversion ===
def convert_label_to_mot(label_file, frame_img_path, frame_id, out_txt_path):
    if not label_file.exists():
        return  # no predictions to write
    with open(label_file, "r") as f, open(out_txt_path, "a") as out_f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            _, xc, yc, w, h, track_id = parts
            xc, yc, w, h = map(float, [xc, yc, w, h])
            track_id = int(track_id)

            img = cv2.imread(str(frame_img_path))
            height, width = img.shape[:2]
            x = (xc - w / 2) * width
            y = (yc - h / 2) * height
            w = w * width
            h = h * height
            out_f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1\n")

# === MOT Evaluation ===
def evaluate(seq_name, result_txt, gt_txt):
    acc = mm.MOTAccumulator(auto_id=True)
    mh = mm.metrics.create()
    gt_frame, pred_frame = {}, {}

    with open(gt_txt) as f:
        for line in f:
            fid, tid, x, y, w, h, _, _, _ = map(float, line.strip().split(','))
            gt_frame.setdefault(int(fid), []).append((int(tid), (x, y, w, h)))

    with open(result_txt) as f:
        for line in f:
            fid, tid, x, y, w, h, _, _, _ = map(float, line.strip().split(','))
            pred_frame.setdefault(int(fid), []).append((int(tid), (x, y, w, h)))

    frames = sorted(set(gt_frame.keys()) | set(pred_frame.keys()))
    for fid in frames:
        gt_ids, gt_boxes = zip(*gt_frame.get(fid, [])) if fid in gt_frame else ([], [])
        pr_ids, pr_boxes = zip(*pred_frame.get(fid, [])) if fid in pred_frame else ([], [])
        dist = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5) if gt_boxes and pr_boxes else []
        acc.update(gt_ids, pr_ids, dist)

    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=seq_name)
    print(f"\n📊 Evaluation for {seq_name}:\n{summary.to_string()}\n")
    return summary

# === MAIN LOOP ===
all_summaries = []

for seq_dir in sorted(runs_dir.glob("*")):
    seq_name = seq_dir.name
    label_dir = seq_dir / "labels"
    out_dir = output_root / seq_name
    out_img_dir = out_dir / "img1"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_path = out_dir / "results.txt"

    print(f"🔄 Processing sequence: {seq_name}")
    if out_label_path.exists():
        out_label_path.unlink()

    all_frames = sorted(seq_dir.glob("*.jpg"))
    for img_path in all_frames:
        frame_id = int(img_path.stem)
        lbl_path = label_dir / f"{img_path.stem}.txt"

        # Copy image
        shutil.copy2(img_path, out_img_dir / img_path.name)

        # Convert label
        convert_label_to_mot(lbl_path, img_path, frame_id, out_label_path)

    # Copy video if available
    print(f"Saving tracking video for {seq_name}...")
    frame_files = sorted(out_img_dir.glob("*.jpg"))
    if frame_files:
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]
        video_path = out_dir / "video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid = cv2.VideoWriter(str(video_path), fourcc, 15.0, (width, height))

        for frame in frame_files:
            img = cv2.imread(str(frame))
            out_vid.write(img)

        out_vid.release()
        print(f"Saved video to {video_path}")
    else:
        print("No frames found to render video.")

    # Evaluate if GT exists
    gt_file = test_data_root / seq_name / "gt/gt.txt"
    if gt_file.exists():
        summary = evaluate(seq_name, out_label_path, gt_file)
        all_summaries.append(summary)
    else:
        print(f"No ground truth for {seq_name}, skipping evaluation.")

# === Save all metrics to CSV ===
if all_summaries:
    full_df = pd.concat(all_summaries, axis=0)
    full_df.to_csv(metrics_csv_path)
    print(f"\nSaved summary metrics to: {metrics_csv_path}")
else:
    print("No metrics saved — no gt.txt files found.")

print("All sequences processed.")
