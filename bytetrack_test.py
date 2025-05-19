import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import motmetrics as mm
import pandas as pd

def load_mot_file(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            obj_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            bbox = [x, y, x + w, y + h]  # convert to x1, y1, x2, y2
            data.append((frame, obj_id, bbox))
    return data

def evaluate_mot(gt_path, pred_path, seq_name):
    acc = mm.MOTAccumulator(auto_id=True)
    gt = load_mot_file(gt_path)
    pred = load_mot_file(pred_path)

    # Group by frame
    gt_by_frame = {}
    pred_by_frame = {}
    for frame, obj_id, box in gt:
        gt_by_frame.setdefault(frame, []).append((obj_id, box))
    for frame, obj_id, box in pred:
        pred_by_frame.setdefault(frame, []).append((obj_id, box))

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    for frame in all_frames:
        gt_objs = gt_by_frame.get(frame, [])
        pred_objs = pred_by_frame.get(frame, [])

        gt_ids = [o[0] for o in gt_objs]
        gt_boxes = [o[1] for o in gt_objs]

        pred_ids = [o[0] for o in pred_objs]
        pred_boxes = [o[1] for o in pred_objs]

        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=seq_name)
    return summary

# CONFIG
MODEL_PATH = "ftvsd6.pt"
ROOT_DIR = "data/test"
SAVE_ROOT = "bytetrack_results2"
os.makedirs(SAVE_ROOT, exist_ok=True)

# Tracker parameters
tracker_args = SimpleNamespace(
    track_thresh=0.25,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=30,
    mot20=False
)

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Prepare color palette
np.random.seed(42)
colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)

def run_tracking(seq_path):
    img_dir = seq_path / "img1"
    gt_path = seq_path / "gt" / "gt.txt"
    out_dir = Path(SAVE_ROOT) / seq_path.name
    out_img_dir = out_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir / "output.mp4"
    out_label_path = out_dir / "results.txt"

    img_paths = sorted(img_dir.glob("*.png"))
    tracker = BYTETracker(tracker_args, frame_rate=tracker_args.frame_rate)

    frame_size = None
    results = []

    for frame_id, img_path in enumerate(tqdm(img_paths, desc=f"Tracking {seq_path.name}"), 1):
        frame = cv2.imread(str(img_path))
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])  # width, height

        img_info = (frame.shape[0], frame.shape[1])
        img_size = (frame.shape[0], frame.shape[1])

        # Inference
        preds = model.predict(source=frame, conf=0.25, iou=0.5, verbose=False)[0].boxes
        if preds is None or preds.shape[0] == 0:
            cv2.imwrite(str(out_img_dir / img_path.name), frame)
            continue

        boxes = preds.xyxy
        scores = preds.conf.unsqueeze(1)
        dets = torch.cat([boxes, scores], dim=1).cpu().numpy()

        # Track
        online_targets = tracker.update(dets, img_info, img_size)

        # Draw and save
        for t in online_targets:
            if not t.is_activated:
                continue
            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.tlbr)
            color = colors[tid % len(colors)].tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{tid}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            results.append(f"{frame_id},{tid},{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n")

        cv2.imwrite(str(out_img_dir / img_path.name), frame)

    # Save video
    out = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                          tracker_args.frame_rate, frame_size)
    for img_path in sorted(out_img_dir.glob("*.png")):
        frame = cv2.imread(str(img_path))
        out.write(frame)
    out.release()

    # Save tracking labels
    with open(out_label_path, "w") as f:
        f.writelines(results)

    return out_label_path, gt_path


# === Run everything ===
all_seq_dirs = [p for p in Path(ROOT_DIR).iterdir() if p.is_dir()]
all_summaries = []

for seq in all_seq_dirs:
    result_file, gt_file = run_tracking(seq)
    print(f"\nEvaluating {seq.name}...")
    summary = evaluate_mot(gt_file, result_file, seq.name)
    print(summary.to_string(float_format="%.3f"))
    all_summaries.append(summary)

# Optional: save full evaluation to CSV
final_summary = pd.concat(all_summaries)
final_summary.to_csv("tracking_summary.csv")
print("\nSaved full evaluation summary to tracking_summary.csv")
