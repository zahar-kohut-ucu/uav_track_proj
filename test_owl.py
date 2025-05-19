import os
import uuid
from PIL import Image
from transformers import pipeline
from pathlib import Path
import numpy as np
import glob
from PIL import ImageDraw, ImageFont

SAVE_EXAMPLES = True
EXAMPLES_DIR = "zero_shot_examples"
os.makedirs(EXAMPLES_DIR, exist_ok=True)
MAX_EXAMPLES = 100
saved_count = 0

# --- Utility functions ---

def rescale_to(image, max_size=800):
    width, height = image.size
    if max(width, height) > max_size:
        new_width = max_size if width > height else int(width * max_size / height)
        new_height = max_size if height > width else int(height * max_size / width)
        image = image.resize((new_width, new_height))
    return image

def iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def yolo_to_box(yolo_line, img_w, img_h):
    parts = yolo_line.strip().split()
    xc, yc, w, h = map(float, parts[1:])
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return [x1, y1, x2, y2]

# --- Detection & Evaluation ---

def detect_owlv2(image, labels, save_path=None, conf_thres=0.35):
    image = image.convert("RGB")
    img_w, img_h = image.size

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    predictions = detector(image, candidate_labels=CANDIDATE_LABELS)

    preds = []
    for pred in predictions:
        score = pred["score"]
        if score < conf_thres:
            continue
        box = pred["box"]
        box_xyxy = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
        preds.append((box_xyxy, score))

    gts = [yolo_to_box(line, img_w, img_h) for line in labels if line.strip()]

    matched_gt = set()
    tp = 0
    for pred_box, score in preds:
        for idx, gt_box in enumerate(gts):
            if idx in matched_gt:
                continue
            if iou(pred_box, gt_box) >= 0.5:
                tp += 1
                matched_gt.add(idx)
                break

    fp = len(preds) - tp
    fn = len(gts) - tp

    if save_path:
        # Draw predictions in RED
        for box, score in preds:
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1] - 10), f"{score:.2f}", fill="red", font=font)

        # Draw GT in GREEN
        for gt in gts:
            draw.rectangle(gt, outline="green", width=2)

        image.save(save_path)

    return tp, fp, fn

# --- Main Evaluation ---

CANDIDATE_LABELS = ["vehicle", "car", "military truck", "tank"]
detector = pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection", device="cuda")

IMAGE_DIR = "original_yolo_format/images/test"
LABEL_DIR = "original_yolo_format/labels/test"

all_images = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
TP_total, FP_total, FN_total = 0, 0, 0

for image_path in all_images:
    image = Image.open(image_path)
    base = Path(image_path).stem
    label_path = os.path.join(LABEL_DIR, f"{base}.txt")

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        gt_labels = f.readlines()

    image = rescale_to(image, max_size=712)
    save_path = None
    if SAVE_EXAMPLES and saved_count < MAX_EXAMPLES:
        save_path = os.path.join(EXAMPLES_DIR, f"{base}.jpg")
        saved_count += 1

    tp, fp, fn = detect_owlv2(image, gt_labels, save_path)

    TP_total += tp
    FP_total += fp
    FN_total += fn

# --- Metrics ---
precision = TP_total / (TP_total + FP_total + 1e-9)
recall = TP_total / (TP_total + FN_total + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("\n📊 Zero-Shot OWL-V2 Evaluation:")
print(f"True Positives : {TP_total}")
print(f"False Positives: {FP_total}")
print(f"False Negatives: {FN_total}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
