import os
import shutil
import uuid
from configparser import ConfigParser
from tqdm import tqdm
from PIL import Image
import cv2

def parse_seqinfo(seqinfo_path):
    config = ConfigParser()
    config.read(seqinfo_path)
    width = int(config['Sequence']['imWidth'])
    height = int(config['Sequence']['imHeight'])
    return width, height

def convert_bbox_to_yolo(x, y, w, h, img_w, img_h):
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h

def compute_avg_box_area(data_root):
    total_area = 0
    count = 0
    for split in ['train', 'valid']:
        split_dir = os.path.join(data_root, split)
        for sample in os.listdir(split_dir):
            sample_path = os.path.join(split_dir, sample)
            gt_path = os.path.join(sample_path, 'gt', 'gt.txt')
            seqinfo_path = os.path.join(sample_path, 'seqinfo.ini')
            if not os.path.exists(gt_path) or not os.path.exists(seqinfo_path):
                continue
            img_w, img_h = parse_seqinfo(seqinfo_path)
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    x, y, w, h = map(float, parts[2:6])
                    area = (w * h) / (img_w * img_h)
                    total_area += area
                    count += 1
    return total_area / count if count else 0

def process_original_dataset(src_root, out_base_dir, avg_area):
    counter = 0
    for split in ['train', 'valid', 'test']:
        out_img_dir = os.path.join(out_base_dir, 'images', split)
        out_lbl_dir = os.path.join(out_base_dir, 'labels', split)
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)
        split_dir = os.path.join(src_root, split)
        for sample in tqdm(os.listdir(split_dir), desc=f"Processing Original {split}"):
            sample_path = os.path.join(split_dir, sample)
            img_dir = os.path.join(sample_path, 'img1')
            gt_path = os.path.join(sample_path, 'gt', 'gt.txt')
            seqinfo_path = os.path.join(sample_path, 'seqinfo.ini')
            if not os.path.exists(gt_path) or not os.path.exists(seqinfo_path):
                continue
            img_w, img_h = parse_seqinfo(seqinfo_path)

            annotations = {}
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    frame_id = int(parts[0])
                    x, y, w, h = map(float, parts[2:6])
                    box_area = (w * h) / (img_w * img_h)
                    yolo_box = convert_bbox_to_yolo(x, y, w, h, img_w, img_h)
                    if frame_id not in annotations:
                        annotations[frame_id] = []
                    annotations[frame_id].append((box_area, yolo_box))

            for frame_id in sorted(annotations.keys()):
                if frame_id % 2 != 0:
                    continue
                boxes = annotations[frame_id]
                uid = str(uuid.uuid4())
                img_name = f'{frame_id:06d}.png'
                src_img_path = os.path.join(img_dir, img_name)
                dst_img_path = os.path.join(out_img_dir, f'{uid}.png')
                dst_lbl_path = os.path.join(out_lbl_dir, f'{uid}.txt')

                if not os.path.exists(src_img_path):
                    continue
                    
                shutil.copy2(src_img_path, dst_img_path)
                counter += 1
                with open(dst_lbl_path, 'w') as f:
                    for _, box in boxes:
                        f.write(f"0 {' '.join(f'{v:.6f}' for v in box)}\n")
    return counter

def process_yolo_dataset(yolo_root, split_name, out_base_dir, avg_area):
    counter = 0
    in_img_dir = os.path.join(yolo_root, split_name, 'images')
    in_lbl_dir = os.path.join(yolo_root, split_name, 'labels')

    out_img_dir = os.path.join(out_base_dir, 'images', split_name)
    out_lbl_dir = os.path.join(out_base_dir, 'labels', split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    for file in tqdm(os.listdir(in_img_dir), desc=f"Processing {yolo_root}/{split_name}"):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        base = os.path.splitext(file)[0]
        img_path = os.path.join(in_img_dir, file)
        lbl_path = os.path.join(in_lbl_dir, f'{base}.txt')
        if not os.path.exists(lbl_path):
            continue

        # Area filtering
        keep = True
        lines = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, w, h = map(float, parts)
                if w * h > 4 * avg_area:
                    keep = False
                    break
                if 'data4' in yolo_root:
                    cls_id = '0'
                lines.append(f"0 {xc} {yc} {w} {h}")
        if not keep:
            continue

        uid = str(uuid.uuid4())
        out_img_path = os.path.join(out_img_dir, f'{uid}.png')
        out_lbl_path = os.path.join(out_lbl_dir, f'{uid}.txt')

        if 'data2' in yolo_root:
            # Apply median filter
            image = cv2.imread(img_path)
            if image is None:
                continue
            filtered = cv2.medianBlur(image, 3)
            cv2.imwrite(out_img_path, filtered)
        else:
            # Just copy as-is
            shutil.copy2(img_path, out_img_path)

        with open(out_lbl_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        counter += 1
    return counter


# === MAIN ===

original_dataset = 'data'
combined_dir = 'combined_dataset_4'
data2_dir = 'data2'
data3_dir = 'data3'
data4_dir = 'data4'


print("📏 Calculating average box area from original dataset...")
avg_box_area = compute_avg_box_area(original_dataset)
print(f"Average normalized box area: {avg_box_area:.6f}")

# Process original dataset
inner_counter = process_original_dataset(original_dataset, combined_dir, avg_box_area)
print(f"{inner_counter} images from {original_dataset}")

# Process data2/data3 (train and val only), with filtering
for dset in [data2_dir, data3_dir, data4_dir]:
    for split in ['train', 'valid', 'test']:
        outer_counter = process_yolo_dataset(dset, split, combined_dir, avg_box_area)
        print(f"{outer_counter} images from {dset}")

print("Combined dataset ready at: combined_dataset/")
