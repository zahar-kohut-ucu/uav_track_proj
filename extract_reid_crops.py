import os
import cv2
from pathlib import Path
from tqdm import tqdm

# === Paths ===
DATA_ROOT = "data"
BASE_OUTPUT_DIR = Path("reid_data")
os.makedirs(Path("reid_data"), exist_ok=True)

splits = ["train", "valid"]  # you can add "test" if needed

def extract_crops():
    for split in splits:
        split_path = Path(DATA_ROOT) / split
        for seq_dir in tqdm(list(split_path.iterdir()), desc=f"Processing {split}"):
            img_dir = seq_dir / "img1"
            gt_file = seq_dir / "gt" / "gt.txt"
            if not gt_file.exists():
                continue

            # Read ground truth annotations
            with open(gt_file, "r") as f:
                lines = [line.strip().split(",") for line in f if line.strip()]
                for line in lines:
                    frame_id = int(line[0])
                    track_id = int(line[1])
                    x, y, w, h = map(int, map(float, line[2:6]))

                    img_path = img_dir / f"{frame_id:06d}.png"
                    if not img_path.exists():
                        continue

                    img = cv2.imread(str(img_path))
                    crop = img[y:y+h, x:x+w]
                    if crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue  

                    global_id = f"{seq_dir.name}_{track_id}"
                    out_dir = BASE_OUTPUT_DIR / split / global_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{seq_dir.name}_{frame_id:06d}.jpg"
                    cv2.imwrite(str(out_dir / crop_name), crop)

if __name__ == "__main__":
    extract_crops()
    print(f"✅ ReID crops saved to: {BASE_OUTPUT_DIR}")
