import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# === CONFIGURATION ===
YOLO_MODEL = "tracking/weights/yolov8_vd.pt"
REID_MODEL = "tracking/weights/osnet_x1_0_uavtrack.pt"
TRACKING_METHOD = "deepocsort"

TEST_ROOT = Path("../data/test")
IMG_SUBDIR = "img1"

# Find all sequences under data/test/*/img1
sequences = sorted(TEST_ROOT.glob("*/" + IMG_SUBDIR))
if not sequences:
    print("No sequences found under data/test/*/img1")
    exit(1)

# Loop and run tracking
for seq in tqdm(sequences, desc="Tracking sequences"):
    seq_path = str(seq.resolve())
    seq_name = seq.parent.name  # e.g., 101_1_1135531

    print(f"\nRunning DeepOCSORT on: {seq_name}")

    cmd = f"""
    python tracking/track.py \
        --source "{seq_path}" \
        --yolo-model "{YOLO_MODEL}" \
        --reid-model "{REID_MODEL}" \
        --tracking-method "{TRACKING_METHOD}" \
        --name "{seq_name}" \
        --save --save-txt
    """
    subprocess.run(cmd, shell=True, check=True)

    print(f"Tracking for {seq_name} complete.\n")

print("All sequences processed.")
