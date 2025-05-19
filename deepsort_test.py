# full_tracking.py
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
import torchreid
import torch.nn.functional as F

# === Config ===
YOLO_MODEL_PATH = "ftvsd6.pt"  # Your YOLOv8 detector
REID_MODEL_PATH = "log/reid_model/model/model.pth.tar-25"  # Your trained Torchreid model
SEQUENCE_ROOT = "data/test"  # Folder with sequences
OUTPUT_ROOT = "deepsort_results"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# === Load ReID Model ===
class ReIDEmbedder:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = torchreid.models.build_model('osnet_x1_0', num_classes=783, pretrained=False)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract(self, image):
        image = self.preprocess(image).to(self.device).unsqueeze(0)
        feat = self.model(image)
        return F.normalize(feat, dim=1).squeeze(0).cpu().numpy()

# === Main Tracker Class ===
def run_tracking(sequence_path):
    reid = ReIDEmbedder(REID_MODEL_PATH, DEVICE)
    model = YOLO(YOLO_MODEL_PATH)
    deepsort = DeepSort(model_type='osnet_x1_0', use_cuda=(DEVICE == 'cuda'))

    img_dir = Path(sequence_path) / "img1"
    out_img_dir = Path(OUTPUT_ROOT) / Path(sequence_path).name / "images"
    out_vid_path = Path(OUTPUT_ROOT) / Path(sequence_path).name / "output.mp4"
    out_txt_path = Path(OUTPUT_ROOT) / Path(sequence_path).name / "results.txt"
    os.makedirs(out_img_dir, exist_ok=True)

    image_paths = sorted(img_dir.glob("*.png"))
    writer = None
    results = []

    for frame_id, img_path in enumerate(tqdm(image_paths), 1):
        frame = cv2.imread(str(img_path))
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_vid_path), cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))

        # Detection
        dets = model.predict(source=frame, conf=0.4, iou=0.5, verbose=False)[0].boxes
        if dets.shape[0] == 0:
            writer.write(frame)
            continue

        bboxes = dets.xyxy.cpu().numpy()[:, :4]
        confs = dets.conf.cpu().numpy()

        # Extract ReID features
        features = []
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                features.append(np.random.rand(512))  # fallback
                continue
            features.append(reid.extract(crop))
        features = np.stack(features)

        # Update tracker
        outputs = deepsort.update_tracks(bboxes, confs, features, frame)

        # Draw results and save
        for track in outputs:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            results.append(f"{frame_id},{tid},{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n")

        writer.write(frame)
        cv2.imwrite(str(out_img_dir / img_path.name), frame)

    # Save labels
    with open(out_txt_path, 'w') as f:
        f.writelines(results)

    writer.release()
    print(f"✅ Saved to {out_vid_path}")

# === Run on all test sequences ===
for seq in sorted(Path(SEQUENCE_ROOT).iterdir()):
    if not (seq / 'img1').exists():
        continue
    print(f"\n🚀 Tracking {seq.name}...")
    run_tracking(str(seq))
