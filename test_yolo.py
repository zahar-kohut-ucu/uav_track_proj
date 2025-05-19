from ultralytics import YOLO

model = YOLO("/home/zkohu/Desktop/CV2025/uav_track_proj/runs/detect/train_comb_7/weights/best.pt")
metrics = model.val(data="/home/zkohu/Desktop/CV2025/uav_track_proj/inf_mil.yaml", split="test")
