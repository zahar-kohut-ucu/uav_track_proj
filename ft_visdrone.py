from ultralytics import YOLO

model = YOLO("visdrone_pretrain.pt")
model.train(data="/home/zkohu/Desktop/CV2025/uav_track_proj/mil_vehicles.yaml", epochs=50, imgsz=640, batch=16)
# model = YOLO("runs/detect/train/weights/last.pt")
# model.train(data="data/mil_vehicles.yaml", epochs=30, batch=32)