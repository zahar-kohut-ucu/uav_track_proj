from ultralytics import YOLO
import os



model = YOLO("/home/zkohu/Desktop/CV2025/uav_track_proj/runs/detect/ft_visdrone_6/weights/best.pt")

metrics = model.val(
    data="inf_mil.yaml",
    split='test',
    save=True,         
    save_txt=True,   
    conf=0.25,          # confidence threshold       
    imgsz=640       
)
