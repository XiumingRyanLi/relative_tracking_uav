# save as ov_one_shot.py
import os, cv2, numpy as np
from ultralytics import YOLO
os.environ["AUTOINSTALL"]="0"; os.environ["YOLOv5_AUTOINSTALL"]="0"
os.environ["OV_CPU_THREADS_NUM"]="2"; os.environ["OMP_NUM_THREADS"]="2"
model = YOLO("/path/to/yolo11n_openvino_model")  # directory or .xml
img = np.zeros((256,256,3),dtype=np.uint8)
res = model(img, imgsz=256, verbose=False, device="cpu")
print("OK", res[0].boxes if res else None)
