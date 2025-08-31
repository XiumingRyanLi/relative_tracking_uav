from ultralytics import YOLO

# Export YOLO11n model to NCNN format with 320x320 resolution
model = YOLO('yolo11n.pt')
model.export(format='ncnn', imgsz=320)  # creates 'yolo11n_ncnn_model' directory
print("Export complete. Copy the 'yolo11n_ncnn_model' directory to your Raspberry Pi.")
