# yolo_openvino_stream.py
import os, time, collections, threading, cv2, numpy as np
from ultralytics import YOLO

# --- keep runtime tame on Pi 5 ---
os.environ["AUTOINSTALL"] = "0"
os.environ["YOLOv5_AUTOINSTALL"] = "0"
os.environ["OV_CPU_THREADS_NUM"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
# ----------------------------------

MODEL_PATH = "/home/mitchell/Documents/PhD/circumnavigation_ws/yolo11n_openvino_model"  # folder or .xml
CAM_INDEX  = 0
IMGSZ      = 256       # try 160 if you still see instability
SHOW_WIN   = True      # flip False to test headless
PROC_EVERY = 1         # 1 = every frame, 2 = every 2nd frame (lighter)

print(f"[INFO] Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMGSZ)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMGSZ)
if not cap.isOpened():
    print("[ERROR] Could not open camera."); exit(1)

# Non-blocking capture: keep only the latest frame (drop backlog)
q = collections.deque(maxlen=1)
stop = False

def grabber():
    while not stop:
        ok, frame = cap.read()
        if not ok: continue
        # resize fast to inference size; RGB not required for Ultralytics
        frame = cv2.resize(frame, (IMGSZ, IMGSZ), interpolation=cv2.INTER_NEAREST)
        q.append(frame)

t = threading.Thread(target=grabber, daemon=True)
t.start()

print("[INFO] Press 'q' to quit.")
n, t0 = 0, time.time()
proc_counter = 0
last_annot = None

try:
    while True:
        if not q:
            # no frame yet; small sleep prevents busy-wait
            time.sleep(0.001)
            continue

        frame = q[-1]  # latest frame
        proc_counter += 1

        # Inference (optionally skip frames to lighten load)
        if proc_counter % PROC_EVERY == 0:
            results = model(frame, imgsz=IMGSZ, verbose=False, device="cpu")
            last_annot = results[0].plot()
        # If skipping, still display the last annotated frame
        disp = last_annot if last_annot is not None else frame

        # FPS overlay (display loop)
        n += 1
        fps = n / max(1e-6, (time.time()-t0))
        cv2.putText(disp, f"FPS:{fps:.1f} imgsz:{IMGSZ} every:{PROC_EVERY}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if SHOW_WIN:
            cv2.imshow("YOLO OpenVINO Stream", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # headless mode: small sleep to avoid 100% busy-loop
            time.sleep(0.001)

except KeyboardInterrupt:
    pass
finally:
    stop = True; t.join(timeout=1.0)
    cap.release()
    if SHOW_WIN: cv2.destroyAllWindows()
