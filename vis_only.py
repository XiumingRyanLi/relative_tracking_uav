# vis_only.py
import cv2, time

CAM_INDEX = 0
W, H, FPS = 320, 320, 30

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

if not cap.isOpened():
    print("[ERROR] Could not open camera."); exit(1)

t0 = time.time(); n = 0
print("[INFO] Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok: print("[WARN] read fail"); break
    n += 1
    fps = n / max(1e-6, (time.time()-t0))
    cv2.putText(frame, f"VIS FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("VIS ONLY", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()
