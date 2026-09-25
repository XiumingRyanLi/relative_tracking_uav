# DOPE weights (Audi R8)

`audi_droneview_v2_epoch0725.pth` is the DOPE model the simulation used up to
2026-09-25 (`DOPE_WEIGHTS` in `sim_launch.py` points at the original in
`~/Desktop/ryan/Deep_Object_Pose/train/output/weights_droneview_v2/net_epoch_0725.pth`).
It is kept here as the baseline to compare retrained models against.

The `.pth` file (201 MB) is **not committed** (`*.pth` is in `.gitignore`;
GitHub rejects files over 100 MB without Git LFS). Only this README and the
training header are in git.

| | |
|---|---|
| File | `audi_droneview_v2_epoch0725.pth` |
| SHA-256 (first 16 hex) | `33e629c1032459a5` |
| Object | `Audi`, cuboid 203.82 x 123.95 x 441.46 cm (w x h x l) |
| Training data | `~/data/AudiDroneView_all` = v1 (300 frames) + v2 (3000 frames) |
| Viewpoints trained | camera 3-75 deg above the car, 7-45 m, car always fully in frame |
| Training | fine-tuned from droneview_v1 epoch 650 to 750, image size 448, lr 1e-4 (see `audi_droneview_v2_header.txt`) |

Known gaps (runs 20260925_131648 / _131943): detections are lost when the
car is partly outside the image, i.e. closer than ~9 m or nearly overhead
(75-90 deg was never trained). See "DOPE model and training data" in the
top-level README.
