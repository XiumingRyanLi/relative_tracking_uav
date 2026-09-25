# DOPE weights (Audi R8)

Local copies of the DOPE models for the sim (`DOPE_WEIGHTS` in `sim_launch.py`).
The `.pth` files (201 MB each) are **not committed** (`*.pth` is in `.gitignore`;
GitHub rejects files over 100 MB without Git LFS). Only these notes and the
training headers are in git.

| | v3 (in use since 2026-09-25) | v2 (previous) |
|---|---|---|
| File | `audi_droneview_v3_epoch0850.pth` | `audi_droneview_v2_epoch0725.pth` |
| SHA-256 (first 16 hex) | `7601e5e9a97b2e76` | `33e629c1032459a5` |
| Original | `~/Desktop/ryan/Deep_Object_Pose/train/output/weights_droneview_v3/net_epoch_0850.pth` | `.../weights_droneview_v2/net_epoch_0725.pth` |
| Training data | `~/data/AudiDroneView_all_v3`: v1 + v2 + v3_overhead + v3_close (5700 frames) | `~/data/AudiDroneView_all`: v1 + v2 (3300 frames) |
| Viewpoints trained | 3-89 deg above the car, 4-45 m, incl. car partly out of frame | 3-75 deg, 7-45 m, car always fully in frame |
| Training | fine-tuned from v2 epoch 750 to 850 (`audi_droneview_v3_header.txt`) | fine-tuned from v1 epoch 650 to 750 (`audi_droneview_v2_header.txt`) |

Object: `Audi`, cuboid 203.82 x 123.95 x 441.46 cm (w x h x l).

## Held-out test set

`~/data/AudiDroneView_test_v3` (150 frames per view type, never trained on);
full results in `EVAL_RESULTS.txt` there, rerun with `scripts/eval_dope_weights.py`.

| Test set | v2 epoch 725 detected / flips | v3 epoch 850 detected / flips |
|---|---|---|
| overhead (55-89 deg, 6-20 m) | 93% / 1% | 98% / 0% |
| close (20-89 deg, 4-10 m) | 56% / 6% | 65% / 0% |
| standard (3-75 deg, 7-45 m) | 99% / 1% | 99% / 0% |

Median position error ~0.6 m and cuboid keypoint error ~5 px for both. The
remaining close-range misses are frames with at most 3 of the 8 car corners
inside the image (0% for both models, a limit of the cuboid-keypoint method);
with 4-5 corners visible v3 detects 75% (v2 54%), with 6+ corners ~100%.
