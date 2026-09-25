#!/usr/bin/env python3
"""Compare DOPE weights on rendered test frames (BlenderProc datagen output).

Usage:
    python3 scripts/eval_dope_weights.py --weights a.pth b.pth --data DIR [DIR ...] [--max 300]

For every <name>.png/<name>.json pair it runs the same DOPE inference as
dope_detector (vendored circumnavigation_controller.dope, same thresholds,
full 500x500 frame with the json's camera intrinsics) and reports per weights
and per data dir:
  detected   fraction of frames with a pose
  pos_err    camera-frame position error (m), detected frames
  kp_err     mean pixel error of the 9 reprojected cuboid points vs the
             ground-truth projected_cuboid (large = wrong orientation, e.g. a
             front/back flip)
  flips      fraction of detections with kp_err > 25 px
Ground truth: json 'location' is in the Blender camera frame (x right, y up,
z back, metres); DOPE returns OpenCV camera frame (x right, y down, z forward)
in the cuboid's units (cm).
"""
import argparse
import glob
import json
import os
import sys
from collections import OrderedDict
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src",
                                "circumnavigation_controller"))
from circumnavigation_controller.dope.detector import DopeNetwork, ObjectDetector  # noqa: E402
from circumnavigation_controller.dope.cuboid import Cuboid3d  # noqa: E402
from circumnavigation_controller.dope.cuboid_pnp_solver import CuboidPNPSolver  # noqa: E402

CUBOID_CM = [203.82, 123.95, 441.46]      # Audi, as in sim_launch.py
CONFIG = SimpleNamespace(mask_edges=1, mask_faces=1, vertex=1, threshold=0.5, softmax=1000,
                         thresh_angle=0.5, thresh_map=0.01, sigma=3, thresh_points=0.1)


def load_net(path):
    sd = torch.load(path, map_location="cpu")
    if any(k.startswith("module.") for k in sd):
        sd = OrderedDict((k[len("module."):] if k.startswith("module.") else k, v) for k, v in sd.items())
    net = DopeNetwork()
    net.load_state_dict(sd)
    return net.cuda().eval()


def frames(data_dir, max_n):
    js = sorted(glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True))
    return js[:max_n] if max_n else js


def evaluate(net, json_paths):
    det, pos, kp = 0, [], []
    solver = CuboidPNPSolver("Audi", cuboid3d=Cuboid3d(CUBOID_CM))
    for jp in json_paths:
        d = json.load(open(jp))
        gt = next(o for o in d["objects"] if o.get("class", "").lower() == "audi")
        K = d["camera_data"]["intrinsics"]
        solver.set_camera_intrinsic_matrix(np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1.0]]))
        img = cv2.cvtColor(cv2.imread(jp[:-5] + ".png"), cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            res, _ = ObjectDetector.detect_object_in_image(net, solver, img, CONFIG)
        res = [r for r in res if r.get("location") is not None]
        if not res:
            continue
        det += 1
        r = max(res, key=lambda r: r.get("confidence", 0) or 0)
        gl = np.array(gt["location"], dtype=float)
        gt_cv = np.array([gl[0], -gl[1], -gl[2]])
        pos.append(np.linalg.norm(np.array(r["location"], dtype=float) / 100.0 - gt_cv))
        pp = np.array([p if p is not None else [np.nan, np.nan] for p in r["projected_points"]], dtype=float)
        gp = np.array(gt["projected_cuboid"], dtype=float)
        n = min(len(pp), len(gp))
        kp.append(np.nanmean(np.linalg.norm(pp[:n] - gp[:n], axis=1)))
    n = len(json_paths)
    kp = np.array(kp)
    return dict(n=n, detected=det / n if n else float("nan"),
                pos_err=float(np.median(pos)) if pos else float("nan"),
                kp_err=float(np.median(kp)) if len(kp) else float("nan"),
                flips=float((kp > 25).mean()) if len(kp) else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--max", type=int, default=0, help="max frames per data dir (0 = all)")
    a = ap.parse_args()
    sets = {os.path.basename(os.path.normpath(p)): frames(p, a.max) for p in a.data}
    print(f"{'weights':42s} {'test set':30s} {'n':>5s} {'detected':>9s} {'pos_err m':>10s} {'kp_err px':>10s} {'flips':>6s}")
    for w in a.weights:
        net = load_net(w)
        for name, js in sets.items():
            r = evaluate(net, js)
            print(f"{os.path.basename(os.path.dirname(w)) + '/' + os.path.basename(w):42s} {name:30s} {r['n']:5d} "
                  f"{100 * r['detected']:8.0f}% {r['pos_err']:10.2f} {r['kp_err']:10.1f} {100 * r['flips']:5.0f}%")
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
