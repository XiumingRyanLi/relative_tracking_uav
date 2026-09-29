#!/usr/bin/env python3
"""Offline test: does the target estimator follow a car that stops and reverses?

The car's NOSE heading is what DOPE measures; reversing means moving against it.
Scenarios (DOPE-like noise, nose-to-tail flips injected on 3 % of frames):
  straight     forward 10 m/s, brake to 0, reverse at 4 m/s for 6 s, stop, forward
  three_point  reverse 3 m/s while turning 25 deg/s, stop, forward while turning back
  reverse_gap  as straight, with a 2.5 s detection gap in the middle of the reverse
  forward      forward only with turns (regression: flips must still be caught)
  flipped_start          forward at 5 m/s, the FIRST detection flipped (too slow for
                         the reverse-speed guard; the start-up vote must catch it)
  flipped_start_reverse  reversing at 3 m/s from the start, first detection flipped

Pass criteria per scenario: heading within 20 deg of the nose on >= 95 % of
frames, speed with the right sign while |v| > 1 m/s on >= 90 %, >= 95 % of the
injected flips turned back, position error median < 1 m.

    python3 scripts/test_reverse.py
"""
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from benchmark_target_filter import make_estimator, pose_matrix, wrap   # noqa: E402

RATE = 15.0


def profile(name):
    """[(duration s, speed start, speed end, turn rate deg/s)] segments."""
    if name in ("straight", "reverse_gap"):
        return [(5, 10, 10, 0), (2.5, 10, 0, 0), (1, 0, -4, 0), (6, -4, -4, 0), (1, -4, 0, 0),
                (1, 0, 4, 0), (4, 4, 10, 0)]
    if name == "three_point":
        return [(4, 6, 6, 0), (1.5, 6, 0, 0), (1, 0, -3, 25), (4, -3, -3, 25), (1, -3, 0, 25),
                (1, 0, 3, -20), (4, 3, 6, -20)]
    if name == "flipped_start":
        return [(12, 5, 5, 0)]
    if name == "flipped_start_reverse":
        return [(8, -3, -3, 10), (1, -3, 0, 0), (3, 0, 5, 0)]
    if name == "forward":
        return [(4, 8, 12, 0), (3, 12, 8, 20), (3, 8, 12, -15), (3, 12, 12, 0), (3, 12, 6, 30),
                (4, 6, 12, 0)]
    raise ValueError(name)


def truth(name, dt=0.001):
    """Integrate the segments: arrays t, x, y, nose heading, speed."""
    t, x, y, psi, v = [0.0], [0.0], [0.0], [0.0], [profile(name)[0][1]]
    for dur, v0, v1, w in profile(name):
        n = int(round(dur / dt))
        for k in range(n):
            vk = v0 + (v1 - v0) * (k + 0.5) / n
            x.append(x[-1] + vk * math.cos(psi[-1]) * dt)
            y.append(y[-1] + vk * math.sin(psi[-1]) * dt)
            psi.append(psi[-1] + math.radians(w) * dt)
            v.append(vk)
            t.append(t[-1] + dt)
    return tuple(np.array(a) for a in (t, x, y, psi, v))


def run(name, seed=0, range_m=20.0):
    t, x, y, psi, v = truth(name)
    rs = np.random.default_rng(seed)
    est = make_estimator("ctra")
    gap = (12.0, 14.5) if name == "reverse_gap" else None
    rows, injected, caught_before = [], 0, None
    bias = np.zeros(2)
    for tk in np.arange(0.0, t[-1], 1.0 / RATE):
        i = int(np.searchsorted(t, tk))
        if gap and gap[0] <= tk < gap[1]:
            continue
        a = math.exp(-1.0 / RATE / 1.5)
        bias = a * bias + math.sqrt(1 - a * a) * 0.035 * range_m * rs.standard_normal(2)
        mx = x[i] + bias[0] + 0.035 * range_m * rs.standard_normal()
        my = y[i] + bias[1] + 0.035 * range_m * rs.standard_normal()
        mpsi = psi[i] + math.radians(3.0) * rs.standard_normal()
        flip = rs.random() < 0.03 or (name.startswith("flipped_start") and not rows and injected == 0)
        if flip:
            mpsi += math.pi
            injected += 1
        flips_before = est.heading_flips
        est.update(pose_matrix(mx, my, wrap(mpsi)), range_m, tk)
        caught = est.heading_flips > flips_before
        if tk < 1.0:
            rows.append(None)
            continue
        rows.append((tk, abs(math.degrees(wrap(est.heading - psi[i]))), math.hypot(est.x - x[i], est.y - y[i]),
                     est.ctra.speed, v[i], flip, caught))
    r = np.array([row for row in rows if row is not None], dtype=float)
    hdg_ok = np.mean(r[:, 1] <= 20.0)
    moving = np.abs(r[:, 4]) > 1.0
    sign_ok = np.mean(np.sign(r[moving, 3]) == np.sign(r[moving, 4]))
    rev = r[:, 4] < -1.0
    flip_rows = r[:, 5] == 1
    caught = np.mean(r[flip_rows, 6] == 1) if flip_rows.any() else 1.0
    false_flips = int(np.sum((r[:, 6] == 1) & (r[:, 5] == 0)))
    return dict(hdg_ok=hdg_ok, hdg_med=np.median(r[:, 1]), hdg_max=r[:, 1].max(), pos_med=np.median(r[:, 2]),
                sign_ok=sign_ok, caught=caught, injected=int(flip_rows.sum()), false_flips=false_flips,
                rev_speed=np.median(r[rev, 3]) if rev.any() else float("nan"),
                rev_true=np.median(r[rev, 4]) if rev.any() else float("nan"),
                state_flips=est.ctra.state_flips)


def main():
    all_ok = True
    for name in ("straight", "three_point", "reverse_gap", "forward", "flipped_start", "flipped_start_reverse"):
        agg = [run(name, seed) for seed in range(5)]
        m = {k: np.mean([a[k] for a in agg]) for k in agg[0]}
        ok = m["hdg_ok"] >= 0.95 and m["sign_ok"] >= 0.90 and m["caught"] >= 0.95 and m["pos_med"] < 1.0
        all_ok &= ok
        print(f"{'PASS' if ok else 'FAIL'} {name:21} heading within 20 deg {m['hdg_ok'] * 100:5.1f} % "
              f"(median {m['hdg_med']:4.1f}, worst {m['hdg_max']:5.1f} deg) | speed sign right "
              f"{m['sign_ok'] * 100:5.1f} % | reversing: est {m['rev_speed']:+5.2f} vs true {m['rev_true']:+5.2f} m/s"
              f" | flips caught {m['caught'] * 100:5.1f} % of {m['injected']:.0f}, false {m['false_flips']:.1f}"
              f" | pos err median {m['pos_med']:4.2f} m | state flips {m['state_flips']:.1f}")
    print("ALL PASS" if all_ok else "SOME FAILED")


if __name__ == "__main__":
    main()
