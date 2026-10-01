#!/usr/bin/env python3
"""Offline feasibility check of the experiment matrix: the hardest case of each
block, every shot, once, through scripts/mpc_closed_loop_sim.py (same
estimator / coast / MPC modules as the node; fitted drone delay + lag; DOPE
field of view, range, scale cycling and noise; no ROS, no Gazebo).

Cases (fastest speed, or largest radius for the radius sweeps); "~" = the
same on the sine path (the racing line / start line + the matrix's weave):
  A1   parked car, r 20 m
  A2   Silverstone racing line capped at 17 m/s, r 20 m
  A3   Silverstone at 11 m/s, r 45 m
  A4   flat straight launch 0 -> 17 m/s at 8 m/s^2, hold 10 s, brake 6 m/s^2
  B*   Silverstone at 15 m/s with the sim's generic blackouts (1.5 / 3 / 4.5 s
       every ~20 s) -- a stand-in for the tree worlds, not their geometry
Shots come from scripts/run_experiments.py (orbit / overpass included) and
are flown through CinematicPlanner.

    python3 scripts/feasibility_check.py                  # everything (~minutes)
    python3 scripts/feasibility_check.py --cases A2 A2~ --shots behind orbit
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import mpc_closed_loop_sim as sim          # noqa: E402
import run_experiments as rx               # noqa: E402
from benchmark_target_filter import race_truth   # noqa: E402

SHOTS = ['behind', 'front', 'left', 'right', 'orbit', 'overpass', 'weave']
T_GO = 8.0        # flat-field car sets off at this sim time (drone already on its shot)


def sine_offset(s, speed, amplitude, period):
    """race_driver._path: offset left of the start line at distance s."""
    if s <= 0.0:
        return 0.0
    lam = max(speed * period, 1.0)
    r = min(max(s / (0.25 * lam), 0.0), 1.0)
    return amplitude * r * r * (3.0 - 2.0 * r) * math.sin(2.0 * math.pi * s / lam)


def launch_truth(speed, accel, hold, brake, sine=None):
    """Flat-field car: still until T_GO, then race_driver's launch profile from
    (0, 20) along +y, weaving if sine=(amplitude, period). Returns
    (spline, t_end, psi_fn)."""
    dt = 0.01
    t, s, v, phase, t_phase = 0.0, 0.0, 0.0, 'wait', 0.0
    ts, ss = [], []
    while True:
        ts.append(t)
        ss.append(s)
        if phase == 'wait' and t >= T_GO:
            phase = 'accel'
        if phase == 'accel':
            v = min(speed, v + accel * dt)
            if v >= speed:
                phase, t_phase = 'hold', t
        elif phase == 'hold' and t - t_phase >= hold:
            phase = 'brake'
        elif phase == 'brake':
            v = max(0.0, v - brake * dt)
            if v == 0.0:
                phase, t_phase = 'done', t
        if phase == 'done' and t - t_phase > 6.0:
            break
        s += v * dt
        t += dt
    ts, ss = np.array(ts), np.array(ss)
    if sine:
        off = np.array([sine_offset(x, speed, *sine) for x in ss])
        with np.errstate(divide='ignore', invalid='ignore'):   # repeated s while the car is still
            slope = np.nan_to_num(np.gradient(off, ss, edge_order=1))
    else:
        off, slope = np.zeros_like(ss), np.zeros_like(ss)
    xy = np.c_[-off, 20.0 + ss]                     # heading +y: "left" is -x
    psi = math.pi / 2 + np.arctan(slope)
    # Thin the samples where the car is still so the spline stays well posed.
    keep = np.r_[True, np.diff(ss) > 1e-6] | (ts % 0.1 < dt / 2)
    spline = CubicSpline(ts[keep], xy[keep])
    psi_fn = lambda tt: float(np.interp(tt, ts, psi))   # noqa: E731
    return spline, float(ts[-1]), psi_fn


def parked_truth(seconds=62.0):
    """A1: the car parked at (0, 20) facing +y."""
    ts = np.array([0.0, seconds / 2, seconds])
    spline = CubicSpline(ts, np.array([[0.0, 20.0]] * 3))
    return spline, seconds, (lambda tt: math.pi / 2)


def cases(d):
    """Hardest case of each block, each on the plain line and the sine path."""
    a4 = dict(accel=d['launch_accel'], hold=d['launch_hold'], brake=d['launch_brake'])
    amp, period = rx.sine_params(dict(d, car='launch'))
    out = {'A1': dict(kind='parked', speed=0.0, radius=20.0)}
    for path in ('line', 'sine'):
        sfx = '' if path == 'line' else '~'
        out['A2' + sfx] = dict(kind='track', speed=17.0, radius=20.0, path=path)
        out['A3' + sfx] = dict(kind='track', speed=11.0, radius=45.0, path=path)
        out['A4' + sfx] = dict(kind='launch', speed=17.0, radius=20.0, prof=a4, phase_lock=True,
                              sine=(amp, period) if path == 'sine' else None)
        out['B*' + sfx] = dict(kind='track', speed=15.0, radius=20.0, occlusions=True, path=path,
                              shots=['behind', 'right'])
    return out


def run_case(name, c, shot, d, mpc, seed, variant):
    if c['kind'] == 'parked':
        truth = parked_truth()
        t_end = truth[1]
    elif c['kind'] == 'track':
        csv = None
        if c.get('path') == 'sine':
            csv = rx.sine_race_line(dict(d, world='iris_silverstone_exp', speed=c['speed'], car='race'))
        spline, t_end = race_truth('iris_silverstone', csv=csv, max_speed=c['speed'])
        truth = (spline, t_end, None)
    else:
        p = c['prof']
        truth = launch_truth(c['speed'], p['accel'], p['hold'], p['brake'], c['sine'])
        t_end = truth[1]
    seq = rx.shot_sequence(shot, c['radius'], d['height'], t_end, d, parked=c['kind'] == 'parked')
    if c.get('phase_lock') and shot in ('orbit', 'overpass'):
        # The node holds the sequence at its start until the car moves (shot_start_speed).
        seq = [{'type': 'hold_location', 'location': 'back', 'radius': c['radius'], 'height': d['height'],
                'duration': T_GO - 2.0}] + seq
    t0 = time.time()
    m = sim.run(variant, 'iris_silverstone', 1e9, seed, mpc, c.get('occlusions', False), truth=truth, shot_seq=seq)
    err, rad, ang = (np.array(m[k]) for k in ('err', 'rad', 'ang'))
    rng = np.array(m['rng'])
    return {
        'case': name, 'shot': shot, 'speed': c['speed'], 'radius': c['radius'],
        'err_med': np.median(err), 'err_p90': np.percentile(err, 90),
        'dist_held': 100 * np.mean(rad <= 5.0), 'band': 100 * np.mean((rad <= 5.0) & (ang <= 10.0)),
        'angle_med': np.median(ang), 'too_close': 100 * np.mean(rng < 9.4), 'max_range': rng.max(),
        'cmd_acc_med': float(np.median(m['du'])) if m['du'] else float('nan'),
        'yaw_acc_p95': float(np.degrees(np.percentile(m['dyaw'], 95))) if m.get('dyaw') else float('nan'),
        'slips': len(sim.catch_ups(ang)),
        'catch_med': float(np.median([dd for ok, dd in sim.catch_ups(ang) if ok] or [np.nan])),
        'lost': m['lost'], 'solve_max': max(m['ms']) if m['ms'] else float('nan'),
        'sim_s': len(err) * sim.DT_SIM, 'wall_s': time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--cases', nargs='+', default=None)
    ap.add_argument('--shots', nargs='+', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--speed', type=float, default=None, help='override every case\'s car speed (m/s)')
    ap.add_argument('--variant', default='stare', help='mpc_closed_loop_sim variant (stare = current defaults)')
    ap.add_argument('--old', action='store_true',
                    help='before the 2026-10-01 fixes: DOPE heading used from overhead, eased (stop-start) orbits')
    ap.add_argument('--no-skip', action='store_true', help='fold near-overhead detections into the filter')
    ap.add_argument('--no-steep-noise', action='store_true',
                    help='position-only steep updates without the position noise inflation')
    ap.add_argument('--seeds', type=int, default=1)
    args = ap.parse_args()
    d = yaml.safe_load(open(rx.MATRIX))['defaults']
    if args.no_steep_noise:
        sim.STEEP_POS_NOISE_SCALE = 1.0
    if args.no_skip:
        sim.STEEP_SKIP_ELEV_DEG = 90.0
    if args.old:
        sim.HEADING_MAX_ELEV_DEG = 90.0
        sim.STEEP_POS_NOISE_SCALE = 1.0
        sim.STEEP_SKIP_ELEV_DEG = 90.0
        _seq = rx.shot_sequence
        rx.shot_sequence = lambda *a: [{k: v for k, v in act.items() if k != 'easing'} for act in _seq(*a)]
    all_cases = cases(d)
    mpc = sim.DroneMPC()
    rows = []
    hdr = (f"{'case':5} {'shot':9} {'v':>4} {'r':>3} | {'err med/p90 (m)':>15} | {'dist held':>9} | {'band':>5} |"
           f" {'angle med':>9} | {'<9.4 m':>6} | {'max rng':>7} | lost | {'solve max':>9} | sim/wall s")
    print(hdr)
    print('-' * len(hdr))
    for name, c in all_cases.items():
        if args.cases and name not in args.cases:
            continue
        if args.speed:
            c = dict(c, speed=args.speed)
        for shot in c.get('shots', SHOTS):
            if args.shots and shot not in args.shots:
                continue
            for seed in range(args.seed, args.seed + args.seeds):
                r = run_case(name, c, shot, d, mpc, seed, args.variant)
                rows.append(r)
                print(f"{r['case']:5} {r['shot']:9} {r['speed']:4.0f} {r['radius']:3.0f} |"
                      f" {r['err_med']:6.1f} / {r['err_p90']:6.1f} | {r['dist_held']:7.0f} % | {r['band']:3.0f} % |"
                      f" {r['angle_med']:7.1f} deg | {r['too_close']:4.1f} % | {r['max_range']:5.0f} m |"
                      f" {r['lost']:4d} | {r['solve_max']:6.1f} ms | {r['sim_s']:4.0f}/{r['wall_s']:4.0f}", flush=True)
    lost = [r for r in rows if r['lost']]
    print(f"\n{len(rows)} cases, {len(lost)} lost the car at least once"
          + (': ' + ', '.join(f"{r['case']} {r['shot']} ({r['lost']}x)" for r in lost) if lost else ''))


if __name__ == '__main__':
    main()
