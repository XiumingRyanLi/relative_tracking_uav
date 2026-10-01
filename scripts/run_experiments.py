#!/usr/bin/env python3
"""Run the Gazebo experiment batches in experiments/matrix.yaml, unattended.

Each run is one headless launch of sim_launch.py (no windows; Gazebo server
only, the camera still renders on the GPU) in its own process group. The
runner waits for the controller's "EXPERIMENT_EVENT tracking_enabled", then
for the run's end condition (car done / duration / search-timeout LAND),
stops the whole group and files everything under

    experiments/<block>/<run_name>/
        <run_name>.csv   controller log (relative_pid format + shot columns)
        launch.log       every node's output
        run.json         configuration, status, events, git commit, times
        sitl/            ArduCopter.log (dataflash .BIN deleted unless --keep-sitl-logs)

Runs whose run.json says "done" or "lost" are skipped, so a batch can be
stopped (Ctrl+C) and started again. Analyse with scripts/analyse_experiments.py.

    python3 scripts/run_experiments.py --dry-run                 # list the runs, estimated time
    python3 scripts/run_experiments.py --block A2 --workers 6    # 6 sims in parallel (default: auto)
    python3 scripts/run_experiments.py --block A1 --only rep01 --limit 2   # pilot
    python3 scripts/run_experiments.py --block A2 --block A3
    python3 scripts/run_experiments.py --only right_11ms         # substring filter on run names
    python3 scripts/run_experiments.py --rerun-failed            # retry timeouts / crashes

Needs: colcon build after code changes (the launch uses the installed
package), tools/build_experiment_assets.py once, and NO other simulation
running: SITL / MAVROS use fixed ports, so the runner refuses to start next
to one. It never kills processes it did not start (they carry a marker in
their environment).
"""
import argparse
import copy
import datetime
import glob
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'src', 'circumnavigation_controller', 'circumnavigation_controller'))
from race_line import RaceLine  # noqa: E402

MATRIX = os.path.join(REPO, 'experiments', 'matrix.yaml')
OUT_ROOT = os.path.join(REPO, 'experiments')
RACE_LINES = os.path.join(REPO, 'src', 'circumnavigation_controller', 'config', 'race_lines')
EEPROM = os.path.expanduser('~/ardupilot/eeprom.bin')
MARKER = 'EXPERIMENT_RUN_ID'

# Wall-clock budget. Startup = launch -> "tracking enabled": Gazebo 10 s,
# SITL 30 s, MAVROS/controller, arm, takeoff, hover (~70-90 s in practice).
STARTUP_TIMEOUT = 300.0
OVERHEAD_SEC = 110.0          # per run, for the time estimate
POST_END_SEC = 3.0            # keep logging this long after the end event
FOREIGN_PATTERNS = ['gz sim', 'gz-sim-main', 'sim_vehicle.py', 'arducopter', 'mavros_node', 'mavproxy']

SHOT_SIDES = {'behind': 'back', 'front': 'front', 'left': 'left', 'right': 'right'}

# Parallel runs: worker w flies SITL instance w + 1 (instance 0's start-up
# runs sim_vehicle's process cleanup). Instance N's ports are all + 10 N,
# including the Gazebo JSON link the drone model listens on (9002), so each
# instance gets its own copy of the drone model and of every world.
ARDUPILOT_WORLDS = os.path.expanduser('~/ardupilot_gazebo/worlds')
INSTANCE_MODELS = os.path.join(OUT_ROOT, 'instances', 'models')
DRONE_MODEL = 'iris_with_dope_gimbal_exp'
_asset_lock = threading.Lock()


def instance_world(world, instance):
    """World name for SITL `instance`: <world>_i<N> (written once), with a
    drone model listening on 9002 + 10 N."""
    if instance == 0:
        return world
    name = f'{world}_i{instance}'
    model = f'{DRONE_MODEL}_i{instance}'
    with _asset_lock:
        mdir = os.path.join(INSTANCE_MODELS, model)
        os.makedirs(mdir, exist_ok=True)
        src = os.path.join(REPO, 'models', DRONE_MODEL)
        sdf = open(os.path.join(src, 'model.sdf')).read()
        assert sdf.count('<fdm_port_in>9002</fdm_port_in>') == 1
        sdf = sdf.replace('<fdm_port_in>9002</fdm_port_in>', f'<fdm_port_in>{9002 + 10 * instance}</fdm_port_in>')
        open(os.path.join(mdir, 'model.sdf'), 'w').write(sdf)
        cfg = open(os.path.join(src, 'model.config')).read()
        open(os.path.join(mdir, 'model.config'), 'w').write(
            cfg.replace(f'<name>{DRONE_MODEL}</name>', f'<name>{model}</name>'))
        w = open(os.path.join(ARDUPILOT_WORLDS, world + '.sdf')).read()
        assert f'<world name="{world}">' in w and f'model://{DRONE_MODEL}<' in w
        w = w.replace(f'<world name="{world}">', f'<world name="{name}">')
        w = w.replace(f'model://{DRONE_MODEL}<', f'model://{model}<')
        path = os.path.join(ARDUPILOT_WORLDS, name + '.sdf')
        if not os.path.isfile(path) or open(path).read() != w:
            open(path, 'w').write(w)
    return name


# ---------------------------------------------------------------------------
# Matrix -> runs
# ---------------------------------------------------------------------------
def world_tag(world):
    return world.replace('iris_', '').replace('_exp', '')


def run_name(block, world, shot, speed, radius, rep, car, path='line'):
    sp = 'parked' if car == 'parked' else f'{speed:g}ms'
    w = world_tag(world) + ('_sine' if path == 'sine' else '')
    return f'{block}_{w}_{shot}_{sp}_r{radius:g}m_rep{rep:02d}'


def race_line_file(world):
    return os.path.join(RACE_LINES, world + '.csv')


def sine_params(run):
    """(amplitude m, period s) of the weave, the same in every block:
    sine_amplitude at peak lateral acceleration sine_lat_accel, so period =
    2 pi sqrt(A / a) -- one weave per period at the target speed."""
    a = float(run['sine_amplitude'])
    return a, 2.0 * math.pi * math.sqrt(a / float(run['sine_lat_accel']))


def sine_race_line(run):
    """The world's racing line with a sine weave on top, for this speed cap;
    written to experiments/race_lines/ (deterministic) and returned. The
    weave is lam = speed * period long, ramps in and out over the first /
    last quarter wavelength (the car starts straight and the lap closes), and where
    config/race_lines/<world>_room.csv exists its amplitude is cut to the
    room to the wall minus sine_wall_clearance on the tighter side."""
    world, speed = run['world'], float(run['speed'])
    amp, period = sine_params(run)
    lam = max(speed * period, 10.0)
    pts = np.loadtxt(race_line_file(world), delimiter=',', comments='#')
    seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)[:-1]]
    length = float(seg.sum())
    t = np.gradient(pts, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    nrm = np.c_[-t[:, 1], t[:, 0]]                       # left of the direction of travel
    a = np.full(len(pts), amp)
    room_file = os.path.join(RACE_LINES, world + '_room.csv')
    if os.path.isfile(room_file):
        room = np.loadtxt(room_file, delimiter=',', comments='#')
        if len(room) == len(pts):
            avail = np.minimum(room[:, 2], room[:, 3]) - float(run['sine_wall_clearance'])
            # Fit a whole half-weave: the tightest room within +-lam/2, then smoothed.
            k = max(1, int(round(0.5 * lam / max(np.mean(seg), 1e-3))))
            tight = np.array([avail[np.arange(i - k, i + k + 1) % len(avail)].min() for i in range(len(avail))])
            kern = np.ones(2 * k + 1) / (2 * k + 1)
            tight = np.convolve(np.r_[tight[-k:], tight, tight[:k]], kern, mode='valid')
            a = np.clip(np.minimum(a, tight), 0.0, None)
    ramp = np.clip(np.minimum(s, length - s) / (0.25 * lam), 0.0, 1.0)
    env = ramp * ramp * (3.0 - 2.0 * ramp)
    off = a * env * np.sin(2.0 * np.pi * s / lam)
    out = pts + nrm * off[:, None]
    d = os.path.join(OUT_ROOT, 'race_lines')
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'{world}_sine_{speed:g}ms.csv')
    np.savetxt(path, out, fmt='%.3f', delimiter=',',
               header=f'{world} racing line + sine weave: up to {amp:.1f} m, {lam:.0f} m wavelength '
                      f'({period:.1f} s at {speed:g} m/s), max offset used {np.abs(off).max():.1f} m\nx,y')
    return path


def expected_seconds(run):
    """Sim seconds from the shot start to the end condition."""
    if run['car'] == 'parked':
        return float(run['duration'])
    if run['car'] == 'launch':
        v = run['speed']
        return v / run['launch_accel'] + run['launch_hold'] + v / run['launch_brake'] + 8.0
    line = RaceLine.from_csv(run['race_line'], max_speed=float(run['speed']))
    return line.lap_time() * run['laps'] + 8.0      # + race_driver's start delay


def shot_sequence(shot, r, h, total_sec, d, parked=False):
    """Controller shot_sequence (list of actions) covering total_sec.
    parked: chained orbits at one constant speed ("easing": "linear") -- the
    eased ones stop behind a parked car every lap (A1: drone speed swung
    1-11 m/s). Round a moving car the eased orbit tracked better offline
    (its pause behind the car lets the drone recover), so it stays there."""
    total = total_sec + 30.0
    if shot in SHOT_SIDES:
        # The planner holds the last offset after the sequence ends.
        return [{'type': 'hold_location', 'location': SHOT_SIDES[shot], 'radius': r, 'height': h,
                 'duration': 10.0}]
    if shot == 'orbit':
        t = 2.0 * math.pi * r / d['orbit_speed']
        orbit = {'type': 'orbit', 'start': 'back', 'radius': r, 'height': h, 'angle_deg': 360.0,
                 'direction': 'counterclockwise', 'duration': round(t, 2)}
        if parked:
            orbit['easing'] = 'linear'
        return [orbit] * int(math.ceil(total / t))
    if shot == 'weave':
        # Sideways sine round the car from behind (radius kept), one action
        # per <= 300 s (the schema's duration limit), a whole number of periods.
        per = float(d['weave_period'])
        n_per = max(1, int(288.0 // per))
        n_act = int(math.ceil(total / (n_per * per)))
        return [{'type': 'weave', 'location': 'back', 'radius': r, 'height': h, 'axis': d['weave_axis'],
                 'amplitude': float(d['weave_amplitude']), 'period': per, 'duration': round(n_per * per, 2)}] * n_act
    if shot == 'overpass':
        t = 2.0 * r / d['overpass_speed']
        seq = []
        for i in range(int(math.ceil(total / t))):
            a, b = ('back', 'front') if i % 2 == 0 else ('front', 'back')
            seq.append({'type': 'overpass', 'from': a, 'to': b, 'radius': r, 'start_height': h,
                        'peak_height': d['overpass_peak'], 'end_height': h, 'duration': round(t, 2)})
        return seq
    if shot == 'showcase':
        cycle = [
            {'type': 'hold_location', 'location': 'back', 'radius': r, 'height': h, 'duration': 20.0},
            {'type': 'orbit', 'start': 'back', 'radius': r, 'height': h, 'angle_deg': 360.0,
             'direction': 'counterclockwise', 'duration': round(2.0 * math.pi * r / d['orbit_speed'], 2)},
            {'type': 'move_location', 'from': 'back', 'to': 'front', 'via': 'right', 'radius': r,
             'height': h, 'duration': round(math.pi * r / d['orbit_speed'], 2)},
            {'type': 'hold_location', 'location': 'front', 'radius': r, 'height': h, 'duration': 20.0},
            {'type': 'overpass', 'from': 'front', 'to': 'back', 'radius': r, 'start_height': h,
             'peak_height': d['overpass_peak'], 'end_height': h,
             'duration': round(2.0 * r / d['overpass_speed'], 2)},
        ]
        t = sum(a['duration'] for a in cycle)
        return cycle * int(math.ceil(total / t))
    raise ValueError(f'unknown shot {shot!r}')


def expand(matrix):
    d0 = matrix['defaults']
    runs = []
    for block, spec in matrix['blocks'].items():
        b = dict(d0)
        b.update({k: v for k, v in spec.items() if k not in ('extra', 'runs')})
        worlds = b.get('worlds') or [b['world']]
        radii = b.get('radii') or [b['radius']]
        # Fastest first, then down: the limit of the controller / vision shows
        # up first, and a stopped batch already has the hardest cases.
        speeds = sorted(b.get('speeds') or [0.0], reverse=True)
        # Car paths: line = the racing line (race) / the straight start line
        # (launch); sine = the same with a sine weave. Parked cars have none.
        paths = ['line'] if b['car'] == 'parked' else list(b.get('paths') or ['line'])
        combos = []        # (world, shot, path, speed, radius, repeats)
        if 'runs' in spec:
            for r in spec['runs']:
                for path in ([r['path']] if 'path' in r else paths):
                    combos.append((r.get('world', worlds[0]), r['shot'], path, float(r.get('speed', 0.0)),
                                   float(r.get('radius', radii[0])), int(r.get('repeats', b['repeats']))))
        else:
            for w in worlds:
                for radius in radii:
                    for shot in b['shots']:
                        for path in paths:
                            for sp in speeds:
                                n = int((b.get('repeats_by_speed') or {}).get(sp, b['repeats']))
                                combos.append((w, shot, path, float(sp), float(radius), n))
        for r in spec.get('extra', []):
            for path in ([r['path']] if 'path' in r else paths):
                combos.append((r.get('world', worlds[0]), r['shot'], path, float(r.get('speed', speeds[0])),
                               float(r.get('radius', radii[0])), int(r.get('repeats', b['repeats']))))
        first = int(b.get('first_rep', 1))
        for w, shot, path, sp, radius, n in combos:
            for rep in range(first, first + n):
                run = copy.deepcopy(b)
                run.update(block=block, world=w, shot=shot, path=path, speed=sp, radius=radius, rep=rep)
                for k in ('worlds', 'radii', 'speeds', 'shots', 'repeats_by_speed', 'description', 'paths'):
                    run.pop(k, None)
                if run['car'] == 'race':
                    run['race_line'] = sine_race_line(run) if path == 'sine' else race_line_file(w)
                if path == 'sine':
                    run['sine_amplitude_used'], run['sine_period_used'] = sine_params(run)
                run['name'] = run_name(block, w, shot, sp, radius, rep, b['car'], path)
                run['expected_sec'] = expected_seconds(run)
                run['shot_sequence'] = shot_sequence(shot, radius, float(b['height']), run['expected_sec'], b,
                                                     parked=b['car'] == 'parked')
                runs.append(run)
    names = [r['name'] for r in runs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise ValueError(f'duplicate run names: {sorted(dup)[:5]}')
    return runs


def launch_args(run, run_dir):
    a = {
        'headless': 'true',
        'world': run['world'],
        'detector': run['detector'],
        'controller': run['controller'],
        'shot_sequence': json.dumps(run['shot_sequence'], separators=(',', ':')),
        'run_name': run['name'],
        'log_dir': run_dir,
        'sitl_dir': os.path.join(run_dir, 'sitl'),
        'sitl_rebuild': 'false',
        'shot_start_speed': str(run.get('shot_start_speed', 0.0)),
    }
    if run['car'] == 'parked':
        a.update(race='false', experiment_duration=str(run['duration']))
    elif run['car'] == 'race':
        a.update(race='true', car_mode='race', max_speed=str(run['speed']), laps=str(run['laps']),
                 race_line_file=run['race_line'])
    elif run['car'] == 'launch':
        a.update(race='true', car_mode='launch', launch_speed=str(run['speed']),
                 launch_accel=str(run['launch_accel']), launch_hold=str(run['launch_hold']),
                 launch_brake=str(run['launch_brake']),
                 launch_path='sine' if run['path'] == 'sine' else 'straight')
        if run['path'] == 'sine':
            amp, period = sine_params(run)
            a.update(sine_amplitude=f'{amp:g}', sine_period=f'{period:.3f}')
    else:
        raise ValueError(run['car'])
    return a


# ---------------------------------------------------------------------------
# Processes
# ---------------------------------------------------------------------------
def marked_pids(run_id=None):
    """PIDs whose environment carries the runner's marker (any run, or one)."""
    out = []
    for env_path in glob.glob('/proc/[0-9]*/environ'):
        try:
            with open(env_path, 'rb') as f:
                env = f.read().split(b'\0')
        except OSError:
            continue
        for e in env:
            if e.startswith(MARKER.encode() + b'='):
                if run_id is None or e == f'{MARKER}={run_id}'.encode():
                    out.append(int(env_path.split('/')[2]))
                break
    return [p for p in out if p != os.getpid()]


def foreign_sim_processes():
    mine = set(marked_pids())
    found = []
    for pat in FOREIGN_PATTERNS:
        res = subprocess.run(['pgrep', '-af', pat], capture_output=True, text=True)
        for line in res.stdout.splitlines():
            pid = int(line.split()[0])
            if pid not in mine and pid != os.getpid() and 'pgrep' not in line and 'run_experiments' not in line:
                found.append(line)
    return sorted(set(found))


def stop_group(proc, run_id, log):
    """SIGINT the launch's process group, then escalate; finally kill any
    process still carrying this run's marker (e.g. something that left the group)."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None
    for sig, wait in ((signal.SIGINT, 25.0), (signal.SIGTERM, 8.0), (signal.SIGKILL, 3.0)):
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                pass
        deadline = time.time() + wait
        while time.time() < deadline:
            if proc.poll() is not None and not marked_pids(run_id):
                return 0
            time.sleep(0.5)
    left = marked_pids(run_id)
    for pid in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1.0)
    left = marked_pids(run_id)
    if left:
        log(f'WARNING: {len(left)} process(es) of this run survived SIGKILL: {left}')
    return len(left)


def git_info():
    def g(*a):
        return subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {'commit': g('rev-parse', 'HEAD'), 'dirty': bool(g('status', '--porcelain', '--untracked-files=no'))}


class LogWatcher:
    """Reads new lines of launch.log and collects EXPERIMENT_EVENT lines."""

    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.events = []        # (wall s since start, event text)
        self.flags = set()
        self.t0 = time.time()
        self.stopping = False    # set before the runner stops the launch: shutdown noise isn't a fault

    def poll(self):
        try:
            with open(self.path, 'r', errors='replace') as f:
                f.seek(self.pos)
                chunk = f.read()
                self.pos = f.tell()
        except OSError:
            return []
        new = []
        for line in chunk.splitlines():
            if 'EXPERIMENT_EVENT' in line:
                ev = line.split('EXPERIMENT_EVENT', 1)[1].strip()
                self.events.append((round(time.time() - self.t0, 1), ev))
                new.append(ev)
            elif self.stopping:
                continue
            elif 'MPC unavailable' in line:
                self.flags.add('mpc_unavailable')
            elif 'Traceback (most recent call last)' in line:
                self.flags.add('traceback')
            elif 'process has died' in line:
                self.flags.add('process_died')
        return new

    def has(self, prefix):
        return any(e.startswith(prefix) for _, e in self.events)


def attempt(run, run_dir, args, log, instance=0, stop=None):
    """One launch on SITL `instance`. Returns (status, info dict)."""
    run_id = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(run_dir, 'sitl'), exist_ok=True)
    if os.path.isfile(EEPROM):
        # Same stored parameters as the interactive flights, but a private copy.
        shutil.copy(EEPROM, os.path.join(run_dir, 'sitl', 'eeprom.bin'))
    la = launch_args(run, run_dir)
    la['world'] = instance_world(run['world'], instance)
    la['instance'] = str(instance)
    cmd = ['ros2', 'launch', 'circumnavigation_controller', 'sim_launch.py'] + [f'{k}:={v}' for k, v in la.items()]
    env = dict(os.environ)
    env[MARKER] = run_id
    env['GZ_PARTITION'] = f'exp_{run_id}'
    env['ROS_DOMAIN_ID'] = str(args.domain_id + instance)        # one ROS graph per parallel run
    env['GZ_SIM_RESOURCE_PATH'] = INSTANCE_MODELS + ':' + env.get('GZ_SIM_RESOURCE_PATH', '')
    log_path = os.path.join(run_dir, 'launch.log')
    t_start = time.time()
    with open(log_path, 'a') as logf:
        logf.write(f'\n===== {datetime.datetime.now().isoformat()} run {run_id}: {" ".join(cmd)}\n')
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                env=env, cwd=run_dir, start_new_session=True)
    watch = LogWatcher(log_path)
    watch.pos = os.path.getsize(log_path)
    status, reason = None, ''
    t_track = t_end = None
    budget = 2.0 * run['expected_sec'] + 120.0
    try:
        while True:
            time.sleep(1.0)
            watch.poll()
            now = time.time()
            if stop is not None and stop.is_set():
                status, reason = 'interrupted', 'batch stopped'
                break
            if proc.poll() is not None:
                status, reason = 'crashed', f'launch exited with {proc.returncode}'
                break
            if t_track is None:
                if watch.has('tracking_enabled'):
                    t_track = now
                    log(f'  tracking enabled after {now - t_start:.0f} s')
                elif now - t_start > STARTUP_TIMEOUT:
                    status, reason = 'startup_failed', f'no tracking after {STARTUP_TIMEOUT:.0f} s'
                    break
                continue
            lost = [e for _, e in watch.events if e.startswith('flight_end') and 'no target found' in e]
            if lost:
                status, reason = 'lost', lost[0]
            elif watch.has('car_done') or watch.has('duration_done'):
                status, reason = 'done', 'car_done' if watch.has('car_done') else 'duration_done'
            elif watch.has('flight_end'):
                status, reason = 'aborted', [e for _, e in watch.events if e.startswith('flight_end')][0]
            elif now - t_track > budget:
                status, reason = 'timeout', f'no end condition after {budget:.0f} s of tracking'
            if status is not None:
                t_end = now
                time.sleep(POST_END_SEC)
                break
    except KeyboardInterrupt:
        status, reason = 'interrupted', 'Ctrl+C'
    finally:
        watch.poll()
        watch.stopping = True
        survivors = stop_group(proc, run_id, log)
        watch.poll()
    info = {
        'run_id': run_id, 'reason': reason, 'events': watch.events, 'flags': sorted(watch.flags),
        'wall_start': datetime.datetime.fromtimestamp(t_start).isoformat(timespec='seconds'),
        'wall_sec': round(time.time() - t_start, 1),
        'startup_sec': round(t_track - t_start, 1) if t_track else None,
        'tracking_wall_sec': round(t_end - t_track, 1) if (t_end and t_track) else None,
        'survivors': survivors, 'launch_args': la,
    }
    return status, info


def tidy_sitl(run_dir, keep):
    if keep:
        return
    for f in glob.glob(os.path.join(run_dir, 'sitl', '**', '*.BIN'), recursive=True):
        os.remove(f)
    for f in ('eeprom.bin',):
        p = os.path.join(run_dir, 'sitl', f)
        if os.path.exists(p):
            os.remove(p)


def installed_ok():
    """The launch runs the colcon-installed package: check it has this code."""
    try:
        share = subprocess.run(['ros2', 'pkg', 'prefix', 'circumnavigation_controller'],
                               capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return False, 'ros2 not found (source the workspace: source install/setup.bash)'
    if not share:
        return False, 'circumnavigation_controller not found (source install/setup.bash)'
    hits = glob.glob(os.path.join(share, 'lib', 'python3*', 'site-packages', 'circumnavigation_controller',
                                  'race_driver.py'))
    launch = os.path.join(share, 'share', 'circumnavigation_controller', 'launch', 'sim_launch.py')
    for path, needle in ((hits[0] if hits else '', 'EXPERIMENT_EVENT'), (launch, "'headless'")):
        try:
            if needle not in open(path).read():
                return False, f'{path} is out of date: run colcon build'
        except OSError:
            return False, f'{path or "installed race_driver.py"} missing: run colcon build'
    return True, ''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--matrix', default=MATRIX)
    ap.add_argument('--block', action='append', help='only these blocks (repeatable)')
    ap.add_argument('--only', action='append', help='only run names containing this (repeatable, any)')
    ap.add_argument('--limit', type=int, default=0, help='at most this many runs')
    ap.add_argument('--dry-run', action='store_true', help='list the runs and the estimated time')
    ap.add_argument('--rerun-failed', action='store_true', help='also re-run timeouts / crashes / aborted')
    ap.add_argument('--retries', type=int, default=2, help='retries after a startup failure / crash')
    ap.add_argument('--keep-sitl-logs', action='store_true', help='keep the dataflash .BIN logs')
    ap.add_argument('--domain-id', type=int, default=77,
                    help='ROS_DOMAIN_ID base: parallel sim N uses base + N (isolation)')
    ap.add_argument('--workers', type=int, default=0,
                    help='parallel sims (0 = auto: ~4 cores and ~6 GB RAM each; 6 on this machine)')
    ap.add_argument('--stagger', type=float, default=30.0, help='seconds between the parallel sims\' first starts')
    ap.add_argument('--force', action='store_true', help='start even with another simulation running')
    ap.add_argument('--allow-dirty', action='store_true',
                    help='run with uncommitted changes to tracked files (testing only: the results '
                         "can't be tied to a commit)")
    args = ap.parse_args()

    with open(args.matrix) as f:
        runs = expand(yaml.safe_load(f))
    if args.block:
        runs = [r for r in runs if r['block'] in args.block]
    if args.only:
        runs = [r for r in runs if any(s in r['name'] for s in args.only)]

    def finished(r):
        p = os.path.join(OUT_ROOT, r['block'], r['name'], 'run.json')
        if not os.path.isfile(p):
            return False
        st = json.load(open(p)).get('status')
        if st in ('done', 'lost'):
            return True
        return not args.rerun_failed       # failed runs: only again with --rerun-failed
    todo = [r for r in runs if not finished(r)]
    if args.limit:
        todo = todo[:args.limit]

    est = sum(r['expected_sec'] + OVERHEAD_SEC for r in todo)
    if args.dry_run:
        by_block = {}
        for r in todo:
            by_block.setdefault(r['block'], []).append(r)
        for blk, rs in by_block.items():
            h = sum(r['expected_sec'] + OVERHEAD_SEC for r in rs) / 3600.0
            print(f'\n{blk}: {len(rs)} runs, ~{h:.1f} h')
            for r in rs:
                print(f'  {r["name"]:60s} {r["car"]:7s} ~{r["expected_sec"]:5.0f} s  '
                      f'{len(r["shot_sequence"])} action(s)')
        w = args.workers if args.workers > 0 else auto_workers()
        print(f'\n{len(todo)} of {len(runs)} selected runs to do, ~{est / 3600.0:.1f} h of sim one at a time '
              f'(+{OVERHEAD_SEC:.0f} s startup/shutdown each) -> ~{est / 3600.0 / w:.1f} h on {w} parallel sims '
              f'if each still ran at full speed')
        return

    ok, why = installed_ok()
    if not ok:
        sys.exit(f'Not ready: {why}')
    git = git_info()
    if git['dirty'] and not args.allow_dirty:
        sys.exit(f"Uncommitted changes in tracked files: commit them first, so every run.json names the "
                 f"exact code (commit {git['commit'][:8]} + changes). --allow-dirty overrides, for tests.\n"
                 + subprocess.run(['git', 'status', '--short', '--untracked-files=no'], cwd=REPO,
                                  capture_output=True, text=True).stdout)
    foreign = foreign_sim_processes()
    if foreign and not args.force:
        sys.exit('Another simulation is running (fixed SITL/MAVROS ports); stop it first:\n  '
                 + '\n  '.join(foreign[:10]))
    batch_log = open(os.path.join(OUT_ROOT, 'batch.log'), 'a')
    log_lock = threading.Lock()

    def log(msg):
        line = f'{datetime.datetime.now().strftime("%H:%M:%S")} {msg}'
        with log_lock:
            print(line, flush=True)
            batch_log.write(line + '\n')
            batch_log.flush()

    workers = args.workers if args.workers > 0 else auto_workers()
    workers = max(1, min(workers, len(todo)))
    log(f'{len(todo)} run(s) on {workers} parallel sim(s), estimated {est / 3600.0 / workers:.1f} h '
        f'(if they all ran at full speed), code {git["commit"][:8]}'
        + (' + UNCOMMITTED CHANGES' if git['dirty'] else ''))
    queue = list(todo)
    qlock = threading.Lock()
    stop = threading.Event()
    done_count = [0]
    t_batch = time.time()

    def worker(w):
        instance = w + 1
        # Stagger the start-ups: SITL / EKF alignment is the slow part under load.
        if stop.wait(w * args.stagger):
            return
        while not stop.is_set():
            with qlock:
                if not queue:
                    return
                run = queue.pop(0)
                pos = len(todo) - len(queue)
            run_dir = os.path.join(OUT_ROOT, run['block'], run['name'])
            os.makedirs(run_dir, exist_ok=True)
            log(f'[{pos}/{len(todo)}] sim {instance}: {run["name"]}')
            tries = []
            status = None
            for _ in range(args.retries + 1):
                status, info = attempt(run, run_dir, args, log, instance, stop)
                tries.append({'status': status, 'instance': instance, **info})
                log(f'  sim {instance}: {run["name"]} -> {status} ({info["reason"]}), {info["wall_sec"]:.0f} s')
                if status in ('done', 'lost', 'interrupted', 'timeout', 'aborted'):
                    break
            tidy_sitl(run_dir, args.keep_sitl_logs)
            csv = os.path.join(run_dir, run['name'] + '.csv')
            record = {
                'name': run['name'], 'status': status, 'config': run, 'git': git,
                'csv': os.path.basename(csv) if os.path.isfile(csv) else None,
                'attempts': tries, 'finished': datetime.datetime.now().isoformat(timespec='seconds'),
            }
            with open(os.path.join(run_dir, 'run.json'), 'w') as f:
                json.dump(record, f, indent=1)
            if status == 'interrupted':
                return
            with qlock:
                done_count[0] += 1
                n = done_count[0]
            rate = (time.time() - t_batch) / n
            log(f'  {n}/{len(todo)} done in {(time.time() - t_batch) / 3600.0:.1f} h, '
                f'~{rate * (len(todo) - n) / 3600.0:.1f} h left')

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(workers)]
    for th in threads:
        th.start()
    try:
        while any(th.is_alive() for th in threads):
            for th in threads:
                th.join(timeout=1.0)
    except KeyboardInterrupt:
        log('Ctrl+C: stopping the running sims (re-run the same command to continue)')
        stop.set()
        for th in threads:
            th.join()
    log('batch stopped' if stop.is_set() else 'batch finished')


def auto_workers():
    """As many parallel sims as the machine holds: ~4 cores and ~6 GB of RAM each."""
    cores = os.cpu_count() or 4
    try:
        avail_gb = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2 ** 30
    except (ValueError, OSError):
        avail_gb = 16.0
    return max(1, min(cores // 4, int(avail_gb // 6)))


if __name__ == '__main__':
    main()
