#!/usr/bin/python3
"""
Build the Gazebo models and worlds used by the experiment batches
(scripts/run_experiments.py, experiments/matrix.yaml). Deterministic: a
re-run writes the same files.

Models (repo models/, on GZ_SIM_RESOURCE_PATH via sim_launch.py):
  gimbal_small_3d_dope_noise  gimbal_small_3d_dope + Gaussian image noise
                              (stddev CAMERA_NOISE_STD of full scale)
  iris_with_dope_gimbal_exp   iris_with_dope_gimbal on that gimbal
  audi_r8_launch              audi_r8 for the flat-field launch / sine runs
                              (A4, A2S-A4S): all four wheels driven, tyre mu
                              1.2, plugin limits +-8 m/s^2. The stock car is
                              rear-drive with mu2 = 1, so it cannot pull more
                              than ~5 m/s^2. The front wheels are driven at the
                              rear-axle speed: fine on a straight or the gentle
                              sine (radius > 100 m), not for tight turns.

No wind: the runs are wind-free. (Had it been wanted: ArduPilot's SIM_WIND_*
does nothing with the Gazebo JSON backend -- SIM_JSON.cpp zeroes it -- so it
would need Gazebo's WindEffects system in the world.)

Worlds (worlds/ and ~/ardupilot_gazebo/worlds; <world name> = file name):
  iris_silverstone_exp, iris_silverstone_trees_exp
      the track worlds with the experiment drone (noisy camera)
  iris_flat_exp
      flat field, audi_r8_launch at (0, 20) facing +y: A4 straight line and
      the sine-path runs (race_driver mode:=launch launch_path:=sine)
  iris_oval_trees_{short,medium,long}_exp
      flat oval (400 m straights, 90 m radius ends, counter-clockwise from
      (0, 20) along +y), tree clusters on the OUTSIDE (the car's right) of
      both straights, CLUSTER_OFFSET m from the lane: between the car and a
      right-side shot. short / medium / long = about 10 / 30 / 60 m of trees.

Racing lines (config/race_lines/<world>.csv) for every world above that has
a track, and for the walled tracks the free room either side of the line
(config/race_lines/<world>_room.csv: x, y, left, right in m), which
scripts/run_experiments.py uses to fit the sine weave inside the lane. Run once, then colcon build (the runner passes the source CSV path,
so the race lines also work without a rebuild):

    /usr/bin/python3 tools/build_experiment_assets.py
"""

import math
import os
import re
import shutil

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_MODELS = os.path.join(REPO, 'models')
REPO_WORLDS = os.path.join(REPO, 'worlds')
ARDUPILOT_MODELS = os.path.expanduser('~/ardupilot_gazebo/models')
ARDUPILOT_WORLDS = os.path.expanduser('~/ardupilot_gazebo/worlds')
RACE_LINES = os.path.join(REPO, 'src', 'circumnavigation_controller', 'config', 'race_lines')

CAMERA_NOISE_STD = 0.01

OVAL_STRAIGHT = 400.0
OVAL_RADIUS = 90.0
OVAL_Y0 = -100.0            # first straight from y = OVAL_Y0 to OVAL_Y0 + OVAL_STRAIGHT along x = 0
CLUSTER_OFFSET = 9.0        # trunk line, m outside the lane centre
# Where the clusters start, m along each straight (clear of the start at y 0..20).
CLUSTER_STARTS = (190.0, 330.0)
LAYOUTS = {'short': 10.0, 'medium': 30.0, 'long': 60.0}   # m of trees per cluster
TREE_SPACING = 7.0          # big broadleaf (veg_tree_2, 11 m canopy, 12 m tall), overlapping


def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)
    print('wrote', os.path.relpath(path, REPO) if path.startswith(REPO) else path)


def model_source(name):
    """The copy sim_launch.py actually loads (ardupilot_gazebo first on the path)."""
    for d in (ARDUPILOT_MODELS, REPO_MODELS):
        if os.path.isfile(os.path.join(d, name, 'model.sdf')):
            return os.path.join(d, name)
    raise FileNotFoundError(name)


def make_model(src_name, dst_name, edit):
    src = model_source(src_name)
    dst = os.path.join(REPO_MODELS, dst_name)
    os.makedirs(dst, exist_ok=True)
    sdf = edit(read(os.path.join(src, 'model.sdf')))
    write(os.path.join(dst, 'model.sdf'), sdf)
    cfg = read(os.path.join(src, 'model.config'))
    cfg = re.sub(r'<name>.*?</name>', f'<name>{dst_name}</name>', cfg, count=1)
    write(os.path.join(dst, 'model.config'), cfg)


def build_models():
    def gimbal(sdf):
        noise = (f'\n          <noise>\n            <type>gaussian</type>\n            <mean>0.0</mean>\n'
                 f'            <stddev>{CAMERA_NOISE_STD}</stddev>\n          </noise>')
        out, n = re.subn(r'(<camera>.*?</image>)', lambda m: m.group(1) + noise, sdf, count=1, flags=re.S)
        assert n == 1
        return out
    make_model('gimbal_small_3d_dope', 'gimbal_small_3d_dope_noise', gimbal)

    def iris(sdf):
        out = sdf.replace('model://gimbal_small_3d_dope<', 'model://gimbal_small_3d_dope_noise<')
        assert 'gimbal_small_3d_dope_noise' in out
        return out
    make_model('iris_with_dope_gimbal', 'iris_with_dope_gimbal_exp', iris)

    def car(sdf):
        out = re.sub(r'<mu>[\d.]+</mu>(\s*<mu2>)[\d.]+(</mu2>\s*<slip1>)',
                     r'<mu>1.2</mu>\g<1>1.2\g<2>', sdf)
        assert out.count('<mu>1.2</mu>') == 4, out.count('<mu>1.2</mu>')
        out = out.replace('<left_joint>rear_left_wheel_joint</left_joint>',
                          '<left_joint>rear_left_wheel_joint</left_joint>\n'
                          '            <left_joint>front_left_wheel_joint</left_joint>')
        out = out.replace('<right_joint>rear_right_wheel_joint</right_joint>',
                          '<right_joint>rear_right_wheel_joint</right_joint>\n'
                          '            <right_joint>front_right_wheel_joint</right_joint>')
        for tag, val in (('min_acceleration', -8), ('max_acceleration', 8), ('min_jerk', -40), ('max_jerk', 40)):
            out, n = re.subn(rf'<{tag}>[-\d.]+</{tag}>', f'<{tag}>{val}</{tag}>', out)
            assert n == 1, tag
        return out
    make_model('audi_r8', 'audi_r8_launch', car)


def experiment_world(src_text, src_name, name, car_uri=None):
    out = src_text.replace(f'<world name="{src_name}">', f'<world name="{name}">')
    assert f'<world name="{name}">' in out
    out = out.replace('<uri>model://iris_with_dope_gimbal</uri>', '<uri>model://iris_with_dope_gimbal_exp</uri>')
    assert 'iris_with_dope_gimbal_exp' in out
    if car_uri:
        out = out.replace('<uri>model://audi_r8</uri>', f'<uri>model://{car_uri}</uri>')
        assert car_uri in out
    return out


def save_world(name, text):
    write(os.path.join(REPO_WORLDS, name + '.sdf'), text)
    if os.path.isdir(ARDUPILOT_WORLDS):
        shutil.copy(os.path.join(REPO_WORLDS, name + '.sdf'), ARDUPILOT_WORLDS)


def flat_base():
    """iris_silverstone.sdf without the track mesh: grey plane, drone, car."""
    src = read(os.path.join(REPO_WORLDS, 'iris_silverstone.sdf'))
    out, n = re.subn(r'\s*<!-- Silverstone.*?-->\s*<include>\s*<uri>model://race_track_silverstone</uri>.*?</include>',
                     '', src, count=1, flags=re.S)
    assert n == 1
    return out


def oval_line(step=1.0):
    """Counter-clockwise oval through (0, 20) heading +y, sampled every `step` m,
    starting at (0, 20). Returns (n, 2) and the straights' (start, direction, outward normal)."""
    L, R, y0 = OVAL_STRAIGHT, OVAL_RADIUS, OVAL_Y0
    pts = []
    # Right straight x = 0, +y; top bend centre (-R, y0 + L); left straight x = -2R, -y; bottom bend.
    segs = []
    n_s = int(round(L / step))
    n_b = int(round(math.pi * R / step))
    for i in range(n_s):
        pts.append((0.0, y0 + i * L / n_s))
    for i in range(n_b):
        a = math.pi * i / n_b
        pts.append((-R + R * math.cos(a), y0 + L + R * math.sin(a)))
    for i in range(n_s):
        pts.append((-2 * R, y0 + L - i * L / n_s))
    for i in range(n_b):
        a = math.pi + math.pi * i / n_b
        pts.append((-R + R * math.cos(a), y0 + R * math.sin(a)))
    pts = np.array(pts)
    k = int(np.argmin(np.hypot(pts[:, 0], pts[:, 1] - 20.0)))
    pts = np.roll(pts, -k, axis=0)
    segs = [((0.0, y0), (0.0, 1.0), (1.0, 0.0)),             # outward (car's right) = +x
            ((-2 * R, y0 + L), (0.0, -1.0), (-1.0, 0.0))]    # outward = -x
    return pts, segs


def lane_room(world, track_world):
    """Free room (m) to the wall either side of the racing line of
    `track_world`'s lane, saved as config/race_lines/<world>_room.csv."""
    import sys
    sys.path.insert(0, os.path.join(REPO, 'tools'))
    import build_race_lines as rl
    model, tx, ty, yaw = rl.track_include(os.path.join(rl.WORLDS_DIR, track_world + '.sdf'))
    stl, sx, sy = rl.track_mesh(model)
    v = rl.load_stl(stl)[:, :, :2] * (sx, sy)
    c, s = math.cos(yaw), math.sin(yaw)
    xy = np.stack([c * v[..., 0] - s * v[..., 1] + tx, s * v[..., 0] + c * v[..., 1] + ty], axis=-1)
    lane, x0, y1 = rl.lane_mask(xy)
    race = np.loadtxt(os.path.join(RACE_LINES, world + '.csv'), delimiter=',', comments='#')
    t = np.gradient(race, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    nrm = np.c_[-t[:, 1], t[:, 0]]             # left of the direction of travel

    def room(sign):
        out = []
        for p, n in zip(race, nrm):
            r = 0.0
            while r < 40.0:
                i, j = rl.to_cell(p + sign * n * (r + 0.1), x0, y1)
                if not (0 <= i < lane.shape[0] and 0 <= j < lane.shape[1]) or not lane[i, j]:
                    break
                r += 0.1
            out.append(r)
        return np.array(out)
    out = np.c_[race, room(1.0), room(-1.0)]
    path = os.path.join(RACE_LINES, world + '_room.csv')
    np.savetxt(path, out, fmt='%.2f', delimiter=',',
               header=f'{world}: free room to the wall left / right of the racing line\nx,y,left,right')
    print('wrote', os.path.relpath(path, REPO),
          f'(lane {np.median(out[:, 2] + out[:, 3]):.1f} m wide, median)')


def build_worlds():
    for base in ('iris_silverstone', 'iris_silverstone_trees'):
        name = base + '_exp'
        save_world(name, experiment_world(read(os.path.join(REPO_WORLDS, base + '.sdf')), base, name))
        shutil.copy(os.path.join(RACE_LINES, base + '.csv'), os.path.join(RACE_LINES, name + '.csv'))
        lane_room(name, 'iris_silverstone')

    flat = flat_base()
    save_world('iris_flat_exp', experiment_world(flat, 'iris_silverstone', 'iris_flat_exp', car_uri='audi_r8_launch'))

    pts, segs = oval_line()
    lap = float(np.sum(np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)))
    for layout, length in LAYOUTS.items():
        name = f'iris_oval_trees_{layout}_exp'
        trees = [f'    <!-- Tree clusters ({layout}: {length:.0f} m each, {CLUSTER_OFFSET:.0f} m outside the '
                 f'lane on both straights) from tools/build_experiment_assets.py; visual only. -->']
        i = 0
        for (sx, sy), (dx, dy), (nx, ny) in segs:
            for s0 in CLUSTER_STARTS:
                n_trees = max(1, int(round(length / TREE_SPACING)) + 1) if length > TREE_SPACING else 1
                for j in range(n_trees):
                    s = s0 + (j * length / (n_trees - 1) if n_trees > 1 else 0.0)
                    x = sx + dx * s + nx * CLUSTER_OFFSET
                    y = sy + dy * s + ny * CLUSTER_OFFSET
                    trees.append(f'    <include><uri>model://veg_tree_2</uri><name>veg_tree_2_{i}</name>'
                                 f'<pose>{x:.2f} {y:.2f} 0 0 0 {0.7 * i % 6.283:.3f}</pose></include>')
                    i += 1
        text = experiment_world(flat, 'iris_silverstone', name)
        text = text.replace('\n  </world>', '\n' + '\n'.join(trees) + '\n\n  </world>')
        save_world(name, text)
        hdr = (f'# {name} racing line: counter-clockwise oval, {OVAL_STRAIGHT:.0f} m straights, '
               f'{OVAL_RADIUS:.0f} m bends; lap {lap:.1f} m\n# x,y\n')
        write(os.path.join(RACE_LINES, name + '.csv'),
              hdr + '\n'.join(f'{x:.3f},{y:.3f}' for x, y in pts) + '\n')


def main():
    build_models()
    build_worlds()
    print('\nNext: colcon build (installs the race lines); the worlds are already in ~/ardupilot_gazebo/worlds.')


if __name__ == '__main__':
    main()
