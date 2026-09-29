#!/usr/bin/python3
"""
Make tree-lined versions of the race-track worlds, to test tracking when
trees block the drone's view of the car.

For each track world worlds/iris_<track>.sdf this writes
worlds/iris_<track>_trees.sdf: the same world (track, drone, car) plus
veg_* trees and bushes (models/veg_*, from gazebo-vegetation) placed

  - in a few dense "forest" sections along the lap, on both sides, and
  - as scattered single trees everywhere else,

off the lane (canopies may overhang it; the models are visual only, so
neither the car nor the drone can hit them) and clear of the start area,
so takeoff and the first lock-on on the car are unobstructed. Placement is
random but seeded, so a re-run gives the same world.

    /usr/bin/python3 tools/build_tree_worlds.py                 # all tracks
    /usr/bin/python3 tools/build_tree_worlds.py iris_monza      # one track
    ... --seed 3 --density 1.5 --plot  # another layout, 1.5x the trees, PNG

It also copies each world to ~/ardupilot_gazebo/worlds (what sim_launch.py
runs) and writes its racing line (tools/build_race_lines.py), so
`world:=iris_<track>_trees race:=true` works after a colcon build.
"""

import argparse
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

import build_race_lines as rl

ARDUPILOT_WORLDS = os.path.expanduser('~/ardupilot_gazebo/worlds')

# model: (canopy width m, height m, weight in the forests, weight as singles)
TREES = {
    'veg_tree_1': (5.9, 5.4, 2, 2),
    'veg_tree_2': (11.3, 12.3, 3, 3),   # big broadleaf, the best blocker
    'veg_tree_3': (7.7, 7.0, 2, 2),
    'veg_tree_5': (4.5, 6.4, 3, 1),     # conifer, dense down to the ground
    'veg_tree_9': (7.5, 18.4, 2, 2),    # tall pine, sparse canopy
    'veg_bush_0': (3.4, 3.6, 2, 1),
    'veg_bush_1': (4.6, 4.6, 2, 1),
}
START_CLEAR = 40.0      # m round the drone (0,0) / car (0,20) spawn
EDGE_MIN = 1.5          # trunk at least this far outside the lane, m
# The drone flies ~10 m from the car at 3-10 m up, so the trees that block
# its view are the ones within ~12 m of the lane; placement favours those.
FOREST_DEPTH = 14.0     # forests reach about this far past the lane, m
FORESTS_PER_KM = 2.5
FOREST_LEN = (60.0, 130.0)   # m of track per forest
FOREST_TREES_PER_100M = 26   # both sides together
SINGLE_EVERY = 35.0     # m of track per scattered tree
SINGLE_DEPTH = 8.0


def distance_outside(lane):
    """Distance (m) from each cell to the lane; 0 on the lane."""
    return ndimage.distance_transform_edt(~lane) * rl.RES


def place(world, seed, density):
    rng = np.random.default_rng(seed)
    inc = rl.track_include(os.path.join(rl.WORLDS_DIR, world + '.sdf'))
    model, tx, ty, yaw = inc
    stl, sx, sy = rl.track_mesh(model)
    v = rl.load_stl(stl)[:, :, :2] * (sx, sy)
    c, s = math.cos(yaw), math.sin(yaw)
    xy = np.stack([c * v[..., 0] - s * v[..., 1] + tx,
                   s * v[..., 0] + c * v[..., 1] + ty], axis=-1)
    lane, x0, y1 = rl.lane_mask(xy)
    d_out = distance_outside(lane)
    line = np.loadtxt(os.path.join(rl.OUT_DIR, world + '.csv'), delimiter=',')
    seg = np.linalg.norm(np.roll(line, -1, axis=0) - line, axis=1)
    s_line = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    length = seg.sum()
    tang = np.roll(line, -1, axis=0) - np.roll(line, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])

    def dist_to_lane(p):
        r, cc = int((y1 - p[1]) / rl.RES), int((p[0] - x0) / rl.RES)
        if 0 <= r < d_out.shape[0] and 0 <= cc < d_out.shape[1]:
            return d_out[r, cc]
        return 1e9

    placed = []  # (name, x, y, yaw)

    def try_add(name, p):
        width = TREES[name][0]
        if min(math.dist(p, (0.0, 0.0)), math.dist(p, (0.0, 20.0))) < START_CLEAR:
            return False
        if dist_to_lane(p) < EDGE_MIN:
            return False
        for other, ox, oy, _ in placed:
            # Canopies may touch/overlap a little, like a real tree line.
            if math.dist(p, (ox, oy)) < 0.4 * (width + TREES[other][0]):
                return False
        placed.append((name, float(p[0]), float(p[1]),
                       float(rng.uniform(-math.pi, math.pi))))
        return True

    def pick(col):
        names = list(TREES)
        w = np.array([TREES[n][col] for n in names], float)
        return names[rng.choice(len(names), p=w / w.sum())]

    def sample_near(i, depth):
        side = rng.choice([-1.0, 1.0])
        # Lateral offset from the racing line, skewed towards the lane edge
        # (the line can sit anywhere in the lane; try_add rejects points
        # still on it).
        off = 4.0 + depth * rng.beta(1.2, 2.2)
        along = rng.uniform(-0.5, 0.5)
        return line[i] + side * off * normal[i] + along * tang[i]

    # Forest sections, spread round the lap but not on the start straight.
    n_forest = max(1, int(round(FORESTS_PER_KM * length / 1000.0)))
    usable = (START_CLEAR + 40.0, length - START_CLEAR - 40.0)
    centres = np.linspace(*usable, n_forest + 2)[1:-1]
    centres += rng.uniform(-0.3, 0.3, n_forest) * (usable[1] - usable[0]) / (n_forest + 1)
    forests = []
    for sc in centres:
        flen = rng.uniform(*FOREST_LEN)
        forests.append((sc - flen / 2, sc + flen / 2))
        target = int(FOREST_TREES_PER_100M * density * flen / 100.0)
        idx = np.where((s_line >= sc - flen / 2) & (s_line <= sc + flen / 2))[0]
        added, tries = 0, 0
        while added < target and tries < 60 * target:
            tries += 1
            added += try_add(pick(2), sample_near(rng.choice(idx), FOREST_DEPTH))

    # Scattered singles everywhere else.
    for sc in np.arange(0.0, length, SINGLE_EVERY / density):
        if any(a <= sc <= b for a, b in forests):
            continue
        i = int(np.argmin(np.abs(s_line - (sc + rng.uniform(0, SINGLE_EVERY / density)) % length)))
        for _ in range(20):
            if try_add(pick(3), sample_near(i, SINGLE_DEPTH)):
                break
    return placed, forests, length, (lane, x0, y1, line)


def write_world(world, placed, seed, density):
    src = open(os.path.join(rl.WORLDS_DIR, world + '.sdf')).read()
    name = world + '_trees'
    src = src.replace(f'<world name="{world}">', f'<world name="{name}">')
    assert f'<world name="{name}">' in src, world
    trees = [f'    <!-- {len(placed)} trees/bushes from tools/build_tree_worlds.py '
             f'(seed {seed}, density {density}); visual only. -->']
    for i, (m, x, y, yaw) in enumerate(placed):
        trees.append(f'    <include><uri>model://{m}</uri><name>{m}_{i}</name>'
                     f'<pose>{x:.2f} {y:.2f} 0 0 0 {yaw:.3f}</pose></include>')
    out = src.replace('\n  </world>', '\n' + '\n'.join(trees) + '\n\n  </world>')
    assert out != src
    path = os.path.join(rl.WORLDS_DIR, name + '.sdf')
    open(path, 'w').write(out)
    if os.path.isdir(ARDUPILOT_WORLDS):
        shutil.copy(path, ARDUPILOT_WORLDS)
    # Same racing line as the base world, under the new world name.
    shutil.copy(os.path.join(rl.OUT_DIR, world + '.csv'),
                os.path.join(rl.OUT_DIR, name + '.csv'))
    return name


def plot(name, placed, forests, geo):
    lane, x0, y1, line = geo
    img = Image.fromarray(np.where(lane, 255, 60).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(img)
    cells = [((x - x0) / rl.RES, (y1 - y) / rl.RES) for x, y in line]
    draw.line(cells + cells[:1], fill=(170, 170, 170), width=3)
    for m, x, y, _ in placed:
        r = TREES[m][0] / 2 / rl.RES
        cx, cy = (x - x0) / rl.RES, (y1 - y) / rl.RES
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(40, 150, 50))
    sx, sy = (0 - x0) / rl.RES, (y1 - 10) / rl.RES
    r = START_CLEAR / rl.RES
    draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline=(30, 90, 220), width=6)
    img.thumbnail((1400, 1400))
    img.save(name + '.png')
    print(f'  plot -> {os.path.abspath(name + ".png")}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('worlds', nargs='*',
                    default=['iris_monza', 'iris_silverstone', 'iris_oschersleben'])
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--density', type=float, default=1.0)
    ap.add_argument('--plot', action='store_true',
                    help='save <world>_trees.png (top view) in the cwd')
    a = ap.parse_args()
    for world in a.worlds:
        placed, forests, length, geo = place(world, a.seed, a.density)
        name = write_world(world, placed, a.seed, a.density)
        counts = {}
        for m, *_ in placed:
            counts[m] = counts.get(m, 0) + 1
        print(f'{name}: {len(placed)} trees ({len(forests)} forest sections, '
              f'{sum(b - a for a, b in forests):.0f} of {length:.0f} m), '
              + ', '.join(f'{k[4:]} {v}' for k, v in sorted(counts.items())))
        if a.plot:
            plot(name, placed, forests, geo)


if __name__ == '__main__':
    main()
