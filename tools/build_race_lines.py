#!/usr/bin/python3
"""
Extract the lane centre line of each race-track world for race_driver.

For every worlds/iris_<track>.sdf that includes a model://race_track_* model,
this reads the track's pose from the world and the mesh + scale from the
model, rasterises the mesh (a slab with the lane cut out) top-down in world
coordinates, and traces the lane centre line from the car's spawn point
(0, 20) driving +y (the direction the car faces) once around the lap. It
then fits a minimum-curvature racing line inside the lane (MARGIN from the
walls) and writes it to
src/circumnavigation_controller/config/race_lines/<world>.csv as rows of
x, y (m, world frame, ~1 m spacing, closed loop, first row nearest the
spawn point). race_driver builds its speed profile from this line.

Re-run after changing a track's pose or scale in a world file:
    /usr/bin/python3 tools/build_race_lines.py            # all track worlds
    /usr/bin/python3 tools/build_race_lines.py iris_monza # just one
    ... --plot            # also save <world>_race_line.png in the cwd

Needs numpy, scipy and Pillow (the ROS /usr/bin/python3 has them).
"""

import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLDS_DIR = os.path.join(REPO, 'worlds')
MODELS_DIR = os.path.join(REPO, 'models')
OUT_DIR = os.path.join(
    REPO, 'src', 'circumnavigation_controller', 'config', 'race_lines')

RES = 0.1                 # raster cell size, m
START = (0.0, 20.0)       # car spawn in every track world, facing +y
STEP = 0.5                # centre-line tracing step, m
SPACING = 1.0             # output point spacing, m
MARGIN = 2.0              # racing line to wall: ~1 m half car + 1 m, m


def load_stl(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    if 84 + 50 * n != len(data):
        raise ValueError(f'{path}: not a binary STL')
    rec = np.frombuffer(data[84:], dtype=np.dtype(
        [('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]), count=n)
    return rec['v'].astype(float)  # (n, 3 vertices, xyz)


def track_include(world_path):
    """(model name, x, y, yaw) of the race_track_* include in a world."""
    root = ET.parse(world_path).getroot()
    for inc in root.iter('include'):
        m = re.match(r'model://(race_track_\w+)', inc.findtext('uri', ''))
        if m:
            x, y, _, _, _, yaw = map(float, inc.findtext('pose').split())
            return m.group(1), x, y, yaw
    return None


def track_mesh(model):
    """Mesh path and xy scale from models/<model>/model.sdf (visual)."""
    root = ET.parse(os.path.join(MODELS_DIR, model, 'model.sdf')).getroot()
    mesh = root.find('.//visual/geometry/mesh')
    rel = mesh.findtext('uri').split(f'model://{model}/', 1)[1]
    sx, sy, _ = map(float, (mesh.findtext('scale') or '1 1 1').split())
    return os.path.join(MODELS_DIR, model, rel), sx, sy


def lane_mask(tris_xy):
    """Boolean raster (True = drivable lane) and its world origin."""
    pts = tris_xy.reshape(-1, 2)
    x0, y0 = pts.min(axis=0) - 5.0
    x1, y1 = pts.max(axis=0) + 5.0
    w, h = int((x1 - x0) / RES) + 1, int((y1 - y0) / RES) + 1
    img = Image.new('1', (w, h), 0)
    draw = ImageDraw.Draw(img)
    for tri in tris_xy:
        draw.polygon([((px - x0) / RES, (y1 - py) / RES) for px, py in tri],
                     fill=1)
    free = ~np.array(img, dtype=bool)
    # The lane is the free region containing the spawn point; this drops
    # the open ground outside the slab.
    labels, _ = ndimage.label(free)
    r, c = to_cell(START, x0, y1)
    return labels == labels[r, c], x0, y1


def to_cell(p, x0, y1):
    return int((y1 - p[1]) / RES), int((p[0] - x0) / RES)


def trace_centre(lane, x0, y1):
    """Walk the ridge of the distance-to-wall field once round the lap."""
    dist = ndimage.distance_transform_edt(lane) * RES

    def d_at(p):
        r, c = to_cell(p, x0, y1)
        if 0 <= r < dist.shape[0] and 0 <= c < dist.shape[1]:
            return dist[r, c]
        return 0.0

    def centre_across(p, heading):
        # Slide sideways onto the lane centre (max wall distance).
        nx, ny = -math.sin(heading), math.cos(heading)
        best, best_d = p, d_at(p)
        for off in np.arange(-3.0, 3.01, 0.1):
            q = (p[0] + off * nx, p[1] + off * ny)
            dq = d_at(q)
            if dq > best_d + 1e-9:
                best, best_d = q, dq
        return best

    heading = math.pi / 2
    pts = [centre_across(START, heading)]
    travelled = 0.0
    while True:
        p = pts[-1]
        q = (p[0] + STEP * math.cos(heading), p[1] + STEP * math.sin(heading))
        q = centre_across(q, heading)
        if d_at(q) <= 0.0:
            raise RuntimeError(f'left the lane at {q}')
        # Heading from the last few metres, smooth but able to turn hairpins.
        back = pts[max(0, len(pts) - 4)]
        heading = math.atan2(q[1] - back[1], q[0] - back[0])
        pts.append(q)
        travelled += STEP
        if travelled > 50.0 and math.dist(q, pts[0]) < 1.5 * STEP:
            break
        if travelled > 20000.0:
            raise RuntimeError('did not close the lap')
    pts = np.array(pts[:-1])
    half_w = np.array([d_at(p) for p in pts])
    return pts, half_w


def resample(pts, spacing=SPACING):
    """Resample a closed loop at even spacing, starting from pts[0]."""
    loop = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    s_new = np.linspace(0.0, s[-1], int(s[-1] / spacing), endpoint=False)
    return np.column_stack([np.interp(s_new, s, loop[:, i])
                            for i in range(loop.shape[1])]), s[-1]


def smooth(pts, k=7):
    """Closed-loop moving average over k points."""
    n = len(pts)
    kern = np.ones(k) / k
    ext = np.concatenate([pts[-k:], pts, pts[:k]])
    return np.column_stack([np.convolve(ext[:, i], kern, 'same')[k:k + n]
                            for i in range(pts.shape[1])])


def racing_line(centre, half_w, margin, iters=5):
    """Minimum-curvature line inside the lane.

    Slides each centre-line point along its (fixed) normal by a_i, with
    |a_i| <= half_w - margin, to minimise the summed squared second
    difference of the closed path, i.e. roughly its squared curvature: the
    classic outside-apex-outside line that takes corners at the widest
    radius the lane allows. Linear in a, so a box-bounded least-squares
    solve (L-BFGS-B). Points bunch up where the line moves to the inside
    of a corner, which makes a second difference understate curvature
    there, so each row is re-weighted by 1/spacing^2 from the previous
    solve and the solve repeated.
    """
    from scipy import sparse
    from scipy.optimize import minimize

    n = len(centre)
    # Second-difference operator on a closed loop.
    d2 = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(n, n)).tolil()
    d2[0, n - 1] = 1.0
    d2[n - 1, 0] = 1.0
    d2 = d2.tocsr()
    tang = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])
    room = np.maximum(half_w - margin, 0.0)
    rhs0 = -np.concatenate([d2 @ centre[:, 0], d2 @ centre[:, 1]])
    off = np.zeros(n)
    for _ in range(iters):
        path = centre + normal * off[:, None]
        seg = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
        spacing = 0.5 * (seg + np.roll(seg, 1))
        w = sparse.diags(np.tile(1.0 / spacing ** 2, 2))
        # Path = centre + off * normal; minimise |W D2 path|^2 over off.
        a = w @ sparse.vstack([d2 @ sparse.diags(normal[:, 0]),
                               d2 @ sparse.diags(normal[:, 1])]).tocsr()
        rhs = w @ rhs0

        def cost(x):
            r = a @ x - rhs
            return r @ r, 2.0 * (a.T @ r)

        off = minimize(cost, off, jac=True, method='L-BFGS-B',
                       bounds=list(zip(-room, room)),
                       options={'maxiter': 50000, 'ftol': 1e-15,
                                'gtol': 1e-12}).x
    return centre + normal * off[:, None]


def build(world):
    inc = track_include(os.path.join(WORLDS_DIR, world + '.sdf'))
    if inc is None:
        return None
    model, tx, ty, yaw = inc
    stl, sx, sy = track_mesh(model)
    v = load_stl(stl)[:, :, :2] * (sx, sy)
    c, s = math.cos(yaw), math.sin(yaw)
    xy = np.stack([c * v[..., 0] - s * v[..., 1] + tx,
                   s * v[..., 0] + c * v[..., 1] + ty], axis=-1)
    lane, x0, y1 = lane_mask(xy)
    pts, half_w = trace_centre(lane, x0, y1)
    centre, _ = resample(smooth(np.column_stack([pts, half_w])))
    race = racing_line(centre[:, :2], centre[:, 2], MARGIN)
    race, length = resample(smooth(race, k=3))
    # Start the lap at the point nearest the car's spawn.
    race = np.roll(race, -int(np.argmin(
        np.linalg.norm(race - START, axis=1))), axis=0)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, world + '.csv')
    np.savetxt(out, race, fmt='%.3f', delimiter=',',
               header=f'{world} racing line from {model} ({MARGIN:.1f} m '
                      f'wall margin); lap {length:.1f} m\nx,y')
    print(f'{world}: {model}, lap {length:.0f} m, lane half width '
          f'{centre[:, 2].min():.1f}-{centre[:, 2].max():.1f} m, '
          f'min radius centre {min_radius(centre[:, :2]):.1f} m -> racing '
          f'line {min_radius(race):.1f} m -> {os.path.relpath(out, REPO)}')
    return lane, x0, y1, centre, race


def min_radius(pts, k=3):
    """Tightest corner radius (circumcircle over +-k points)."""
    a, c = np.roll(pts, k, axis=0), np.roll(pts, -k, axis=0)
    cross = ((pts[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
             - (pts[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    prod = (np.linalg.norm(pts - a, axis=1) * np.linalg.norm(c - pts, axis=1)
            * np.linalg.norm(c - a, axis=1))
    return 1.0 / (2.0 * np.abs(cross) / prod).max()


def plot(world, lane, x0, y1, centre, race):
    img = Image.fromarray(np.where(lane, 255, 40).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(img)
    for line, colour in ((centre, (170, 170, 170)), (race, (220, 30, 30))):
        cells = [((x - x0) / RES, (y1 - y) / RES) for x, y in line[:, :2]]
        draw.line(cells + cells[:1], fill=colour, width=3)
    sx, sy = cells[0]
    draw.ellipse([sx - 12, sy - 12, sx + 12, sy + 12], outline=(30, 90, 220), width=4)
    path = world + '_race_line.png'
    img.save(path)
    print(f'  plot -> {os.path.abspath(path)}')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    worlds = args or sorted(f[:-4] for f in os.listdir(WORLDS_DIR)
                            if f.endswith('.sdf'))
    for world in worlds:
        res = build(world)
        if res and '--plot' in sys.argv:
            plot(world, *res)


if __name__ == '__main__':
    main()
