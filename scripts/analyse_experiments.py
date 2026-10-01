#!/usr/bin/env python3
"""Analyse the experiment runs filed by scripts/run_experiments.py.

Reads experiments/<block>/<run>/run.json + <run>.csv and writes
experiments/results/:

  results.csv     one row per run: config + metrics (below)
  sections.csv    every run's rows split by what the car was doing
                  (straight / sweeping / tight corner, left / right) -- the
                  track's corners stand in for motion types
  blackouts.csv   every detection blackout >= BLACKOUT_SEC: start, length,
                  re-acquired or not, car speed, shot
  tables.tex      LaTeX tables per block (median over repeats)
  fig_*.png       speed curves (A2), radius curves (A3), launch time series
                  (A4), occlusion curves (B1/B2), section bars -- line and
                  sine path runs side by side (path column everywhere)

Metrics (shot phase only: from SETTLE_SEC (10 s) after the shot start --
the fly-in from the takeoff spot -- until the end event):
  dev_*        horizontal distance from the drone to the TRUE shot point
               (Gazebo car pose + the planner's shot offset), m: median,
               rms, p95, max
  dist_err_*   |drone - car| - shot radius (horizontal), m (median of |.|, p95)
  angle_err_*  bearing of the drone from the car vs the shot bearing, deg
               (only where the shot radius > 5 m)
  band_pct     % of time with |dist err| <= 5 m and |angle err| <= 10 deg
  detect_pct   % of time with a detection younger than 0.25 s ("in frame")
  min_range_m  closest 3D distance to the car; min_alt_m lowest altitude
  cap_pct      % of time the drone flew >= 14 m/s (its limit is 15)
  blackouts / reacquired / longest_blackout_s
  success      run status "done" and never fell into search

The car truth is interpolated at each row from its odometry stamps
(car_gt_stamp): it arrives slower than the 6.7 Hz rows.

    python3 scripts/analyse_experiments.py
    python3 scripts/analyse_experiments.py --block A2 --plot-run A2_silverstone_right_11ms_r20m_rep01
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, 'experiments')
BAND_DIST = 5.0
BAND_ANGLE = 10.0
DETECT_AGE = 0.25
BLACKOUT_SEC = 0.5
CAP_SPEED = 14.0
# Car curvature classes (1/m): straight < 1/300 m, sweeping < 1/80 m, else tight.
K_STRAIGHT = 1.0 / 300.0
K_SWEEP = 1.0 / 80.0


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def car_truth(d):
    """Car x, y, yaw (rad), speed at each row's time, interpolated from the
    odometry samples (by their stamps if logged)."""
    t = d['time'].to_numpy(float)
    cols = ['car_gt_x', 'car_gt_y', 'car_gt_yaw_deg']
    g = d[cols + (['car_gt_stamp'] if 'car_gt_stamp' in d else [])].copy()
    g['row_t'] = t
    g = g.dropna(subset=cols)
    if 'car_gt_stamp' in g and g['car_gt_stamp'].notna().any():
        g = g.dropna(subset=['car_gt_stamp']).drop_duplicates('car_gt_stamp')
        ts = g['car_gt_stamp'].to_numpy(float)
    else:
        # Older logs: a new sample shows up as a change; stamp it with the row time.
        changed = (g[cols].diff().abs().sum(axis=1) > 0) | (np.arange(len(g)) == 0)
        g = g[changed.to_numpy()]
        ts = g['row_t'].to_numpy(float)
    if len(g) < 2:
        return None
    order = np.argsort(ts)
    ts = ts[order]
    x = g['car_gt_x'].to_numpy(float)[order]
    y = g['car_gt_y'].to_numpy(float)[order]
    yaw = np.unwrap(np.radians(g['car_gt_yaw_deg'].to_numpy(float)[order]))
    cx, cy, cyaw = (np.interp(t, ts, v) for v in (x, y, yaw))
    # Speed / curvature from the samples, smoothed over ~1 s.
    dt = np.maximum(np.diff(ts), 1e-3)
    v = np.hypot(np.diff(x), np.diff(y)) / dt
    w = np.diff(yaw) / dt
    tm = 0.5 * (ts[1:] + ts[:-1])
    k = max(1, int(round(1.0 / max(np.median(dt), 1e-3))))
    kern = np.ones(k) / k
    vs = np.convolve(v, kern, mode='same')
    ws = np.convolve(w, kern, mode='same')
    speed = np.interp(t, tm, vs)
    curv = np.interp(t, tm, ws / np.maximum(vs, 0.5))
    curv[np.interp(t, tm, vs) < 1.0] = 0.0
    return cx, cy, cyaw, speed, curv


def blackouts(t, age, end_t):
    """Detection gaps: (start, length, reacquired) with no detection younger
    than BLACKOUT_SEC; age NaN = no detection yet / at all."""
    lost = ~(np.nan_to_num(age, nan=1e9) <= BLACKOUT_SEC)
    out = []
    i, n = 0, len(t)
    while i < n:
        if lost[i]:
            j = i
            while j < n and lost[j]:
                j += 1
            start = t[i] - BLACKOUT_SEC
            stop = t[j] if j < n else end_t
            out.append((start, stop - start, j < n))
            i = j
        else:
            i += 1
    return out


def section_label(curv):
    a = abs(curv)
    if a < K_STRAIGHT:
        return 'straight'
    kind = 'sweeping' if a < K_SWEEP else 'tight'
    return f'{kind}_{"left" if curv > 0 else "right"}'


SETTLE_SEC = 10.0   # first seconds of the shot (flying in from the takeoff spot) left out


def run_frame(d, cfg, settle_sec=SETTLE_SEC):
    """Per-row evaluation frame over the shot phase, or None."""
    need = ['shot_ox', 'shot_oy', 'shot_oz']
    if any(c not in d for c in need):
        return None
    truth = car_truth(d)
    if truth is None:
        return None
    cx, cy, cyaw, cspeed, curv = truth
    f = pd.DataFrame({'t': d['time'].to_numpy(float), 'stage': d['stage'].astype(str)})
    f['cx'], f['cy'], f['cyaw'], f['car_speed'], f['curv'] = cx, cy, cyaw, cspeed, curv
    for c in ('drone_x', 'drone_y', 'drone_z', 'shot_ox', 'shot_oy', 'shot_oz', 'dope_age_s', 'shot_clock'):
        f[c] = pd.to_numeric(d[c], errors='coerce').to_numpy(float) if c in d else np.nan
    # Shot phase: rows with a shot offset after the shot started (shot_clock > 0
    # once the sequence runs; parked/launch runs may hold at 0 while waiting).
    f = f[f['shot_ox'].notna() & f['stage'].isin(['stage2', 'coast', 'search', 'hold'])]
    if len(f) < 5:
        return None
    started = f['shot_clock'] > 0.0
    if started.any():
        f = f.loc[started.idxmax():]
    c, s = np.cos(f['cyaw']), np.sin(f['cyaw'])
    ox, oy = f['shot_ox'], f['shot_oy']
    f['ref_x'] = f['cx'] + c * ox - s * oy
    f['ref_y'] = f['cy'] + s * ox + c * oy
    f['ref_z'] = f['shot_oz']
    f['dev'] = np.hypot(f['drone_x'] - f['ref_x'], f['drone_y'] - f['ref_y'])
    rx, ry = f['drone_x'] - f['cx'], f['drone_y'] - f['cy']
    r_shot = np.hypot(ox, oy)
    f['r_shot'] = r_shot
    f['dist_err'] = np.hypot(rx, ry) - r_shot
    ang = np.degrees(np.arctan2(ry, rx) - np.arctan2(f['ref_y'] - f['cy'], f['ref_x'] - f['cx']))
    f['angle_err'] = np.where(r_shot > 5.0, wrap_deg(ang), np.nan)
    f['range3d'] = np.sqrt(rx ** 2 + ry ** 2 + (f['drone_z'] - 0.6) ** 2)
    f['detected'] = np.nan_to_num(f['dope_age_s'], nan=1e9) < DETECT_AGE
    dt = np.gradient(f['t'].to_numpy(float))
    dt = np.where(dt > 0, dt, np.nan)
    raw = np.hypot(np.gradient(f['drone_x'].to_numpy(float)), np.gradient(f['drone_y'].to_numpy(float))) / dt
    # Pose updates jitter against the rows: ~1 s rolling median.
    f['drone_speed'] = pd.Series(raw, index=f.index).rolling(7, center=True, min_periods=1).median()
    f['section'] = [section_label(k) for k in f['curv']]
    f = f[f['t'] >= f['t'].iloc[0] + settle_sec]
    return f if len(f) >= 5 else None


def pct(mask):
    mask = np.asarray(mask, dtype=bool)
    return 100.0 * mask.mean() if len(mask) else np.nan


def metrics(f, status):
    dev = f['dev'].to_numpy(float)
    de = f['dist_err'].to_numpy(float)
    ae = f['angle_err'].to_numpy(float)
    band = (np.abs(de) <= BAND_DIST) & ((np.abs(ae) <= BAND_ANGLE) | np.isnan(ae))
    end_t = float(f['t'].iloc[-1])
    bo = blackouts(f['t'].to_numpy(float), f['dope_age_s'].to_numpy(float), end_t)
    return {
        'duration_s': end_t - float(f['t'].iloc[0]),
        'dev_median': np.nanmedian(dev), 'dev_rms': float(np.sqrt(np.nanmean(dev ** 2))),
        'dev_p95': np.nanpercentile(dev, 95), 'dev_max': np.nanmax(dev),
        'dist_err_median': np.nanmedian(np.abs(de)), 'dist_err_p95': np.nanpercentile(np.abs(de), 95),
        'angle_err_median': np.nanmedian(np.abs(ae)) if np.isfinite(ae).any() else np.nan,
        'angle_err_p95': np.nanpercentile(np.abs(ae), 95) if np.isfinite(ae).any() else np.nan,
        'band_pct': pct(band),
        'detect_pct': pct(f['detected']),
        'min_range_m': np.nanmin(f['range3d']),
        'min_alt_m': np.nanmin(f['drone_z']),
        'cap_pct': pct(f['drone_speed'] >= CAP_SPEED),
        'car_speed_mean': np.nanmean(f['car_speed']),
        'blackouts': len(bo),
        'reacquired': sum(1 for b in bo if b[2]),
        'longest_blackout_s': max((b[1] for b in bo), default=0.0),
        'searched': bool((f['stage'] == 'search').any()),
        'success': status == 'done' and not bool((f['stage'] == 'search').any()),
    }, bo


# ---------------------------------------------------------------------------
# All runs
# ---------------------------------------------------------------------------
def load_runs(root, blocks=None):
    out = []
    for path in sorted(glob.glob(os.path.join(root, '*', '*', 'run.json'))):
        rec = json.load(open(path))
        # Only experiments/<block>/<run>/ -- not runs moved aside (e.g. old_A1/).
        if os.path.basename(os.path.dirname(os.path.dirname(path))) != rec['config']['block']:
            continue
        if blocks and rec['config']['block'] not in blocks:
            continue
        rec['dir'] = os.path.dirname(path)
        out.append(rec)
    return out


def analyse(recs):
    rows, sec_rows, bo_rows, frames = [], [], [], {}
    for rec in recs:
        cfg = rec['config']
        base = {k: cfg.get(k) for k in ('block', 'world', 'shot', 'speed', 'radius', 'height', 'rep', 'car')}
        base['path'] = cfg.get('path', 'line')
        base.update(name=rec['name'], status=rec['status'])
        csv = os.path.join(rec['dir'], rec['csv']) if rec.get('csv') else None
        f = run_frame(pd.read_csv(csv, low_memory=False), cfg) if csv and os.path.isfile(csv) else None
        if f is None:
            rows.append({**base, 'success': False})
            continue
        frames[rec['name']] = f
        m, bo = metrics(f, rec['status'])
        rows.append({**base, **m})
        for start, length, re in bo:
            i = int(np.searchsorted(f['t'].to_numpy(float), start))
            i = min(max(i, 0), len(f) - 1)
            bo_rows.append({**base, 'start_s': start - f['t'].iloc[0], 'length_s': length, 'reacquired': re,
                            'car_speed': f['car_speed'].iloc[i]})
        for sec, g in f.groupby('section'):
            if len(g) < 5:
                continue
            ae = g['angle_err'].to_numpy(float)
            sec_rows.append({**base, 'section': sec, 'rows': len(g),
                             'dev_median': np.nanmedian(g['dev']),
                             'dist_err_median': np.nanmedian(np.abs(g['dist_err'])),
                             'angle_err_median': np.nanmedian(np.abs(ae)) if np.isfinite(ae).any() else np.nan,
                             'band_pct': pct((np.abs(g['dist_err']) <= BAND_DIST)
                                             & ((np.abs(ae) <= BAND_ANGLE) | np.isnan(ae))),
                             'detect_pct': pct(g['detected'])})
    return pd.DataFrame(rows), pd.DataFrame(sec_rows), pd.DataFrame(bo_rows), frames


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
TABLE_COLS = [('dev_median', 'dev. med (m)'), ('dev_p95', 'dev. p95 (m)'), ('angle_err_median', 'angle (deg)'),
              ('band_pct', 'band (\\%)'), ('detect_pct', 'in frame (\\%)'), ('success', 'success')]


def latex_tables(res):
    out = []
    for block, g in res.groupby('block'):
        keys = [k for k in ('world', 'shot', 'path', 'speed', 'radius') if g[k].nunique() > 1] or ['shot']
        agg = {c: 'median' for c, _ in TABLE_COLS if c != 'success'}
        agg['success'] = 'mean'
        t = g.groupby(keys).agg({**agg, 'name': 'count'}).rename(columns={'name': 'n'}).reset_index()
        head = ' & '.join([k for k in keys] + [h for _, h in TABLE_COLS] + ['n'])
        lines = [f'% {block}', '\\begin{tabular}{' + 'l' * len(keys) + 'r' * (len(TABLE_COLS) + 1) + '}',
                 '\\hline', head + ' \\\\', '\\hline']
        for _, r in t.iterrows():
            cells = [str(r[k]).replace('_', '\\_') if not isinstance(r[k], float) else f'{r[k]:g}' for k in keys]
            for c, _ in TABLE_COLS:
                v = r[c]
                cells.append(f'{100 * v:.0f}\\%' if c == 'success' else ('--' if pd.isna(v) else f'{v:.1f}'))
            cells.append(str(int(r['n'])))
            lines.append(' & '.join(cells) + ' \\\\')
        lines += ['\\hline', '\\end{tabular}', '']
        out.append('\n'.join(lines))
    return '\n'.join(out)


def curve_figure(res, block, x, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = res[res['block'].isin(block)]
    if g.empty or 'dev_median' not in g:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = {}
    for (shot, path), gs in g.groupby(['shot', 'path']):
        col = colors.setdefault(shot, f'C{len(colors)}')
        ls = '--' if path == 'sine' else '-'
        lab = shot + (' (sine)' if path == 'sine' else '')
        q = gs.groupby(x).agg(dev=('dev_median', 'median'), lo=('dev_median', lambda v: np.nanpercentile(v, 25)),
                              hi=('dev_median', lambda v: np.nanpercentile(v, 75)),
                              band=('band_pct', 'median'), det=('detect_pct', 'median')).reset_index()
        axes[0].errorbar(q[x], q['dev'], yerr=[q['dev'] - q['lo'], q['hi'] - q['dev']], marker='o', capsize=3,
                         label=lab, color=col, ls=ls)
        axes[1].plot(q[x], q['band'], marker='o', label=lab, color=col, ls=ls)
        axes[2].plot(q[x], q['det'], marker='o', label=lab, color=col, ls=ls)
    for ax, lab in zip(axes, ('median deviation to shot point (m)', 'in band (%)', 'car in frame (%)')):
        ax.set_xlabel(x)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(title + ' -- solid: line, dashed: sine path')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def a4_figure(res, frames, recs, path, block='A4'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a4 = [r for r in recs if r['config']['block'] == block and r['name'] in frames]
    if not a4:
        return
    speeds = sorted({r['config']['speed'] for r in a4})
    shots = sorted({r['config']['shot'] for r in a4})
    fig, axes = plt.subplots(len(speeds), 3, figsize=(15, 3.6 * len(speeds)), squeeze=False)
    colors = dict(zip(shots, plt.rcParams['axes.prop_cycle'].by_key()['color']))
    for i, sp in enumerate(speeds):
        for r in a4:
            if r['config']['speed'] != sp:
                continue
            f = frames[r['name']]
            moving = f['car_speed'] > 0.5
            t0 = f['t'][moving].iloc[0] if moving.any() else f['t'].iloc[0]
            t = f['t'] - t0
            sh = r['config']['shot']
            axes[i][0].plot(t, f['car_speed'], color='k', alpha=0.3, lw=0.8)
            axes[i][1].plot(t, f['dist_err'], color=colors[sh], alpha=0.6, lw=0.9,
                            label=sh if r['config']['rep'] == 1 else None)
            axes[i][2].plot(t, np.abs(f['angle_err']), color=colors[sh], alpha=0.6, lw=0.9)
        axes[i][0].set_ylabel(f'{sp:g} m/s\ncar speed (m/s)')
        axes[i][1].set_ylabel('distance error (m)')
        axes[i][2].set_ylabel('|angle error| (deg)')
        axes[i][1].legend(fontsize=8)
        for ax in axes[i]:
            ax.set_xlim(-5, None)
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel('time from launch (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def occlusion_figure(bo, res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = res[res['block'].isin(['B1', 'B2'])]
    if g.empty or bo.empty:
        return
    b = bo[bo['block'].isin(['B1', 'B2'])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for (blk, world, shot, path), gs in b.groupby(['block', 'world', 'shot', 'path']):
        q = gs.groupby('speed').agg(re=('reacquired', 'mean'), length=('length_s', 'median')).reset_index()
        lab = f'{blk} {world.replace("iris_", "").replace("_exp", "")} {shot}' + (' sine' if path == 'sine' else '')
        axes[0].plot(q['speed'], 100 * q['re'], marker='o', label=lab)
        axes[1].plot(q['speed'], q['length'], marker='o', label=lab)
    axes[0].set_ylabel('blackouts re-acquired (%)')
    axes[1].set_ylabel('median blackout (s)')
    for ax in axes:
        ax.set_xlabel('car speed cap (m/s)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def section_figure(sec, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = sec[sec['block'].isin(['A2'])] if not sec.empty else sec
    if s.empty:
        return
    q = s.groupby(['path', 'section', 'shot'])['dev_median'].median().unstack('shot')
    ax = q.plot.bar(figsize=(11, 4.2), rot=0)
    ax.set_ylabel('median deviation to shot point (m)')
    ax.set_title('A2 by car motion (curvature class), line and sine path')
    ax.grid(alpha=0.3, axis='y')
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=130)
    plt.close(ax.figure)


def plot_run(f, name, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0][0]
    ax.plot(f['cx'], f['cy'], 'k-', lw=1, label='car (truth)')
    ax.plot(f['ref_x'], f['ref_y'], 'g--', lw=0.8, label='shot point (truth)')
    ax.plot(f['drone_x'], f['drone_y'], 'b-', lw=1, label='drone')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.set_title(name)
    t = f['t'] - f['t'].iloc[0]
    axes[0][1].plot(t, f['dev'], label='deviation')
    axes[0][1].plot(t, f['dist_err'], label='distance error')
    axes[0][1].axhspan(-BAND_DIST, BAND_DIST, color='g', alpha=0.08)
    axes[0][1].set_ylabel('m')
    axes[0][1].legend(fontsize=8)
    axes[1][0].plot(t, f['angle_err'])
    axes[1][0].axhspan(-BAND_ANGLE, BAND_ANGLE, color='g', alpha=0.08)
    axes[1][0].set_ylabel('angle error (deg)')
    axes[1][1].plot(t, f['car_speed'], 'k', label='car speed')
    axes[1][1].plot(t, f['drone_speed'], 'b', alpha=0.6, label='drone speed')
    det = f['detected'].to_numpy(bool)
    axes[1][1].fill_between(t, 0, 2, where=~det, color='r', alpha=0.3, label='no detection')
    axes[1][1].legend(fontsize=8)
    for ax in axes.flat[1:]:
        ax.set_xlabel('shot time (s)')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default=ROOT, help='experiments directory')
    ap.add_argument('--block', action='append')
    ap.add_argument('--out', default=None, help='default: <root>/results')
    ap.add_argument('--plot-run', action='append', help='also plot these runs (names)')
    args = ap.parse_args()
    args.out = args.out or os.path.join(args.root, 'results')
    recs = load_runs(args.root, args.block)
    if not recs:
        raise SystemExit(f'no runs under {args.root}')
    os.makedirs(args.out, exist_ok=True)
    res, sec, bo, frames = analyse(recs)
    res.to_csv(os.path.join(args.out, 'results.csv'), index=False)
    sec.to_csv(os.path.join(args.out, 'sections.csv'), index=False)
    bo.to_csv(os.path.join(args.out, 'blackouts.csv'), index=False)
    ok = res[res.get('dev_median', pd.Series(dtype=float)).notna()] if 'dev_median' in res else res.iloc[0:0]
    if not ok.empty:
        with open(os.path.join(args.out, 'tables.tex'), 'w') as f:
            f.write(latex_tables(ok))
        curve_figure(ok, ['A2'], 'speed', os.path.join(args.out, 'fig_A2_speed.png'), 'A2: speed sweep (r = 20 m)')
        rad = pd.concat([ok[ok['block'] == 'A3'], ok[(ok['block'] == 'A2') & (ok['speed'] == 11)]])
        curve_figure(rad.assign(block='A3'), ['A3'], 'radius', os.path.join(args.out, 'fig_A3_radius.png'),
                     'A3: radius sweep (11 m/s)')
        for path in ('line', 'sine'):
            a4_figure(ok, frames, [r for r in recs if r['config'].get('path', 'line') == path],
                      os.path.join(args.out, f'fig_A4_launch_{path}.png'))
        occlusion_figure(bo, ok, os.path.join(args.out, 'fig_B_occlusion.png'))
        section_figure(sec, os.path.join(args.out, 'fig_A2_sections.png'))
    for name in args.plot_run or []:
        if name in frames:
            plot_run(frames[name], name, os.path.join(args.out, f'run_{name}.png'))
        else:
            print(f'no data for {name}')
    print(f'{len(res)} runs ({int(res["success"].fillna(False).sum())} successful) -> {args.out}')
    st = res['status'].value_counts().to_dict()
    print('status:', st)
    if not ok.empty:
        summary = ok.groupby(['block', 'shot', 'path']).agg(
            runs=('name', 'count'), dev_median=('dev_median', 'median'), band_pct=('band_pct', 'median'),
            detect_pct=('detect_pct', 'median'), success_pct=('success', lambda v: 100.0 * np.mean(v)))
        print(summary.round(1).to_string())


if __name__ == '__main__':
    main()
