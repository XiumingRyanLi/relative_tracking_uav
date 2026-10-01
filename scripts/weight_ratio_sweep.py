#!/usr/bin/env python3
"""Offline sweep of the MPC's distance / angle weight ratio (mpc_w_radial /
mpc_w_tangential), through the closed-loop sim (scripts/feasibility_check.py,
scripts/mpc_closed_loop_sim.py).

Both errors are divided by their band (5 m; radius x 10 deg), so 50/50 costs
one band-width of either the same. The ratio only matters where the drone
cannot hold both -- so the cases are where it binds: Silverstone at 8 / 11 /
15 / 17 m/s, every shot at r 20 m, and r 45 m at 11 m/s (right, orbit).
Every ratio runs the same seeds, so the comparisons are paired.

Score: % of time within +-5 m and +-10 deg of the shot ("band"); also
losses, too close (< 9.4 m), slips past 20 deg and the time back within
10 deg, command jitter.

    python3 scripts/weight_ratio_sweep.py                # ~15-20 min on 24 cores
    python3 scripts/weight_ratio_sweep.py --plot-only    # re-plot the saved CSV
Writes experiments/offline/ratio_sweep/{runs.csv, summary.csv, ratio_sweep.png}.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
OUT = os.path.join(REPO, 'experiments', 'offline', 'ratio_sweep')
RATIOS = [(50, 50), (60, 40), (70, 30), (80, 20), (90, 10), (95, 5)]
DU = (1.0, 1.0, 10.0)            # change weights: velocity, vertical, yaw rate (the new defaults)
SPEEDS = [8.0, 11.0, 15.0, 17.0]
SHOTS = ['behind', 'right', 'left', 'orbit', 'overpass']

_W = {}


def _init():
    import yaml
    import feasibility_check as fc
    import mpc_closed_loop_sim as sim
    import run_experiments as rx
    _W.update(fc=fc, sim=sim, d=yaml.safe_load(open(rx.MATRIX))['defaults'], mpc=sim.DroneMPC())
    for wr, wt in RATIOS:
        sim.VARIANTS[f'r{wr}'] = dict(sim.VARIANTS['stare'], w=(float(wr), float(wt)), du=DU)


def _job(args):
    wr, speed, radius, shot, seed = args
    fc = _W['fc']
    c = dict(fc.cases(_W['d'])['A2'], speed=speed, radius=radius)
    r = fc.run_case('A2', c, shot, _W['d'], _W['mpc'], seed, f'r{wr}')
    r.update(ratio=f'{wr}/{100 - wr}', w_radial=wr, seed=seed)
    return r


def jobs(seeds):
    cases = [(s, 20.0, sh) for s in SPEEDS for sh in SHOTS] + [(11.0, 45.0, 'right'), (11.0, 45.0, 'orbit')]
    return [(wr, s, r, sh, seed) for wr, _ in RATIOS for s, r, sh in cases for seed in range(seeds)]


def summarise(df):
    df = df.copy()
    df['case'] = df.apply(lambda r: f"{r['speed']:g}ms_r{r['radius']:g}_{r['shot']}", axis=1)
    # Paired differences against 50/50 (same case, same seed).
    base = df[df.w_radial == 50].set_index(['case', 'seed'])['band']
    df['band_vs_50'] = df.apply(lambda r: r['band'] - base.get((r['case'], r['seed']), np.nan), axis=1)
    g = df.groupby('w_radial')
    n = g.size()
    out = pd.DataFrame({
        'ratio': g['ratio'].first(),
        'band_mean': g['band'].mean(),
        'band_vs_50_mean': g['band_vs_50'].mean(),
        'band_vs_50_ci95': 1.96 * g['band_vs_50'].std() / np.sqrt(n),
        'dist_held': g['dist_held'].mean(),
        'angle_med': g['angle_med'].mean(),
        'err_med': g['err_med'].mean(),
        'lost': g['lost'].sum(),
        'too_close_pct': g['too_close'].mean(),
        'slips': g['slips'].sum(),
        'catch_med_s': g['catch_med'].median(),
        'cmd_acc': g['cmd_acc_med'].mean(),
        'runs': n,
    })
    by_speed = df[df.radius == 20].pivot_table(index='w_radial', columns='speed', values='band', aggfunc='mean')
    by_shot = df[df.radius == 20].pivot_table(index='w_radial', columns='shot', values='band', aggfunc='mean')
    r45 = df[df.radius == 45].pivot_table(index='w_radial', columns='shot', values='band', aggfunc='mean')
    return df, out, by_speed, by_shot, r45


def plot(df, out, by_speed, by_shot, r45, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = out.index.values
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    ax[0].errorbar(x, out['band_vs_50_mean'], yerr=out['band_vs_50_ci95'], marker='o', capsize=4)
    ax[0].axhline(0, color='k', lw=0.6)
    ax[0].set_title('time in band vs 50/50 (paired, 95 % CI)')
    ax[0].set_ylabel('percentage points')
    for sp in by_speed.columns:
        ax[1].plot(by_speed.index, by_speed[sp], marker='o', label=f'{sp:g} m/s')
    ax[1].set_title('time in band by speed (r 20 m)')
    ax[1].legend(fontsize=8)
    for sh in by_shot.columns:
        ax[2].plot(by_shot.index, by_shot[sh], marker='o', label=sh)
    for sh in r45.columns:
        ax[2].plot(r45.index, r45[sh], marker='x', ls='--', label=f'{sh} r45 @11')
    ax[2].set_title('time in band by shot')
    ax[2].legend(fontsize=7)
    ax3 = ax[3]
    ax3.bar(x - 1.5, out['lost'], width=3, label='losses (count)')
    ax3.bar(x + 1.5, out['too_close_pct'] * 10, width=3, label='too close (x0.1 %)')
    ax3b = ax3.twinx()
    ax3b.plot(x, out['catch_med_s'], 'k-o', label='time back within 10 deg (s)')
    ax3.set_title('losses / too close / catch-up')
    ax3.legend(fontsize=7, loc='upper left')
    ax3b.legend(fontsize=7, loc='upper right')
    for a in ax:
        a.set_xlabel('distance weight (of 100; angle = rest)')
        a.grid(alpha=0.3)
        a.set_xticks(x)
    ax[1].set_ylabel('% of time within +-5 m, +-10 deg')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runs_csv = os.path.join(OUT, 'runs.csv')
    if not args.plot_only:
        todo = jobs(args.seeds)
        print(f'{len(todo)} runs on {args.workers} workers', flush=True)
        with Pool(args.workers, initializer=_init) as pool:
            rows = []
            for i, r in enumerate(pool.imap_unordered(_job, todo, chunksize=1)):
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f'  {i + 1}/{len(todo)}', flush=True)
        pd.DataFrame(rows).to_csv(runs_csv, index=False)
    df, out, by_speed, by_shot, r45 = summarise(pd.read_csv(runs_csv))
    out.to_csv(os.path.join(OUT, 'summary.csv'))
    plot(df, out, by_speed, by_shot, r45, os.path.join(OUT, 'ratio_sweep.png'))
    pd.set_option('display.width', 200)
    print(out.round(2).to_string())
    print('\ntime in band by speed (r 20 m):\n' + by_speed.round(1).to_string())
    print('\nby shot (r 20 m):\n' + by_shot.round(1).to_string())
    print('\nr 45 m at 11 m/s:\n' + r45.round(1).to_string())


if __name__ == '__main__':
    main()
