"""Tolerance-grid figures (N_val=2, seed 0): filled 5x5 error grids.

Each cell of the local/global tolerance grid is coloured by the final error,
with a colorbar; no 3D panels.  VIB5 (CH3Cl, CH3F) and QeMFi (SCF, EV) are
1x2 figures; ANI is a single panel.

Writes Manuscript/publishable_plots/new_tolerance_<group>.{pdf,png}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from saturatedmfml.dataset_paths import REPO_ROOT, RESULTS_SINGLE_PASS

GRID = REPO_ROOT / 'results' / 'tolerance_grid'
OUT = REPO_ROOT / 'Manuscript' / 'publishable_plots'
LOCAL_TOLS = [0.01, 0.1, 1.0, 2.0, 4.0]
GLOBAL_TOLS = [0.01, 0.1, 1.0, 2.0, 4.0]
VAL_SIZE = 2

VIB5 = np.asarray([296.85, 298.24, 2224.71, 96144.24]) / 3600.0
ANI = np.asarray([9.64615, 16.8199, 40.34815, 157.2737]) / 3600.0
Q5 = np.load(RESULTS_SINGLE_PASS / 'QeMFi_times.npy').mean(axis=0) / 3600.0
QEMFI = Q5[1:]

GROUPS = [
    ('VIB5', [
        ('CH3Cl', VIB5, VIB5[-1], 'MAE [kcal/mol]'),
        ('CH3F', VIB5, VIB5[-1], 'MAE [kcal/mol]'),
    ]),
    ('QeMFi', [
        ('SCF', QEMFI, Q5[4], 'MAE [kcal/mol]'),
        ('EV', QEMFI, Q5[4], 'MAE [kcal/mol]'),
    ]),
    ('ANI', [
        ('ANI', ANI, ANI[-1], 'RMAE'),
    ]),
]


def tag(v):
    return f'{v:g}'


def final_matrix(name):
    fin = np.zeros((len(LOCAL_TOLS), len(GLOBAL_TOLS)))
    for i, l in enumerate(LOCAL_TOLS):
        for j, g in enumerate(GLOBAL_TOLS):
            prefix = GRID / name / f'local{tag(l)}_global{tag(g)}' / '0'
            mae = np.load(prefix.with_name(prefix.name + '_mae.npy'),
                          allow_pickle=True).astype(float)
            fin[i, j] = mae[-1]
    return fin


def panel(ax, cax, name, ylabel):
    fin = final_matrix(name)
    edges = np.arange(-0.5, len(LOCAL_TOLS) + 0.5, 1.0)
    im = ax.pcolormesh(edges, edges, fin, cmap='viridis', shading='flat')
    gx, ly = np.meshgrid(np.arange(len(GLOBAL_TOLS)), np.arange(len(LOCAL_TOLS)))
    ax.scatter(gx.ravel(), ly.ravel(), s=12, c='white', edgecolors='none',
               alpha=0.55, zorder=3)
    ax.set_xticks(np.arange(len(GLOBAL_TOLS)))
    ax.set_xticklabels([tag(v) for v in GLOBAL_TOLS], fontsize=8)
    ax.set_yticks(np.arange(len(LOCAL_TOLS)))
    ax.set_yticklabels([tag(v) for v in LOCAL_TOLS], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(GLOBAL_TOLS) + 0.5, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LOCAL_TOLS) + 0.5, 1.0), minor=True)
    ax.grid(which='minor', color='white', lw=0.8, alpha=0.7)
    ax.tick_params(which='minor', length=0)
    ax.set_xlim(-0.5, len(GLOBAL_TOLS) - 0.5)
    ax.set_ylim(-0.5, len(LOCAL_TOLS) - 0.5)
    ax.set_xlabel('global $\\tau$', fontsize=10)
    ax.set_ylabel('local $\\tau$', fontsize=10)
    ax.set_title(name, fontsize=11)
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(f'final {ylabel}', fontsize=8)
    cb.ax.tick_params(labelsize=7)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for group, entries in GROUPS:
        n = len(entries)
        fig = plt.figure(figsize=(5 * n + 0.7 * n, 4))
        gs = fig.add_gridspec(1, 2 * n, width_ratios=[1, 0.055] * n, wspace=0.42)
        for k, (name, _, _, ylabel) in enumerate(entries):
            ax = fig.add_subplot(gs[0, 2 * k])
            cax = fig.add_subplot(gs[0, 2 * k + 1])
            panel(ax, cax, name, ylabel)
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.14, top=0.90)
        for ext in ['pdf', 'png']:
            fig.savefig(OUT / f'new_tolerance_{group}.{ext}', bbox_inches='tight',
                        dpi=150 if ext == 'png' else None)
        plt.close(fig)
        print('wrote', OUT / f'new_tolerance_{group}.pdf')


if __name__ == '__main__':
    main()
