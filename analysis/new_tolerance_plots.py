"""Legacy-style tolerance plots from the 5x5 grid (N_val=2, seed 0).

One figure per dataset/property (CH3Cl, CH3F, SCF, EV, ANI), 10x4:
  * 3D: x = log10(total time cost), y = local tolerance, z = log10(error),
    one line per (global, local) combination, coloured by global tolerance;
    the log-time axis is the near horizontal axis, as in the legacy figure;
  * scatter: global vs local tolerance, coloured by the final error.

Writes Manuscript/publishable_plots/new_tolerance_<name>.{pdf,png}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

ENTRIES = [
    ('CH3Cl', VIB5, VIB5[-1], 'MAE [kcal/mol]'),
    ('CH3F', VIB5, VIB5[-1], 'MAE [kcal/mol]'),
    ('SCF', QEMFI, Q5[4], 'MAE [kcal/mol]'),
    ('EV', QEMFI, Q5[4], 'MAE [kcal/mol]'),
    ('ANI', ANI, ANI[-1], 'RMAE'),
]


def tag(v):
    return f'{v:g}'


def curve(name, local, glob, cost_vector, val_cost):
    prefix = GRID / name / f'local{tag(local)}_global{tag(glob)}' / '0'
    mae = np.load(prefix.with_name(prefix.name + '_mae.npy'), allow_pickle=True).astype(float)
    sizes = np.load(prefix.with_name(prefix.name + '_trainsizes.npy'), allow_pickle=True).astype(int)
    cost = np.asarray([cost_vector @ s for s in sizes]) + VAL_SIZE * val_cost
    order = np.argsort(cost)
    return cost[order], mae[order]


def panel_3d(ax, cax, name, cost_vector, val_cost, ylabel):
    norm = Normalize(vmin=np.log10(min(GLOBAL_TOLS)), vmax=np.log10(max(GLOBAL_TOLS)))
    cmap = plt.cm.viridis
    for g in GLOBAL_TOLS:
        col = cmap(norm(np.log10(g)))
        for i, l in enumerate(LOCAL_TOLS):
            c, m = curve(name, l, g, cost_vector, val_cost)
            ax.plot(np.log10(c), np.full_like(c, i), np.log10(m),
                    color=col, lw=0.7, alpha=0.85)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks(np.arange(len(LOCAL_TOLS)))
    ax.set_yticklabels([tag(v) for v in LOCAL_TOLS], fontsize=7)
    ax.set_xlabel(r'$\log_{10}T$ [h]', fontsize=8, labelpad=6)
    ax.set_ylabel('local $\\tau$', fontsize=8, labelpad=2)
    ax.set_zlabel('log10 ' + ylabel.split()[0], fontsize=8, labelpad=3)
    ax.set_title(name, fontsize=10)
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect((1.6, 1.0, 1.0))
    ax.tick_params(labelsize=7, pad=0)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(sm, cax=cax)
    cb.set_ticks(np.log10(GLOBAL_TOLS))
    cb.set_ticklabels([tag(v) for v in GLOBAL_TOLS])
    cb.set_label('global $\\tau$', fontsize=8)
    cb.ax.tick_params(labelsize=7)


def panel_scatter(ax, cax, name, cost_vector, val_cost, ylabel):
    xs, ys, zs = [], [], []
    for g in GLOBAL_TOLS:
        for l in LOCAL_TOLS:
            c, m = curve(name, l, g, cost_vector, val_cost)
            xs.append(g); ys.append(l); zs.append(m[-1])
    sc = ax.scatter(xs, ys, c=zs, cmap='viridis', s=65, edgecolors='black', zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(GLOBAL_TOLS); ax.set_xticklabels([tag(v) for v in GLOBAL_TOLS], fontsize=7)
    ax.set_yticks(LOCAL_TOLS); ax.set_yticklabels([tag(v) for v in LOCAL_TOLS], fontsize=7)
    ax.set_xlabel('global $\\tau$', fontsize=9)
    ax.set_ylabel('local $\\tau$', fontsize=9)
    ax.set_title(f'{name}: final {ylabel}', fontsize=10)
    ax.grid(True, which='both', ls='--', alpha=0.35)
    cb = plt.colorbar(sc, cax=cax)
    cb.set_label(f'final {ylabel}', fontsize=8)
    cb.ax.tick_params(labelsize=7)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, cost_vector, val_cost, ylabel in ENTRIES:
        fig = plt.figure(figsize=(10, 4))
        ax3d = fig.add_axes([0.03, 0.17, 0.45, 0.75], projection='3d')
        ax2d = fig.add_axes([0.57, 0.17, 0.30, 0.73])
        cax3 = fig.add_axes([0.92, 0.55, 0.012, 0.30])
        cax2 = fig.add_axes([0.92, 0.15, 0.012, 0.30])
        panel_3d(ax3d, cax3, name, cost_vector, val_cost, ylabel)
        panel_scatter(ax2d, cax2, name, cost_vector, val_cost, ylabel)
        for ext in ['pdf', 'png']:
            fig.savefig(OUT / f'new_tolerance_{name}.{ext}', bbox_inches='tight',
                        dpi=150 if ext == 'png' else None)
        plt.close(fig)
        print('wrote', OUT / f'new_tolerance_{name}.pdf')


if __name__ == '__main__':
    main()
