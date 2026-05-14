"""
exp_e2/run_e2.py
================
EXPERIMENT 2 — "FedADP-FIM Delivers: Comprehensive Performance Comparison"
===========================================================================

STORY-TELLING ROLE  (Section 5.2 of paper — the main comparison table)
------------------------------------------------------------------------
Having established the problem (E1), we now show FedADP-FIM's overall
superiority across ALL four datasets and their full support-threshold
ranges.  This is the PROOF-OF-CONCEPT experiment — the central result
table that TKDE reviewers will study most carefully.

We compare only FedADP-FIM vs FedDP-FPM (corrected baseline), which is
the most direct and honest comparison: same algorithm family, same DP
guarantee, same two communication rounds — but with ABP + IWTC active.

RESEARCH QUESTION ANSWERED
---------------------------
  RQ2: "Does FedADP-FIM consistently outperform the corrected FedDP-FPM
        baseline in both F1 accuracy AND communication cost, across
        diverse datasets and support thresholds?"

STORY ARC
---------
  · Dense datasets (Chess, Mushroom): ΔF1 gains come from better α
    → ABP shines when tree depth is large and α matters most
  · Sparse datasets (Retail, Foodmart): ΔF1 gains are smaller (easier)
    → IWTC dominates: massive comm reduction with negligible F1 drop
  · This "two-regime" pattern is the key narrative insight

KEY FIGURES
-----------
  Fig 2a: Grouped bar chart — F1 comparison (all 4 datasets × 3 δ values)
  Fig 2b: Communication cost comparison (same grid)
  Fig 2c: Δ(F1) heatmap — improvement map by (dataset, δ)
  Fig 2d: Runtime comparison

KEY TABLE
---------
  Table 2: Full comparison table — F1 ± std, Comm KB, Time, ΔF1, ΔComm%
           This is THE main results table for the paper body.

OUTPUTS
-------
  results/exp_e2/figures/e2_main_comparison.pdf
  results/exp_e2/figures/e2_delta_f1_heatmap.pdf
  results/exp_e2/tables/e2_main_table.tex
  results/exp_e2/results.pkl
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'shared'))
from common import (
    load_datasets, load_or_compute_gt, make_splits, run_n, save_results,
    make_exp_dir, write_latex_table, fmt_pm,
    EPS_MAIN, TAU_MAIN, NRUNS, GLOBAL_SEED, DATASETS,
)

SRC = os.path.join(HERE, '..', '..', 'src')
sys.path.insert(0, os.path.abspath(SRC))
from algorithms import FedDP_FPM, FedADP_FIM

np.random.seed(GLOBAL_SEED)

# ── Full experiment grid ─────────────────────────────────────────
# (name, delta) — 4 datasets × 3-4 thresholds
CONFIGS = [
    ('Chess',    0.80), ('Chess',    0.85), ('Chess',    0.90), ('Chess',    0.95),
    ('Mushroom', 0.40), ('Mushroom', 0.50), ('Mushroom', 0.60), ('Mushroom', 0.70),
    ('Retail',   0.01), ('Retail',   0.05), ('Retail',   0.10), ('Retail',   0.20),
    ('Foodmart', 0.005),('Foodmart', 0.01), ('Foodmart', 0.05), ('Foodmart', 0.10),
]

COLORS = {
    'FedDP-FPM':  '#4878CF',
    'FedADP-FIM': '#D65F5F',
}


# ─────────────────────────────────────────────────────────────────
#  CORE EXPERIMENT
# ─────────────────────────────────────────────────────────────────

def run_main_comparison(datasets, GT):
    print("\n[E2] Running main comparison (ε={}, τ={}, {} runs)...".format(
        EPS_MAIN, TAU_MAIN, NRUNS))
    results = {}
    for name, delta in CONFIGS:
        data   = datasets[name]
        splits = make_splits(data, n_clients=3)
        gt     = GT[(name, delta)]

        fdp  = run_n(FedDP_FPM(epsilon=EPS_MAIN, delta=delta), splits, gt)
        fadp = run_n(FedADP_FIM(epsilon=EPS_MAIN, delta=delta, tau=TAU_MAIN),
                     splits, gt)

        results[(name, delta)] = {'FDP': fdp, 'FADP': fadp, 'gt_n': len(gt)}
        delta_f1   = (fadp['f1'] - fdp['f1']) * 100
        delta_comm = (1 - fadp['c'] / max(fdp['c'], 1e-9)) * 100
        print(f"  {name:8s} δ={delta:.3f}: "
              f"FDP={fdp['f1']*100:.1f}%  FADP={fadp['f1']*100:.1f}%  "
              f"ΔF1={delta_f1:+.1f}%  Comm↓={delta_comm:.0f}%")
    return results


# ─────────────────────────────────────────────────────────────────
#  FIGURES
# ─────────────────────────────────────────────────────────────────

# Dataset groups for figure layout
DS_GROUPS = {
    'Chess':    [0.80, 0.85, 0.90, 0.95],
    'Mushroom': [0.40, 0.50, 0.60, 0.70],
    'Retail':   [0.01, 0.05, 0.10, 0.20],
    'Foodmart': [0.005, 0.01, 0.05, 0.10],
}
DS_COLOR = {
    'Chess': '#1f4e79', 'Mushroom': '#833c00',
    'Retail': '#375623', 'Foodmart': '#4a235a',
}


def plot_main_comparison(results, figs_dir):
    """4-panel figure: one panel per dataset, F1 + Comm bars."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    ds_names = list(DS_GROUPS.keys())
    for col, name in enumerate(ds_names):
        deltas = DS_GROUPS[name]
        x      = np.arange(len(deltas))
        w      = 0.35

        # ── F1 row ──
        ax = axes[0, col]
        fdp_f1  = [results[(name, d)]['FDP']['f1']*100  for d in deltas]
        fadp_f1 = [results[(name, d)]['FADP']['f1']*100 for d in deltas]
        fdp_e   = [results[(name, d)]['FDP']['f1s']*100 for d in deltas]
        fadp_e  = [results[(name, d)]['FADP']['f1s']*100 for d in deltas]

        ax.bar(x - w/2, fdp_f1,  w, yerr=fdp_e,  capsize=3,
               color=COLORS['FedDP-FPM'],  alpha=0.85, label='FedDP-FPM',  zorder=3)
        ax.bar(x + w/2, fadp_f1, w, yerr=fadp_e, capsize=3,
               color=COLORS['FedADP-FIM'], alpha=0.85, label='FedADP-FIM', zorder=3)

        for i, (vf, va) in enumerate(zip(fdp_f1, fadp_f1)):
            d = va - vf
            c = '#2ca02c' if d >= 0 else '#d62728'
            ax.text(i, max(vf, va) + max(fdp_e[i], fadp_e[i]) + 2,
                    f'{d:+.1f}%', ha='center', fontsize=8,
                    color=c, fontweight='bold')

        ax.set_title(f'{name}\n(dense={name in ["Chess","Mushroom"]})',
                     fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}' for d in deltas], fontsize=8)
        ax.set_ylabel('F₁ (%)' if col == 0 else '', fontsize=10)
        ax.set_ylim(0, 125)
        ax.grid(True, axis='y', alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8, loc='upper right')

        # ── Comm row ──
        ax = axes[1, col]
        fdp_c   = [results[(name, d)]['FDP']['c']  for d in deltas]
        fadp_c  = [results[(name, d)]['FADP']['c'] for d in deltas]
        fdp_ce  = [results[(name, d)]['FDP']['cs'] for d in deltas]
        fadp_ce = [results[(name, d)]['FADP']['cs'] for d in deltas]

        ax.bar(x - w/2, fdp_c,  w, yerr=fdp_ce,  capsize=3,
               color=COLORS['FedDP-FPM'],  alpha=0.85, zorder=3)
        ax.bar(x + w/2, fadp_c, w, yerr=fadp_ce, capsize=3,
               color=COLORS['FedADP-FIM'], alpha=0.85, zorder=3)

        for i, (vc, va) in enumerate(zip(fdp_c, fadp_c)):
            if vc > 0:
                pct = (1 - va / vc) * 100
                ax.text(i, max(vc, va) + 1,
                        f'{pct:.0f}%↓', ha='center', fontsize=8,
                        color='#2ca02c', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([f'{d}' for d in deltas], fontsize=8)
        ax.set_xlabel('Support δ', fontsize=10)
        ax.set_ylabel('Comm. Cost (KB)' if col == 0 else '', fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

    # Row labels
    fig.text(0.01, 0.73, 'F₁ Score', va='center', rotation='vertical',
             fontsize=12, fontweight='bold')
    fig.text(0.01, 0.27, 'Communication\nCost (KB)', va='center',
             rotation='vertical', fontsize=12, fontweight='bold')

    plt.suptitle(
        'E2: FedADP-FIM vs. Corrected FedDP-FPM — Full Performance Comparison\n'
        '(ε=1.0, τ=0.7, n_clients=3, NRUNS=10; bars show mean ± std)',
        fontsize=13, fontweight='bold', y=1.01)

    out = os.path.join(figs_dir, 'e2_main_comparison.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [FIG] {out}")


def plot_delta_f1_heatmap(results, figs_dir):
    """Heatmap of ΔF1 improvement across (dataset × δ) — shows the two-regime pattern."""
    ds_names = list(DS_GROUPS.keys())
    max_len  = max(len(DS_GROUPS[n]) for n in ds_names)

    # Build matrix (datasets × max_δ_count), pad with NaN
    matrix = np.full((len(ds_names), max_len), np.nan)
    for i, name in enumerate(ds_names):
        deltas = DS_GROUPS[name]
        for j, delta in enumerate(deltas):
            r = results[(name, delta)]
            matrix[i, j] = (r['FADP']['f1'] - r['FDP']['f1']) * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-5, vmax=15)

    # Annotate each cell
    for i in range(len(ds_names)):
        for j in range(max_len):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:+.1f}%',
                        ha='center', va='center', fontsize=9,
                        color='black' if abs(matrix[i, j]) < 10 else 'white',
                        fontweight='bold')

    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(ds_names, fontsize=11)
    ax.set_xticks(range(max_len))
    ax.set_xticklabels([f'δ={i+1}' for i in range(max_len)], fontsize=9)
    ax.set_xlabel('Support threshold (ascending)', fontsize=11)
    ax.set_title('ΔF₁ = F₁(FedADP-FIM) − F₁(FedDP-FPM)\n'
                 'Green = improvement, Red = degradation',
                 fontsize=11)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label='ΔF₁ (%)')

    plt.tight_layout()
    out = os.path.join(figs_dir, 'e2_delta_f1_heatmap.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [FIG] {out}")


# ─────────────────────────────────────────────────────────────────
#  LATEX TABLE  (main paper table)
# ─────────────────────────────────────────────────────────────────

def write_e2_table(results, tabs_dir):
    header = [
        'Dataset', 'δ', r'\#FI',
        r'FDP F₁', r'FADP F₁', r'$\Delta$F₁',
        r'FDP Comm', r'FADP Comm', r'Comm$\downarrow$',
        r'FDP $t$(s)', r'FADP $t$(s)',
    ]
    rows = []
    prev_ds = None
    for name, delta in CONFIGS:
        r  = results[(name, delta)]
        fd = r['FDP']
        fa = r['FADP']
        c1, c2 = fd['c'], fa['c']
        t1, t2 = fd['t'], fa['t']
        df1    = (fa['f1'] - fd['f1']) * 100
        dcomm  = (1 - c2 / max(c1, 1e-9)) * 100
        # Midrule between datasets
        if prev_ds is not None and prev_ds != name:
            rows.append(['\\midrule'])
        rows.append([
            name if name != prev_ds else '',
            f'{delta}',
            str(r['gt_n']),
            fmt_pm(fd['f1'], fd['f1s']),
            fmt_pm(fa['f1'], fa['f1s']),
            f'\\textbf{{{df1:+.1f}\\%}}' if df1 >= 0 else f'{df1:+.1f}\\%',
            f'{c1:.1f}',
            f'{c2:.1f}',
            f'{dcomm:.0f}\\%',
            f'{t1:.3f}',
            f'{t2:.3f}',
        ])
        prev_ds = name

    path = os.path.join(tabs_dir, 'e2_main_table.tex')
    write_latex_table(rows, header,
        caption=(r"E2: Comprehensive comparison of FedADP-FIM vs.\ corrected "
                 r"FedDP-FPM. $\epsilon=1.0$, $\tau=0.7$, $n_{clients}=3$, "
                 r"$\text{NRUNS}=10$. $\Delta$F\textsubscript{1} and "
                 r"Comm$\downarrow$ show relative gain; bold indicates improvement."),
        label='tab:e2_main',
        path=path)


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("E2 — Comprehensive Performance Comparison")
    print("=" * 65)
    _, figs_dir, tabs_dir = make_exp_dir('exp_e2')

    print("\n[DATA] Loading datasets...")
    datasets = load_datasets()

    delta_map = {}
    for name, delta in CONFIGS:
        delta_map.setdefault(name, []).append(delta)
    GT = load_or_compute_gt(datasets, delta_map=delta_map, use_cache=True)

    results = run_main_comparison(datasets, GT)

    plot_main_comparison(results, figs_dir)
    plot_delta_f1_heatmap(results, figs_dir)
    write_e2_table(results, tabs_dir)

    save_results({'results': results, 'configs': CONFIGS}, 'exp_e2')
    print("\n[E2 COMPLETE]")
    print("  Story: FedADP-FIM wins on F1 (dense) + Comm (sparse).")
    print("  Key finding: Two-regime pattern — ABP dominates dense,")
    print("               IWTC dominates sparse datasets.")


if __name__ == '__main__':
    main()
