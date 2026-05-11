"""
run_experiments.py
==================
Full experimental pipeline for FedADP-FIM paper.

Usage
-----
# All experiments (default):
  python scripts/run_experiments.py

# Single experiment:
  python scripts/run_experiments.py --exp e1
  python scripts/run_experiments.py --exp e2
  python scripts/run_experiments.py --exp all

# Custom epsilon:
  python scripts/run_experiments.py --exp e1 --epsilon 1.0

# Skip GT recomputation (use cache):
  python scripts/run_experiments.py --use_cache

Outputs
-------
results/gt_cache.pkl        — cached ground truth (skip slow recomputation)
results/exp_results.pkl     — all experiment results
results/figures/results.png — publication-quality figure
results/tables/summary.csv  — LaTeX-ready CSV table
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# ── Add src/ to path ──────────────────────────────────────────
SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(SRC))

from src.algorithms import FedDP_FPM, FedADP_FIM, DP_PartFIM_Simple
from src.data_utils  import load_spmf, split_non_iid, compute_gt, compute_f1_score

# ── Paths ─────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'results')
FIGS    = os.path.join(RESULTS, 'figures')
TABS    = os.path.join(RESULTS, 'tables')
for d in [RESULTS, FIGS, TABS]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────
#  EXPERIMENT CONFIG
# ─────────────────────────────────────────────────────────────

CFGS = [
    ('Chess',    'chess.txt',    0.60, 3),
    ('Chess',    'chess.txt',    0.85, 3),
    ('Chess',    'chess.txt',    0.90, 3),
    ('Mushroom', 'mushroom.txt', 0.40, 3),
    ('Mushroom', 'mushroom.txt', 0.50, 3),
    ('Mushroom', 'mushroom.txt', 0.60, 3),
    ('Retail',    'retail.txt',    0.01, 3),
    ('Retail',    'retail.txt',    0.05, 3),
    ('Retail',    'retail.txt',    0.15, 3),
    ('Foodmart',    'fruithut.txt',    0.05, 3),
    ('Foodmart',    'fruithut.txt',    0.01, 3),
    ('Foodmart',    'fruithut.txt',    0.005, 3),
]
EPSILONS  = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]
TAUS      = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_CLIENTS = [3, 5, 10]
EPS_MAIN  = 1.0
TAU_MAIN  = 0.7
NRUNS     = 10


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def run_once(algo, splits, gt):
    pred, st = algo.run(splits)
    _, _, f  = compute_f1_score(pred, gt)
    return f, st['comm_mb'] * 1024, st['mem_client_mb'] * 1024, st['time']

def run_n(algo, splits, gt, n=NRUNS):
    F, C, M, T = [], [], [], []
    for _ in range(n):
        f, c, m, t = run_once(algo, splits, gt)
        F.append(f); C.append(c); M.append(m); T.append(t)
    return {'f1': np.mean(F), 'f1s': np.std(F),
            'c':  np.mean(C), 'cs':  np.std(C),
            'm':  np.mean(M), 'ms':  np.std(M),
            't':  np.mean(T), 'ts':  np.std(T)}


# ─────────────────────────────────────────────────────────────
#  GROUND TRUTH  (with caching)
# ─────────────────────────────────────────────────────────────

def load_or_compute_gt(datasets: dict, use_cache: bool = True) -> dict:
    cache_path = os.path.join(RESULTS, 'gt_cache.pkl')

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            GT = pickle.load(f)
        print(f"[GT] Loaded cache from {cache_path}")
        # Check all keys exist
        missing = [(n, d) for n, _, d, _ in CFGS if (n, d) not in GT]
        if not missing:
            return GT
        print(f"[GT] Missing keys: {missing} — will recompute those")
    else:
        GT = {}

    for name, fname, delta, _ in CFGS:
        key = (name, delta)
        if key in GT:
            continue
        data   = datasets[fname]
        splits = split_non_iid(data)
        all_t  = [t for s in splits for t in s]
        print(f"\n[GT] Computing {name} δ={delta}  n={len(all_t)} ...")
        gt = compute_gt(all_t, delta, max_time=120.0, verbose=True)
        GT[key] = gt

    with open(cache_path, 'wb') as f:
        pickle.dump(GT, f)
    print(f"\n[GT] Saved to {cache_path}")
    return GT


# ─────────────────────────────────────────────────────────────
#  EXPERIMENT RUNNERS
# ─────────────────────────────────────────────────────────────

def run_e1(datasets, GT):
    """E1: Main comparison table (ε=EPS_MAIN, τ=TAU_MAIN)."""
    print(f"\n{'='*70}")
    print(f"E1: Main Comparison  (ε={EPS_MAIN}, τ={TAU_MAIN}, {NRUNS} runs)")
    print(f"{'='*70}")
    E1 = {}
    for name, fname, delta, nc in CFGS:
        splits = split_non_iid(datasets[fname], n_clients=nc)
        gt     = GT[(name, delta)]
        row    = {}
        for tag, cls, kw in [
            ('FDP',  FedDP_FPM,         dict(epsilon=EPS_MAIN, delta=delta)),
            ('FADP', FedADP_FIM,         dict(epsilon=EPS_MAIN, delta=delta, tau=TAU_MAIN)),
            ('PART', DP_PartFIM_Simple,  dict(epsilon=EPS_MAIN, delta=delta)),
        ]:
            row[tag] = run_n(cls(**kw), splits, gt)
        E1[(name, delta)] = row
        e   = row
        df  = (e['FADP']['f1'] - e['FDP']['f1']) * 100
        dc  = (1 - e['FADP']['c'] / max(e['FDP']['c'], 1e-9)) * 100
        print(f"  {name} δ={delta}  GT={len(gt):5d} | "
              f"FDP={e['FDP']['f1']*100:.1f}%  FADP={e['FADP']['f1']*100:.1f}%  "
              f"PART={e['PART']['f1']*100:.1f}% | "
              f"ΔF1={df:+.1f}%  Comm↓={dc:.1f}%")
    return E1


def run_e2(datasets, GT):
    """E2: ε sensitivity."""
    print(f"\n{'='*70}")
    print("E2: Epsilon Sensitivity")
    print(f"{'='*70}")
    E2 = {}
    for name, fname, delta in [('Chess', 'chess.txt', 0.85),
                                 ('Mushroom', 'mushroom.txt', 0.50)]:
        splits = split_non_iid(datasets[fname])
        gt     = GT[(name, delta)]
        E2[(name, delta)] = {k: [] for k in ['fdp', 'fadp', 'part']}
        for eps in EPSILONS:
            for tag, cls, kw in [
                ('fdp',  FedDP_FPM,        dict(epsilon=eps, delta=delta)),
                ('fadp', FedADP_FIM,        dict(epsilon=eps, delta=delta, tau=TAU_MAIN)),
                ('part', DP_PartFIM_Simple, dict(epsilon=eps, delta=delta)),
            ]:
                r = run_n(cls(**kw), splits, gt, n=3)
                E2[(name, delta)][tag].append(r['f1'])
        vals = ' | '.join(
            f"ε={e}: {E2[(name,delta)]['fdp'][i]*100:.0f}/{E2[(name,delta)]['fadp'][i]*100:.0f}"
            for i, e in enumerate(EPSILONS)
        )
        print(f"  {name}: {vals}")
    return E2


def run_e3(datasets, GT):
    """E3: τ sensitivity (IWTC compression ratio)."""
    print(f"\n{'='*70}")
    print("E3: τ Sensitivity (IWTC compression ratio)")
    print(f"{'='*70}")
    E3 = {}
    for name, fname, delta in [('Chess', 'chess.txt', 0.85),
                                 ('Mushroom', 'mushroom.txt', 0.50)]:
        splits   = split_non_iid(datasets[fname])
        gt       = GT[(name, delta)]
        fdp_ref  = run_n(FedDP_FPM(epsilon=EPS_MAIN, delta=delta), splits, gt, n=3)
        E3[(name, delta)] = {
            'f1': [], 'comm': [], 'mem': [],
            'fdp_f1': fdp_ref['f1'] * 100,
            'fdp_comm': fdp_ref['c'],
        }
        for tau in TAUS:
            r = run_n(FedADP_FIM(epsilon=EPS_MAIN, delta=delta, tau=tau), splits, gt, n=3)
            E3[(name, delta)]['f1'].append(r['f1'] * 100)
            E3[(name, delta)]['comm'].append(r['c'])
            E3[(name, delta)]['mem'].append(r['m'])
        best = TAUS[int(np.argmax(E3[(name, delta)]['f1']))]
        print(f"  {name}: best τ={best}  "
              f"F1={max(E3[(name,delta)]['f1']):.1f}%  "
              f"(FDP ref={fdp_ref['f1']*100:.1f}%)")
    return E3


def run_e4(datasets, GT):
    """E4: Client scalability."""
    print(f"\n{'='*70}")
    print("E4: Client Scalability")
    print(f"{'='*70}")
    gt_c = GT[('Chess', 0.85)]
    E4 = {k: [] for k in ['fdp_c', 'fadp_c', 'fdp_f1', 'fadp_f1']}
    for nc in N_CLIENTS:
        ratios = [1.0 / nc] * nc
        sp     = split_non_iid(datasets['chess.txt'], ratios=ratios, n_clients=nc)
        for tag, cls, kw in [
            ('fdp',  FedDP_FPM,  dict(epsilon=EPS_MAIN, delta=0.85)),
            ('fadp', FedADP_FIM, dict(epsilon=EPS_MAIN, delta=0.85, tau=TAU_MAIN)),
        ]:
            r = run_n(cls(**kw), sp, gt_c, n=3)
            E4[f'{tag}_c'].append(r['c'])
            E4[f'{tag}_f1'].append(r['f1'] * 100)
        dc = (1 - E4['fadp_c'][-1] / max(E4['fdp_c'][-1], 1e-9)) * 100
        print(f"  n={nc}: FDP={E4['fdp_f1'][-1]:.1f}%  "
              f"FADP={E4['fadp_f1'][-1]:.1f}%  Comm↓={dc:.1f}%")
    return E4


def run_e5(datasets, GT):
    """E5: Global FI Loss Rate (GLoss)."""
    print(f"\n{'='*70}")
    print("E5: Global FI Loss Rate (GLoss)")
    print(f"{'='*70}")
    E5 = {}
    for name, fname, delta in [('Chess', 'chess.txt', 0.85),
                                 ('Mushroom', 'mushroom.txt', 0.50)]:
        splits = split_non_iid(datasets[fname])
        gt     = GT[(name, delta)]
        # Local loss per client
        local_losses = []
        for ds in splits:
            nd = len(ds)
            lc = defaultdict(int)
            for t in ds:
                for it in set(t):
                    lc[it] += 1
            lf1 = {it for it, c in lc.items() if c / nd >= delta}
            miss = sum(1 for fs in gt if not all(it in lf1 for it in fs))
            local_losses.append(miss / max(len(gt), 1))
        p_fdp, _  = FedDP_FPM(epsilon=EPS_MAIN, delta=delta).run(splits)
        p_fadp, _ = FedADP_FIM(epsilon=EPS_MAIN, delta=delta, tau=TAU_MAIN).run(splits)
        E5[(name, delta)] = {
            'local': local_losses,
            'fdp':   len(gt - p_fdp)  / max(len(gt), 1),
            'fadp':  len(gt - p_fadp) / max(len(gt), 1),
            'gt_n':  len(gt),
        }
        d = E5[(name, delta)]
        print(f"  {name}: local={[f'{l*100:.0f}%' for l in d['local']]}  "
              f"FDP={d['fdp']*100:.1f}%  FADP={d['fadp']*100:.1f}%")
    return E5


# ─────────────────────────────────────────────────────────────
#  FIGURE
# ─────────────────────────────────────────────────────────────

def make_figure(E1, E2, E3, E4, E5, GT):
    import math

    BG    = '#0B0F1A'; PANEL = '#131929'; GRID  = '#1E2A3A'; TEXT = '#C8D8F0'
    C_FDP = '#3B9CF5'; C_FADP= '#FF6B35'; C_PART= '#56C596'; GOLD = '#FFD700'

    fig = plt.figure(figsize=(22, 24))
    fig.patch.set_facecolor(BG)
    gs  = GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.34,
                   left=0.07, right=0.97, top=0.94, bottom=0.04)

    def sa(ax, title, xl='', yl=''):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8.5)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.set_title(title, color='#E8F0FF', fontsize=10.5,
                     fontweight='bold', pad=8)
        ax.set_xlabel(xl, fontsize=8.5); ax.set_ylabel(yl, fontsize=8.5)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.8)

    CFGS_K = [('Chess', 0.85), ('Chess', 0.90),
              ('Mushroom', 0.40), ('Mushroom', 0.50), ('Mushroom', 0.60)]
    NAMES  = [f'{n}\nδ={d}' for n, d in CFGS_K]
    x = np.arange(len(CFGS_K)); W = 0.26

    # Row 0: F1, Comm, Memory
    ax = fig.add_subplot(gs[0, 0]); sa(ax, 'F1 Score  (ε=1.0, τ=0.7)', 'Dataset', 'F1 (%)')
    vf = [E1[k]['FDP']['f1'] * 100 for k in CFGS_K]
    va = [E1[k]['FADP']['f1'] * 100 for k in CFGS_K]
    vp = [E1[k]['PART']['f1'] * 100 for k in CFGS_K]
    ax.bar(x - W, vf, W, color=C_FDP, alpha=0.88, label='FedDP-FPM', zorder=3)
    ax.bar(x,     va, W, color=C_FADP, alpha=0.88, label='FedADP-FIM', zorder=3)
    ax.bar(x + W, vp, W, color=C_PART, alpha=0.88, label='DP-PartFIM', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=8); ax.set_ylim(0, 120)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    for xi, (f, a) in enumerate(zip(vf, va)):
        d = a - f; col = '#76FF03' if d >= 0 else '#FF6B6B'
        ax.text(xi, max(f, a) + 3.5, f'{d:+.1f}%',
                ha='center', fontsize=7.5, color=col, fontweight='bold')

    ax = fig.add_subplot(gs[0, 1]); sa(ax, 'Communication Cost  (ε=1.0)', 'Dataset', 'KB')
    vc  = [E1[k]['FDP']['c']  for k in CFGS_K]
    vca = [E1[k]['FADP']['c'] for k in CFGS_K]
    vcp = [E1[k]['PART']['c'] for k in CFGS_K]
    ax.bar(x - W, vc, W, color=C_FDP, alpha=0.88, label='FedDP-FPM', zorder=3)
    ax.bar(x,    vca, W, color=C_FADP, alpha=0.88, label='FedADP-FIM', zorder=3)
    ax.bar(x + W,vcp, W, color=C_PART, alpha=0.88, label='DP-PartFIM', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=8)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    for xi, (c, ca) in enumerate(zip(vc, vca)):
        if c > 0:
            dc = (1 - ca / c) * 100
            ax.text(xi, max(c, ca) * 1.09, f'{dc:+.0f}%', ha='center',
                    fontsize=7.5, color='#76FF03' if dc > 0 else '#FF6B6B',
                    fontweight='bold')

    ax = fig.add_subplot(gs[0, 2]); sa(ax, 'Peak Client Memory  (ε=1.0)', 'Dataset', 'KB')
    vm  = [E1[k]['FDP']['m']  for k in CFGS_K]
    vma = [E1[k]['FADP']['m'] for k in CFGS_K]
    ax.bar(x - W/2, vm,  W, color=C_FDP,  alpha=0.88, label='FedDP-FPM', zorder=3)
    ax.bar(x + W/2, vma, W, color=C_FADP, alpha=0.88, label='FedADP-FIM', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=8)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    for xi, (m, ma) in enumerate(zip(vm, vma)):
        if m > 0:
            dm = (1 - ma / m) * 100
            ax.text(xi, max(m, ma) * 1.09, f'{dm:+.0f}%', ha='center',
                    fontsize=7.5, color='#76FF03' if dm > 0 else '#FF6B6B',
                    fontweight='bold')

    # Row 1: eps sensitivity x2 + runtime
    for col, (name, delta) in enumerate([('Chess', 0.85), ('Mushroom', 0.50)]):
        ax = fig.add_subplot(gs[1, col])
        sa(ax, f'ε Sensitivity — {name} δ={delta}', 'ε (log scale)', 'F1 (%)')
        d = E2[(name, delta)]
        ax.plot(EPSILONS, [v * 100 for v in d['fdp']],
                'o-', color=C_FDP,  lw=2.2, ms=7, label='FedDP-FPM')
        ax.plot(EPSILONS, [v * 100 for v in d['fadp']],
                's-', color=C_FADP, lw=2.2, ms=7, label='FedADP-FIM')
        ax.plot(EPSILONS, [v * 100 for v in d['part']],
                '^--', color=C_PART, lw=1.8, ms=6, label='DP-PartFIM')
        ax.set_xscale('log'); ax.set_ylim(0, 110)
        fadp_a = np.array([v * 100 for v in d['fadp']])
        fdp_a  = np.array([v * 100 for v in d['fdp']])
        ax.fill_between(EPSILONS, fdp_a, fadp_a,
                        where=fadp_a >= fdp_a, alpha=0.12, color=C_FADP)
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

    ax = fig.add_subplot(gs[1, 2]); sa(ax, 'Runtime  (ε=1.0)', 'Dataset', 'Time (s)')
    vt  = [E1[k]['FDP']['t']  for k in CFGS_K]
    vta = [E1[k]['FADP']['t'] for k in CFGS_K]
    ax.bar(x - W/2, vt,  W, color=C_FDP,  alpha=0.88, label='FedDP-FPM', zorder=3)
    ax.bar(x + W/2, vta, W, color=C_FADP, alpha=0.88, label='FedADP-FIM', zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(NAMES, fontsize=8)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    for xi, (t1, t2) in enumerate(zip(vt, vta)):
        spd = t1 / max(t2, 1e-9)
        ax.text(xi, max(t1, t2) * 1.1, f'{spd:.1f}×',
                ha='center', fontsize=8,
                color=GOLD if spd > 1 else '#FF6B6B', fontweight='bold')

    # Row 2: tau x2 + scalability
    for col, (name, delta) in enumerate([('Chess', 0.85), ('Mushroom', 0.50)]):
        ax = fig.add_subplot(gs[2, col])
        sa(ax, f'τ Sensitivity — {name} δ={delta}', 'τ', 'F1 (%)')
        d = E3[(name, delta)]
        ax.plot(TAUS, d['f1'], 'D-', color=C_FADP, lw=2.5, ms=9,
                label='FedADP-FIM', zorder=4)
        ax.axhline(d['fdp_f1'], color=C_FDP, ls='--', lw=2,
                   label=f"FedDP-FPM ({d['fdp_f1']:.0f}%)")
        ax.set_xticks(TAUS)
        ax.set_ylim(0, max(max(d['f1']), d['fdp_f1']) * 1.35)
        best = TAUS[int(np.argmax(d['f1']))]
        ax.axvline(best, color=GOLD, ls=':', lw=1.8, alpha=0.8,
                   label=f'Sweet spot τ={best}')
        for tv, fv in zip(TAUS, d['f1']):
            ax.annotate(f'{fv:.0f}%', (tv, fv + 1.5),
                        ha='center', fontsize=8, color=TEXT)
        ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

    ax = fig.add_subplot(gs[2, 2])
    sa(ax, 'Client Scalability — Chess δ=0.85', '# Clients', 'KB')
    ax.plot(N_CLIENTS, E4['fdp_c'],  'o-', color=C_FDP,  lw=2.2, ms=9, label='FedDP-FPM')
    ax.plot(N_CLIENTS, E4['fadp_c'], 's-', color=C_FADP, lw=2.2, ms=9, label='FedADP-FIM')
    ax2 = ax.twinx()
    ax2.plot(N_CLIENTS, E4['fdp_f1'],  'o--', color=C_FDP,  lw=1.5, ms=6, alpha=0.6)
    ax2.plot(N_CLIENTS, E4['fadp_f1'], 's--', color=C_FADP, lw=1.5, ms=6, alpha=0.6)
    ax2.set_ylabel('F1 (%)', fontsize=8, color=TEXT)
    ax2.tick_params(colors=TEXT, labelsize=8)
    ax2.spines['right'].set_edgecolor(GRID)
    ax.set_xticks(N_CLIENTS)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    for i, nc in enumerate(N_CLIENTS):
        dc = (1 - E4['fadp_c'][i] / max(E4['fdp_c'][i], 1e-9)) * 100
        ax.text(nc, max(E4['fdp_c'][i], E4['fadp_c'][i]) * 1.1,
                f'{dc:+.0f}%', ha='center', fontsize=8,
                color='#76FF03', fontweight='bold')

    # Row 3: GLoss + ABP theory + summary
    ax = fig.add_subplot(gs[3, 0]); sa(ax, 'GLoss — Missing FI Rate  (ε=1.0)', '', 'GLoss (%)')
    for ci, (name, delta) in enumerate([('Chess', 0.85), ('Mushroom', 0.50)]):
        d = E5[(name, delta)]
        lb   = [f'P{i+1}' for i in range(len(d['local']))] + ['FedDP', 'FedADP']
        vals = [l * 100 for l in d['local']] + [d['fdp'] * 100, d['fadp'] * 100]
        cols = ['#666'] * len(d['local']) + [C_FDP, C_FADP]
        xpos = np.arange(len(lb))
        for xb, v, c in zip(xpos + ci * 5, vals, cols):
            ax.bar(xb, v, 0.7, color=c, alpha=0.85, zorder=3)
            ax.text(xb, v + 0.5, f'{v:.0f}%', ha='center', fontsize=7, color=TEXT)
    ax.set_xlim(-0.6, 10)
    ax.text(1.5, -12, 'Chess δ=0.85', ha='center', fontsize=8, color=TEXT, style='italic')
    ax.text(7,   -12, 'Mushroom δ=0.5', ha='center', fontsize=8, color=TEXT, style='italic')

    ax = fig.add_subplot(gs[3, 1])
    sa(ax, 'ABP Theorem: Optimal α*(β, L)', 'Tree Depth L', 'α*')
    L_vals = np.linspace(2, 42, 100)
    for beta, col, lab in [
        (5,  C_FADP, 'β=5 (Chess-like)'),
        (10, GOLD,   'β=10'),
        (20, '#C080FF', 'β=20'),
    ]:
        alphas = [min(2.0, max(0.05, math.log(beta) / L)) if L > 5 else 0.30
                  for L in L_vals]
        ax.plot(L_vals, alphas, '-', color=col, lw=2.5, label=lab)
    ax.axhline(0.5, color=C_FDP, ls='--', lw=2, alpha=0.9,
               label='FedDP-FPM (fixed α=0.5)')
    ax.axvspan(0, 5, alpha=0.07, color='white')
    ax.text(2.5, 1.6, 'Chess\nL≈37', ha='center', fontsize=8, color=TEXT, alpha=0.8)
    ax.set_xlim(0, 42); ax.set_ylim(0, 2.1)
    ax.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)
    ax.text(0.63, 0.96,
            'Theorem: α* = ln(β)/L\nminimises total MSE\nunder budget constraint',
            transform=ax.transAxes, ha='center', va='top', fontsize=7.5,
            color='#C0D8FF',
            bbox=dict(facecolor='#0D1829', edgecolor=GRID, alpha=0.85, pad=4))

    ax = fig.add_subplot(gs[3, 2]); ax.set_facecolor(PANEL); ax.axis('off')
    ax.set_title('Summary Table  (ε=1.0, τ=0.7)', color='#E8F0FF',
                 fontsize=10, fontweight='bold', pad=8)
    cols_t = ['Config', 'GT', 'FDP\nF1', 'FADP\nF1', 'ΔF1', 'Comm\nFDP', 'Comm\nFADP', 'Comm↓']
    rows_t = []
    for name, delta in CFGS_K:
        e = E1[(name, delta)]
        c1 = e['FDP']['c']; c2 = e['FADP']['c']
        rows_t.append([
            f'{name} δ={delta}', str(len(GT[(name, delta)])),
            f"{e['FDP']['f1']*100:.1f}%", f"{e['FADP']['f1']*100:.1f}%",
            f"{(e['FADP']['f1']-e['FDP']['f1'])*100:+.1f}%",
            f'{c1:.0f}KB', f'{c2:.0f}KB',
            f'{(1-c2/max(c1,1e-9))*100:+.0f}%',
        ])
    tbl = ax.table(cellText=rows_t, colLabels=cols_t, loc='center',
                   cellLoc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if ri == 0:
            cell.set_facecolor('#112244')
            cell.set_text_props(color='#90C8FF', fontweight='bold')
        elif ci == 4:
            try:
                v = float(rows_t[ri - 1][4].replace('%', '').replace('+', ''))
                cell.set_facecolor('#0A2A0A' if v >= 0 else '#2A0A0A')
                cell.set_text_props(
                    color='#76FF03' if v >= 0 else '#FF6B6B', fontweight='bold')
            except Exception:
                pass
        else:
            cell.set_facecolor(PANEL if ri % 2 == 0 else '#0F1A2A')
            cell.set_text_props(color=TEXT)
        cell.set_height(0.17)

    plt.suptitle(
        'FedDP-FPM  vs  FedADP-FIM  vs  DP-PartFIM — Comprehensive Experimental Results\n'
        'Datasets: Chess & Mushroom (SPMF)   |   Non-IID Federated Setting   |   '
        'ABP + IWTC modules',
        fontsize=12.5, fontweight='bold', color='white', y=0.975)

    out = os.path.join(FIGS, 'results.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=BG, edgecolor='none')
    print(f"\n[FIG] Saved → {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────
#  CSV TABLE
# ─────────────────────────────────────────────────────────────

def save_csv(E1, GT):
    import csv
    path = os.path.join(TABS, 'summary.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Config', 'GT', 'FDP_F1', 'FADP_F1', 'PART_F1',
                    'Delta_F1', 'FDP_Comm_KB', 'FADP_Comm_KB', 'Comm_Reduction',
                    'FDP_Mem_KB', 'FADP_Mem_KB', 'Mem_Reduction',
                    'FDP_Time_s', 'FADP_Time_s'])
        for name, delta in [('Chess', 0.85), ('Chess', 0.90),
                             ('Mushroom', 0.40), ('Mushroom', 0.50), ('Mushroom', 0.60)]:
            e = E1[(name, delta)]
            c1 = e['FDP']['c']; c2 = e['FADP']['c']
            m1 = e['FDP']['m']; m2 = e['FADP']['m']
            w.writerow([
                f'{name}_d{int(delta*100)}', len(GT[(name, delta)]),
                f"{e['FDP']['f1']*100:.2f}", f"{e['FADP']['f1']*100:.2f}",
                f"{e['PART']['f1']*100:.2f}",
                f"{(e['FADP']['f1']-e['FDP']['f1'])*100:+.2f}",
                f'{c1:.2f}', f'{c2:.2f}',
                f'{(1-c2/max(c1,1e-9))*100:.1f}%',
                f'{m1:.3f}', f'{m2:.3f}',
                f'{(1-m2/max(m1,1e-9))*100:.1f}%',
                f"{e['FDP']['t']:.4f}", f"{e['FADP']['t']:.4f}",
            ])
    print(f"[CSV] Saved → {path}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    global EPS_MAIN, TAU_MAIN, NRUNS

    parser = argparse.ArgumentParser(description='FedADP-FIM Experiments')
    parser.add_argument('--exp',       default='all',
                        choices=['all', 'e1', 'e2', 'e3', 'e4', 'e5'],
                        help='Which experiment to run')
    parser.add_argument('--epsilon',   type=float, default=EPS_MAIN,
                        help='Epsilon for main comparison (default 1.0)')
    parser.add_argument('--tau',       type=float, default=TAU_MAIN,
                        help='Tau for IWTC (default 0.7)')
    parser.add_argument('--nruns',     type=int,   default=NRUNS,
                        help='Number of repetitions (default 5)')
    parser.add_argument('--use_cache', action='store_true',
                        help='Load GT from cache instead of recomputing')
    args = parser.parse_args()

    EPS_MAIN = args.epsilon
    TAU_MAIN = args.tau
    NRUNS    = args.nruns

    np.random.seed(42)

    print("=" * 70)
    print("FedADP-FIM Experiment Runner")
    print(f"  ε={EPS_MAIN}  τ={TAU_MAIN}  n_runs={NRUNS}")
    print("=" * 70)

    # Load datasets
    print("\n[DATA] Loading datasets...")
    datasets = {}
    for name, fname, delta, _ in CFGS:
        if fname not in datasets:
            fp = os.path.join(DATA, fname)
            if not os.path.exists(fp):
                print(f"  ERROR: {fp} not found. "
                      f"Download from https://www.philippe-fournier-viger.com/spmf/")
                sys.exit(1)
            datasets[fname] = load_spmf(fp)
            print(f"  {fname}: {len(datasets[fname])} transactions")

    # Ground truth
    print("\n[GT] Computing / loading ground truth...")
    GT = load_or_compute_gt(datasets, use_cache=args.use_cache)

    # Run experiments
    results = {'GT': GT, 'EPSILONS': EPSILONS, 'TAUS': TAUS, 'NS': N_CLIENTS}

    if args.exp in ('all', 'e1'):
        results['E1'] = run_e1(datasets, GT)
    if args.exp in ('all', 'e2'):
        results['E2'] = run_e2(datasets, GT)
    if args.exp in ('all', 'e3'):
        results['E3'] = run_e3(datasets, GT)
    if args.exp in ('all', 'e4'):
        results['E4'] = run_e4(datasets, GT)
    if args.exp in ('all', 'e5'):
        results['E5'] = run_e5(datasets, GT)

    # Save pickle
    pkl_path = os.path.join(RESULTS, 'exp_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n[SAVE] Results → {pkl_path}")

    # Figure + CSV (only if E1 present)
    if 'E1' in results and 'E2' in results:
        # Fill missing experiments with empty dicts for figure
        for k in ('E2', 'E3', 'E4', 'E5'):
            if k not in results:
                results[k] = {}
        make_figure(results['E1'], results['E2'],
                    results.get('E3', {}), results.get('E4', {}),
                    results.get('E5', {}), GT)
        save_csv(results['E1'], GT)

    print("\n[DONE] All experiments complete.")


if __name__ == '__main__':
    main()
