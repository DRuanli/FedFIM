"""
shared/common.py
================
Shared utilities for all FedADP-FIM experiments.

Conventions
-----------
- All experiments import from this module.
- GT cache stored at results/gt_cache.pkl (auto-created).
- Each experiment writes its own results/exp_XX/  subfolder.
- Seeding: np.random.seed(GLOBAL_SEED) at top of each experiment script.
"""

import os, sys, time, pickle, math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set, Tuple

# ── Resolve project root (fedadp_experiments/../) ────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)               # fedadp_experiments/
SRC     = os.path.join(ROOT, '..', 'src')     # ../src/
DATA    = os.path.join(ROOT, '..', 'data')    # ../data/
RESULTS = os.path.join(ROOT, '..', 'results') # ../results/

sys.path.insert(0, os.path.abspath(SRC))

from algorithms import FedDP_FPM, FedADP_FIM
from data_utils  import (load_spmf, split_non_iid,
                         compute_gt, compute_f1_score)

# ── Global constants ──────────────────────────────────────────────
GLOBAL_SEED = 42

# Datasets: (display_name, filename, support_thresholds)
# Covers dense → sparse spectrum required by TKDE reviewers
DATASETS = {
    'Chess':    ('chess.txt',    [0.80, 0.85, 0.90, 0.95]),   # dense,  75 items
    'Mushroom': ('mushroom.txt', [0.40, 0.50, 0.60, 0.70]),   # medium, 119 items
    'Retail':   ('retail.txt',  [0.01, 0.05, 0.10, 0.20]),    # sparse, 16470 items
    'Foodmart': ('fruithut.txt',[0.005,0.01, 0.05, 0.10]),    # sparse, varies
}

# Core hyperparameters used across experiments
EPS_MAIN  = 1.0          # representative privacy budget
TAU_MAIN  = 0.7          # IWTC compression ratio (sweet-spot from E3)
NRUNS     = 10           # repetitions for mean ± std
EPSILONS  = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]
TAUS      = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_CLIENTS = [3, 5, 10, 20, 50]


# ─────────────────────────────────────────────────────────────────
#  RESULT HELPERS
# ─────────────────────────────────────────────────────────────────

def make_exp_dir(exp_name: str) -> Tuple[str, str, str]:
    """Create results/exp_name/{figures,tables} and return paths."""
    base = os.path.join(RESULTS, exp_name)
    figs = os.path.join(base, 'figures')
    tabs = os.path.join(base, 'tables')
    for d in [base, figs, tabs]:
        os.makedirs(d, exist_ok=True)
    return base, figs, tabs


def save_results(data: dict, exp_name: str) -> str:
    base, _, _ = make_exp_dir(exp_name)
    path = os.path.join(base, 'results.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"  [SAVED] {path}")
    return path


def load_results(exp_name: str) -> dict:
    path = os.path.join(RESULTS, exp_name, 'results.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────
#  DATASET LOADING & GT CACHING
# ─────────────────────────────────────────────────────────────────

def load_datasets(names=None) -> Dict[str, list]:
    """Load requested datasets (default: all). Returns {name: transactions}."""
    if names is None:
        names = list(DATASETS.keys())
    datasets = {}
    for name in names:
        fname, _ = DATASETS[name]
        fpath = os.path.join(DATA, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Dataset not found: {fpath}\n"
                f"Download from https://www.philippe-fournier-viger.com/spmf/"
            )
        datasets[name] = load_spmf(fpath)
        n = len(datasets[name])
        items = len({it for t in datasets[name] for it in t})
        avg   = np.mean([len(t) for t in datasets[name]])
        print(f"  Loaded {name:10s}: {n:7d} trans | {items:5d} items | avg_len={avg:.1f}")
    return datasets


def load_or_compute_gt(datasets: dict,
                       delta_map: dict = None,
                       cache_path: str = None,
                       use_cache: bool = True,
                       max_time: float = 300.0) -> dict:
    """
    GT cache keyed by (dataset_name, delta).
    delta_map: {name: [d1, d2, ...]}  — defaults to DATASETS thresholds.
    """
    if cache_path is None:
        cache_path = os.path.join(RESULTS, 'gt_cache.pkl')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    GT = {}
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            GT = pickle.load(f)
        print(f"  [GT] Loaded {len(GT)} entries from cache.")

    if delta_map is None:
        delta_map = {name: DATASETS[name][1] for name in datasets}

    dirty = False
    for name, deltas in delta_map.items():
        data = datasets[name]
        for delta in deltas:
            if (name, delta) in GT:
                continue
            all_t = [t for t in data]
            print(f"  [GT] Computing {name} δ={delta} (n={len(all_t)}) ...")
            t0 = time.time()
            GT[(name, delta)] = compute_gt(all_t, delta,
                                           max_time=max_time, verbose=True)
            print(f"       → {len(GT[(name, delta)])} FIs in {time.time()-t0:.1f}s")
            dirty = True

    if dirty:
        with open(cache_path, 'wb') as f:
            pickle.dump(GT, f)
        print(f"  [GT] Cache updated → {cache_path}")
    return GT


# ─────────────────────────────────────────────────────────────────
#  RUN HELPERS
# ─────────────────────────────────────────────────────────────────

def run_once(algo, splits, gt):
    """Single run → (f1, comm_kb, mem_kb, time_s, n_frequent)."""
    pred, st = algo.run(splits)
    _, _, f  = compute_f1_score(pred, gt)
    return (f,
            st['comm_mb'] * 1024,
            st.get('mem_client_mb', 0) * 1024,
            st['time'],
            st.get('n_frequent', len(pred)))


def run_n(algo, splits, gt, n=NRUNS):
    """n runs → dict of mean ± std for F1, comm, mem, time."""
    rows = [run_once(algo, splits, gt) for _ in range(n)]
    F, C, M, T, NF = zip(*rows)
    return {
        'f1':  np.mean(F),  'f1s':  np.std(F),
        'c':   np.mean(C),  'cs':   np.std(C),   # comm KB
        'm':   np.mean(M),  'ms':   np.std(M),   # mem  KB
        't':   np.mean(T),  'ts':   np.std(T),   # time s
        'nf':  np.mean(NF),
    }


def make_splits(data, n_clients=3, ratios=None, seed=GLOBAL_SEED):
    return split_non_iid(data, ratios=ratios, n_clients=n_clients, seed=seed)


# ─────────────────────────────────────────────────────────────────
#  DIRICHLET NON-IID SPLIT  (for E6)
# ─────────────────────────────────────────────────────────────────

def split_dirichlet(data: list, n_clients: int = 3,
                    alpha_dir: float = 0.5, seed: int = GLOBAL_SEED) -> list:
    """
    True Dirichlet Non-IID split: α→0 = maximally skewed, α→∞ = IID.
    Standard in FL literature (McMahan et al. Pathological Non-IID).
    """
    rng = np.random.default_rng(seed)
    n   = len(data)

    # Compute global item frequencies, assign each transaction a "group"
    # based on its most frequent (globally) item
    item_freq = defaultdict(int)
    for t in data:
        for it in t:
            item_freq[it] += 1
    sorted_items = sorted(item_freq, key=lambda x: item_freq[x], reverse=True)
    item_to_group = {it: i % n_clients
                     for i, it in enumerate(sorted_items)}

    # Dirichlet proportions per group
    props = rng.dirichlet([alpha_dir] * n_clients, size=n_clients)

    buckets = [[] for _ in range(n_clients)]
    for idx in rng.permutation(n):
        t = data[idx]
        if not t:
            buckets[0].append(t)
            continue
        dominant = max(t, key=lambda x: item_freq.get(x, 0))
        g     = item_to_group[dominant]
        p     = props[g]
        p     = p / p.sum()
        client = rng.choice(n_clients, p=p)
        buckets[client].append(t)

    # Guarantee no empty client
    for i, b in enumerate(buckets):
        if not b:
            buckets[i] = [data[0]]
    return buckets


# ─────────────────────────────────────────────────────────────────
#  LATEX TABLE HELPERS
# ─────────────────────────────────────────────────────────────────

def fmt(v, decimals=1):
    return f"{v:.{decimals}f}"

def fmt_pm(mean, std, decimals=1, pct=True):
    suffix = "\\%" if pct else ""
    scale  = 100 if pct else 1
    return f"{mean*scale:.{decimals}f}$\\pm${std*scale:.{decimals}f}{suffix}"

def write_latex_table(rows, header, caption, label, path):
    ncols = len(header)
    col_spec = "l" + "r" * (ncols - 1)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  [LaTeX] {path}")
