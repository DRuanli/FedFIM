"""
Data utilities for FedADP-FIM experiments.

Functions
---------
load_spmf        : Load SPMF-format transaction files (Chess, Mushroom, etc.)
split_non_iid    : Split transactions into non-IID federated clients
compute_gt       : Compute exact ground-truth frequent itemsets (Apriori)
compute_f1_score : Precision / Recall / F1 between predicted and GT sets
"""

import numpy as np
import time
from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Set, Tuple, Optional


# ─────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_spmf(filepath: str) -> List[List[int]]:
    """
    Load a transaction file in SPMF format.

    Supports both plain format and utility format:
      Plain  : "item1 item2 item3"
      Utility: "item1 item2 : total_util : util1 util2"
               (only item part is used)

    Parameters
    ----------
    filepath : path to .txt file

    Returns
    -------
    transactions : list of lists of integer item IDs
    """
    transactions = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle SPMF utility format (split on ':')
            items_part = line.split(':')[0].strip()
            items = list(map(int, items_part.split()))
            if items:
                transactions.append(items)
    return transactions


def dataset_stats(transactions: List[List[int]]) -> dict:
    """Print and return basic dataset statistics."""
    n = len(transactions)
    lengths = [len(t) for t in transactions]
    items = set(it for t in transactions for it in t)
    stats = {
        'n_transactions': n,
        'n_items':        len(items),
        'avg_length':     float(np.mean(lengths)),
        'max_length':     max(lengths),
        'density':        float(np.mean(lengths)) / len(items) * 100,
    }
    print(f"  Transactions : {stats['n_transactions']}")
    print(f"  Unique items : {stats['n_items']}")
    print(f"  Avg length   : {stats['avg_length']:.1f}")
    print(f"  Density      : {stats['density']:.1f}%")
    return stats


# ─────────────────────────────────────────────────────────────
#  DATA SPLITTING
# ─────────────────────────────────────────────────────────────

def split_non_iid(data: List[List[int]],
                  ratios: List[float] = None,
                  n_clients: int = 3,
                  seed: int = 42) -> List[List[List[int]]]:
    """
    Split transactions into non-IID federated client datasets.

    The split is non-IID in the sense that client sizes differ
    (default 70/20/10 for 3 clients), simulating realistic
    data heterogeneity.

    Parameters
    ----------
    data      : full transaction dataset
    ratios    : list of fractions summing to 1.0
                (default: [0.70, 0.20, 0.10] for 3 clients,
                           equal split for other n_clients)
    n_clients : number of clients (used when ratios=None)
    seed      : random seed for reproducibility

    Returns
    -------
    splits : list of n_clients transaction lists
    """
    if ratios is None:
        if n_clients == 3:
            ratios = [0.70, 0.20, 0.10]
        else:
            ratios = [1.0 / n_clients] * n_clients

    rng = np.random.default_rng(seed)
    n   = len(data)
    idx = rng.permutation(n)
    splits = []
    s = 0
    for i, r in enumerate(ratios):
        if i < len(ratios) - 1:
            sz = max(1, int(n * r))
        else:
            sz = n - s
        chunk = [data[j] for j in rng.permutation(idx[s:s + sz])]
        splits.append(chunk)
        s += sz
    return splits


# ─────────────────────────────────────────────────────────────
#  GROUND TRUTH  (exact Apriori)
# ─────────────────────────────────────────────────────────────

def compute_gt(transactions: List[List[int]],
               delta: float,
               max_time: float = 120.0,
               verbose: bool = True) -> Set[frozenset]:
    """
    Compute exact global ground-truth frequent itemsets using Apriori.

    Parameters
    ----------
    transactions : all transactions (concatenated from all clients)
    delta        : minimum support threshold (fraction, e.g. 0.85)
    max_time     : wall-clock timeout in seconds (stops at current k)
    verbose      : print progress per k-level

    Returns
    -------
    frequent : set of frozensets of integer item IDs

    Notes
    -----
    For dense datasets (Chess δ=0.85), k can reach 7–8 and takes ~60s.
    For sparse datasets (Mushroom δ=0.50), done in < 5s.
    Use the pre-computed cache (gt_cache.pkl) to skip recomputation.
    """
    n  = len(transactions)
    mc = n * delta

    # Frequent 1-itemsets
    cnt = defaultdict(int)
    for t in transactions:
        for it in set(t):
            cnt[it] += 1
    f1 = {it for it, c in cnt.items() if c >= mc}

    ts = [sorted([it for it in t if it in f1]) for t in transactions]
    frequent: Set[frozenset] = {frozenset([it]) for it in f1}

    if verbose:
        print(f"    k=1: {len(f1)} frequent items")

    k  = 2
    t0 = time.time()
    while k <= len(f1) and (time.time() - t0) < max_time:
        cnt_k: Dict[frozenset, int] = defaultdict(int)
        for t in ts:
            for combo in combinations(t, k):
                cnt_k[frozenset(combo)] += 1
        new = {
            c for c, v in cnt_k.items()
            if v >= mc and all(
                frozenset(sub) in frequent
                for sub in combinations(c, k - 1)
            )
        }
        if not new:
            break
        frequent.update(new)
        if verbose:
            print(f"    k={k}: {len(new)} new → total {len(frequent)}  "
                  f"({time.time() - t0:.1f}s)")
        k += 1

    if verbose:
        by_k: Dict[int, int] = defaultdict(int)
        for fs in frequent:
            by_k[len(fs)] += 1
        print(f"    TOTAL: {len(frequent)} frequent itemsets "
              f"(k-dist: {dict(sorted(by_k.items()))})")
    return frequent


# ─────────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────────

def compute_f1_score(predicted: Set[frozenset],
                     ground_truth: Set[frozenset]
                     ) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, F1 between predicted and GT frequent itemsets.

    Returns
    -------
    precision, recall, f1  (all in [0, 1])
    """
    if not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    tp = len(predicted & ground_truth)
    fp = len(predicted - ground_truth)
    fn = len(ground_truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1
