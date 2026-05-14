"""
FedDP-FPM — Baseline Algorithm
================================
Yu et al., "FedDP-FPM: A federated frequent itemset mining algorithm
under differential privacy", Knowledge-Based Systems 335 (2026) 115142.

Implementation notes
--------------------
Three bugs present in a naive transcription of the paper are corrected:

  BUG-1  F1 noise scale should be Lap(2/ε) = Lap(1/(ε/2)), not Lap(4/ε).
         Paper Eq.(3): c̃_i = c_i + Lap(2Δf/ε), Δf = 1, budget half = ε/2.

  BUG-2  Level index for exponential decay must use child.depth (level l)
         not parent.depth.  Depth-1 nodes receive the highest budget
         (no decay); deeper nodes receive exponentially less.

  BUG-3  count_itemset must scan ALL nodes whose item-name equals the
         target item and check the full bitmap mask — not a "leaf-only"
         heuristic.  The leaf heuristic misses transactions that pass
         through an intermediate target-node before continuing to other
         items, causing ~93 % error on dense datasets.

Server-side mining uses Apriori candidate generation rather than the
NegFIN / NegNodeset structure of Algorithms 3–5 in the paper.  The
support oracle (count_itemset) is mathematically equivalent, so F1
scores are faithful; however, runtime is higher than the paper reports
for dense datasets and should not be compared directly.

The 1-itemset frequent set is taken directly from GF1 (Round 1
aggregation), consistent with the paper's Algorithm 4 / Step 2.
"""

import math
import pickle
import time
from collections import defaultdict
from itertools import combinations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _lap(scale: float) -> float:
    """Draw one sample from Laplace(0, scale)."""
    return np.random.laplace(0.0, scale)


def _obj_size(obj) -> int:
    """Serialised byte size of *obj* (used for communication cost tracking)."""
    return len(pickle.dumps(obj))


# ─────────────────────────────────────────────────────────────────────────────
#  BMC Tree  (Algorithm 2, Yu et al. 2026)
# ─────────────────────────────────────────────────────────────────────────────

class BMCNode:
    """
    Node in a Bitmap-Mapped Count tree.

    Fields (Definition 5 in paper):
      item    — item name (None for the root sentinel)
      count   — number of transactions passing through this path
      bitmap  — accumulated OR of item-index bits along the path from root
      children — dict mapping item → child BMCNode
      depth   — distance from root (root = 0, first level = 1)
    """
    __slots__ = ('item', 'count', 'bitmap', 'children', 'depth')

    def __init__(self, item=None, count: int = 0,
                 bitmap: int = 0, depth: int = 0):
        self.item     = item
        self.count    = count
        self.bitmap   = bitmap
        self.children: dict = {}
        self.depth    = depth


def build_bmc_tree(transactions: list,
                   gf1_asc: list,
                   item_to_idx: dict) -> BMCNode:
    """
    Construct a BMC tree for one participant's local dataset.

    Parameters
    ----------
    transactions : list of iterables — local transaction database D_i
    gf1_asc      : items in GF1 sorted ascending by global support count
                   (lowest support → index 0, used for L1 vector)
    item_to_idx  : mapping item → position in gf1_asc

    Tree construction (Algorithm 2):
      Items in each transaction are filtered to GF1, then sorted in
      *descending* support order (reverse of L1) so that higher-support
      items are inserted closer to the root.  Each node's bitmap
      accumulates the OR of its ancestors' item-bits, encoding the full
      prefix path.
    """
    root = BMCNode(depth=0)
    for trans in transactions:
        # Keep only global-frequent items; sort descending by support.
        filtered = [it for it in trans if it in item_to_idx]
        filtered.sort(key=lambda x: item_to_idx[x], reverse=True)

        cur = root
        bmp = 0
        for item in filtered:
            idx  = item_to_idx[item]
            bmp |= (1 << idx)
            if item not in cur.children:
                cur.children[item] = BMCNode(
                    item=item, count=0, bitmap=bmp,
                    depth=cur.depth + 1
                )
            cur.children[item].count += 1
            cur = cur.children[item]
    return root


def _tree_max_depth(root: BMCNode) -> int:
    """Return the maximum depth of any node in the tree."""
    max_d, stack = 0, [root]
    while stack:
        n    = stack.pop()
        max_d = max(max_d, n.depth)
        stack.extend(n.children.values())
    return max_d


# ─────────────────────────────────────────────────────────────────────────────
#  Support counting  (BUG-3 corrected)
# ─────────────────────────────────────────────────────────────────────────────

def count_itemset(itemset, trees: list, item_to_idx: dict) -> float:
    """
    Estimate the support of *itemset* across a list of (noisy) BMC trees.

    Correctness argument
    --------------------
    Let P = itemset, target = the item in P with the lowest support
    (earliest in gf1_asc, smallest index).  Every other item o ∈ P has
    higher support, so o is inserted *above* target in the tree (closer
    to the root, due to descending-support ordering).

    For any transaction T ⊇ P:
      • T contributes +count to exactly one target-node n_target.
      • Because o was inserted before target, bit_{idx(o)} is set in
        n_target.bitmap if and only if o ∈ T.

    Therefore:
      support(P) = Σ { n.count : n.item == target
                                  AND (n.bitmap & mask) == mask }

    where mask = OR of bit_{idx(o)} for all o ∈ P \ {target}.

    BUG-3 (original): summed only "leaf" target-nodes (those with no
    child covering the mask).  This misses transactions that pass through
    an intermediate target-node before continuing to items not in P.
    """
    if not itemset:
        return 0.0

    # target = item with lowest support = lowest index in gf1_asc
    items  = sorted(itemset, key=lambda x: item_to_idx.get(x, 999_999))
    target = items[0]

    mask = 0
    for it in items[1:]:
        idx = item_to_idx.get(it)
        if idx is None:
            return 0.0
        mask |= (1 << idx)

    total = 0.0
    for root in trees:
        stack = [root]
        while stack:
            n = stack.pop()
            if n.item == target and (n.bitmap & mask) == mask:
                total += n.count
            stack.extend(n.children.values())
    return max(0.0, total)


# ─────────────────────────────────────────────────────────────────────────────
#  Post-processing  (Section 4.3)
# ─────────────────────────────────────────────────────────────────────────────

def _post_process(root: BMCNode) -> None:
    """
    Apply post-processing to a noisy BMC tree in-place.

    Rules (paper Section 4.3):
      1. Prune any node whose noisy count ≤ 0 (along with all descendants).
      2. Enforce that a child's count does not exceed its parent's count:
           c̄_{n_i} = min(c_{n_i}, c_{n_j})  where n_j is n_i's parent.

    Post-processing preserves ε-differential privacy (standard DP
    post-processing theorem — no additional privacy cost).
    """
    def _recurse(node: BMCNode, parent_count: int) -> None:
        for item in list(node.children):
            child = node.children[item]
            if child.count <= 0:
                del node.children[item]
            else:
                child.count = min(child.count, parent_count)
                _recurse(child, child.count)

    for item in list(root.children):
        child = root.children[item]
        if child.count <= 0:
            del root.children[item]
        else:
            _recurse(child, child.count)


# ─────────────────────────────────────────────────────────────────────────────
#  Server-side mining  (Apriori + bitmap oracle)
# ─────────────────────────────────────────────────────────────────────────────

def _mine_frequent_itemsets(gf1: dict,
                             trees: list,
                             total_trans: int,
                             delta: float,
                             item_to_idx: dict) -> set:
    """
    Mine global frequent itemsets from aggregated noisy BMC trees.

    1-itemsets are taken directly from GF1 (Round 1 result), consistent
    with the paper's Algorithm 4 / Step 2.  Higher-order candidates are
    generated via Apriori join and pruned with count_itemset as the
    support oracle.

    Returns
    -------
    set of frozensets — every globally frequent itemset found.
    """
    min_count = total_trans * delta
    gf1_asc   = sorted(gf1, key=lambda x: gf1[x])

    # ── 1-itemsets: take directly from GF1 (paper Section 4.1 / Step 2) ──
    frequent: set = {frozenset([item]) for item in gf1_asc}

    # ── k-itemsets (k ≥ 2): Apriori candidate generation ──
    k = 2
    while True:
        prev_list = sorted([sorted(fs) for fs in frequent if len(fs) == k - 1])
        if not prev_list:
            break

        candidates: set = set()
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                a, b = prev_list[i], prev_list[j]
                if a[:-1] == b[:-1]:           # Apriori join on common prefix
                    cand = frozenset(a) | frozenset(b)
                    if len(cand) == k:
                        candidates.add(cand)

        if not candidates:
            break

        new_freq: set = set()
        for cand in candidates:
            # Apriori pruning: all (k-1)-subsets must be frequent
            if not all(frozenset(sub) in frequent
                       for sub in combinations(cand, k - 1)):
                continue
            if count_itemset(cand, trees, item_to_idx) >= min_count:
                new_freq.add(cand)

        if not new_freq:
            break

        frequent.update(new_freq)
        k += 1

    return frequent


# ─────────────────────────────────────────────────────────────────────────────
#  FedDP-FPM  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

class FedDP_FPM:
    """
    FedDP-FPM — federated frequent itemset mining under differential privacy.

    Reference
    ---------
    Yu Z., Huo Z., Wang T., Li X.
    "FedDP-FPM: A federated frequent itemset mining algorithm under
    differential privacy."
    Knowledge-Based Systems 335 (2026) 115142.

    Parameters
    ----------
    epsilon : float
        Total privacy budget ε shared across both communication rounds.
        Each round receives ε/2 (Section 4.2 / 4.3).
    delta : float
        Minimum support threshold δ.
    gamma : float or None
        Pre-large threshold γ for local client mining (Definition 3).
        Default: δ/2, following [23] (Wu et al., ACM TOSN 2023).
    alpha : float
        Exponential decay factor α for BMC-tree level budget allocation
        (Eq. 4).  Larger α → more noise at deeper levels.  Default: 0.5.
    """

    def __init__(self,
                 epsilon: float = 1.0,
                 delta:   float = 0.4,
                 gamma:   float | None = None,
                 alpha:   float = 0.5):
        self.epsilon = epsilon
        self.delta   = delta
        self.gamma   = gamma if gamma is not None else delta / 2.0
        self.alpha   = alpha
        self.name    = "FedDP-FPM"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _mine_f1_local(self, transactions: list) -> dict:
        """
        Client-side: mine frequent 1-itemset using pre-large threshold γ.
        Returns {item: absolute_count}.
        """
        n   = len(transactions)
        cnt: dict = defaultdict(int)
        for t in transactions:
            for it in t:
                cnt[it] += 1
        return {it: c for it, c in cnt.items() if c / n >= self.gamma}

    def _add_f1_noise(self, f1: dict) -> dict:
        """
        BUG-1 FIX — Eq.(3): c̃_i = c_i + Lap(2Δf/ε) = c_i + Lap(1/(ε/2)).
        Sensitivity Δf = 1 for count queries.
        """
        half  = self.epsilon / 2.0                    # budget for F1 round
        scale = 1.0 / half                             # = 2/ε
        return {it: max(0, round(c + _lap(scale)))
                for it, c in f1.items()}

    def _level_budget(self, level_l: int, L: int) -> float:
        """
        Eq.(4): ε_l = (ε/2)·e^{-α(l-1)} / Σ_{k=1}^{L} e^{-α(k-1)}.
        Σ_l ε_l = ε/2 (verified by Theorem 3 / Eq. 5).
        """
        half  = self.epsilon / 2.0
        denom = sum(math.exp(-self.alpha * (k - 1)) for k in range(1, L + 1))
        return half * math.exp(-self.alpha * (level_l - 1)) / denom

    def _add_tree_noise(self, root: BMCNode) -> BMCNode:
        """
        BUG-2 FIX — add Laplace noise to every BMC-tree node count,
        using child.depth (= l) as the level index for Eq.(4).

        Constructs and returns a new (noisy) tree; the original is not
        modified.  Post-processing is applied after noise injection.
        """
        L       = max(1, _tree_max_depth(root))
        noisy_r = BMCNode(depth=0)

        def _recurse(orig: BMCNode, new: BMCNode) -> None:
            for item, child in orig.children.items():
                eps_l = self._level_budget(child.depth, L)   # BUG-2 fixed
                scale = 1.0 / max(eps_l, 1e-12)
                nc    = max(0, round(child.count + _lap(scale)))
                c2    = BMCNode(item=item, count=nc,
                                bitmap=child.bitmap, depth=child.depth)
                new.children[item] = c2
                _recurse(child, c2)

        _recurse(root, noisy_r)
        _post_process(noisy_r)
        return noisy_r

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, client_datasets: list) -> tuple[set, dict]:
        """
        Execute FedDP-FPM over a list of client datasets.

        Parameters
        ----------
        client_datasets : list of list-of-transactions
            Each element is the local dataset D_i held by participant P_i.

        Returns
        -------
        frequent : set of frozensets
            Global frequent itemsets GF.
        stats : dict with keys
            time          — wall-clock seconds
            comm_mb       — total communication (client → server) in MB
            mem_client_mb — peak serialised BMC-tree size per client (MB)
            mem_server_mb — peak serialised trees-list size on server (MB)
            n_frequent    — |GF|
        """
        t0   = time.time()
        N    = sum(len(ds) for ds in client_datasets)
        comm = 0
        mems = []

        # ── Round 1: each client sends noisy local F1 (Step 1 / Algorithm 1) ─
        nf1_list = []
        for ds in client_datasets:
            f1  = self._mine_f1_local(ds)
            nf1 = self._add_f1_noise(f1)
            comm += _obj_size(nf1)
            nf1_list.append(nf1)

        # ── Server: aggregate → GF1 (Step 2 / Algorithm 1 lines 3–5) ─────────
        agg: dict = defaultdict(float)
        for nf1 in nf1_list:
            for it, c in nf1.items():
                agg[it] += c
        gf1 = {it: c for it, c in agg.items() if c >= N * self.delta}

        if not gf1:
            return set(), {
                'time':           time.time() - t0,
                'comm_mb':        comm / 1e6,
                'mem_client_mb':  0.0,
                'mem_server_mb':  0.0,
                'n_frequent':     0,
            }

        gf1_asc  = sorted(gf1, key=lambda x: gf1[x])      # ascending support
        item2idx = {it: i for i, it in enumerate(gf1_asc)} # L1 index mapping

        # ── Round 2: each client builds and sends noisy BMC tree (Step 3) ─────
        trees = []
        for ds in client_datasets:
            bmc    = build_bmc_tree(ds, gf1_asc, item2idx)
            noisy  = self._add_tree_noise(bmc)
            sz     = _obj_size(noisy)
            comm  += sz
            mems.append(sz / 1e6)
            trees.append(noisy)

        mem_srv = _obj_size(trees) / 1e6

        # ── Server: mine GF from noisy trees (Step 4 / Algorithm 1 lines 8–12)
        gf = _mine_frequent_itemsets(gf1, trees, N, self.delta, item2idx)

        return gf, {
            'time':           time.time() - t0,
            'comm_mb':        comm / 1e6,
            'mem_client_mb':  max(mems) if mems else 0.0,
            'mem_server_mb':  mem_srv,
            'n_frequent':     len(gf),
        }
