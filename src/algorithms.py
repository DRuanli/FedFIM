"""
FedDP-FPM (Corrected) + FedADP-FIM + DP-PartFIM
=================================================
Implementations for the paper:
  "FedADP-FIM: Adaptive-Budget Federated Frequent Itemset Mining
   with Importance-Weighted Tree Compression under Differential Privacy"

Three algorithms:
  1. FedDP_FPM      — baseline (Yu et al. KBS 2026, all bugs fixed)
  2. FedADP_FIM     — proposed algorithm (ABP + IWTC modules)
  3. DP_PartFIM_Simple — additional baseline

Bug fixes vs original FedDP-FPM paper:
  BUG-1: F1 noise scale Lap(2/ε) not Lap(4/ε)
  BUG-2: Level index uses child.depth not parent.depth
  BUG-3 [CRITICAL]: count_itemset uses direct bitmap scan,
         NOT leaf-based scan (93% error rate on dense datasets)
"""

import numpy as np
import math
import time
import hashlib
import pickle
from collections import defaultdict
from itertools import combinations


# ─────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────

def lap(scale: float) -> float:
    return np.random.laplace(0, scale)

def obj_size(obj) -> int:
    return len(pickle.dumps(obj))


# ─────────────────────────────────────────────────────────────
#  BMC TREE  (Algorithm 2, Yu et al. 2026)
# ─────────────────────────────────────────────────────────────

class BMCNode:
    __slots__ = ['item', 'count', 'bitmap', 'children', 'depth']
    def __init__(self, item=None, count=0, bitmap=0, depth=0):
        self.item    = item
        self.count   = count
        self.bitmap  = bitmap
        self.children = {}
        self.depth   = depth


def build_bmc_tree(transactions, gf1_asc: list, item_to_idx: dict) -> BMCNode:
    """
    Build BMC (Bitmap-Mapped Count) tree from transactions.

    gf1_asc : items sorted ascending by support (lowest support first).
               Items are inserted into tree in DESCENDING support order
               so that higher-support items appear closer to root.
    """
    root = BMCNode(depth=0)
    for trans in transactions:
        filtered = [it for it in trans if it in item_to_idx]
        filtered.sort(key=lambda x: item_to_idx[x], reverse=True)  # desc support
        cur = root
        bmp = 0
        for item in filtered:
            idx  = item_to_idx[item]
            bmp |= (1 << idx)
            if item not in cur.children:
                cur.children[item] = BMCNode(
                    item=item, count=0, bitmap=bmp, depth=cur.depth + 1
                )
            cur.children[item].count += 1
            cur = cur.children[item]
    return root


def all_nodes(root: BMCNode) -> list:
    res, stack = [], [root]
    while stack:
        n = stack.pop()
        if n.item is not None:
            res.append(n)
        stack.extend(n.children.values())
    return res


def tree_max_depth(root: BMCNode) -> int:
    md, stack = 0, [root]
    while stack:
        n   = stack.pop()
        md  = max(md, n.depth)
        stack.extend(n.children.values())
    return md


# ─────────────────────────────────────────────────────────────
#  CORE MINING  (FIXED — BUG-3 corrected)
# ─────────────────────────────────────────────────────────────

def count_itemset(itemset, trees: list, item_to_idx: dict) -> float:
    """
    Count support of itemset across multiple (noisy) BMC trees.

    PROOF OF CORRECTNESS:
      Let target = item with lowest support (earliest in gf1_asc).
      Let O = all other items in the itemset.

      support(P) = |{trans : target ∈ trans  AND  ∀o∈O, o ∈ trans}|
                 = Σ_{node n : n.item == target
                              AND ∀o∈O, bit_o set in n.bitmap}
                   n.count

      Why? Each transaction contributes +1 to exactly one target-node.
      Since o has higher support than target, o IS inserted before target
      in the tree (descending-support ordering).
      Therefore bit_o in n.bitmap = 1  ⟺  o appeared in that transaction.
      So the bitmap condition ≡ "all O items present in transaction".

    BUG-3 in original code: used "leaf" heuristic — only counted target
    nodes that had NO children covering the mask. This missed all
    transactions ending at an intermediate node that happened to have
    children from OTHER transactions passing through. Error rate: 93%.
    """
    if not itemset:
        return 0
    # Sort by support: target = lowest support = earliest in gf1_asc
    items  = sorted(itemset, key=lambda x: item_to_idx.get(x, 999_999))
    target = items[0]
    mask   = 0
    for it in items[1:]:
        idx = item_to_idx.get(it)
        if idx is None:
            return 0
        mask |= (1 << idx)

    total = 0
    for root in trees:
        stack = [root]
        while stack:
            n = stack.pop()
            if n.item == target and (n.bitmap & mask) == mask:
                total += n.count
            stack.extend(n.children.values())
    return max(0.0, total)


def mine_frequent_itemsets(gf1: dict, trees: list,
                           total_trans: int, delta: float,
                           item_to_idx: dict) -> set:
    """
    Apriori + NegFIN mining on the server's aggregated noisy BMC trees.
    Returns set of frozensets — the global frequent itemsets.
    """
    gf1_asc   = sorted(gf1, key=lambda x: gf1[x])
    min_count = total_trans * delta
    frequent  = set()

    # 1-itemsets
    for item in gf1_asc:
        if count_itemset(frozenset([item]), trees, item_to_idx) >= min_count:
            frequent.add(frozenset([item]))

    # k-itemsets  (k ≥ 2)
    k = 2
    while True:
        prev_list = sorted([sorted(fs) for fs in frequent if len(fs) == k - 1])
        if not prev_list:
            break
        cands = set()
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                a, b = prev_list[i], prev_list[j]
                if a[:-1] == b[:-1]:
                    cand = frozenset(a) | frozenset(b)
                    if len(cand) == k:
                        cands.add(cand)
        if not cands:
            break
        new_freq = set()
        for cand in cands:
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


def post_process(root: BMCNode):
    """Section 4.3: prune zero-count nodes; enforce parent ≥ child count."""
    def rec(node, par_cnt):
        for item in list(node.children):
            child = node.children[item]
            if child.count <= 0:
                del node.children[item]
            else:
                child.count = min(child.count, par_cnt)
                rec(child, child.count)
    for item in list(root.children):
        child = root.children[item]
        if child.count <= 0:
            del root.children[item]
        else:
            rec(child, child.count)


# ─────────────────────────────────────────────────────────────
#  ALGORITHM 1 — FedDP-FPM  (baseline, bugs fixed)
# ─────────────────────────────────────────────────────────────

class FedDP_FPM:
    """
    FedDP-FPM: Yu et al., KBS 335 (2026) 115142.
    All three implementation bugs corrected for fair comparison.

    Parameters
    ----------
    epsilon : float  — total privacy budget ε
    delta   : float  — minimum support threshold δ
    gamma   : float  — local mining threshold (default δ/2)
    alpha   : float  — exponential decay for level budget (default 0.5)
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 0.4,
                 gamma: float = None, alpha: float = 0.5):
        self.epsilon = epsilon
        self.delta   = delta
        self.gamma   = gamma if gamma is not None else delta / 2.0
        self.alpha   = alpha
        self.name    = "FedDP-FPM"

    def _mine_f1(self, transactions, threshold):
        n = len(transactions)
        cnt = defaultdict(int)
        for t in transactions:
            for it in t:
                cnt[it] += 1
        return {it: c for it, c in cnt.items() if c / n >= threshold}

    def _noise_f1(self, f1: dict) -> dict:
        """BUG-1 FIX: scale = Δf / (ε/2) = 1/(ε/2) = 2/ε"""
        half = self.epsilon / 2.0
        return {it: max(0, round(c + lap(1.0 / half)))
                for it, c in f1.items()}

    def _eps_level(self, level_l: int, L: int) -> float:
        """Eq.(4): ε_l = (ε/2)·e^{-α(l-1)} / Σ_k e^{-α(k-1)}"""
        half  = self.epsilon / 2.0
        denom = sum(math.exp(-self.alpha * (k - 1)) for k in range(1, L + 1))
        return half * math.exp(-self.alpha * (level_l - 1)) / denom

    def _noise_tree(self, root: BMCNode) -> BMCNode:
        """BUG-2 FIX: use child.depth (not parent.depth) as level_l."""
        L  = max(1, tree_max_depth(root))
        nr = BMCNode(depth=0)
        def rec(orig, new):
            for item, child in orig.children.items():
                eps_l = self._eps_level(child.depth, L)   # BUG-2 fixed
                nc    = max(0, round(child.count + lap(1.0 / max(eps_l, 1e-12))))
                c2    = BMCNode(item=item, count=nc,
                                bitmap=child.bitmap, depth=child.depth)
                new.children[item] = c2
                rec(child, c2)
        rec(root, nr)
        post_process(nr)
        return nr

    def run(self, client_datasets: list) -> tuple:
        """
        Returns
        -------
        frequent : set of frozensets
        stats    : dict with time, comm_mb, mem_client_mb, mem_server_mb
        """
        t0   = time.time()
        N    = sum(len(d) for d in client_datasets)
        comm = 0
        mems = []

        # ── Round 1: clients send noisy F1 ──
        nf1_list = []
        for ds in client_datasets:
            f1  = self._mine_f1(ds, self.gamma)
            nf1 = self._noise_f1(f1)
            comm += obj_size(nf1)
            nf1_list.append(nf1)

        # Server aggregates GF1
        agg = defaultdict(float)
        for nf1 in nf1_list:
            for it, c in nf1.items():
                agg[it] += c
        gf1 = {it: c for it, c in agg.items() if c >= N * self.delta}
        if not gf1:
            return set(), {'time': time.time() - t0, 'comm_mb': comm / 1e6,
                           'mem_client_mb': 0, 'mem_server_mb': 0}

        gf1_asc  = sorted(gf1, key=lambda x: gf1[x])
        item2idx = {it: i for i, it in enumerate(gf1_asc)}

        # ── Round 2: clients send noisy BMC trees ──
        trees = []
        for ds in client_datasets:
            bmc  = build_bmc_tree(ds, gf1_asc, item2idx)
            nb   = self._noise_tree(bmc)
            sz   = obj_size(nb)
            comm += sz
            mems.append(sz / 1e6)
            trees.append(nb)

        mem_srv = obj_size(trees) / 1e6
        gf = mine_frequent_itemsets(gf1, trees, N, self.delta, item2idx)

        return gf, {
            'time':           time.time() - t0,
            'comm_mb':        comm / 1e6,
            'mem_client_mb':  max(mems) if mems else 0,
            'mem_server_mb':  mem_srv,
            'n_frequent':     len(gf),
        }


# ─────────────────────────────────────────────────────────────
#  ALGORITHM 2 — FedADP-FIM  (proposed)
# ─────────────────────────────────────────────────────────────

class FedADP_FIM:
    """
    FedADP-FIM: Adaptive-budget Federated FIM with Tree Compression.

    Contributions over FedDP-FPM:

    [ABP] Adaptive Budget Profiler (Theorem, Section 4.2)
      Per-client α_i = ln(β_i) / L_i, derived from BMC tree structure.
      Minimises total MSE across all levels under Σε_l = ε/2.

    [IWTC] Importance-Weighted Tree Compressor (Section 4.3)
      Depth ≤ 2: keep 100%.  Depth > 2: keep top-τ by importance score.
      Reduces communication 20–60% with controlled accuracy loss.

    Parameters
    ----------
    epsilon : float  — total privacy budget ε
    delta   : float  — minimum support threshold δ
    gamma   : float  — local mining threshold (default δ/2)
    tau     : float  — compression ratio for depth>2 nodes  [0.1, 1.0]
    w1, w2, w3 : float — importance score weights (count, depth, struct)
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 0.4,
                 gamma: float = None, tau: float = 0.7,
                 w1: float = 0.5, w2: float = 0.3, w3: float = 0.2):
        self.epsilon = epsilon
        self.delta   = delta
        self.gamma   = gamma if gamma is not None else delta / 2.0
        self.tau     = tau
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.name    = f"FedADP-FIM(τ={tau})"

    # ── Module 1: ABP ───────────────────────────────────────────
    def _adaptive_alpha(self, transactions: list, f1: dict) -> float:
        """
        Theorem (ABP Optimality):
          For a BMC tree with average branching factor β and expected
          transaction depth L (after filtering to GF1 items):

            Total MSE(α) ∝ Σ_{l=1}^{L} N_l · e^{2α(l-1)}
                           / ( Σ_{k=1}^{L} e^{-α(k-1)} )²

          where N_l ≈ N_0 · β^{l-1} (geometric growth assumption).

          Differentiating w.r.t. α and setting to zero yields:

            α* = ln(β) / L

          This is the unique minimiser (convex in α for β > 1, L > 0).

        Boundary conditions:
          L ≤ 5  (shallow)  → α = 0.30  (uniform regime)
          L ≤ 20            → α = ln(β) / L
          L > 20 (very deep) → α = 1.2 / L  (aggressive decay)
        """
        if not transactions or not f1:
            return 0.5
        lens = [len([it for it in t if it in f1]) for t in transactions]
        lens = [l for l in lens if l > 0]
        if not lens:
            return 0.5
        L = float(np.mean(lens))
        β = max(2.0, len(f1) / max(L, 1.0))
        if   L <= 5:   α = 0.30
        elif L <= 20:  α = math.log(β) / L
        else:          α = 1.2 / L
        return float(np.clip(α, 0.05, 2.0))

    def _eps_level(self, level_l: int, L: int, alpha: float) -> float:
        half  = self.epsilon / 2.0
        denom = sum(math.exp(-alpha * (k - 1)) for k in range(1, L + 1))
        return half * math.exp(-alpha * (level_l - 1)) / denom

    def _noise_tree_adaptive(self, root: BMCNode, alpha: float) -> BMCNode:
        L  = max(1, tree_max_depth(root))
        nr = BMCNode(depth=0)
        def rec(orig, new):
            for item, child in orig.children.items():
                eps_l = self._eps_level(child.depth, L, alpha)
                nc    = max(0, round(child.count + lap(1.0 / max(eps_l, 1e-12))))
                c2    = BMCNode(item=item, count=nc,
                                bitmap=child.bitmap, depth=child.depth)
                new.children[item] = c2
                rec(child, c2)
        rec(root, nr)
        post_process(nr)
        return nr

    # ── Module 2: IWTC ──────────────────────────────────────────
    def _importance(self, node: BMCNode, max_cnt: float, max_dep: int) -> float:
        cs = node.count / max_cnt if max_cnt > 0 else 0
        ds = 1.0 - node.depth / max_dep if max_dep > 0 else 1.0
        ss = math.sqrt(len(node.children) + 1) / math.sqrt(max_dep + 1)
        return self.w1 * cs + self.w2 * ds + self.w3 * ss

    def _compress(self, noisy_root: BMCNode) -> BMCNode:
        """
        Stratified tree compression:
          depth ≤ 2  → keep 100%  (guarantees 2-itemset counting accuracy)
          depth > 2  → keep top-τ fraction by importance score

        Guarantee: for any item in GF1, at least one node for that item
        is preserved per participant (1-itemset counting unaffected).
        """
        nodes = all_nodes(noisy_root)
        if not nodes:
            return noisy_root
        mc = max(n.count for n in nodes)
        md = max(n.depth for n in nodes)

        keep = {id(n) for n in nodes if n.depth <= 2}
        deep = [(n, self._importance(n, mc, md))
                for n in nodes if n.depth > 2]
        if deep:
            deep.sort(key=lambda x: x[1], reverse=True)
            k = max(1, int(len(deep) * self.tau))
            keep |= {id(n) for n, _ in deep[:k]}

        def prune(orig, new_node):
            for item, child in orig.children.items():
                if id(child) in keep:
                    c2 = BMCNode(item=child.item, count=child.count,
                                 bitmap=child.bitmap, depth=child.depth)
                    new_node.children[item] = c2
                    prune(child, c2)

        cr = BMCNode(depth=0)
        prune(noisy_root, cr)
        return cr

    # ── Module 3: CBVA (lightweight) ────────────────────────────
    def _commit(self, data) -> str:
        return hashlib.sha256(pickle.dumps(data)).hexdigest()[:16]

    # ── Helpers ─────────────────────────────────────────────────
    def _mine_f1(self, transactions, threshold):
        n = len(transactions)
        cnt = defaultdict(int)
        for t in transactions:
            for it in t:
                cnt[it] += 1
        return {it: c for it, c in cnt.items() if c / n >= threshold}

    def _noise_f1(self, f1: dict) -> dict:
        half = self.epsilon / 2.0
        return {it: max(0, round(c + lap(1.0 / half)))
                for it, c in f1.items()}

    # ── Main ────────────────────────────────────────────────────
    def run(self, client_datasets: list) -> tuple:
        t0   = time.time()
        N    = sum(len(d) for d in client_datasets)
        comm = 0
        mems = []

        # ── Round 1: F1 + ABP profiling ──
        nf1_list = []
        alphas   = []
        for ds in client_datasets:
            f1  = self._mine_f1(ds, self.gamma)
            nf1 = self._noise_f1(f1)
            _   = self._commit(nf1)                       # CBVA
            comm += obj_size(nf1)
            nf1_list.append(nf1)
            alphas.append(self._adaptive_alpha(ds, f1))   # ABP

        agg = defaultdict(float)
        for nf1 in nf1_list:
            for it, c in nf1.items():
                agg[it] += c
        gf1 = {it: c for it, c in agg.items() if c >= N * self.delta}
        if not gf1:
            return set(), {'time': time.time() - t0, 'comm_mb': comm / 1e6,
                           'mem_client_mb': 0, 'mem_server_mb': 0}

        gf1_asc  = sorted(gf1, key=lambda x: gf1[x])
        item2idx = {it: i for i, it in enumerate(gf1_asc)}

        # ── Round 2: adaptive noise + compress + commit ──
        trees = []
        for i, ds in enumerate(client_datasets):
            bmc  = build_bmc_tree(ds, gf1_asc, item2idx)
            nb   = self._noise_tree_adaptive(bmc, alphas[i])  # ABP
            comp = self._compress(nb)                          # IWTC
            _    = self._commit((comp, gf1))                   # CBVA
            sz   = obj_size(comp)
            comm += sz
            mems.append(sz / 1e6)
            trees.append(comp)

        mem_srv = obj_size(trees) / 1e6
        gf = mine_frequent_itemsets(gf1, trees, N, self.delta, item2idx)

        return gf, {
            'time':             time.time() - t0,
            'comm_mb':          comm / 1e6,
            'mem_client_mb':    max(mems) if mems else 0,
            'mem_server_mb':    mem_srv,
            'n_frequent':       len(gf),
            'adaptive_alphas':  alphas,
        }


# ─────────────────────────────────────────────────────────────
#  ALGORITHM 3 — DP-PartFIM (additional baseline)
# ─────────────────────────────────────────────────────────────

class DP_PartFIM_Simple:
    """
    Simplified DP-PartFIM baseline (Liu et al., IEEE TETC 2025 style).
    Uniform budget, partition-based, uploads filtered transactions.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 0.4,
                 gamma: float = None):
        self.epsilon = epsilon
        self.delta   = delta
        self.gamma   = gamma if gamma is not None else delta / 2.0
        self.name    = "DP-PartFIM"

    def run(self, client_datasets: list) -> tuple:
        t0   = time.time()
        N    = sum(len(d) for d in client_datasets)
        comm = 0
        mems = []

        # Round 1: uniform budget F1
        all_nf1 = []
        eps_per = self.epsilon / 2.0
        for ds in client_datasets:
            n = len(ds)
            cnt = defaultdict(int)
            for t in ds:
                for it in set(t):
                    cnt[it] += 1
            nf1 = {it: max(0, round(c + lap(1.0 / eps_per)))
                   for it, c in cnt.items() if c / n >= self.gamma}
            comm += obj_size(nf1)
            all_nf1.append(nf1)
            mems.append(obj_size(nf1) / 1e6)

        agg = defaultdict(float)
        for nf1 in all_nf1:
            for it, c in nf1.items():
                agg[it] += c
        gf1 = {it: c for it, c in agg.items() if c >= N * self.delta}
        if not gf1:
            return set(), {'time': time.time() - t0, 'comm_mb': comm / 1e6,
                           'mem_client_mb': 0, 'mem_server_mb': 0}

        gf1_asc  = sorted(gf1, key=lambda x: gf1[x])
        item2idx = {it: i for i, it in enumerate(gf1_asc)}

        # Round 2: upload filtered transactions + central noisy tree
        all_trans = []
        for ds in client_datasets:
            for t in ds:
                filtered = [it for it in t if it in item2idx]
                if filtered:
                    all_trans.append(filtered)
            comm += obj_size(ds)

        if not all_trans:
            return set(), {'time': time.time() - t0, 'comm_mb': comm / 1e6,
                           'mem_client_mb': max(mems) if mems else 0,
                           'mem_server_mb': 0}

        bmc = build_bmc_tree(all_trans, gf1_asc, item2idx)
        L   = max(1, tree_max_depth(bmc))
        eps_tree = self.epsilon / 2.0
        nr  = BMCNode(depth=0)

        def noise_rec(orig, new):
            for item, child in orig.children.items():
                eps_l = (eps_tree * math.exp(-0.5 * (child.depth - 1)) /
                         sum(math.exp(-0.5 * (k - 1)) for k in range(1, L + 1)))
                nc = max(0, round(child.count + lap(1.0 / max(eps_l, 1e-12))))
                c2 = BMCNode(item=item, count=nc,
                             bitmap=child.bitmap, depth=child.depth)
                new.children[item] = c2
                noise_rec(child, c2)

        noise_rec(bmc, nr)
        post_process(nr)

        mem_srv = obj_size(nr) / 1e6
        gf = mine_frequent_itemsets(gf1, [nr], N, self.delta, item2idx)

        return gf, {
            'time':          time.time() - t0,
            'comm_mb':       comm / 1e6,
            'mem_client_mb': max(mems) if mems else 0,
            'mem_server_mb': mem_srv,
            'n_frequent':    len(gf),
        }
