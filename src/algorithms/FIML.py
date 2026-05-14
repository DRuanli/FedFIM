"""
FIML — Frequent Itemset Mining with Local Differential Privacy
================================================================
Li J., Gan W., Gui Y., Wu Y., Yu P.S.
"Frequent Itemset Mining with Local Differential Privacy"
CIKM '22, October 17-21, 2022, Atlanta, GA, USA. ACM, pp. 1146-1155.
DOI: 10.1145/3511808.3557327

Privacy model
-------------
Local Differential Privacy (LDP): each user holds one transaction
record and applies a randomization algorithm locally before sending
data to an untrusted data collector.  The data collector never sees
raw transactions.  The whole algorithm satisfies ε-LDP via parallel
composition (three disjoint user groups, each using budget ε).

Comparison note
---------------
FIML is a cross-device LDP algorithm; FedDP-FPM is a cross-silo
federated algorithm with central DP.  The two operate under different
privacy models and deployment settings.  FIML is used as a baseline
in Yu et al. (KBS 2026, Table 4) with this same distinction noted.

Algorithm overview (Section 4, three stages)
--------------------------------------------
  Stage 1 — Domain pruning
    1/3 of users report via PSFO(L=1, OLH, ε).
    DC estimates item frequencies and selects top-1.5k items → S.

  Stage 2 — Candidate 1-item frequency estimation
    1/3 of users answer a binary query (is candidate v in your
    transaction?) using Randomized Response (RR).
    DC estimates true frequencies → selects top-k items → S', F'.

  Stage 3 — Frequent itemset mining
    Candidate itemsets IS are built from S' using frequency product
    heuristic (Eq. 11).  1/3 of users answer binary RR queries.
    DC estimates itemset frequencies → top-k frequent itemsets.

Interface
---------
FIML.run(client_datasets) mirrors FedDP_FPM.run().
client_datasets is flattened: each transaction becomes one user's
record, which matches the LDP single-user-per-transaction assumption.
"""

import hashlib
import math
import pickle
import time
from collections import defaultdict
from itertools import combinations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _obj_size(obj) -> int:
    return len(pickle.dumps(obj))


# ─────────────────────────────────────────────────────────────────────────────
#  OLH primitives  (Section 3.3.2)
# ─────────────────────────────────────────────────────────────────────────────

def _olh_hash(val: int, seed: int, g: int) -> int:
    """
    Deterministic hash H_{seed}: Z → {0,...,g-1}.
    Used by each user to map their item index to the OLH domain.
    """
    raw = int(hashlib.md5(f"{val}_{seed}".encode()).hexdigest(), 16)
    return raw % g


def _olh_perturb(item_idx: int, g: int, seed: int,
                  rng: np.random.Generator) -> tuple[int, int]:
    """
    OLH perturbation (Section 3.3.2, Eq. 5):
      g = ⌈e^ε + 1⌉, p = 0.5, q = 1/g.
      Map item to y = H_seed(item) ∈ {0,...,g-1}.
      With prob p=0.5 report y; else report uniform random in {0,...,g-1}.

    Returns (perturbed_hash_value, seed).
    """
    y = _olh_hash(item_idx, seed, g)
    if rng.random() < 0.5:          # p = 0.5
        return y, seed
    return int(rng.integers(g)), seed


def _olh_estimate(reports: list[tuple[int, int]],
                  item_indices: list[int],
                  g: int) -> dict[int, float]:
    """
    Aggregate OLH reports and return estimated count for each item index.

    Eq. (6): f̂_x = (C(x) - n/g) / (p - 1/g)
    where p=0.5, q=1/g, and
    C(x) = |{j : H_{seed_j}(x) = y_j}|  — reports consistent with x.
    """
    n = len(reports)
    p = 0.5
    q = 1.0 / g
    denom = p - q

    estimates = {}
    for idx in item_indices:
        C = sum(1 for (y, seed) in reports if _olh_hash(idx, seed, g) == y)
        estimates[idx] = (C - n * q) / denom
    return estimates


# ─────────────────────────────────────────────────────────────────────────────
#  Randomized Response (RR)  (Section 3.3.1 / Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def _rr_perturb(b: int, p: float, rng: np.random.Generator) -> int:
    """
    Binary GRR (d=2): send true bit b with probability p, flip with 1-p.
    p = e^ε / (e^ε + 1),  q = 1 / (e^ε + 1).
    """
    return b if rng.random() < p else (1 - b)


def _rr_estimate(C: int, n: int, p: float) -> float:
    """
    Unbiased estimate of true count from binary RR reports (Eq. 3).
    C = number of reported 1s among n users.
    Returns estimated count of true 1s.
    """
    q = 1.0 - p
    return (C - n * q) / (p - q)


# ─────────────────────────────────────────────────────────────────────────────
#  FIML
# ─────────────────────────────────────────────────────────────────────────────

class FIML:
    """
    FIML — Frequent Itemset Mining with Local Differential Privacy.

    Parameters
    ----------
    epsilon : float
        Privacy budget ε.  Each of the three user groups satisfies
        ε-LDP; the overall algorithm satisfies ε-LDP by parallel
        composition (Theorem 4.1 in the paper).
    delta : float
        Minimum support threshold δ, used to derive k = ⌊N·δ⌋ where
        N is the total number of users (transactions).  This converts
        the paper's top-k formulation to a min-support threshold for
        fair comparison with FedDP-FPM.
    seed : int or None
        NumPy random seed for reproducibility.
    """

    def __init__(self,
                 epsilon: float = 1.0,
                 delta:   float = 0.4,
                 seed:    int | None = None):
        self.epsilon = epsilon
        self.delta   = delta
        self.name    = "FIML"
        self._rng    = np.random.default_rng(seed)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _rr_p(self) -> float:
        """RR true-bit probability p = e^ε / (e^ε + 1)."""
        e = math.exp(self.epsilon)
        return e / (e + 1.0)

    @property
    def _rr_min_freq(self) -> float:
        """
        Minimum possible estimated frequency from binary RR.
        a = -1 / (e^ε - 1)  (Section 4.2.3, normalization step).
        """
        return -1.0 / (math.exp(self.epsilon) - 1.0)

    def _build_candidate_itemsets(self,
                                   S_prime: list,
                                   freq_norm: dict,
                                   k: int) -> list:
        """
        Eq. (11): IS = {x ⊆ S', 1 < |x| < log2(k), Π_{i∈x} A'(i) > t}
        where |IS| = 1.5k by choosing threshold t.

        All qualifying itemsets are scored by their product of normalised
        frequencies; the top-1.5k by score are returned.
        """
        max_size = max(2, int(math.log2(max(k, 2))))  # upper bound on |x|
        target   = min(int(1.5 * k), 500)             # cap to avoid OOM

        candidates = []
        S_list = list(S_prime)

        for size in range(2, max_size + 1):
            for combo in combinations(S_list, size):
                score = 1.0
                for item in combo:
                    score *= freq_norm.get(item, 0.0)
                if score > 0:
                    candidates.append((frozenset(combo), score))

        # Sort by guessed frequency, take top 1.5k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cand for cand, _ in candidates[:target]]

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, client_datasets: list) -> tuple[set, dict]:
        """
        Execute FIML on client datasets.

        The datasets are flattened into individual transactions; each
        transaction represents one LDP user's record, consistent with
        the paper's single-user-per-record assumption.

        Parameters
        ----------
        client_datasets : list of list-of-transactions

        Returns
        -------
        frequent : set of frozensets — globally frequent itemsets found
        stats    : dict with keys
            time, comm_mb, mem_client_mb, mem_server_mb, n_frequent
        """
        t0 = time.time()

        # Flatten: each transaction = one user
        all_trans = [t for ds in client_datasets for t in ds]
        N = len(all_trans)
        if N < 3:
            return set(), {
                'time': time.time() - t0, 'comm_mb': 0.0,
                'mem_client_mb': 0.0, 'mem_server_mb': 0.0,
                'n_frequent': 0,
            }

        # k from min-support threshold (convert top-k ↔ min-support)
        k = max(1, int(N * self.delta))

        # Split users into three equal disjoint groups
        n1 = N // 3
        n2 = N // 3
        n3 = N - n1 - n2
        g1 = all_trans[:n1]
        g2 = all_trans[n1:n1 + n2]
        g3 = all_trans[n1 + n2:]

        # Global item universe and index mapping
        all_items = sorted({it for t in all_trans for it in t})
        item_to_idx = {it: i for i, it in enumerate(all_items)}
        idx_to_item = {i: it for it, i in item_to_idx.items()}
        d = len(all_items)

        comm = 0   # total bytes: users → DC + DC → users

        # OLH domain size (best g = ⌈e^ε + 1⌉, Section 3.3.2)
        g_olh = max(2, math.ceil(math.exp(self.epsilon) + 1))

        # ── Stage 1 — Domain Pruning via PSFO(L=1, OLH, ε) ──────────────────
        # Each user: randomly sample 1 item from transaction → OLH perturb.
        # PSFO with L=1: no padding needed; estimate × L=1 (no correction).

        reports_s1: list[tuple[int, int]] = []
        for trans in g1:
            if not trans:
                # Empty transaction: uniform random report
                seed = int(self._rng.integers(0, 2**31))
                reports_s1.append((int(self._rng.integers(g_olh)), seed))
            else:
                item = all_items[int(self._rng.integers(len([it for it in trans
                                                              if it in item_to_idx]))
                                     if [it for it in trans if it in item_to_idx]
                                     else 0)]
                # Pick a valid item from the transaction
                valid = [it for it in trans if it in item_to_idx]
                if not valid:
                    seed = int(self._rng.integers(0, 2**31))
                    reports_s1.append((int(self._rng.integers(g_olh)), seed))
                    continue
                item = valid[int(self._rng.integers(len(valid)))]
                seed = int(self._rng.integers(0, 2**31))
                y, seed = _olh_perturb(item_to_idx[item], g_olh, seed, self._rng)
                reports_s1.append((y, seed))

        # Each report: (int, int) ≈ 8 bytes
        comm += len(reports_s1) * 8

        # DC estimates frequencies of all items
        est_s1 = _olh_estimate(reports_s1, list(range(d)), g_olh)
        # {item_idx: estimated_count}; multiply by L=1 (PSFO correction)

        # Select top 1.5k candidate items → S
        k_cand = min(d, max(1, int(1.5 * k)))
        top_indices = sorted(est_s1, key=lambda i: est_s1[i], reverse=True)[:k_cand]
        S = [idx_to_item[i] for i in top_indices]  # candidate item set
        S_set = set(S)

        # DC broadcasts S to group 2 (one item per user query)
        comm += _obj_size(S)

        # ── Stage 2 — Candidate Frequency Estimation via Binary RR ───────────
        # For each user in g2: DC sends random v ∈ S; user reports
        # b = I(v ∈ transaction) via RR; DC estimates f(v) for each v in S.

        p_rr    = self._rr_p
        counts_s2: dict[str, list[int]] = defaultdict(list)

        if S:
            for trans in g2:
                candidate = S[int(self._rng.integers(len(S)))]
                comm += _obj_size(candidate)      # DC → user: 1 item name

                b = 1 if candidate in set(trans) else 0
                r = _rr_perturb(b, p_rr, self._rng)
                counts_s2[candidate].append(r)
                comm += 1                         # user → DC: 1 bit

        # Estimate frequency (as proportion) for each item in S
        freq_s2: dict[str, float] = {}
        for item in S:
            responses = counts_s2[item]
            n_item    = len(responses)
            if n_item == 0:
                freq_s2[item] = 0.0
            else:
                C = sum(responses)
                count_est     = _rr_estimate(C, n_item, p_rr)
                freq_s2[item] = count_est / n_item  # normalised ∈ [0, 1]

        # Select top-k items → S', with their estimated frequencies F'
        S_prime_items = sorted(freq_s2, key=lambda x: freq_s2[x], reverse=True)[:k]
        S_prime = set(S_prime_items)

        if not S_prime:
            return set(), {
                'time':           time.time() - t0,
                'comm_mb':        comm / 1e6,
                'mem_client_mb':  0.0,
                'mem_server_mb':  0.0,
                'n_frequent':     0,
            }

        # ── Stage 3 — Frequent Itemset Mining ────────────────────────────────
        # 3a. Normalise Stage 2 frequencies (Section 4.2.3, Eq. after Eq. 11)
        #     A'(x) = 0.9 * (A(x) - a) / (max A(x) - a)
        #     a = minimum possible RR estimate = -1/(e^ε - 1)

        a = self._rr_min_freq
        max_freq = max(freq_s2.get(it, 0.0) for it in S_prime_items)
        denom_norm = max_freq - a if max_freq > a else 1e-9

        freq_norm: dict[str, float] = {}
        for it in S_prime_items:
            raw = freq_s2.get(it, 0.0)
            freq_norm[it] = 0.9 * (raw - a) / denom_norm

        # 3b. Build candidate itemsets IS (Eq. 11)
        IS = self._build_candidate_itemsets(S_prime_items, freq_norm, k)

        mem_srv = _obj_size(IS) / 1e6

        if not IS:
            # Return 1-itemsets only
            frequent = {frozenset([it]) for it in S_prime_items}
            return frequent, {
                'time':           time.time() - t0,
                'comm_mb':        comm / 1e6,
                'mem_client_mb':  0.0,
                'mem_server_mb':  mem_srv,
                'n_frequent':     len(frequent),
            }

        # 3c. Binary RR on candidate itemsets (Algorithm 1, same as Stage 2)
        counts_s3: dict[int, list[int]] = defaultdict(list)   # IS index → bits

        for trans in g3:
            if not IS:
                break
            cand_idx  = int(self._rng.integers(len(IS)))
            candidate = IS[cand_idx]
            comm += _obj_size(candidate)          # DC → user

            b = 1 if candidate.issubset(set(trans)) else 0
            r = _rr_perturb(b, p_rr, self._rng)
            counts_s3[cand_idx].append(r)
            comm += 1                             # user → DC: 1 bit

        # 3d. Estimate frequency of each candidate itemset (Algorithm 2)
        itemset_freq: list[tuple[frozenset, float]] = []
        for idx, itemset in enumerate(IS):
            responses = counts_s3[idx]
            n_item    = len(responses)
            if n_item == 0:
                itemset_freq.append((itemset, 0.0))
            else:
                C = sum(responses)
                count_est = _rr_estimate(C, n_item, p_rr)
                freq_est  = count_est / n_item    # proportion ∈ [0, 1]
                itemset_freq.append((itemset, freq_est))

        # 3e. Select top-k frequent itemsets
        itemset_freq.sort(key=lambda x: x[1], reverse=True)

        # Convert top-k back to min-support threshold for fair comparison.
        # An itemset is retained if its estimated frequency ≥ delta.
        min_support = self.delta
        frequent: set[frozenset] = set()

        # Always include 1-itemsets from S' (Stage 2 result)
        for it in S_prime_items:
            if freq_s2.get(it, 0.0) >= min_support:
                frequent.add(frozenset([it]))

        # Include k-itemsets (k≥2) from Stage 3 that pass the threshold
        for itemset, freq in itemset_freq:
            if freq >= min_support:
                frequent.add(itemset)

        # If nothing met the threshold, fall back to top-k
        if not frequent:
            for it in S_prime_items[:k]:
                frequent.add(frozenset([it]))
            for itemset, _ in itemset_freq[:k]:
                frequent.add(itemset)

        return frequent, {
            'time':           time.time() - t0,
            'comm_mb':        comm / 1e6,
            'mem_client_mb':  0.0,   # LDP: no local state retained after send
            'mem_server_mb':  mem_srv,
            'n_frequent':     len(frequent),
        }
