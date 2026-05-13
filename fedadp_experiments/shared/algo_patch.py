"""
shared/algo_patch.py
====================
Minimal patch to FedADP_FIM that adds the `force_alpha` parameter
needed by Experiment E3 (ablation study).

USAGE
-----
Import this module AFTER importing from algorithms.py.
It monkey-patches FedADP_FIM in-place without modifying the source file.

    from algorithms import FedADP_FIM
    from algo_patch import patch_fedadp_fim
    patch_fedadp_fim()
    # Now FedADP_FIM accepts force_alpha=<float>

WHAT IT DOES
------------
Adds optional `force_alpha` parameter to FedADP_FIM.__init__.
When set, it overrides the adaptive α computed by _adaptive_alpha()
with the specified fixed value — enabling clean ablation of IWTC alone.

This is the correct way to isolate IWTC contribution:
  - force_alpha=0.5 pins α to FedDP-FPM default → only IWTC is active
  - tau=1.0 disables compression → only ABP is active
  - Both default → full FedADP-FIM
"""

import sys, os

# Ensure src/ is in path before patching
_SRC = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
sys.path.insert(0, os.path.abspath(_SRC))

from algorithms import FedADP_FIM


def patch_fedadp_fim():
    """Apply force_alpha patch to FedADP_FIM class."""

    # Store original __init__ and run
    _orig_init = FedADP_FIM.__init__
    _orig_run  = FedADP_FIM.run

    def _patched_init(self, epsilon=1.0, delta=0.4, gamma=None,
                      tau=0.7, w1=0.5, w2=0.3, w3=0.2,
                      force_alpha=None):
        """Extended __init__ with optional force_alpha parameter."""
        _orig_init(self, epsilon=epsilon, delta=delta, gamma=gamma,
                   tau=tau, w1=w1, w2=w2, w3=w3)
        self.force_alpha = force_alpha
        # Update name to reflect ablation state
        if force_alpha is not None:
            self.name = f"FedADP-FIM(+IWTC_only, α={force_alpha}, τ={tau})"
        elif tau >= 1.0:
            self.name = f"FedADP-FIM(+ABP_only, τ={tau})"

    def _patched_run(self, client_datasets):
        """
        Patched run: if force_alpha is set, override adaptive α per client.
        All other logic unchanged.
        """
        import time
        import numpy as np
        from collections import defaultdict
        from algorithms import (build_bmc_tree, obj_size, lap,
                                mine_frequent_itemsets, BMCNode)

        t0   = time.time()
        N    = sum(len(d) for d in client_datasets)
        comm = 0
        mems = []

        # Round 1: F1 + (optionally overridden) ABP
        nf1_list = []
        alphas   = []
        for ds in client_datasets:
            f1  = self._mine_f1(ds, self.gamma)
            nf1 = self._noise_f1(f1)
            comm += obj_size(nf1)
            nf1_list.append(nf1)

            # KEY PATCH: use force_alpha if set, else adaptive
            if self.force_alpha is not None:
                alphas.append(float(self.force_alpha))
            else:
                alphas.append(self._adaptive_alpha(ds, f1))

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

        # Round 2: adaptive noise + compress
        trees = []
        for i, ds in enumerate(client_datasets):
            bmc  = build_bmc_tree(ds, gf1_asc, item2idx)
            nb   = self._noise_tree_adaptive(bmc, alphas[i])
            comp = self._compress(nb)
            sz   = obj_size(comp)
            comm += sz
            mems.append(sz / 1e6)
            trees.append(comp)

        import pickle
        mem_srv = len(pickle.dumps(trees)) / 1e6
        gf = mine_frequent_itemsets(gf1, trees, N, self.delta, item2idx)

        return gf, {
            'time':            time.time() - t0,
            'comm_mb':         comm / 1e6,
            'mem_client_mb':   max(mems) if mems else 0,
            'mem_server_mb':   mem_srv,
            'n_frequent':      len(gf),
            'adaptive_alphas': alphas,
            'force_alpha':     self.force_alpha,
        }

    # Apply patches
    FedADP_FIM.__init__ = _patched_init
    FedADP_FIM.run      = _patched_run
    print("[PATCH] FedADP_FIM patched: force_alpha parameter enabled.")


# Auto-apply when imported
patch_fedadp_fim()
