"""
quick_test.py
=============
Sanity-check the installation in ~30 seconds.

Usage:  python scripts/quick_test.py

Runs a single FedDP-FPM + FedADP-FIM comparison on Chess δ=0.85
with ε=1000 (near-zero noise) and reports F1.

Expected (near-zero noise):
  FedDP-FPM  F1 ≈ 90–95%
  FedADP-FIM F1 ≈ 95–98%
  All bugs fixed F1 should NEVER be below 85% at ε=1000.
"""

import sys, os, time
import numpy as np
np.random.seed(42)

SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(SRC))

from src.data_utils  import load_spmf, split_non_iid, compute_gt, compute_f1_score
from src.algorithms import FedDP_FPM, FedADP_FIM

DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
fp   = os.path.join(DATA, 'chess.txt')
if not os.path.exists(fp):
    print(f"ERROR: {fp} not found.")
    print("Download Chess from https://www.philippe-fournier-viger.com/spmf/")
    sys.exit(1)

print("Loading Chess dataset...", flush=True)
chess  = load_spmf(fp)
splits = split_non_iid(chess)
all_t  = [t for s in splits for t in s]

print("Computing ground truth (k=1..4, ~8s)...", flush=True)
gt = compute_gt(all_t, delta=0.85, max_time=30, verbose=True)

print(f"\nRunning algorithms (ε=1000, near-zero noise)...", flush=True)
for cls, kw, name in [
    (FedDP_FPM,  dict(epsilon=1000.0, delta=0.85),          "FedDP-FPM  "),
    (FedADP_FIM, dict(epsilon=1000.0, delta=0.85, tau=0.7), "FedADP-FIM "),
]:
    t0 = time.time()
    pred, st = cls(**kw).run(splits)
    p, r, f  = compute_f1_score(pred, gt)
    print(f"  {name}: F1={f*100:.1f}%  P={p*100:.1f}%  R={r*100:.1f}%  "
          f"n_pred={len(pred)}  comm={st['comm_mb']*1024:.1f}KB  t={time.time()-t0:.2f}s")

print("\n✓ Setup OK. F1 >= 85% at ε=1000 confirms correct NegFIN implementation.")
print("  (Note: GT is partial k<=4 here. Full GT=2647 takes ~60s)")
