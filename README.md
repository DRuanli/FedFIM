# FedADP-FIM

**Adaptive-Budget Federated Frequent Itemset Mining with Importance-Weighted Tree Compression under Differential Privacy**

Implements and compares three algorithms:
| Algorithm | Description |
|---|---|
| **FedDP-FPM** | Baseline — Yu et al. KBS 2026, all implementation bugs corrected |
| **FedADP-FIM** | Proposed — ABP + IWTC modules for adaptive budget and tree compression |
| **DP-PartFIM** | Additional baseline — Liu et al. IEEE TETC 2025 style |

---

## Folder Structure

```
fedadp_fim/
├── src/
│   ├── algorithms.py       ← All three algorithm implementations
│   └── data_utils.py       ← Data loading, splitting, GT, evaluation
├── scripts/
│   ├── run_experiments.py  ← Full experiment runner (E1–E5)
│   └── quick_test.py       ← Sanity check (~30s)
├── data/                   ← Put chess.txt and mushroom.txt here
│   ├── chess.txt
│   └── mushroom.txt
├── results/                ← Auto-created by experiments
│   ├── gt_cache.pkl        ← Cached ground truth (saves recomputation)
│   ├── exp_results.pkl     ← All experiment results
│   ├── figures/
│   │   └── results.png     ← Publication figure
│   └── tables/
│       └── summary.csv     ← LaTeX-ready CSV
└── requirements.txt
```

---

## Installation

```bash
# 1. Create virtualenv (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

Only `numpy` and `matplotlib` are required — no deep learning frameworks.

---

## Download Datasets

Download from [SPMF Dataset Page](https://www.philippe-fournier-viger.com/spmf/index.php?link=datasets.php):

| File | Dataset | Transactions | Items | Format |
|---|---|---|---|---|
| `chess.dat` | Chess | 3,196 | 75 | Plain SPMF |
| `mushroom.dat` | Mushroom | 8,124 | 119 | Plain SPMF |

Rename to `chess.txt` and `mushroom.txt`, place in `data/`.

---

## Quick Test (verify setup)

```bash
python scripts/quick_test.py
```

Expected output:
```
FedDP-FPM  : F1=90.x%  comm=xx.xKB  t=0.0xs
FedADP-FIM : F1=95.x%  comm=xx.xKB  t=0.0xs
✓ Setup OK. F1 >= 85% at ε=1000 confirms correct NegFIN implementation.
```

If F1 < 85% at ε=1000, something is wrong with the dataset file.

---

## Run Full Experiments

```bash
# All 5 experiments (E1–E5) — takes ~10–20 min depending on hardware
python scripts/run_experiments.py

# Use cached GT (skip ~60s recomputation after first run)
python scripts/run_experiments.py --use_cache

# Single experiment
python scripts/run_experiments.py --exp e1   # Main comparison table
python scripts/run_experiments.py --exp e2   # ε sensitivity
python scripts/run_experiments.py --exp e3   # τ sensitivity
python scripts/run_experiments.py --exp e4   # Client scalability
python scripts/run_experiments.py --exp e5   # GLoss analysis

# Custom parameters
python scripts/run_experiments.py --epsilon 3.0 --tau 0.5 --nruns 10
```

Outputs are saved to `results/`.

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `epsilon` | 1.0 | Privacy budget ε (lower = more private, less accurate) |
| `delta` | 0.85 | Minimum support threshold δ |
| `tau` | 0.7 | IWTC compression ratio (0.1–1.0; 1.0 = no compression) |
| `alpha` | auto | Budget decay (FedDP-FPM: fixed 0.5; FedADP-FIM: adaptive) |
| `nruns` | 5 | Repetitions for mean±std (use 10+ for paper submission) |

---

## Using the API Directly

```python
import numpy as np
from src.algorithms import FedDP_FPM, FedADP_FIM, DP_PartFIM_Simple
from src.data_utils  import load_spmf, split_non_iid, compute_gt, compute_f1_score

np.random.seed(42)

# Load data
chess  = load_spmf('data/chess.txt')
splits = split_non_iid(chess)                  # 3 clients: 70/20/10%

# Ground truth (exact, may take ~60s for Chess δ=0.85)
all_t  = [t for s in splits for t in s]
gt     = compute_gt(all_t, delta=0.85)

# Run algorithms
fdp_pred,  fdp_stats  = FedDP_FPM(epsilon=1.0, delta=0.85).run(splits)
fadp_pred, fadp_stats = FedADP_FIM(epsilon=1.0, delta=0.85, tau=0.7).run(splits)
part_pred, part_stats = DP_PartFIM_Simple(epsilon=1.0, delta=0.85).run(splits)

# Evaluate
for name, pred, stats in [('FedDP-FPM', fdp_pred, fdp_stats),
                           ('FedADP-FIM', fadp_pred, fadp_stats),
                           ('DP-PartFIM', part_pred, part_stats)]:
    p, r, f = compute_f1_score(pred, gt)
    print(f"{name}: F1={f*100:.1f}%  Comm={stats['comm_mb']*1024:.1f}KB  "
          f"t={stats['time']:.3f}s")
```

---

## Ground Truth Caching

First run computes GT using Apriori (slow for Chess δ=0.85, ~60s).
Result is cached to `results/gt_cache.pkl` automatically.

Subsequent runs with `--use_cache` skip GT computation.

To recompute from scratch:
```bash
rm results/gt_cache.pkl
python scripts/run_experiments.py
```

---

## Implementation Notes

### Bug Fixes vs Original FedDP-FPM Paper (Yu et al. KBS 2026)

Three bugs were identified and corrected:

**BUG-3 [CRITICAL]** `count_itemset` using leaf-based scan  
→ Error rate: **93% on Chess** (228/246 pairs wrong)  
→ Fix: direct bitmap scan — support(P) = Σ count of target-item nodes  
   where bitmap covers all other items  
→ F1 at ε=1000: 29% → **92.7%** after fix

**BUG-1** F1 noise scale: `Lap(4/ε)` → `Lap(2/ε)`  
**BUG-2** Level index: `parent.depth` → `child.depth` in budget allocation

### FedADP-FIM Contributions

**[ABP] Adaptive Budget Profiler**  
Per-client α_i = ln(β_i)/L_i from tree structure.  
Theorem: minimises total MSE under budget constraint Σε_l = ε/2.

**[IWTC] Importance-Weighted Tree Compressor**  
Stratified: depth ≤ 2 kept fully (correct 2-itemset counting),  
depth > 2 kept top-τ by importance = w₁·count + w₂·depth + w₃·struct.

---

## Citation

```bibtex
@article{fedadpfim2026,
  title   = {FedADP-FIM: Adaptive-Budget Federated Frequent Itemset Mining
             with Importance-Weighted Tree Compression under Differential Privacy},
  author  = {[Author Names]},
  journal = {Expert Systems with Applications},
  year    = {2026}
}
```
# FedFIM
