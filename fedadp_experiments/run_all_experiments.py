"""
run_all_experiments.py
======================
Master runner for all FedADP-FIM experiments.

Runs E1 → E7 in sequence, with timing and a final summary report.

USAGE
-----
    # Run all experiments:
    python run_all_experiments.py

    # Run specific experiments:
    python run_all_experiments.py --exps e1 e2 e3

    # Skip GT recomputation (use cache from prior run):
    python run_all_experiments.py --use_cache

    # Quick mode (fewer runs, for testing):
    python run_all_experiments.py --quick

EXPERIMENT NARRATIVE ARC (for paper Section 5)
-----------------------------------------------
  E1  → Motivation: why fixed-α is sub-optimal  (Sec 5.1)
  E2  → Overall performance: FedADP-FIM wins     (Sec 5.2)
  E3  → Ablation: ABP vs IWTC contributions      (Sec 5.3)
  E4  → Privacy-utility: Pareto frontier         (Sec 5.4)
  E5  → Scalability: clients + τ sensitivity     (Sec 5.5)
  E6  → Non-IID robustness + GLoss analysis      (Sec 5.6)
  E7  → Theory validation + complexity           (Sec 5.7)

Each experiment is self-contained and writes to results/exp_eX/
"""

import argparse
import importlib
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'shared'))

# Experiment modules (in narrative order)
EXP_MODULES = {
    'e1': 'exp_e1.run_e1',
    'e2': 'exp_e2.run_e2',
    'e3': 'exp_e3.run_e3',
    'e4': 'exp_e4.run_e4',
    'e5': 'exp_e5.run_e5',
    'e6': 'exp_e6.run_e6',
    'e7': 'exp_e7.run_e7',
}

EXP_TITLES = {
    'e1': 'Motivation: Fixed-Decay Limitation',
    'e2': 'Overall Performance Comparison',
    'e3': 'Ablation: ABP vs IWTC',
    'e4': 'Privacy-Utility Pareto Frontier',
    'e5': 'Scalability & τ Sensitivity',
    'e6': 'Non-IID Robustness & GLoss',
    'e7': 'Theory Validation & Complexity',
}

EXP_RQS = {
    'e1': 'RQ1: Is fixed α=0.5 sub-optimal?',
    'e2': 'RQ2: Does FedADP-FIM consistently outperform?',
    'e3': 'RQ3: What does ABP vs IWTC each contribute?',
    'e4': 'RQ4: Does advantage hold across all privacy regimes?',
    'e5': 'RQ5: Is FedADP-FIM scalable? Is τ robust?',
    'e6': 'RQ6: Robust under real Non-IID heterogeneity?',
    'e7': 'RQ7: Is Theorem 1 empirically validated?',
}


def run_experiment(exp_id, quick=False):
    """Import and run a single experiment module."""
    module_path = os.path.join(HERE, *EXP_MODULES[exp_id].split('.'))
    module_path += '.py'
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Experiment file not found: {module_path}")

    # Patch NRUNS if quick mode
    if quick:
        import common
        common.NRUNS = 3
        print(f"  [QUICK] NRUNS reduced to 3")

    # Dynamically import and run
    spec = importlib.util.spec_from_file_location(
        EXP_MODULES[exp_id], module_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def print_summary(results):
    """Print a clean summary table of all experiment outcomes."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE SUMMARY")
    print("=" * 70)
    print(f"{'Exp':5s}  {'Title':38s}  {'Status':8s}  {'Time':8s}")
    print("-" * 70)
    total_time = 0
    for exp_id, (status, elapsed, error) in results.items():
        title = EXP_TITLES[exp_id][:38]
        t_str = f"{elapsed:.1f}s" if elapsed else "—"
        s_str = "✓ OK" if status else "✗ FAIL"
        print(f"{exp_id.upper():5s}  {title:38s}  {s_str:8s}  {t_str:8s}")
        if error:
            print(f"       Error: {str(error)[:60]}")
        if elapsed:
            total_time += elapsed
    print("-" * 70)
    n_ok   = sum(1 for s, _, _ in results.values() if s)
    n_fail = sum(1 for s, _, _ in results.values() if not s)
    print(f"{'TOTAL':5s}  {n_ok}/{len(results)} passed  "
          f"{'':28s}  {total_time:.0f}s")
    print("=" * 70)

    if n_fail == 0:
        print("\n✓ All experiments complete. Ready to compile paper figures.")
        print(f"\nOutput directories:")
        results_dir = os.path.join(HERE, '..', 'results')
        for exp_id in results:
            exp_dir = os.path.join(results_dir, f'exp_{exp_id}')
            if os.path.exists(exp_dir):
                n_figs = len([f for f in os.listdir(
                    os.path.join(exp_dir, 'figures'))
                    if f.endswith('.pdf')])
                n_tabs = len([f for f in os.listdir(
                    os.path.join(exp_dir, 'tables'))
                    if f.endswith('.tex')])
                print(f"  {exp_id.upper()}: {n_figs} figures, "
                      f"{n_tabs} LaTeX tables → {exp_dir}")
    else:
        print(f"\n⚠ {n_fail} experiment(s) failed. Check error messages above.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='FedADP-FIM Master Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Narrative arc:
  E1 → Motivation → E2 → Main Results → E3 → Ablation →
  E4 → Privacy → E5 → Scalability → E6 → Non-IID → E7 → Theory
        """
    )
    parser.add_argument('--exps', nargs='+',
                        choices=list(EXP_MODULES.keys()),
                        default=list(EXP_MODULES.keys()),
                        help='Which experiments to run (default: all)')
    parser.add_argument('--use_cache', action='store_true',
                        help='Use cached GT (skip slow Apriori computation)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: NRUNS=3 for testing')
    args = parser.parse_args()

    print("=" * 70)
    print("FedADP-FIM — Full Experimental Suite")
    print("Targeting: IEEE TKDE")
    print("=" * 70)
    print(f"\nExperiments to run: {', '.join(e.upper() for e in args.exps)}")
    if args.quick:
        print("Mode: QUICK (NRUNS=3)")
    if args.use_cache:
        print("GT: Using cache")
    print()

    # Print narrative overview
    print("Narrative Arc:")
    for exp_id in args.exps:
        rq = EXP_RQS[exp_id]
        print(f"  [{exp_id.upper()}] {EXP_TITLES[exp_id]}")
        print(f"       {rq}")
    print()

    # Run experiments
    results = {}
    for exp_id in args.exps:
        print("\n" + "─" * 65)
        print(f"Running [{exp_id.upper()}] — {EXP_TITLES[exp_id]}")
        print(f"Research Question: {EXP_RQS[exp_id]}")
        print("─" * 65)

        t0 = time.time()
        try:
            # Inject use_cache flag into common before each exp
            if args.use_cache:
                os.environ['FEDADP_USE_CACHE'] = '1'

            run_experiment(exp_id, quick=args.quick)
            elapsed = time.time() - t0
            results[exp_id] = (True, elapsed, None)
            print(f"\n  ✓ [{exp_id.upper()}] Complete in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            results[exp_id] = (False, elapsed, e)
            print(f"\n  ✗ [{exp_id.upper()}] FAILED after {elapsed:.1f}s")
            traceback.print_exc()

    # Summary
    print_summary(results)


if __name__ == '__main__':
    main()
