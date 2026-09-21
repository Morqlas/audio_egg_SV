"""
Paper artifact: the cross-speaker shuffled-EGG identity control (10/10 low-SNR cells).

Paired t-test for the shuffled-EGG sanity check (pre-registered protocol).

Pre-registered decision rule:
  - Paired t-test across folds, one-tailed, alpha=0.05
  - H0: real-EGG and shuffled-EGG produce equal fused SG-EER
  - H1: real-EGG produces LOWER SG-EER than shuffled-EGG
  - Tested at all -10 dB and -5 dB conditions (10 cells)
  - Decision: gate "uses EGG identity" if H0 rejected on >= 8 of 10 cells

Also reports all 26 conditions for completeness (exploratory, not pre-registered).

Inputs: the per-fold flat CSVs written by src/evaluate.py, committed as
    results/gate_primary/per_fold.csv           (real EGG)
    results/control_shuffled_egg/per_fold.csv   (cross-speaker shuffled EGG)

Each has columns: fold, noise_type, snr_db, n_pairs, sg_eer, sg_min_dcf, ...

Run (defaults already point at the committed CSVs):
    python -m src.controls.shuffled_egg_ttest
"""

import argparse
import csv
from collections import defaultdict

import numpy as np
from scipy import stats
from src import config


def load_per_fold(fn):
    """Returns dict: (noise_type, snr_db_str) -> {fold: sg_eer}."""
    out = defaultdict(dict)
    with open(fn) as f:
        for r in csv.DictReader(f):
            key = (r['noise_type'], str(r['snr_db']))
            out[key][int(r['fold'])] = float(r['sg_eer'])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', default=str(config.RESULTS_DIR / "gate_primary" / "per_fold.csv"))
    parser.add_argument('--shuffled', default=str(config.RESULTS_DIR / "control_shuffled_egg" / "per_fold.csv"))
    parser.add_argument('--output_csv', default=str(config.RESULTS_DIR / "control_shuffled_egg" / "ttest.csv"))
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    real = load_per_fold(args.real)
    shuf = load_per_fold(args.shuffled)

    # Pre-registered cells: all -10 and -5 dB conditions
    NOISE_TYPES = ["cafeteria", "metro", "office", "park", "traffic"]
    # snr_db may be stored as '-10', '-10.0', etc. Normalize by float.
    def find_key(store, noise, snr_val):
        for (n, s) in store:
            if n == noise:
                try:
                    if float(s) == snr_val:
                        return (n, s)
                except ValueError:
                    continue
        return None

    preregistered = []
    for nt in NOISE_TYPES:
        for snr in [-10.0, -5.0]:
            preregistered.append((nt, snr))

    rows = []
    reject_count = 0

    print("=" * 90)
    print("PRE-REGISTERED CELLS (one-tailed paired t-test, H1: real < shuffled)")
    print("=" * 90)
    print(f"{'Condition':<20} {'Real mean':<12} {'Shuf mean':<12} {'t-stat':<10} {'p (1-tail)':<12} {'Reject H0?'}")
    print("-" * 90)

    for nt, snr_val in preregistered:
        rk = find_key(real, nt, snr_val)
        sk = find_key(shuf, nt, snr_val)
        if rk is None or sk is None:
            print(f"  MISSING: {nt} {snr_val}")
            continue

        folds = sorted(set(real[rk].keys()) & set(shuf[sk].keys()))
        real_vals = np.array([real[rk][f] for f in folds])
        shuf_vals = np.array([shuf[sk][f] for f in folds])

        # Paired t-test. scipy two-sided; convert to one-tailed for H1: real < shuffled
        t_stat, p_two = stats.ttest_rel(real_vals, shuf_vals)
        # One-tailed p for H1: real < shuffled (i.e., shuffled > real, so t_stat negative)
        if t_stat < 0:
            p_one = p_two / 2
        else:
            p_one = 1 - p_two / 2

        reject = p_one < args.alpha
        if reject:
            reject_count += 1

        label = f"{nt}_{int(snr_val)}dB"
        print(f"{label:<20} {real_vals.mean()*100:<12.4f} {shuf_vals.mean()*100:<12.4f} "
              f"{t_stat:<10.3f} {p_one:<12.5f} {'YES' if reject else 'no'}")

        rows.append({
            'condition': label, 'noise_type': nt, 'snr_db': int(snr_val),
            'preregistered': True,
            'real_mean': real_vals.mean(), 'shuf_mean': shuf_vals.mean(),
            't_stat': t_stat, 'p_one_tailed': p_one, 'reject_h0': reject,
            'n_folds': len(folds),
        })

    print()
    print(f"PRE-REGISTERED DECISION: H0 rejected on {reject_count}/10 cells "
          f"(threshold: >= 8 to conclude gate uses EGG identity)")
    print(f"OUTCOME: {'PASS - gate uses EGG identity' if reject_count >= 8 else 'FAIL'}")

    # Exploratory: all 26 conditions
    print()
    print("=" * 90)
    print("ALL 26 CONDITIONS (exploratory, not pre-registered)")
    print("=" * 90)
    print(f"{'Condition':<20} {'Real mean':<12} {'Shuf mean':<12} {'t-stat':<10} {'p (1-tail)':<12} {'Reject?'}")
    print("-" * 90)

    all_conditions = [('clean', None)]
    for nt in NOISE_TYPES:
        for snr in [-10.0, -5.0, 0.0, 5.0, 10.0]:
            all_conditions.append((nt, snr))

    for nt, snr_val in all_conditions:
        if snr_val is None:
            # clean: snr_db stored as '' or 'nan'
            rk = next((k for k in real if k[0] == 'clean'), None)
            sk = next((k for k in shuf if k[0] == 'clean'), None)
            label = 'clean'
        else:
            rk = find_key(real, nt, snr_val)
            sk = find_key(shuf, nt, snr_val)
            label = f"{nt}_{int(snr_val)}dB"
        if rk is None or sk is None:
            continue

        folds = sorted(set(real[rk].keys()) & set(shuf[sk].keys()))
        real_vals = np.array([real[rk][f] for f in folds])
        shuf_vals = np.array([shuf[sk][f] for f in folds])
        t_stat, p_two = stats.ttest_rel(real_vals, shuf_vals)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        reject = p_one < args.alpha

        # Skip duplicate print of preregistered ones? No—show all for the full table
        is_prereg = snr_val in (-10.0, -5.0) and nt in NOISE_TYPES
        print(f"{label:<20} {real_vals.mean()*100:<12.4f} {shuf_vals.mean()*100:<12.4f} "
              f"{t_stat:<10.3f} {p_one:<12.5f} {'YES' if reject else 'no'}"
              f"{'  *prereg' if is_prereg else ''}")

        if not is_prereg:  # preregistered already in rows
            rows.append({
                'condition': label, 'noise_type': nt,
                'snr_db': int(snr_val) if snr_val is not None else '',
                'preregistered': False,
                'real_mean': real_vals.mean(), 'shuf_mean': shuf_vals.mean(),
                't_stat': t_stat, 'p_one_tailed': p_one, 'reject_h0': reject,
                'n_folds': len(folds),
            })

    # Save
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()