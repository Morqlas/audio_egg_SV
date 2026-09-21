#!/usr/bin/env python3
"""
Paper artifact: the per-noise-type severity figure.

noise_type_effect.py
====================
Per-noise-type SG-EER vs SNR for a single fusion system (default: the gate),
on a LOG y-axis, with per-cell fold error bars and an optional EGG-only
baseline band.

Answers: "how does noise type affect this system?" -- shows the severity
ordering (which noise is hardest) and how it converges as SNR rises.

Input system CSV must have per-condition rows with columns:
    noise_type, snr_db, sg_eer_mean, sg_eer_std   (EER in [0,1]; clean row ignored)

Optional EGG summary CSV (for the baseline band): a 'metric,mean,std,...' file
containing a test_sg_eer row, OR pass --egg_mean / --egg_std directly.

USAGE
-----
    python -m src.figures.noise_type_effect \
        --system no_trunk_vector_gate_test_aggregated.csv \
        --egg    test_metrics_summary.csv \
        --label  "Vector gate (primary)" \
        --outdir figures

    # or give the EGG baseline numbers directly, skip the file:
    python -m src.figures.noise_type_effect --system gate.csv --egg_mean 2.11 --egg_std 0.42

Notes
-----
* Log y-axis spreads out the low-EER high-SNR points so all noise types stay
  distinguishable across the whole range.
* Lower error bars are clipped at the axis floor: on a log scale a symmetric
  (mean - std) can go <= 0 where the mean is tiny, which would draw a spurious
  spike to zero. Clipping shows the real upward spread without implying a
  meaningless near-zero/negative EER.
* Error bars here are per-CELL fold std (noise type and SNR both fixed), which
  is a coherent uncertainty -- unlike a std pooled across noise types.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from src import _console  # noqa: F401  (UTF-8 safe stdout)

SNR_ORDER = [-10.0, -5.0, 0.0, 5.0, 10.0]
AXIS_FLOOR = 0.04  # just below the smallest y-tick; lower bars clip here

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2,
    "lines.markersize": 8,
    "figure.dpi": 150,
})

# fixed worst->best severity order and consistent styling
NOISE_ORDER = ["cafeteria", "metro", "park", "traffic", "office"]
STYLE = {
    "cafeteria": ("#1f77b4", "o"),
    "metro":     ("#ff7f0e", "s"),
    "park":      ("#d62728", "D"),
    "traffic":   ("#9467bd", "v"),
    "office":    ("#2ca02c", "^"),
}


def _fmt_snr(s):
    """Signed SNR label, but a plain '0' for zero."""
    i = int(s)
    return "0" if i == 0 else f"{i:+d}"


def load_system(path):
    """Return {noise: {snr: (eer_pct, std_pct)}} from a per-condition CSV."""
    if not os.path.exists(path):
        sys.exit(f"[error] system file not found: {path}")
    out = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for need in ("noise_type", "snr_db", "sg_eer_mean"):
            if need not in (reader.fieldnames or []):
                sys.exit(f"[error] {path} missing column '{need}'")
        for r in reader:
            nt = r["noise_type"].strip().lower()
            if nt == "clean" or r["snr_db"].strip() == "":
                continue
            eer = float(r["sg_eer_mean"]) * 100.0
            std = float(r.get("sg_eer_std", 0) or 0) * 100.0
            out.setdefault(nt, {})[float(r["snr_db"])] = (eer, std)
    if not out:
        sys.exit(f"[error] no per-condition rows parsed from {path}")
    return out


def load_egg_baseline(path, mean_arg, std_arg):
    """Resolve EGG baseline mean/std from CLI args or a summary CSV."""
    if mean_arg is not None:
        return mean_arg, std_arg
    if not path:
        return None, None
    if not os.path.exists(path):
        print(f"[warn] EGG file not found, no baseline drawn: {path}", file=sys.stderr)
        return None, None
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r.get("metric", "").strip().lower() in ("test_sg_eer", "sg_eer"):
            mean = float(r["mean"]) * 100.0
            std = float(r["std"]) * 100.0 if r.get("std") else None
            return mean, std
    print(f"[warn] no test_sg_eer row in {path}; no baseline drawn", file=sys.stderr)
    return None, None


def make_plot(per_noise, outpath, egg_mean=None, egg_std=None,
              system_label="Vector gate (primary)", bars_low_snr_only=False):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    # EGG baseline band behind everything
    if egg_mean is not None:
        if egg_std is not None:
            ax.axhspan(egg_mean - egg_std, egg_mean + egg_std,
                       color="gray", alpha=0.25, zorder=1, label="EGG-only ±std")
        ax.axhline(egg_mean, color="black", ls="--", lw=1.5, zorder=2,
                   label=f"EGG-only ({egg_mean:.2f}%)")

    for nt in NOISE_ORDER:
        if nt not in per_noise:
            continue
        color, marker = STYLE.get(nt, ("#444", "o"))
        xs, ys, es = [], [], []
        for s in SNR_ORDER:
            if s in per_noise[nt]:
                xs.append(s)
                ys.append(per_noise[nt][s][0])
                es.append(per_noise[nt][s][1])
        mean_eer = sum(v[0] for v in per_noise[nt].values()) / len(per_noise[nt])

        # build clipped asymmetric error bars (log-axis safe)
        if bars_low_snr_only:
            lower = [e if x <= -5 else 0.0 for x, e in zip(xs, es)]
            upper = [e if x <= -5 else 0.0 for x, e in zip(xs, es)]
        else:
            upper = es
            lower = [min(e, max(y - AXIS_FLOOR, 0.0)) for y, e in zip(ys, es)]

        ax.errorbar(xs, ys, yerr=[lower, upper], color=color, marker=marker,
                    capsize=3, label=f"{nt} (mean {mean_eer:.2f}%)", zorder=3)

    ax.set_yscale("log")
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 3]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.set_xticks(SNR_ORDER)
    ax.set_xticklabels([_fmt_snr(s) for s in SNR_ORDER])
    ax.set_xlabel("Acoustic SNR level (dB)")
    ax.set_ylabel("SG-EER (%)")
    ax.set_title(f"Noise-type effect on the gate ({system_label})")
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(title="Noise type", frameon=True)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {outpath}  (+ .pdf)")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Per-noise-type EER plot for one system.")
    ap.add_argument("--system", required=True,
                    help="per-condition CSV for the system (e.g. the gate)")
    ap.add_argument("--egg", help="EGG summary CSV for the baseline band")
    ap.add_argument("--egg_mean", type=float, help="EGG baseline EER %% (overrides --egg)")
    ap.add_argument("--egg_std", type=float, help="EGG baseline std %%")
    ap.add_argument("--label", default="Vector gate (primary)", help="system label")
    ap.add_argument("--low_snr_bars_only", action="store_true",
                    help="draw error bars only at -10/-5 dB (cleaner at high SNR)")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--outfile", default="noise_type_effect.png",
                    help="output filename within --outdir")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    per_noise = load_system(args.system)
    egg_mean, egg_std = load_egg_baseline(args.egg, args.egg_mean, args.egg_std)

    # echo the severity ordering so you SEE what the plot claims
    means = {nt: sum(v[0] for v in d.values()) / len(d) for nt, d in per_noise.items()}
    print("\nSeverity ordering (mean SG-EER over SNRs, worst -> best):")
    for nt, m in sorted(means.items(), key=lambda x: -x[1]):
        print(f"  {nt:<10} {m:.3f}%")
    if egg_mean is not None:
        print(f"\nEGG-only baseline: {egg_mean:.3f}%"
              + (f" ± {egg_std:.3f}" if egg_std is not None else ""))
    print()

    make_plot(per_noise, os.path.join(args.outdir, args.outfile),
              egg_mean=egg_mean, egg_std=egg_std, system_label=args.label,
              bars_low_snr_only=args.low_snr_bars_only)


if __name__ == "__main__":
    main()