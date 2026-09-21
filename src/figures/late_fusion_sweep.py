#!/usr/bin/env python3
"""
Paper artifact: the late-fusion weight-sweep motivation figure (optimal fixed alpha migrates with SNR).

Late-fusion weight-sweep figure (styled to match the fusion-results crossover panel).

Reads the per-weight aggregated CSVs
    late_fusion_aggregated_audio_weight_0_1.csv  ...  _0_9.csv
(columns: noise_type, snr_db, sg_eer_mean, sg_eer_std, ...) and produces a
two-panel figure:

  (a) Mean SG-EER vs audio weight alpha, one curve per SNR level (averaged
      over the 5 noise types at that SNR). Each curve's minimum is starred.
      Shows that the optimal fixed weight migrates with SNR.

  (b) The distilled trend: the best (oracle-selected) fixed alpha vs SNR,
      showing the optimal static audio weight sliding from audio-heavy
      (clean) to EGG-heavy (heavy noise).

Both panels use only score-level late fusion (a fixed, non-adaptive
combination). Their point is the *motivation* for an input-dependent gate,
not a performance claim for late fusion. Panel (b) is an ORACLE: the best
alpha is chosen using the known test SNR, so it is a diagnostic of the shift,
not an achievable operating point.

Usage:
    python -m src.figures.late_fusion_sweep --sweep_dir . --output late_fusion_sweep.pdf

Notes:
  - sg_eer values in the CSVs are FRACTIONS (0.010 = 1.0%); multiplied by 100.
  - Log y-axis on panel (a): the per-SNR curves span ~0.1% to ~25%.
  - The sweep covers alpha in {0.1,...,0.9}. Pure-modality endpoints
    (alpha=0 EGG-only, alpha=1 audio-only) are the unimodal baselines of the
    previous figure and are intentionally not plotted here.
  - Fold std is omitted from these panels for legibility; std is fold
    dispersion, not a CI. Per-condition std is in the CSVs.
"""

import argparse
import csv
import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from src import config


NOISE_TYPES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_LEVELS = [-10, -5, 0, 5, 10]

SNR_STYLE = {
    -10: {"color": "#1f77b4", "marker": "o", "ls": "-",  "label": "-10 dB"},
    -5:  {"color": "#ff7f0e", "marker": "s", "ls": "--", "label": " -5 dB"},
    0:   {"color": "#2ca02c", "marker": "^", "ls": "-.", "label": "  0 dB"},
    5:   {"color": "#9467bd", "marker": "D", "ls": ":",  "label": " +5 dB"},
    10:  {"color": "#d62728", "marker": "v", "ls": "-",  "label": "+10 dB"},
}


def load_csv(path):
    d = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            nt = r["noise_type"].strip()
            snr = r.get("snr_db", "").strip()
            key = "clean" if (nt == "clean" or snr in ("", "nan")) else (nt, int(float(snr)))
            d[key] = float(r["sg_eer_mean"]) * 100.0
    return d


def load_sweep(directory):
    pattern = os.path.join(directory, "late_fusion_aggregated_audio_weight_*.csv")
    files = {}
    for fn in glob.glob(pattern):
        m = re.search(r"weight_(\d).(\d)", os.path.basename(fn))
        if not m:
            continue
        files[float(f"{m.group(1)}.{m.group(2)}")] = load_csv(fn)
    if not files:
        raise FileNotFoundError(f"No late-fusion weight CSVs found under {directory!r}")
    return files


def mean_eer_at(files, alpha, snr):
    d = files[alpha]
    return np.mean([d[(nt, snr)] for nt in NOISE_TYPES])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", default=str(config.RESULTS_DIR / "score_level_late_fusion"))
    parser.add_argument("--output", default=str(config.FIGURES_DIR / "late_fusion_sweep.pdf"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    files = load_sweep(args.sweep_dir)
    alphas = sorted(files)

    curves = {snr: [mean_eer_at(files, a, snr) for a in alphas] for snr in SNR_LEVELS}
    best = {}
    for snr in SNR_LEVELS:
        i = int(np.argmin(curves[snr]))
        best[snr] = (alphas[i], curves[snr][i])
    clean_curve = [files[a]["clean"] for a in alphas]
    i = int(np.argmin(clean_curve))
    best_clean = (alphas[i], clean_curve[i])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.6))

    for snr in SNR_LEVELS:
        st = SNR_STYLE[snr]
        axA.plot(alphas, curves[snr], linestyle=st["ls"], marker=st["marker"],
                 color=st["color"], markersize=6, linewidth=1.8,
                 label=st["label"], zorder=3)
        ba, bv = best[snr]
        axA.plot(ba, bv, marker="*", markersize=13, zorder=5,
                 markeredgecolor="black", markerfacecolor="none")

    axA.set_yscale("log")
    axA.set_xlabel(r"Audio weight", fontsize=12)
    axA.set_ylabel("Mean SG-EER (%)", fontsize=12)
    axA.set_title("EER vs fixed weight, per SNR", fontsize=12)
    axA.set_xticks(alphas)
    yticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    axA.yaxis.set_major_locator(mticker.FixedLocator(yticks))
    axA.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    axA.yaxis.set_minor_locator(mticker.NullLocator())
    axA.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
    axA.legend(fontsize=9, loc="upper center", framealpha=0.95,
               title="Test SNR", title_fontsize=9, ncol=2)

    xs = [f"{s:+d}" for s in SNR_LEVELS] + ["clean"]
    ys = [best[s][0] for s in SNR_LEVELS] + [best_clean[0]]
    axB.plot(range(len(xs)), ys, linestyle="-", marker="o",
             color="0.25", markersize=7, linewidth=1.8, zorder=3)
    for x, y in zip(range(len(xs)), ys):
        axB.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=9)
    axB.axhline(0.5, linestyle="--", linewidth=1.0, color="0.5", zorder=1)
    axB.text(len(xs) - 1, 0.515, "equal weight", ha="right", va="bottom",
             fontsize=8, color="0.5", style="italic")
    axB.set_xticks(range(len(xs)))
    axB.set_xticklabels(xs)
    axB.set_ylim(0.0, 1.0)
    axB.set_xlabel("Acoustic SNR (dB)", fontsize=12)
    axB.set_ylabel(r"Best audio weight", fontsize=12)
    axB.set_title("Best static weight, per SNR", fontsize=12)
    axB.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    png = args.output.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print(f"Saved: {args.output} and {png}")

    print("\nBest (oracle) fixed alpha per SNR (mean over 5 noise types):")
    print(f"  clean : alpha*={best_clean[0]:.1f}  (EER {best_clean[1]:.3f}%)")
    for s in SNR_LEVELS:
        ba, bv = best[s]
        print(f"  {s:+3d} dB: alpha*={ba:.1f}  (EER {bv:.3f}%)")
    overall = {a: np.mean(list(files[a].values())) for a in alphas}
    ba = min(overall, key=overall.get)
    print(f"\nBest alpha by overall 26-condition mean: alpha*={ba:.1f} (mean {overall[ba]:.3f}%)")


if __name__ == "__main__":
    main()