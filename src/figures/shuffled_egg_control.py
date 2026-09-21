#!/usr/bin/env python3
"""
Paper artifact: the shuffled-EGG identity-control figure.

shuffled_egg_control.py
=======================
Identity control: real EGG vs cross-speaker-shuffled EGG, per-condition SG-EER
across the full SNR sweep (Form 2).

The question: does the gate rely on genuine EGG *speaker identity*, or just on an
EGG-shaped signal? Permuting EGG across speakers preserves signal statistics but
destroys identity. If real >> shuffled, the gate is using identity.

Pre-registered acceptance rule: real better than shuffled in >= 8/10 low-SNR
cells (-10, -5 dB x 5 noise types). Reported result: 10/10 (and 26/26 overall).

Both input CSVs are per-condition aggregates with columns:
    noise_type, snr_db, sg_eer_mean, sg_eer_std   (EER in [0,1]; clean row ignored)

The "real" file is the normal primary gate; the "shuffled" file is the separately
trained shuffled-EGG model. NOTE: the shuffled model is a DIAGNOSTIC CONTROL, not
a system -- never present its EER as a result of the proposed method.

USAGE
-----
    python -m src.figures.shuffled_egg_control \
        --real     no_trunk_vector_gate_test_aggregated.csv \
        --shuffled no_trunk_vector_gate_shuffled_test_aggregated.csv \
        --outdir   figures

Log y-axis (real ~2%, shuffled up to ~38% -- an order of magnitude apart).
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from src import config

SNR_ORDER = [-10.0, -5.0, 0.0, 5.0, 10.0]
LOW_SNR = (-10.0, -5.0)
AXIS_FLOOR = 0.04

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12,
    "lines.linewidth": 2.2, "lines.markersize": 8, "figure.dpi": 150,
})

NOISE_ORDER = ["cafeteria", "metro", "park", "traffic", "office"]
NOISE_COLOR = {
    "cafeteria": "#1f77b4", "metro": "#ff7f0e", "park": "#d62728",
    "traffic": "#9467bd", "office": "#2ca02c",
}

OFFSETS = {
    "cafeteria": -0.80,
    "metro":     -0.40,
    "office":     0.80,
    "park":       0.00,
    "traffic":    0.40,
}

STYLE = {
    "cafeteria": {"color": "#1f77b4", "marker": "o", "label": "cafeteria"},
    "metro":     {"color": "#ff7f0e", "marker": "s", "label": "metro"},
    "park":      {"color": "#d62728", "marker": "D", "label": "park"},
    "traffic":   {"color": "#9467bd", "marker": "v", "label": "traffic"},
    "office":    {"color": "#2ca02c", "marker": "^", "label": "office"},
}


def _fmt_snr(s):
    i = int(s)
    return "0" if i == 0 else f"{i:+d}"


def load(path):
    """Return {(noise, snr_float): (eer_pct, std_pct)} ignoring the clean row."""
    if not os.path.exists(path):
        sys.exit(f"[error] file not found: {path}")
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
            out[(nt, float(r["snr_db"]))] = (eer, std)
    return out


def report_pass(real, shuf):
    """Print the acceptance-rule outcome so the test result is visible."""
    low = [k for k in real if k[1] in LOW_SNR]
    allk = [k for k in real if k in shuf]
    lw = sum(1 for k in low if real[k][0] < shuf[k][0])
    aw = sum(1 for k in allk if real[k][0] < shuf[k][0])
    print(f"\nShuffled-EGG identity control:")
    print(f"  low-SNR cells: real better in {lw}/{len(low)} "
          f"(pre-registered rule: >= 8/10)  ->  {'PASS' if lw >= 8 else 'FAIL'}")
    print(f"  all conditions: real better in {aw}/{len(allk)}")
    print(f"\n  -10 dB blowups (real -> shuffled):")
    for n in NOISE_ORDER:
        k = (n, -10.0)
        if k in real and k in shuf:
            r, s = real[k][0], shuf[k][0]
            print(f"    {n:<10} {r:6.2f} -> {s:6.2f}  ({s / r:.0f}x)")
    print()


def _yerr(means, stds, floor):
    """Asymmetric yerr clamped so the lower bar never reaches <= 0 on a log axis."""
    lower, upper = [], []
    for m, sd in zip(means, stds):
        lo = min(sd, max(0.0, m - floor))   # bar stops at the axis floor
        lower.append(lo)
        upper.append(sd)
    return [lower, upper]


def make_plot(real, shuf, outpath):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    for nt in NOISE_ORDER:
        color = NOISE_COLOR[nt]
        st = STYLE[nt]
        # real: solid line + filled markers
        snrs_r = [s for s in SNR_ORDER if (nt, s) in real]
        xs_r = [s + OFFSETS[nt] for s in snrs_r]
        ys_r = [real[(nt, s)][0] for s in snrs_r]
        es_r = [real[(nt, s)][1] for s in snrs_r]
        ax.errorbar(xs_r, ys_r, yerr=_yerr(ys_r, es_r, AXIS_FLOOR),
                    color=color, marker=st["marker"], ls="None",
                    capsize=3, capthick=1.0, elinewidth=1.0, zorder=3)

        # shuffled: dashed line + open markers
        snrs_s = [s for s in SNR_ORDER if (nt, s) in shuf]
        xs_s = [s + OFFSETS[nt] for s in snrs_s]
        ys_s = [shuf[(nt, s)][0] for s in snrs_s]
        es_s = [shuf[(nt, s)][1] for s in snrs_s]
        ax.errorbar(xs_s, ys_s, yerr=_yerr(ys_s, es_s, AXIS_FLOOR),
                    color=color, marker=st["marker"], ls="None",
                    markerfacecolor="white", markeredgecolor=color,
                    capsize=3, capthick=1.0, elinewidth=1.0, zorder=2)

    ax.set_yscale("log")
    ax.set_ylim(bottom=AXIS_FLOOR)        # <-- enforce the floor the bars clamp to
    ticks = [0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 40]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.set_xticks(SNR_ORDER)
    ax.set_xticklabels([_fmt_snr(s) for s in SNR_ORDER])
    ax.set_xlabel("Audio SNR (dB)")
    ax.set_ylabel("EER (%)")
    ax.set_title("Identity Control: Real vs Cross-Speaker-Shuffled EGG")

    # two legends: one for noise color, one for the line-style meaning
    from matplotlib.lines import Line2D
    style_handles = [
    Line2D([0], [0], color="0.3", ls="None", marker="o",
           markerfacecolor="0.3", label="Real EGG"),
    Line2D([0], [0], color="0.3", ls="None", marker="o",
           markerfacecolor="white", markeredgecolor="0.3", label="Shuffled EGG"),
]
    noise_handles = [Line2D([0], [0], color=NOISE_COLOR[n], ls="None", marker="o",
                        label=n) for n in NOISE_ORDER]
    leg1 = ax.legend(handles=style_handles, loc="upper right",
                     frameon=True, title="EGG source")
    ax.add_artist(leg1)
    ax.legend(handles=noise_handles, loc="center right", frameon=True, bbox_to_anchor=(0.99,0.6),
              title="Noise type", fontsize=11)

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.12)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {outpath}  (+ .pdf)")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Shuffled-EGG identity control plot.")
    ap.add_argument("--real", default=str(config.RESULTS_DIR / "gate_primary" / "per_condition.csv"), help="primary gate per-condition CSV")
    ap.add_argument("--shuffled", default=str(config.RESULTS_DIR / "control_shuffled_egg" / "per_condition.csv"), help="shuffled-EGG per-condition CSV")
    ap.add_argument("--outdir", default=str(config.FIGURES_DIR))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    real = load(args.real)
    shuf = load(args.shuffled)
    report_pass(real, shuf)
    make_plot(real, shuf, os.path.join(args.outdir, "shuffled_egg_control.png"))


if __name__ == "__main__":
    main()