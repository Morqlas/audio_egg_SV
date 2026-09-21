#!/usr/bin/env python3
r"""
Paper artifact: the fusion-results figures (unimodal vs fusion, and the MLP/gate crossover).

fusion_results.py
======================
Build the two fusion-results figures from per-condition SG-EER CSVs.

Plot 1  (motivation):          audio-only, EGG-only, late fusion, and MLP fusion
                                across SNR. Wide y-axis -- shows audio collapsing,
                                EGG flat, naive late fusion dragged down by audio,
                                and MLP (trained) tracking well below both.
                                Motivates an adaptive gate. (needs --audio/--egg)
                                Optionally overlays the Ćirović et al. (2010)
                                multimodal baseline (--cirovic), drawn ONLY over
                                the 0/+5/+10 dB range they report.

Plot 2  (trained fusion):       MLP vs gate across SNR, zoomed to the low-EER
                                band -- shows the crossover.
                                Answers "which fusion, and where?"

Each input CSV must have columns:
    noise_type, snr_db, sg_eer_mean   (sg_eer in [0,1]; clean row has empty snr_db)

USAGE
-----
    python -m src.figures.fusion_results \
        --gate   no_trunk_vector_gate_test_aggregated.csv \
        --mlp    mlp_test_aggregated.csv \
        --audio  audio_only_test_aggregated.csv \   # optional, for plot 1
        --egg    egg_only_test_aggregated.csv \     # optional, for plot 1
        --late   late_fusion_test_aggregated.csv \  # optional, adds to plot 2
        --cirovic \                                 # optional, overlays plot 1
        --outdir figures

Anything not supplied is simply skipped (plot 1 needs --audio/--egg to be useful).

Design notes
------------
* SNR axis is ordered -10, -5, 0, +5, +10 (left = worst).
* Values are SNR-averaged across the five noise types at each SNR level.
* 'clean' is reported separately in the title/annotation, NOT placed on the SNR
  axis (it has no SNR and would distort the x-scale).
* No interpolation: only the five measured SNR points are plotted, joined by
  straight segments purely as a visual guide.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNR_ORDER = [-10.0, -5.0, 0.0, 5.0, 10.0]

# External baseline: Ćirović et al. (2010), Table 2, System 3 (GAD, EGG + audio
# features) -- SV error rate (%) at 0/5/10 dB. Their paper reports down to 0 dB
# only, so this is plotted over a subset of our SNR range (0/+5/+10 dB); -10/-5
# dB are intentionally left blank rather than extrapolated.
CIROVIC_2010 = {0.0: 8.33, 5.0: 6.85, 10.0: 4.4}

# ----- presentation-scale styling (readable from the back of a room) ---------
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "lines.linewidth": 2.4,
    "lines.markersize": 8,
    "figure.dpi": 150,
})

# consistent colors across both plots
COLORS = {
    "audio": "#C0392B",   # red   -- the modality that collapses
    "egg":   "#27AE60",   # green -- the stable modality
    "gate":  "#2C6FB5",   # blue  -- primary
    "mlp":   "#E08E0B",   # amber -- strong baseline
    "late":  "#7F8C8D",   # gray  -- untrained floor
    "cirovic": "#000000", # black -- external reference, visually distinct
}
LABELS = {
    "audio": "ECAPA2",
    "egg":   "EGG-only",
    "gate":  "Vector Gated Fusion (primary)",
    "mlp":   "MLP Fusion",
    "late":  "Late Fusion",
    "cirovic": "Ćirović et al. (2010)",
}
# redundant (non-color) encoding: line style + marker shape per series, so
# identity survives grayscale print and red/green color-vision deficiency
# (audio and EGG are red/green -- a classic confusable pair on color alone).
LINESTYLES = {
    "audio": "-", "late": "--", "mlp": "-.", "gate": "-",
    "egg": ":", "cirovic": ":",
}
MARKERS = {
    "audio": "o", "late": "s", "mlp": "^", "gate": "D", "cirovic": "x",
}


def load_csv(path):
    """Return (snr_means_dict, clean_pct_or_None).

    snr_means_dict maps snr_float -> mean SG-EER % averaged across noise types.
    """
    if path is None:
        return None, None
    if not os.path.exists(path):
        print(f"  [warn] file not found, skipping: {path}", file=sys.stderr)
        return None, None

    by_snr = {s: [] for s in SNR_ORDER}
    clean_vals = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for col in ("noise_type", "snr_db", "sg_eer_mean"):
            if col not in reader.fieldnames:
                print(f"  [error] {path} missing column '{col}'", file=sys.stderr)
                return None, None
        for row in reader:
            eer = float(row["sg_eer_mean"]) * 100.0  # [0,1] -> %
            if row["noise_type"].strip().lower() == "clean" or row["snr_db"].strip() == "":
                clean_vals.append(eer)
                continue
            snr = float(row["snr_db"])
            if snr in by_snr:
                by_snr[snr].append(eer)

    snr_means = {s: (sum(v) / len(v)) for s, v in by_snr.items() if v}
    clean_pct = (sum(clean_vals) / len(clean_vals)) if clean_vals else None
    return snr_means, clean_pct


def series(snr_means):
    """Return x (indices), y (values) in SNR_ORDER, skipping missing points."""
    xs, ys = [], []
    for i, s in enumerate(SNR_ORDER):
        if s in snr_means:
            xs.append(i)
            ys.append(snr_means[s])
    return xs, ys


def load_egg_constant(path):
    """EGG is never noised, so its EER is constant across SNR -> a flat line.

    Accepts either a 'metric,mean,...' summary file (reads test_sg_eer) or a
    per-condition CSV with sg_eer_mean (averages it to one constant).
    Returns a single EER % or None.
    """
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"  [warn] EGG file not found, skipping: {path}", file=sys.stderr)
        return None
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        fns = reader.fieldnames or []
        rows = list(reader)
    if "metric" in fns and "mean" in fns:
        for r in rows:
            if r["metric"].strip().lower() in ("test_sg_eer", "sg_eer"):
                return float(r["mean"]) * 100.0
        print(f"  [warn] no test_sg_eer row in {path}", file=sys.stderr)
        return None
    if "sg_eer_mean" in fns:
        vals = [float(r["sg_eer_mean"]) for r in rows if r.get("sg_eer_mean")]
        if vals:
            return sum(vals) / len(vals) * 100.0
    print(f"  [warn] unrecognized EGG layout: {path}", file=sys.stderr)
    return None


def make_plot1(data, cleans, egg_const, outpath, show_cirovic=False):
    """Unimodal + fusion systems vs SNR -- wide y-axis.

    audio-only, late fusion, and MLP fusion are per-SNR curves; EGG-only is a
    flat horizontal line (it is never noised, so its EER is constant). Late
    fusion sits between audio and EGG and is dragged down by the collapsing
    audio at low SNR; MLP (trained) tracks well below both, still without any
    input-dependent gating -- together the motivation for an adaptive gate.
    Optionally overlays the Ćirović et al. (2010) multimodal baseline over its
    reported 0/+5/+10 dB range only (see CIROVIC_2010).

    Every series carries a distinct line style and marker shape in addition to
    color, so identity survives grayscale print and red/green color-vision
    deficiency (audio and EGG are red/green on color alone).
    """
    has_curves = data.get("audio") or data.get("late") or data.get("mlp")
    if not has_curves and egg_const is None:
        print("  [skip] plot 1 needs audio/late/mlp curves and/or EGG.")
        return False

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for key in ("audio", "late", "mlp"):
        if data.get(key):
            xs, ys = series(data[key])
            ax.plot(xs, ys, marker=MARKERS[key], linestyle=LINESTYLES[key],
                     color=COLORS[key], label=LABELS[key])

    # EGG as a flat reference line across the full SNR axis
    if egg_const is not None:
        ax.axhline(egg_const, color=COLORS["egg"], ls=LINESTYLES["egg"],
                   label=f"{LABELS['egg']} ({egg_const:.1f}%)")

    # external reference: partial range, drawn only where Ćirović et al.
    # report a value (0/+5/+10 dB) -- never extrapolated to -10/-5 dB.
    if show_cirovic:
        idx = {s: i for i, s in enumerate(SNR_ORDER)}
        xs = [idx[s] for s in sorted(CIROVIC_2010) if s in idx]
        ys = [CIROVIC_2010[s] for s in sorted(CIROVIC_2010) if s in idx]
        ax.plot(xs, ys, marker=MARKERS["cirovic"], linestyle=LINESTYLES["cirovic"],
                 color=COLORS["cirovic"], label=f"{LABELS['cirovic']} (multimodal)")

    ax.set_xticks(range(len(SNR_ORDER)))
    ax.set_xticklabels([f"{int(s):+d}" for s in SNR_ORDER])
    ax.set_xlabel("Acoustic SNR (dB)", fontsize=12)
    ax.set_ylabel("SG-EER (%)", fontsize=12)
    ax.set_title("Unimodal vs Fusion Systems", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    '''
    notes = [f"{LABELS[k]} clean: {cleans[k]:.2f}%"
             for k in ("audio", "late", "mlp") if cleans.get(k) is not None]
    if notes:
        ax.text(0.02, 0.97, "\n".join(notes), transform=ax.transAxes,
                va="top", ha="left", fontsize=11,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    '''
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  wrote {outpath}  (+ .pdf)")
    plt.close(fig)
    return True


def make_plot2(data, cleans, outpath):
    """Fusion vs fusion -- zoomed to show the crossover.

    Distinct line style + marker per series (see LINESTYLES/MARKERS), so
    identity survives grayscale print and color-vision deficiency.
    """
    present = [k for k in ("mlp", "gate", "late") if data.get(k)]
    if "mlp" not in present or "gate" not in present:
        print("  [skip] plot 2 needs at least --mlp and --gate.")
        return False

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    for key in ("late", "mlp", "gate"):  # gate last = drawn on top
        if data.get(key) and cleans.get(key) is not None:
            xs, ys = series(data[key])
            ax.plot(xs, ys, marker=MARKERS[key], linestyle=LINESTYLES[key],
                     color=COLORS[key], label=LABELS[key])

    clean_x = len(SNR_ORDER)          # one slot to the right of +10
    for key in ("late", "mlp", "gate"):
        if data.get(key) and cleans.get(key) is not None:
            ax.plot(clean_x, cleans[key], marker=MARKERS[key], color=COLORS[key],
                    markersize=8, zorder=5)
     
    ax.axvline(len(SNR_ORDER) - 0.5, color="0.8", ls=":", lw=1.0, zorder=0)

    ax.set_xticks(range(len(SNR_ORDER) + 1))
    ax.set_xticklabels([f"{int(s):+d}" for s in SNR_ORDER] + ["clean"])
    ax.set_xlabel("Acoustic SNR (dB)")
    ax.set_ylabel("SG-EER (%)")
    ax.set_title("Comparison between Fusion Systems")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    # mark the crossover region (between the SNRs where the leader flips)
    gx, gy = series(data["gate"])
    mx, my = series(data["mlp"])
    common = sorted(set(gx) & set(mx))
    flip_idx = None
    for j in range(1, len(common)):
        a, b = common[j - 1], common[j]
        ga, gb = data["gate"][SNR_ORDER[a]], data["gate"][SNR_ORDER[b]]
        ma, mb = data["mlp"][SNR_ORDER[a]], data["mlp"][SNR_ORDER[b]]
        if (ga - ma) == 0:
            continue
        if (ga - ma) * (gb - mb) < 0:   # sign change => crossover between a and b
            flip_idx = (a + b) / 2.0
    '''        
    if flip_idx is not None:
        ax.axvline(flip_idx, color="0.5", ls="--", lw=1.5, alpha=0.8)
        ax.text(flip_idx, ax.get_ylim()[1] * 0.96, " crossover", color="0.4",
                fontsize=11, va="top", ha="left")
    
    notes = [f"{LABELS[k]} clean: {cleans[k]:.2f}%"
             for k in ("mlp", "gate", "late") if cleans.get(k) is not None]
    if notes:
        ax.text(0.02, 0.97, "\n".join(notes), transform=ax.transAxes,
                va="top", ha="left", fontsize=11,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    '''
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  wrote {outpath}  (+ .pdf)")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Build fusion-results figures from CSVs.")
    ap.add_argument("--gate", required=True, help="primary (no-trunk vector gate) CSV")
    ap.add_argument("--mlp", required=True, help="MLP fusion CSV")
    ap.add_argument("--audio", help="audio-only CSV (for plot 1)")
    ap.add_argument("--egg", help="EGG-only CSV (for plot 1)")
    ap.add_argument("--late", help="late fusion CSV (optional, adds to plot 2)")
    ap.add_argument("--cirovic", action="store_true",
                    help="overlay the Ćirović et al. (2010) multimodal baseline "
                         "on plot 1 (0/+5/+10 dB only; see CIROVIC_2010)")
    ap.add_argument("--outdir", default="figures", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data, cleans = {}, {}
    for key, path in (("gate", args.gate), ("mlp", args.mlp),
                      ("audio", args.audio), ("late", args.late)):
        m, c = load_csv(path)
        if m is not None:
            data[key] = m
            cleans[key] = c

    # EGG is never noised -> a single constant, not a per-condition series
    egg_const = load_egg_constant(args.egg)

    # quick provenance echo so you SEE the numbers the plot is built from
    print("\nMean SG-EER across the SNR points present (NOT the 26-condition")
    print("flat mean -- use a flat 26-row average for the reportable headline):")
    for key in ("audio", "late", "mlp", "gate"):
        if data.get(key):
            allvals = [v for v in data[key].values()]
            extra = cleans[key] if cleans.get(key) is not None else None
            n = len(allvals) + (1 if extra is not None else 0)
            tot = sum(allvals) + (extra if extra is not None else 0)
            cleanstr = f"   (clean {cleans[key]:.3f}%)" if extra is not None else ""
            print(f"  {LABELS[key]:<26} {tot/n:6.3f}%{cleanstr}")
    if egg_const is not None:
        print(f"  {LABELS['egg']:<26} {egg_const:6.3f}%   (SNR-invariant)")
    print()

    make_plot1(data, cleans, egg_const,
               os.path.join(args.outdir, "fusion_vs_unimodal.png"),
               show_cirovic=args.cirovic)
    make_plot2(data, cleans, os.path.join(args.outdir, "fusion_crossover.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()