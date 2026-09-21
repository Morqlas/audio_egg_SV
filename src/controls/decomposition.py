#!/usr/bin/env python3
"""
Paper artifact: the gate-vs-norm decomposition control (gate-driven share 81-96%).

Gate-vs-norm decomposition for the untied vector-gate fusion system.

Addresses the confound: the effective audio share = (g_a*||e_a||) /
(g_a*||e_a|| + g_e*||e_e||) depends on BOTH the gate and the audio norm.
A decline in effective share could therefore be a passive artifact of the
audio embedding norm shrinking under noise, rather than the gate learning
to down-weight audio. This script separates the two.

Three outputs:
  (1) decomposition table: for the +10 -> -10 dB change in effective share,
      how much is gate-driven vs norm-driven (per noise type).
  (2) raw-gate evidence: g_audio (norm-INDEPENDENT) vs SNR, and whether its
      drop orders by noise severity.
  (3) figure: raw g_audio vs SNR (left) and gate/norm decomposition (right).

Aggregation unit = fold (mean across folds). Input = per-fold norm-stats CSVs
with columns: noise_type, snr_db, g_audio_mean, g_egg_mean,
norm_audio_mean, norm_egg_mean, eff_audio_share_mean.

Usage:
    python -m src.controls.decomposition --input-dir ./norm_stats \
        --output-dir ./results --pattern "norm_stats_fold_*_test.csv"
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src import config
from src import _console  # noqa: F401  (UTF-8 safe stdout)

NOISES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_ORDER = [-10, -5, 0, 5, 10]
STYLE = {
    "cafeteria": {"color": "#B85042", "marker": "o"},
    "metro":     {"color": "#065A82", "marker": "s"},
    "office":    {"color": "#2C7A4B", "marker": "^"},
    "park":      {"color": "#C9A227", "marker": "D"},
    "traffic":   {"color": "#9467bd", "marker": "v"},
}

def _clean_val(agg, col):
    """Return the clean-condition value for `col`, or None if no clean row."""
    m = agg["snr_db"].isna() | (agg["noise_type"] == "clean")
    if m.any():
        return float(agg.loc[m, col].mean())
    return None


def share(ga, na, ge, ne):
    return (ga * na) / (ga * na + ge * ne)


def load_and_aggregate(input_dir, pattern):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        sys.exit(f"No files matched {pattern!r} in {input_dir!r}")
    frames = []
    for fi, f in enumerate(files):
        x = pd.read_csv(f); x["fold"] = fi; frames.append(x)
    A = pd.concat(frames, ignore_index=True)
    cols = ["g_audio_mean", "g_egg_mean", "norm_audio_mean",
            "norm_egg_mean", "eff_audio_share_mean"]
    agg = A.groupby(["noise_type", "snr_db"], dropna=False)[cols].mean().reset_index()
    print(f"Loaded {len(files)} folds.")
    return agg


def decompose(agg):
    rows = []
    for nz in NOISES:
        s = agg[agg.noise_type == nz].set_index("snr_db")
        ga_hi, ga_lo = s.loc[10, "g_audio_mean"], s.loc[-10, "g_audio_mean"]
        ge_hi, ge_lo = s.loc[10, "g_egg_mean"],   s.loc[-10, "g_egg_mean"]
        na_hi, na_lo = s.loc[10, "norm_audio_mean"], s.loc[-10, "norm_audio_mean"]
        ne_hi, ne_lo = s.loc[10, "norm_egg_mean"],   s.loc[-10, "norm_egg_mean"]
        base = share(ga_hi, na_hi, ge_hi, ne_hi)
        full = share(ga_lo, na_lo, ge_lo, ne_lo)
        gate_only = share(ga_lo, na_hi, ge_lo, ne_hi)   # only gates vary
        norm_only = share(ga_hi, na_lo, ge_hi, ne_lo)   # only norms vary
        dt = base - full
        dg = base - gate_only
        dn = base - norm_only
        rows.append({
            "noise_type": nz,
            "total_share_drop": dt,
            "gate_driven": dg, "gate_pct": 100 * dg / dt,
            "norm_driven": dn, "norm_pct": 100 * dn / dt,
            "raw_g_audio_drop": ga_hi - ga_lo,
            "g_audio_10dB": ga_hi, "g_audio_-10dB": ga_lo,
            "interaction": dt - dg - dn,
            "interaction_pct": 100 * (dt - dg - dn) / dt,
        })
    return pd.DataFrame(rows)

'''
def make_figure(agg, dec, out_png, out_pdf):
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    })
    xpos = list(range(len(SNR_ORDER)))
    xlabels = [("+%d" % s if s > 0 else str(s)) for s in SNR_ORDER]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # left: RAW gate (norm-independent) vs SNR
    for nz in NOISES:
        s = agg[agg.noise_type == nz].set_index("snr_db")
        y = [s.loc[x, "g_audio_mean"] for x in SNR_ORDER]
        ax1.plot(xpos, y, "-o", color=COLORS[nz], lw=2, ms=5, label=nz)
    ax1.set_xticks(xpos); ax1.set_xticklabels(xlabels)
    ax1.set_xlabel("Audio SNR (dB)")
    ax1.set_ylabel("Average g_audio")
    ax1.set_title("Average g_audio vs Audio SNR - Non Normalized Inputs")
    ax1.legend(frameon=False, ncol=2, loc="lower left")

    # right: stacked gate-driven vs norm-driven share drop
    gate = dec.set_index("noise_type").loc[NOISES, "gate_driven"].values
    norm = dec.set_index("noise_type").loc[NOISES, "norm_driven"].values
    xb = np.arange(len(NOISES))
    ax2.bar(xb, gate, color="#065A82", label="gate-driven")
    ax2.bar(xb, norm, bottom=gate, color="#C9A227", label="norm-driven")
    ax2.set_xticks(xb); ax2.set_xticklabels(NOISES, rotation=20, ha="right")
    ax2.set_ylabel("Effective-share drop, +10 → −10 dB")
    ax2.set_title("Decomposition: gate vs norm")
    for i, nz in enumerate(NOISES):
        ax2.text(i, gate[i] / 2, f"{dec.set_index('noise_type').loc[nz,'gate_pct']:.0f}%",
                 ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax2.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
'''

def make_gate_figure(agg, out_png, out_pdf):
    """Section 4.4.1: raw g_audio AND g_egg vs SNR, per noise type.

    Both gates on one axis so the asymmetry is visible: g_audio plunges as SNR
    falls while g_egg is nearly flat -- the untied gate adapts chiefly by
    suppressing audio, not amplifying EGG. Clean shown as a detached point.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    })
    xpos = list(range(len(SNR_ORDER)))
    xlabels = [("+%d" % s if s > 0 else str(s)) for s in SNR_ORDER]
    clean_x = len(SNR_ORDER)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for nz in NOISES:
        s = agg[agg.noise_type == nz].set_index("snr_db")
        ga = [s.loc[x, "g_audio_mean"] for x in SNR_ORDER]
        ge = [s.loc[x, "g_egg_mean"]   for x in SNR_ORDER]
        st = STYLE[nz]
        ax.plot(xpos, ga, "-", marker=st["marker"], color=st["color"],
                lw=2, ms=5)
        ax.plot(xpos, ge, "--", marker=st["marker"], color=st["color"],
                lw=1.4, ms=4, alpha=0.7)

    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Audio SNR (dB)")
    ax.set_ylabel("Mean Gate Value")
    # Legend 1: noise type (colour). Filter to the solid audio handles only.
    noise_handles = [
        Line2D([0], [0], color=STYLE[nz]["color"], marker=STYLE[nz]["marker"],
               linestyle="-", lw=2, ms=5, label=nz)
        for nz in NOISES
    ]
    style_handles = [
        Line2D([0], [0], color="0.3", linestyle="-",  lw=2,
               label=r"$\bar{g}_a$ (audio)"),
        Line2D([0], [0], color="0.3", linestyle="--", lw=1.4,
               label=r"$\bar{g}_g$ (EGG)"),
    ]

    # stacked: noise-type legend on top, modality legend directly beneath it
    leg1 = ax.legend(handles=noise_handles, frameon=False, ncol=1,
                     loc="upper left", bbox_to_anchor=(0.02, 0.88),
                     title="Noise type")
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, frameon=False, ncol=1,
              loc="upper left", bbox_to_anchor=(0.02, 0.55),
              title="Gate")
    #ax.set_title("Mean Gate Value vs. Audio SNR")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def make_decomp_figure(dec, out_png, out_pdf):
    """Section 4.4.2: stacked gate-driven vs norm-driven share drop.

    The interaction term (|.| <= ~2.7%) is omitted from the bars and reported
    in the table instead; because it is negative, gate% + norm% slightly
    exceeds 100%.
    """
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    })
    d = dec.set_index("noise_type").loc[NOISES]
    gate = d["gate_driven"].values
    norm = d["norm_driven"].values
    xb = np.arange(len(NOISES))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(xb, gate, color="#065A82", label="gate-driven")
    ax.bar(xb, norm, bottom=gate, color="#C9A227", label="norm-driven")
    ax.set_xticks(xb)
    ax.set_xticklabels(NOISES, rotation=20, ha="right")
    ax.set_ylabel(r"Effective-share drop, $+10 \rightarrow -10$ dB")
    for i, nz in enumerate(NOISES):
        ax.text(i, gate[i] / 2, f"{d.loc[nz,'gate_pct']:.0f}%",
                ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Gate vs Norm shares")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", default=str(config.RESULTS_DIR / "control_decomposition"))
    ap.add_argument("--output_dir", default="same")
    ap.add_argument("--pattern", default="norm_stats_fold_*_test.csv")
    args = ap.parse_args()

    if args.output_dir == "same":
        output_dir = args.input_dir
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    agg = load_and_aggregate(args.input_dir, args.pattern)
    dec = decompose(agg)
    dec.to_csv(os.path.join(output_dir, "gate_vs_norm_decomposition.csv"), index=False)

    # text summary
    L = ["GATE vs NORM DECOMPOSITION — effective audio share drop (+10 → −10 dB)",
         "=" * 74,
         "Confound tested: is the effective-share decline driven by the learned",
         "gate, or merely by audio embedding-norm shrinkage under noise?", ""]
    for _, r in dec.iterrows():
        L.append(f"  {r.noise_type:<10} total {r.total_share_drop:+.4f} | "
                 f"GATE {r.gate_driven:+.4f} ({r.gate_pct:3.0f}%) | "
                 f"NORM {r.norm_driven:+.4f} ({r.norm_pct:3.0f}%)")
    L.append("")
    L.append("Raw gate g_audio (norm-INDEPENDENT) drop, ranked by severity:")
    for _, r in dec.sort_values("raw_g_audio_drop", ascending=False).iterrows():
        L.append(f"  {r.noise_type:<10} {r['g_audio_10dB']:.4f} → {r['g_audio_-10dB']:.4f} "
                 f"(Δ {r.raw_g_audio_drop:+.4f})")
    L.append("")
    L.append("Reading: gate-driven % > 50 everywhere => the gate is the primary")
    L.append("driver, not a passive norm artifact. Raw gate ordering by severity")
    L.append("(norm-independent) confirms the gate responds to audio degradation.")
    summary = "\n".join(L)
    with open(os.path.join(output_dir, "gate_vs_norm_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print("\n" + summary + "\n")

    '''
    png = os.path.join(output_dir, "gate_vs_norm_decomposition.png")
    pdf = os.path.join(output_dir, "gate_vs_norm_decomposition.pdf")
    make_figure(agg, dec, png, pdf)
    '''

    make_gate_figure(agg,
        os.path.join(output_dir, "gate_tracks_snr.png"),
        os.path.join(output_dir, "gate_tracks_snr.pdf"))
    make_decomp_figure(dec,
        os.path.join(output_dir, "gate_vs_norm_decomposition.png"),
        os.path.join(output_dir, "gate_vs_norm_decomposition.pdf"))

    print("Wrote: gate_vs_norm_decomposition.(csv|png|pdf), gate_vs_norm_summary.txt")


if __name__ == "__main__":
    main()