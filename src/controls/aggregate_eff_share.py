#!/usr/bin/env python3
"""
Paper artifact: the effective-audio-share aggregation supporting the decomposition control.

Aggregate per-fold norm / effective-share statistics (test set) for the
primary untied vector-gate fusion system.

Produces:
  (1) aggregated_eff_share_by_noise_snr.csv  -- per-condition mean +/- std
      across folds, in the same row structure as the EER aggregation tables.
  (2) eff_share_stats_summary.txt            -- per-noise net drops, difficulty
      ordering, and the audio-norm shrink finding (descriptive only).
  (3) eff_share_vs_snr.(png|pdf)             -- effective audio share vs SNR.
  (4) norm_vs_snr.(png|pdf)                  -- audio embedding norm vs SNR.

Aggregation unit = fold. Within-fold values are already per-sample means;
we average those across folds and take the across-fold std. Raw samples are
NOT pooled across folds (that would ignore the cross-validation structure).

Statistical note: NO significance test is computed here. The effective share
is a confounded quantity (gate x norm); its SNR dependence is resolved by the
gate-vs-norm decomposition, not by a hypothesis test. The one directional
significance claim in the analysis -- that the RAW gate g_audio falls with SNR
-- is tested separately across the five noise types (n=5 Wilcoxon / sign test,
p = 0.031), consistent with the evaluation protocol. Testing the 25 fold x noise
drops as if independent would overstate the evidence and is deliberately avoided.

Usage:
    python -m src.controls.aggregate_eff_share \
        --input-dir  ./norm_stats \
        --output-dir ./results \
        --pattern    "norm_stats_fold_*_test.csv"

Expected input columns (one CSV per fold):
    condition, noise_type, snr_db,
    norm_audio_mean, norm_egg_mean,
    g_audio_mean, g_egg_mean,
    contrib_audio_mean, contrib_egg_mean,
    eff_audio_share_mean, eff_audio_share_std
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from src import config
from src import _console  # noqa: F401  (UTF-8 safe stdout)


# ---- configuration --------------------------------------------------------
NOISES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_ORDER = [-10, -5, 0, 5, 10]            # increasing SNR (-10 on the left)

# colourblind-friendly colour + distinct marker per noise (greyscale-safe)
STYLE = {
    "cafeteria": {"color": "#B85042", "marker": "o"},
    "metro":     {"color": "#065A82", "marker": "s"},
    "office":    {"color": "#2C7A4B", "marker": "^"},
    "park":      {"color": "#C9A227", "marker": "D"},
    "traffic":   {"color": "#9467bd", "marker": "v"},
}
# small horizontal offset per noise so overlapping markers/bars separate
OFFSETS = {"cafeteria": -0.16, "metro": -0.08, "office": 0.0,
           "park": 0.08, "traffic": 0.16}


def load_folds(input_dir, pattern):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        sys.exit(f"No files matched {pattern!r} in {input_dir!r}")
    frames = []
    for fi, f in enumerate(files):
        d = pd.read_csv(f)
        d["fold"] = fi
        frames.append(d)
    print(f"Loaded {len(files)} folds from {input_dir}")
    A = pd.concat(frames, ignore_index=True)
    A["eff_egg_share_mean"] = 1.0 - A["eff_audio_share_mean"]
    return A, len(files)


def aggregate(A, n_folds):
    """Mean +/- std across folds, per (noise_type, snr_db). Clean handled
    separately because its snr_db is NaN and would be dropped by groupby."""
    agg_cols = {
        "eff_audio_share_mean": ["mean", "std"],
        "eff_egg_share_mean":   ["mean", "std"],
        "g_audio_mean":         ["mean", "std"],
        "g_egg_mean":           ["mean", "std"],
        "norm_audio_mean":      ["mean", "std"],
        "norm_egg_mean":        ["mean", "std"],
    }
    noisy = A[A["noise_type"].isin(NOISES)].copy()
    g = noisy.groupby(["noise_type", "snr_db"]).agg(agg_cols)
    g.columns = ["_".join(c).rstrip("_") for c in g.columns]
    g = g.reset_index()
    g["n_folds"] = n_folds

    clean = A[A["noise_type"] == "clean"]
    clean_row = {
        "noise_type": "clean", "snr_db": np.nan,
        "eff_audio_share_mean_mean": clean["eff_audio_share_mean"].mean(),
        "eff_audio_share_mean_std":  clean["eff_audio_share_mean"].std(ddof=1),
        "eff_egg_share_mean_mean":   clean["eff_egg_share_mean"].mean(),
        "eff_egg_share_mean_std":    clean["eff_egg_share_mean"].std(ddof=1),
        "g_audio_mean_mean":         clean["g_audio_mean"].mean(),
        "g_audio_mean_std":          clean["g_audio_mean"].std(ddof=1),
        "g_egg_mean_mean":           clean["g_egg_mean"].mean(),
        "g_egg_mean_std":            clean["g_egg_mean"].std(ddof=1),
        "norm_audio_mean_mean":      clean["norm_audio_mean"].mean(),
        "norm_audio_mean_std":       clean["norm_audio_mean"].std(ddof=1),
        "norm_egg_mean_mean":        clean["norm_egg_mean"].mean(),
        "norm_egg_mean_std":         clean["norm_egg_mean"].std(ddof=1),
        "n_folds": n_folds,
    }
    agg = pd.concat([pd.DataFrame([clean_row]), g], ignore_index=True)
    agg = agg.rename(columns={
        "eff_audio_share_mean_mean": "eff_audio_share_mean",
        "eff_audio_share_mean_std":  "eff_audio_share_std",
        "eff_egg_share_mean_mean":   "eff_egg_share_mean",
        "eff_egg_share_mean_std":    "eff_egg_share_std",
        "g_audio_mean_mean":         "g_audio_mean",
        "g_audio_mean_std":          "g_audio_std",
        "g_egg_mean_mean":           "g_egg_mean",
        "g_egg_mean_std":            "g_egg_std",
        "norm_audio_mean_mean":      "norm_audio_mean",
        "norm_audio_mean_std":       "norm_audio_std",
        "norm_egg_mean_mean":        "norm_egg_mean",
        "norm_egg_mean_std":         "norm_egg_std",
    })
    agg["snr_numeric"] = agg["snr_db"].fillna(np.inf)   # clean sorts as best
    agg = agg.sort_values(["noise_type", "snr_numeric"],
                          ascending=[True, True]).reset_index(drop=True)
    col_order = ["noise_type", "snr_db", "snr_numeric",
                 "eff_audio_share_mean", "eff_audio_share_std",
                 "eff_egg_share_mean", "eff_egg_share_std",
                 "g_audio_mean", "g_audio_std",
                 "g_egg_mean", "g_egg_std",
                 "norm_audio_mean", "norm_audio_std",
                 "norm_egg_mean", "norm_egg_std", "n_folds"]
    return agg[col_order]


def write_summary(agg):
    """Descriptive summary only -- NO significance test (see module docstring)."""
    per_noise = {}
    for nz in NOISES:
        hi = agg[(agg.noise_type == nz) & (agg.snr_db == 10)]["eff_audio_share_mean"].values[0]
        lo = agg[(agg.noise_type == nz) & (agg.snr_db == -10)]["eff_audio_share_mean"].values[0]
        per_noise[nz] = (hi, lo, hi - lo)

    norm_clean = agg[agg.noise_type == "clean"]["norm_audio_mean"].values[0]
    norm_caf10 = agg[(agg.noise_type == "cafeteria") & (agg.snr_db == -10)]["norm_audio_mean"].values[0]

    L = []
    L.append("EFFECTIVE AUDIO SHARE -- DESCRIPTIVE SUMMARY (test set, primary untied vector gate)")
    L.append("=" * 82)
    L.append("Aggregation: mean +/- std across folds, per (noise_type, SNR).")
    L.append("")
    L.append("NOTE: no significance test is reported for the effective share. It is a")
    L.append("confounded quantity (gate x norm) and is resolved by the gate-vs-norm")
    L.append("decomposition. The raw-gate SNR trend is tested separately (n=5 across")
    L.append("noise types, p = 0.031); see the decomposition/gate-tracks-SNR outputs.")
    L.append("")
    L.append("Per-noise aggregate net share drop (+10 dB -> -10 dB), mean across folds:")
    for nz in NOISES:
        hi, lo, dd = per_noise[nz]
        L.append(f"  {nz:<10} {hi:.3f} -> {lo:.3f}   net {dd:+.3f}")
    L.append("")
    ordered = sorted(per_noise.items(), key=lambda kv: -kv[1][2])
    L.append("Drop ranking (largest first; expect cafeteria/metro > ... > office):")
    L.append("  " + ", ".join(f"{k}({v[2]:+.3f})" for k, v in ordered))
    L.append("")
    L.append("Supporting finding -- audio embedding norm shrinks under noise:")
    L.append(f"  clean audio norm = {norm_clean:.1f}  ->  cafeteria -10 dB = {norm_caf10:.1f}")
    L.append("  (part of the down-weighting is emergent in the encoder, before the gate;")
    L.append("   the decomposition quantifies how much of the share drop this accounts for.)")
    return "\n".join(L)


def _series(agg, nz, col):
    return np.array([agg[(agg.noise_type == nz) & (agg.snr_db == s)][col].values[0]
                     for s in SNR_ORDER])


def _base_rc():
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9,
    })


def make_share_figure(agg, out_png, out_pdf):
    """Effective audio share vs SNR, per noise type, with fold-std bars
    and per-noise x-jitter."""
    _base_rc()
    xpos = np.arange(len(SNR_ORDER))
    xlabels = [("+%d" % s if s > 0 else str(s)) for s in SNR_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for nz in NOISES:
        m = _series(agg, nz, "eff_audio_share_mean")
        sd = _series(agg, nz, "eff_audio_share_std")
        st = STYLE[nz]
        ax.errorbar(xpos + OFFSETS[nz], m, yerr=sd, linestyle="None",
                    marker=st["marker"], color=st["color"], lw=2, ms=5,
                    capsize=3, capthick=1.0, elinewidth=1.0, label=nz)

    clean_share = agg[agg.noise_type == "clean"]["eff_audio_share_mean"].values[0]
    ax.axhline(clean_share, ls="--", color="#5A6577", lw=1.2)
    ax.text(0.02, clean_share + 0.006, f"clean $\\approx$ {clean_share:.2f}",
            color="#5A6577", fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.axhline(0.5, ls=":", color="#AAB2BD", lw=1)
    ax.text(len(SNR_ORDER) - 1, 0.5 + 0.006, "equal share",
            color="#AAB2BD", fontsize=8.5, ha="right")

    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Audio SNR (dB)")
    ax.set_ylabel("Effective audio share")
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.legend(frameon=True, ncol=2, loc="lower right", title="Noise type")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def make_norm_figure(agg, out_png, out_pdf):
    """Audio embedding norm vs SNR, per noise type, with per-noise x-jitter.
    Fold std is negligible (<~0.21) so error bars are drawn but effectively
    invisible; the EGG norm is a fixed reference line."""
    _base_rc()
    xpos = np.arange(len(SNR_ORDER))
    xlabels = [("+%d" % s if s > 0 else str(s)) for s in SNR_ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for nz in NOISES:
        m = _series(agg, nz, "norm_audio_mean")
        sd = _series(agg, nz, "norm_audio_std")
        st = STYLE[nz]
        ax.errorbar(xpos, m, yerr=sd, linestyle="-",
                    marker=st["marker"], color=st["color"], lw=2, ms=5,
                    capsize=3, capthick=1.0, elinewidth=1.0, label=nz)

    norm_clean_v = agg[agg.noise_type == "clean"]["norm_audio_mean"].values[0]
    egg_norm_v = agg[agg.noise_type == "clean"]["norm_egg_mean"].values[0]
    ax.axhline(norm_clean_v, ls="--", color="#5A6577", lw=1.2)
    ax.text(0.02, norm_clean_v + 0.4, f"clean audio $\\approx$ {norm_clean_v:.0f}",
            color="#5A6577", fontsize=8.5, transform=ax.get_yaxis_transform())
    ax.axhline(egg_norm_v, ls="-.", color="#444", lw=1)
    ax.text(len(SNR_ORDER) - 1, egg_norm_v + 0.5,
            f"EGG norm $\\approx$ {egg_norm_v:.0f} (fixed)",
            color="#444", fontsize=8.5, ha="right")

    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Audio SNR (dB)")
    ax.set_ylabel("Audio embedding norm")
    ax.legend(frameon=True, ncol=2, loc="lower left", title="Noise type", bbox_to_anchor=(0.02, 0.3))

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default=str(config.RESULTS_DIR / "control_decomposition"),
                    help="Directory containing the per-fold CSVs")
    ap.add_argument("--output-dir", default=str(config.RESULTS_DIR / "control_decomposition"),
                    help="Where to write outputs")
    ap.add_argument("--pattern", default="norm_stats_fold_*_test.csv",
                    help="Glob for the per-fold CSVs")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    A, n_folds = load_folds(args.input_dir, args.pattern)
    agg = aggregate(A, n_folds)

    csv_path = os.path.join(args.output_dir, "aggregated_eff_share_by_noise_snr.csv")
    agg.to_csv(csv_path, index=False)

    summary = write_summary(agg)
    txt_path = os.path.join(args.output_dir, "eff_share_stats_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print("\n" + summary + "\n")

    make_share_figure(agg,
        os.path.join(args.output_dir, "eff_share_vs_snr.png"),
        os.path.join(args.output_dir, "eff_share_vs_snr.pdf"))
    make_norm_figure(agg,
        os.path.join(args.output_dir, "norm_vs_snr.png"),
        os.path.join(args.output_dir, "norm_vs_snr.pdf"))

    print(f"Wrote:\n  {csv_path}\n  {txt_path}\n"
          f"  eff_share_vs_snr.(png|pdf)\n  norm_vs_snr.(png|pdf)")


if __name__ == "__main__":
    main()