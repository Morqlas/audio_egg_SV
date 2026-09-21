#!/usr/bin/env python3
"""Table 1: per-system mean SG-EER / minDCF over the 26 test conditions, with
trainable fusion parameter counts.

Backs paper Table 1. Reads only the committed per-condition CSVs under
results/ and counts parameters by instantiating the fusion modules, so the
table cannot drift from either the recorded results or the shipped code.

Aggregation unit = condition: the flat mean over the 26 rows (clean + 5 noise
types x 5 SNRs), matching how the headline numbers are quoted in the paper.

Run:
    python -m src.figures.table1_main_results
    python -m src.figures.table1_main_results --format latex
"""
import argparse
import csv
import sys
from pathlib import Path

import torch

from src import config
from src.fusion.modules import get_fusion

# label -> (results subdir, fusion factory name or None for score-level/unimodal)
SYSTEMS = [
    ("Audio only (ECAPA2, frozen)", "unimodal_audio",          None),
    ("Late fusion (score, a=0.5)",  "score_level_late_fusion", None),
    ("MLP fusion",                  "mlp",                     "mlp"),
    ("Gate + trunk (ablation)",     "ablation_trunk",          "vector_gate"),
    ("Gate, tied (ablation)",       "ablation_tied",           "no_trunk_tied_vector_gate"),
    ("Gate, scalar (ablation)",     "ablation_scalar",         "no_trunk_scalar_gate"),
    ("Gate (primary)",              "gate_primary",            "no_trunk_vector_gate"),
]

# filename within each results subdir
FILENAMES = {
    "score_level_late_fusion": "late_fusion_aggregated_audio_weight_0.5.csv",
}
DEFAULT_FILENAME = "per_condition.csv"

ARCH = dict(audio_dim=192, egg_dim=192, hidden_dim=384, output_dim=192, dropout=0.3)


def load_mean(path: Path):
    """Flat mean of sg_eer_mean / sg_mdcf_mean across every row in the CSV."""
    if not path.exists():
        return None, None, 0
    eers, dcfs = [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("sg_eer_mean"):
                eers.append(float(r["sg_eer_mean"]))
            if r.get("sg_mdcf_mean"):
                dcfs.append(float(r["sg_mdcf_mean"]))
    if not eers:
        return None, None, 0
    return (sum(eers) / len(eers),
            (sum(dcfs) / len(dcfs)) if dcfs else None,
            len(eers))


def count_params(name):
    """Trainable parameters in the fusion module (encoders are frozen)."""
    if name is None:
        return None
    m = get_fusion(name=name, **ARCH)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=str(config.RESULTS_DIR))
    ap.add_argument("--format", choices=["text", "latex", "csv"], default="text")
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args()

    root = Path(args.results_dir)
    rows = []
    for label, sub, fusion in SYSTEMS:
        fn = FILENAMES.get(sub, DEFAULT_FILENAME)
        eer, dcf, n = load_mean(root / sub / fn)
        if eer is None:
            print(f"[warn] no data for {label} ({sub}/{fn})", file=sys.stderr)
            continue
        rows.append({
            "system": label,
            "mean_sg_eer_pct": 100 * eer,
            "mean_sg_mindcf": dcf,
            "params": count_params(fusion),
            "n_conditions": n,
        })

    if args.format == "latex":
        print(r"\begin{tabular}{lrrr}")
        print(r"\toprule")
        print(r"System & SG-EER (\%) & minDCF & Params \\")
        print(r"\midrule")
        for r in rows:
            p = "--" if r["params"] is None else f"{r['params']:,}"
            d = "--" if r["mean_sg_mindcf"] is None else f"{r['mean_sg_mindcf']:.3f}"
            print(f"{r['system']} & {r['mean_sg_eer_pct']:.2f} & {d} & {p} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    else:
        print(f"{'System':<32} {'SG-EER %':>9} {'minDCF':>8} {'Params':>10} {'n':>4}")
        print("-" * 68)
        for r in rows:
            p = "--" if r["params"] is None else f"{r['params']:,}"
            d = "--" if r["mean_sg_mindcf"] is None else f"{r['mean_sg_mindcf']:.4f}"
            print(f"{r['system']:<32} {r['mean_sg_eer_pct']:>9.4f} {d:>8} {p:>10} {r['n_conditions']:>4}")

    out = args.output_csv or str(root / "table1_main_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
