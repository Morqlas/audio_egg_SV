"""
Paper artifact: produces the per-fold norm_stats CSVs consumed by the gate-vs-norm decomposition control.

Embedding-norm and effective-contribution analysis for vector gate fusion.

The gate values alone do not tell you how much each modality contributes to the
fused embedding, because fusion operates on UN-normalized embeddings:

    e_fused = g_audio (X) e_audio + g_egg (X) e_egg

The honest measure of "reliance" on each modality is the norm of its gated
contribution: ||g_audio (X) e_audio|| vs ||g_egg (X) e_egg||. This script
computes, per condition:

  - raw embedding norms:        ||e_audio||, ||e_egg||
  - gate means:                 mean(g_audio), mean(g_egg)   [for reference]
  - gated contribution norms:   ||g_audio (X) e_audio||, ||g_egg (X) e_egg||
  - effective audio share:      ||g_a (X) e_a|| / (||g_a (X) e_a|| + ||g_e (X) e_e||)

The "effective audio share" is the quantity to cite for cross-modal reliance,
NOT the raw gate ratio.

Run:
    python -m src.controls.norm_stats \
        --checkpoint /path/vector_gate_fold_0.pth \
        --embedding_dir /path/Embeddings --fold 0 --split test \
        --output_csv /path/norm_stats_fold_0_test.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from src.fusion.modules import get_fusion
from src import config
from src import _console  # noqa: F401  (UTF-8 safe stdout)

NOISE_TYPES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_LEVELS = [-10, -5, 0, 5, 10]
CONDITION_LABELS = ["clean"]
for nt in NOISE_TYPES:
    for snr in SNR_LEVELS:
        CONDITION_LABELS.append(f"{nt}_{snr}dB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_type', type=str, required=True, choices=['mlp', 'scalar_gate', 'vector_gate', 'tied_vector_gate', 'no_trunk_vector_gate', 'untied_scalar_gate'])
    parser.add_argument('--gate_input_norm', type=str, required=True, choices=['none', 'l2', 'layernorm'])
    parser.add_argument('--shuffle_egg', action='store_true')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--split', default="test", choices=["train", "val", "test"])
    parser.add_argument('--checkpoint', default=str(config.CHECKPOINT_DIR))
    parser.add_argument('--embedding_dir', default=str(config.EMBEDDING_DIR))
    parser.add_argument('--output_dir', default=str(config.RESULTS_DIR / "control_decomposition"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = args.fusion_type + ("_shuffled" if args.shuffle_egg else "") + (f"_{args.gate_input_norm}" if args.gate_input_norm != 'none' else "")
    ckpt_path = Path(args.checkpoint) / f"{tag}_fold_{args.fold}.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt['args']
    if (ckpt.get('fusion_type', sa.get('fusion_type')) != "vector_gate") and (ckpt.get('fusion_type', sa.get('fusion_type')) != "no_trunk_vector_gate"):
        raise ValueError("This script is for vector_gate/no_trunk_vector_gate checkpoints only.")

    model = get_fusion(
        name=args.fusion_type,
        audio_dim=sa['audio_dim'], egg_dim=sa['egg_dim'],
        hidden_dim=sa['hidden_dim'], output_dim=sa['output_dim'],
        dropout=sa['dropout'], gate_input_norm=args.gate_input_norm,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    emb = Path(args.embedding_dir)
    a_clean = np.load(emb / f"audio_clean_fold{args.fold}.npz", allow_pickle=True)
    a_noisy = np.load(emb / f"audio_noisy_fold{args.fold}.npz", allow_pickle=True)
    e_clean = np.load(emb / f"egg_clean_fold{args.fold}.npz",  allow_pickle=True)

    mask = (a_clean['split'] == args.split)
    e_audio_clean = torch.from_numpy(a_clean['embeddings'][mask]).float().to(device)
    e_audio_noisy = torch.from_numpy(a_noisy['embeddings'][mask]).float().to(device)
    e_egg = torch.from_numpy(e_clean['embeddings'][mask]).float().to(device)

    print(f"Split={args.split}, fold={args.fold}, N={e_audio_clean.shape[0]}")

    rows = []
    with torch.no_grad():
        for ci, label in enumerate(CONDITION_LABELS):
            e_audio = e_audio_clean if ci == 0 else e_audio_noisy[:, ci - 1, :]

            gates = model.get_gate(e_audio, e_egg)
            g_a = gates['g_audio']   # (N, D)
            g_e = gates['g_egg']     # (N, D)

            # Raw embedding norms
            norm_audio = e_audio.norm(dim=-1)           # (N,)
            norm_egg = e_egg.norm(dim=-1)               # (N,)

            # Gated contribution norms (the honest reliance measure)
            contrib_audio = (g_a * e_audio).norm(dim=-1)   # (N,)
            contrib_egg = (g_e * e_egg).norm(dim=-1)       # (N,)

            eff_share = contrib_audio / (contrib_audio + contrib_egg + 1e-12)

            noise_type = "clean" if ci == 0 else label.rsplit("_", 1)[0]
            snr = "" if ci == 0 else int(label.rsplit("_", 1)[1].replace("dB", ""))

            rows.append({
                "condition": label,
                "noise_type": noise_type,
                "snr_db": snr,
                "norm_audio_mean": float(norm_audio.mean()),
                "norm_egg_mean": float(norm_egg.mean()),
                "g_audio_mean": float(g_a.mean()),
                "g_egg_mean": float(g_e.mean()),
                "contrib_audio_mean": float(contrib_audio.mean()),
                "contrib_egg_mean": float(contrib_egg.mean()),
                "eff_audio_share_mean": float(eff_share.mean()),
                "eff_audio_share_std": float(eff_share.std()),
            })

    out = Path(args.output_dir) / f"norm_stats_fold_{args.fold}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out}")

    # Quick interpretation
    clean = rows[0]
    caf10 = rows[1]  # cafeteria -10dB
    print("\nRaw embedding norms (clean):")
    print(f"  ||e_audio|| = {clean['norm_audio_mean']:.3f}")
    print(f"  ||e_egg||   = {clean['norm_egg_mean']:.3f}")
    print(f"  ratio audio/egg = {clean['norm_audio_mean']/clean['norm_egg_mean']:.2f}x")
    print("\nGate means (clean):")
    print(f"  g_audio = {clean['g_audio_mean']:.4f}")
    print(f"  g_egg   = {clean['g_egg_mean']:.4f}")
    print(f"  raw gate ratio egg/audio = {clean['g_egg_mean']/clean['g_audio_mean']:.2f}x")
    print("\nEffective contribution (clean) -- the HONEST reliance measure:")
    print(f"  ||g_a (X) e_a|| = {clean['contrib_audio_mean']:.3f}")
    print(f"  ||g_e (X) e_e|| = {clean['contrib_egg_mean']:.3f}")
    print(f"  effective audio share = {clean['eff_audio_share_mean']:.3f}")
    print(f"\nEffective audio share: clean={clean['eff_audio_share_mean']:.3f} "
          f"-> cafeteria-10dB={caf10['eff_audio_share_mean']:.3f} "
          f"(Δ={caf10['eff_audio_share_mean']-clean['eff_audio_share_mean']:+.3f})")
    print("\nInterpretation guide:")
    print("  - If raw gate ratio (egg/audio) >> effective measure suggests EGG dominance,")
    print("    BUT norms differ, then the gate asymmetry is partly norm compensation.")
    print("  - The effective audio share dropping with SNR is the real reliance signal.")


if __name__ == "__main__":
    main()