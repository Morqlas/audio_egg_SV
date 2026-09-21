"""
Paper artifact: the SG-EER / minDCF numbers in Table 1 and every per-condition CSV under results/.

Unified test evaluation for fusion variants on cached embeddings.

Supports:
    --fusion_type {mlp, scalar_gate, vector_gate}
    --shuffle_egg  (for the sanity-check runs)

For each fold:
  1. Load trained fusion checkpoint matching the requested variant.
  2. Build test pair pool (FOLD_SEED + 1000 + fold_idx, same as all other test scripts).
  3. For each of 26 conditions (clean + 25 noisy):
     - Compute fused embeddings for test utterances.
     - Clean-enroll / noisy-test pairing for noisy conditions.
     - SG-EER, SG-minDCF, CG-FAR via get_scores_stratified.
  4. Save per-fold and aggregated CSVs.

Naming:
    Checkpoint expected at: {checkpoint_dir}/{tag}_fold_{N}.pth
    where tag = fusion_type + ("_shuffled" if --shuffle_egg)

Run all folds:
    python -m src.evaluate --fusion_type vector_gate --all_folds \
        --embedding_dir /path/Embeddings \
        --checkpoint_dir /path/Fusion_Results/vector_gate/training \
        --output_dir /path/Fusion_Results/vector_gate/test
"""

import argparse
import csv
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.fusion.modules import get_fusion
import src.common as helper_functions
from src import config

# ==============================================================================
# CLI
# ==============================================================================

parser = argparse.ArgumentParser(description="Test eval for fusion variants.")
parser.add_argument('--fusion_type', type=str, required=True,
                    choices=['mlp', 'scalar_gate', 'vector_gate', 'tied_vector_gate', 'no_trunk_vector_gate', 'untied_scalar_gate', 'no_trunk_scalar_gate', 'no_trunk_tied_vector_gate'])
parser.add_argument('--shuffle_egg', action='store_true',
                    help="Use shuffled-EGG variant (must match training).")
parser.add_argument('--gate_input_norm', type=str, default='none', choices=['none', 'l2', 'layernorm'])
parser.add_argument('--fold', type=int, default=None, choices=[0, 1, 2, 3, 4])
parser.add_argument('--all_folds', action='store_true')
parser.add_argument('--embedding_dir', type=str, default=str(config.EMBEDDING_DIR))
parser.add_argument('--checkpoint_dir', type=str, default=str(config.CHECKPOINT_DIR),
                    help="Directory containing {tag}_fold_X.pth files.")
parser.add_argument('--output_dir', type=str, default=str(config.RESULTS_DIR / "eval_out"))
parser.add_argument('--pairs_per_category', type=int, default=2000)
args = parser.parse_args()

if args.fold is None and not args.all_folds:
    parser.error("Specify either --fold or --all_folds.")
if args.fold is not None and args.all_folds:
    parser.error("Use either --fold or --all_folds, not both.")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
embedding_dir = Path(args.embedding_dir)
checkpoint_dir = Path(args.checkpoint_dir)

TAG = args.fusion_type + ("_shuffled" if args.shuffle_egg else "") + (f"_{args.gate_input_norm}" if args.gate_input_norm != 'none' else "")

# ==============================================================================
# Constants — must match training and other eval scripts
# ==============================================================================

FOLD_SEED = 42
NOISE_TYPES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_LEVELS = [-10, -5, 0, 5, 10]

CONDITION_LABELS = ["clean"]
for noise in NOISE_TYPES:
    for snr in SNR_LEVELS:
        CONDITION_LABELS.append(f"{noise}_{snr}dB")

# ==============================================================================
# Pair generation — IDENTICAL to all other test scripts
# ==============================================================================

def build_test_pairs(speakers, pairs_per_category, seed):
    rng = random.Random(seed)
    speaker_to_idx = {}
    for i, sp in enumerate(speakers):
        speaker_to_idx.setdefault(sp, []).append(i)
    spk_list = sorted(speaker_to_idx.keys())

    same_pairs = []
    per_spk = max(1, pairs_per_category // len(spk_list))
    for sp in spk_list:
        combos = list(itertools.combinations(speaker_to_idx[sp], 2))
        same_pairs.extend(rng.sample(combos, min(per_spk, len(combos))))

    all_combos = list(itertools.combinations(spk_list, 2))
    sg = [(a, b) for a, b in all_combos if a[0] == b[0]]
    cg = [(a, b) for a, b in all_combos if a[0] != b[0]]

    def neg(combos):
        out = []
        if not combos:
            return out
        per_combo = max(1, pairs_per_category // len(combos))
        for a, b in combos:
            cross = list(itertools.product(speaker_to_idx[a], speaker_to_idx[b]))
            out.extend(rng.sample(cross, min(per_combo, len(cross))))
        return out

    return (
        [(1, 1, i, j) for i, j in same_pairs]
        + [(0, 1, i, j) for i, j in neg(sg)]
        + [(0, 0, i, j) for i, j in neg(cg)]
    )

# ==============================================================================
# Per-fold evaluation
# ==============================================================================

def evaluate_fold(fold_idx, device):
    print(f"\n{'='*72}\nFOLD {fold_idx} | fusion={args.fusion_type} | "
          f"shuffle_egg={args.shuffle_egg}\n{'='*72}")

    # Load cached embeddings
    a_clean_npz = np.load(embedding_dir / f"audio_clean_fold{fold_idx}.npz", allow_pickle=True)
    a_noisy_npz = np.load(embedding_dir / f"audio_noisy_fold{fold_idx}.npz", allow_pickle=True)
    e_clean_npz = np.load(embedding_dir / f"egg_clean_fold{fold_idx}.npz",   allow_pickle=True)

    assert np.array_equal(a_clean_npz['utterance_ids'], a_noisy_npz['utterance_ids'])
    assert np.array_equal(a_clean_npz['utterance_ids'], e_clean_npz['utterance_ids'])

    splits = a_clean_npz['split']
    speakers = a_clean_npz['speaker_ids']
    test_mask = (splits == 'test')
    test_idx = np.where(test_mask)[0]
    test_speakers = speakers[test_mask]
    print(f"Test items: {len(test_idx)}")

    a_clean_test = torch.from_numpy(a_clean_npz['embeddings'][test_idx]).float().to(device)
    a_noisy_test = torch.from_numpy(a_noisy_npz['embeddings'][test_idx]).float().to(device)
    e_clean_test = torch.from_numpy(e_clean_npz['embeddings'][test_idx]).float().to(device)

    # Build test pairs — SAME seed as ALL other test scripts
    test_seed = FOLD_SEED + 1000 + fold_idx
    pairs = build_test_pairs(test_speakers, args.pairs_per_category, test_seed)
    pair_labels = torch.tensor([p[0] for p in pairs])
    pair_gm     = torch.tensor([p[1] for p in pairs])
    pair_a = torch.tensor([p[2] for p in pairs], dtype=torch.long)
    pair_b = torch.tensor([p[3] for p in pairs], dtype=torch.long)
    print(f"Pairs: {len(pairs)} (SG+: {(pair_labels==1).sum()}, "
          f"SG-: {((pair_labels==0)&(pair_gm==1)).sum()}, "
          f"CG-: {((pair_labels==0)&(pair_gm==0)).sum()})")

    # Load checkpoint
    ckpt_path = checkpoint_dir / f"{TAG}_fold_{fold_idx}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = ckpt.get('args', {})

    # Sanity: checkpoint fusion_type matches request
    saved_fusion = ckpt.get('fusion_type', train_args.get('fusion_type'))
    if saved_fusion and saved_fusion != args.fusion_type:
        raise ValueError(f"Checkpoint fusion_type={saved_fusion}, "
                         f"requested {args.fusion_type}. Refusing to mix.")
    saved_shuffled = ckpt.get('shuffle_egg', train_args.get('shuffle_egg', False))
    if saved_shuffled != args.shuffle_egg:
        raise ValueError(f"Checkpoint shuffle_egg={saved_shuffled}, "
                         f"requested {args.shuffle_egg}. Refusing to mix.")

    model = get_fusion(
        name=args.fusion_type,
        audio_dim=train_args.get('audio_dim', 192),
        egg_dim=train_args.get('egg_dim', 192),
        hidden_dim=train_args.get('hidden_dim', 384),
        output_dim=train_args.get('output_dim', 192),
        dropout=train_args.get('dropout', 0.3),
        gate_input_norm=args.gate_input_norm
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('best_epoch', '?')}")

    # If shuffle_egg, replay the SAME training-time permutation on the test split
    # by loading egg_perm from the checkpoint. Note: training saved 'egg_perm_val'
    # and 'egg_perm_train' but NOT a test perm — generate it deterministically here.
    if args.shuffle_egg:
        # Deterministic test-split EGG permutation using the same seed convention.
        # Critical: this must NOT change between runs.
        test_shuffle_seed = FOLD_SEED + 9999 + 2000 + fold_idx  # distinct from train/val seeds
        rng = np.random.default_rng(test_shuffle_seed)
        N = len(test_idx)
        for attempt in range(100):
            perm = rng.permutation(N)
            if all(test_speakers[i] != test_speakers[perm[i]] for i in range(N)):
                break
        else:
            # Force-fix collisions
            perm = rng.permutation(N)
            for i in range(N):
                if test_speakers[i] == test_speakers[perm[i]]:
                    for j in range(i + 1, N):
                        if (test_speakers[j] != test_speakers[i]
                                and test_speakers[perm[j]] != test_speakers[i]):
                            perm[i], perm[j] = perm[j], perm[i]
                            break
        e_clean_test = e_clean_test[torch.from_numpy(perm).long()]
        print(f"  [shuffle_egg] permuted EGG across {N} test utts (seed={test_shuffle_seed})")

    # Pre-compute fused embeddings for clean-audio side (always idx_a side)
    with torch.no_grad():
        fused_clean_all = F.normalize(model(a_clean_test, e_clean_test), dim=-1)

    rows = []

    with torch.no_grad():
        # --- Clean ---
        f_a = fused_clean_all[pair_a].cpu()
        f_b = fused_clean_all[pair_b].cpu()
        results = helper_functions.get_scores_stratified(
            emb=[(f_a, f_b)], true_labels=pair_labels,
            gender_match_labels=pair_gm, verbose=False,
        )
        rows.append(make_row(fold_idx, "clean", None, len(pairs), results))
        print(f"  clean:                SG-EER {results['sg_eer']:.4%}  "
              f"mDCF {results['sg_min_dcf']:.4f}")

        # --- 25 noisy conditions ---
        for ni, noise in enumerate(NOISE_TYPES):
            for si, snr in enumerate(SNR_LEVELS):
                cond_idx = ni * len(SNR_LEVELS) + si
                a_noisy_at_cond = a_noisy_test[:, cond_idx, :]
                fused_noisy_all = F.normalize(model(a_noisy_at_cond, e_clean_test), dim=-1)

                f_a = fused_clean_all[pair_a].cpu()
                f_b = fused_noisy_all[pair_b].cpu()

                results = helper_functions.get_scores_stratified(
                    emb=[(f_a, f_b)], true_labels=pair_labels,
                    gender_match_labels=pair_gm, verbose=False,
                )
                rows.append(make_row(fold_idx, noise, snr, len(pairs), results))
                print(f"  {noise:9s} @ {snr:>3} dB:    "
                      f"SG-EER {results['sg_eer']:.4%}  "
                      f"mDCF {results['sg_min_dcf']:.4f}  "
                      f"CG-FAR {results['cg_far']:.4%}")

    return rows


def make_row(fold_idx, noise_type, snr_db, n_pairs, results):
    return {
        "fold": fold_idx,
        "noise_type": noise_type,
        "snr_db": snr_db if snr_db is not None else "",
        "n_pairs": n_pairs,
        "sg_eer": results['sg_eer'],
        "sg_min_dcf": results['sg_min_dcf'],
        "sg_threshold": results['sg_threshold'],
        "cg_far": results['cg_far'],
        "overall_eer": results['overall_eer'],
        "overall_min_dcf": results['overall_min_dcf'],
        "overall_threshold": results['overall_threshold'],
    }

# ==============================================================================
# Main
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | tag: {TAG}")

folds_to_run = [args.fold] if args.fold is not None else [0, 1, 2, 3, 4]
all_rows = []
for f in folds_to_run:
    all_rows.extend(evaluate_fold(f, device))

# Save per-fold flat CSV
fieldnames = ["fold", "noise_type", "snr_db", "n_pairs",
              "sg_eer", "sg_min_dcf", "sg_threshold", "cg_far",
              "overall_eer", "overall_min_dcf", "overall_threshold"]
csv_path = output_dir / f"{TAG}_test_results.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nSaved per-fold results: {csv_path}")

# Aggregate across folds
if len(folds_to_run) > 1:
    df = pd.DataFrame(all_rows)
    df['snr_db'] = df['snr_db'].replace('', np.nan)
    agg = df.groupby(['noise_type', 'snr_db'], dropna=False).agg(
        sg_eer_mean=('sg_eer', 'mean'),
        sg_eer_std=('sg_eer', 'std'),
        sg_mdcf_mean=('sg_min_dcf', 'mean'),
        sg_mdcf_std=('sg_min_dcf', 'std'),
        cg_far_mean=('cg_far', 'mean'),
        n_folds=('fold', 'count'),
    ).reset_index()
    agg_path = output_dir / f"{TAG}_test_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated: {agg_path}")

    print(f"\n{'='*72}\nMEAN +/- STD ACROSS FOLDS\n{'='*72}")
    print(agg.to_string(index=False))

print("\nDone.")