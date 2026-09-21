"""
Paper artifact: the score-level late-fusion baseline row of Table 1 and the late-fusion curves in the motivation figure.

Late fusion (score averaging) evaluation using cached embeddings.

For each fold and each audio condition (clean + 25 noisy):
  - Compute audio cosine similarity from cached audio embeddings
  - Compute EGG cosine similarity from cached EGG embeddings (always clean)
  - Final score = 0.5 * audio_score + 0.5 * egg_score
  - Compute EER and minDCF on test pairs (same pair pool as unimodal evaluations)

No training. Pure score-level fusion.

Run:
    python -m src.fusion.score_level --embedding_dir /path/to/Embeddings \
                               --output_dir /path/to/Late_Fusion_Results
"""

import argparse
import csv
import itertools
import random
from pathlib import Path

import numpy as np
import torch

import src.common as helper_functions
from src import config

# ==============================================================================
# CLI
# ==============================================================================

parser = argparse.ArgumentParser(description="Late fusion (score averaging) evaluation.")
parser.add_argument('--embedding_dir', type=str, default=str(config.EMBEDDING_DIR),
                    help="Directory with audio_clean_fold*.npz, audio_noisy_fold*.npz, egg_clean_fold*.npz")
parser.add_argument('--output_dir', type=str, default=str(config.RESULTS_DIR / "score_level_late_fusion"))
parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
parser.add_argument('--pairs_per_category', type=int, default=2000)
parser.add_argument('--audio_weight', type=float, default=0.5,
                    help="Weight for audio score. EGG weight = 1 - audio_weight. Default 0.5 = unweighted average.")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
embedding_dir = Path(args.embedding_dir)

FOLD_SEED = 42
NOISE_TYPES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_LEVELS = [-10, -5, 0, 5, 10]

# ==============================================================================
# Pair generation — IDENTICAL to test evaluation scripts
# ==============================================================================

def build_test_pairs(test_utt_ids, test_speakers, pairs_per_category, seed):
    """
    Build (label, gender_match, idx_a, idx_b) pairs over the test split.
    Indices are positions within the test_utt_ids array.
    """
    rng = random.Random(seed)

    # Group test indices by speaker
    speaker_to_idx = {}
    for i, sp in enumerate(test_speakers):
        speaker_to_idx.setdefault(sp, []).append(i)
    speakers = sorted(speaker_to_idx.keys())

    # Same-speaker positives
    same_pairs = []
    per_spk = max(1, pairs_per_category // len(speakers))
    for sp in speakers:
        combos = list(itertools.combinations(speaker_to_idx[sp], 2))
        same_pairs.extend(rng.sample(combos, min(per_spk, len(combos))))

    # Different-speaker by gender match
    all_combos = list(itertools.combinations(speakers, 2))
    sg_combos = [(a, b) for a, b in all_combos if a[0] == b[0]]
    cg_combos = [(a, b) for a, b in all_combos if a[0] != b[0]]

    def sample_negs(combos):
        out = []
        if not combos:
            return out
        per_combo = max(1, pairs_per_category // len(combos))
        for a, b in combos:
            cross = list(itertools.product(speaker_to_idx[a], speaker_to_idx[b]))
            out.extend(rng.sample(cross, min(per_combo, len(cross))))
        return out

    sg_negs = sample_negs(sg_combos)
    cg_negs = sample_negs(cg_combos)

    pairs = (
        [(1, 1, i, j) for i, j in same_pairs]
        + [(0, 1, i, j) for i, j in sg_negs]
        + [(0, 0, i, j) for i, j in cg_negs]
    )
    return pairs

# ==============================================================================
# Score computation
# ==============================================================================

def cosine_pair_scores(emb_a, emb_b):
    """Compute pairwise cosine similarity. Both inputs must be (N, D) and pre-indexed."""
    if isinstance(emb_a, np.ndarray):
        emb_a = torch.from_numpy(emb_a)
    if isinstance(emb_b, np.ndarray):
        emb_b = torch.from_numpy(emb_b)
    return torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)

# ==============================================================================
# Per-fold evaluation
# ==============================================================================

def evaluate_fold(fold_idx):
    print(f"\n{'='*72}\nFOLD {fold_idx} — late fusion (score avg)\n{'='*72}")

    # Load cached embeddings
    audio_clean = np.load(embedding_dir / f"audio_clean_fold{fold_idx}.npz", allow_pickle=True)
    audio_noisy = np.load(embedding_dir / f"audio_noisy_fold{fold_idx}.npz", allow_pickle=True)
    egg_clean   = np.load(embedding_dir / f"egg_clean_fold{fold_idx}.npz",   allow_pickle=True)

    # Sanity checks
    assert np.array_equal(audio_clean['utterance_ids'], audio_noisy['utterance_ids']), \
        "audio_clean and audio_noisy utt_ids mismatch"
    assert np.array_equal(audio_clean['utterance_ids'], egg_clean['utterance_ids']), \
        "audio_clean and egg_clean utt_ids mismatch"

    splits = audio_clean['split']
    speakers = audio_clean['speaker_ids']
    test_mask = (splits == 'test')
    test_idx = np.where(test_mask)[0]
    test_speakers = speakers[test_mask]
    test_utt_ids = audio_clean['utterance_ids'][test_mask]
    print(f"Test items: {len(test_idx)}")

    # Subset embeddings to test split
    a_clean_test = audio_clean['embeddings'][test_idx]                      # (N_test, D)
    a_noisy_test = audio_noisy['embeddings'][test_idx]                      # (N_test, 25, D)
    e_test       = egg_clean['embeddings'][test_idx]                        # (N_test, D_egg)

    # Recover noise/SNR axis labels
    noise_axis = list(audio_noisy['noise_types'])
    snr_axis   = list(audio_noisy['snr_levels'])
    cond_index = {(n, s): i for i, (n, s) in enumerate(zip(noise_axis, snr_axis))}

    # Build test pairs (indexed within test_idx)
    test_seed = FOLD_SEED + 1000 + fold_idx
    pairs = build_test_pairs(test_utt_ids, test_speakers, args.pairs_per_category, test_seed)
    labels = torch.tensor([p[0] for p in pairs])
    gender_match = torch.tensor([p[1] for p in pairs])
    idx_a = np.array([p[2] for p in pairs])
    idx_b = np.array([p[3] for p in pairs])
    print(f"Pairs: {len(pairs)} (SG+: {(labels==1).sum().item()}, "
          f"SG-: {((labels==0)&(gender_match==1)).sum().item()}, "
          f"CG-: {((labels==0)&(gender_match==0)).sum().item()})")

    # EGG scores — same for all audio conditions (EGG is always clean)
    egg_scores = cosine_pair_scores(e_test[idx_a], e_test[idx_b])

    rows = []

    # --- Clean audio condition ---
    audio_scores = cosine_pair_scores(a_clean_test[idx_a], a_clean_test[idx_b])
    fused_scores = args.audio_weight * audio_scores + (1 - args.audio_weight) * egg_scores
   
    results = score_to_metrics(fused_scores, labels, gender_match)
    rows.append(make_row(fold_idx, "clean", None, len(pairs), results))
    print(f"  clean:               SG-EER {results['sg_eer']:.4%}  mDCF {results['sg_min_dcf']:.4f}")

    # --- 25 noisy conditions ---
    for noise in NOISE_TYPES:
        for snr in SNR_LEVELS:
            ci = cond_index[(noise, snr)]
            # Audio side: use clean for "enroll" (idx_a), noisy for "test" (idx_b)
            audio_a = a_clean_test[idx_a]                  # clean enroll
            audio_b = a_noisy_test[:, ci, :][idx_b]        # noisy test
            audio_scores_noisy = cosine_pair_scores(audio_a, audio_b)
            fused_scores = args.audio_weight * audio_scores_noisy + (1 - args.audio_weight) * egg_scores
            results = score_to_metrics(fused_scores, labels, gender_match)
            rows.append(make_row(fold_idx, noise, snr, len(pairs), results))
            print(f"  {noise:9s} @ {snr:>3} dB:    SG-EER {results['sg_eer']:.4%}  mDCF {results['sg_min_dcf']:.4f}")

    return rows


def score_to_metrics(scores, labels, gender_match):
    """
    Reproduces get_scores_stratified's logic but takes raw scores instead of
    (e1, e2) embedding pairs. Returns the same dict structure.
    """
    # get_scores_stratified expects emb_pairs and computes cosine internally.
    # We supply raw scores by faking emb_pairs as (scores, ones) — but that's hacky.
    # Cleaner: directly call helper_functions.EER and helper_functions.min_DCF
    # on the score distributions.

    pos = scores[labels == 1]
    neg = scores[labels == 0]

    # SG = same-gender (gender_match == 1), CG = cross-gender (gender_match == 0)
    sg_mask = gender_match == 1
    cg_mask = gender_match == 0
    sg_pos = scores[(labels == 1) & sg_mask]
    sg_neg = scores[(labels == 0) & sg_mask]
    cg_neg = scores[(labels == 0) & cg_mask]

    sg_eer, sg_threshold = helper_functions.EER(sg_pos, sg_neg)
    sg_min_dcf, _        = helper_functions.minDCF(sg_pos, sg_neg)
    overall_eer, overall_threshold = helper_functions.EER(pos, neg)
    overall_min_dcf, _   = helper_functions.minDCF(pos, neg)

    # CG-FAR at the SG threshold (consistent with other scripts)
    cg_far = (cg_neg >= sg_threshold).float().mean().item() if len(cg_neg) > 0 else 0.0

    return {
        "sg_eer": float(sg_eer),
        "sg_min_dcf": float(sg_min_dcf),
        "sg_threshold": float(sg_threshold),
        "cg_far": float(cg_far),
        "overall_eer": float(overall_eer),
        "overall_min_dcf": float(overall_min_dcf),
        "overall_threshold": float(overall_threshold),
    }


def make_row(fold_idx, noise_type, snr_db, n_pairs, results):
    return {
        "fold": fold_idx,
        "noise_type": noise_type,
        "snr_db": snr_db if snr_db is not None else "",
        "n_pairs": n_pairs,
        **results,
    }

# ==============================================================================
# Main
# ==============================================================================

all_rows = []
for f in args.folds:
    all_rows.extend(evaluate_fold(f))

# Save flat CSV
csv_path = output_dir / f"late_fusion_results_audio_w{args.audio_weight}.csv"
fieldnames = ["fold", "noise_type", "snr_db", "n_pairs",
              "sg_eer", "sg_min_dcf", "sg_threshold", "cg_far",
              "overall_eer", "overall_min_dcf", "overall_threshold"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nSaved: {csv_path}")

# Aggregate across folds per (noise_type, snr_db)
print(f"\n{'='*80}\nAGGREGATED ACROSS FOLDS — late fusion (audio_weight={args.audio_weight})\n{'='*80}")
import pandas as pd
df = pd.DataFrame(all_rows)
df['snr_db'] = df['snr_db'].replace('', np.nan)
agg = df.groupby(['noise_type', 'snr_db'], dropna=False).agg(
    sg_eer_mean=('sg_eer', 'mean'),
    sg_eer_std=('sg_eer', 'std'),
    sg_mdcf_mean=('sg_min_dcf', 'mean'),
    sg_mdcf_std=('sg_min_dcf', 'std'),
    n_folds=('fold', 'count'),
).reset_index()
agg.to_csv(output_dir / f"late_fusion_aggregated_audio_w{args.audio_weight}.csv", index=False)

print(agg.to_string(index=False))
print(f"\nSaved aggregated: {output_dir / f'late_fusion_aggregated_audio_w{args.audio_weight}.csv'}")