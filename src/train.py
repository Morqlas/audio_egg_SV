"""
Paper artifact: shared training recipe behind every row of Table 1 and all three controls (controls differ only by --gate_input_norm / --shuffle_egg).

Unified fusion training using cached embeddings.

Supports three fusion variants (--fusion_type):
    - mlp:          baseline (no gating)
    - scalar_gate:  tied scalar gate (ablation, coarse gating)
    - vector_gate:  untied per-dim gates (primary contribution)

Sanity check support (--shuffle_egg):
    Permutes EGG embeddings across speakers within each batch during BOTH
    training and validation. Same seed per fold for reproducibility.
    Use to test whether the fusion module is actually using EGG identity info.

Per-step audio condition is sampled uniformly from {clean} + 25 noisy.
EGG is always clean (or shuffled-clean if --shuffle_egg).

Run one fold:
    python -m src.train --fusion_type vector_gate --fold 0 \
        --embedding_dir /path/to/Embeddings \
        --output_dir /path/to/Fusion_Results/vector_gate

Run all folds:
    python -m src.train --fusion_type vector_gate --all_folds ...

Sanity check (after primary training is done):
    python -m src.train --fusion_type vector_gate --all_folds \
        --shuffle_egg --output_dir /path/to/Fusion_Results/vector_gate_shuffled
"""

import argparse
import csv
import itertools
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.fusion.modules import get_fusion
from src.encoders.blocks import AAMSoftmax
import src.common as helper_functions
from src import config

# ==============================================================================
# CLI
# ==============================================================================

parser = argparse.ArgumentParser(description="Train fusion module on cached embeddings.")
parser.add_argument('--fusion_type', type=str, required=True,
                    choices=['mlp', 'scalar_gate', 'vector_gate', 'tied_vector_gate', 'no_trunk_vector_gate', 'untied_scalar_gate', 'no_trunk_scalar_gate', 'no_trunk_tied_vector_gate'])
parser.add_argument('--gate_input_norm', type=str, default='none', choices=['none', 'l2', 'layernorm'])
parser.add_argument('--fold', type=int, default=None, choices=[0, 1, 2, 3, 4])
parser.add_argument('--all_folds', action='store_true')
parser.add_argument('--embedding_dir', type=str, default=str(config.EMBEDDING_DIR))
parser.add_argument('--output_dir', type=str, default=str(config.CHECKPOINT_DIR))

# Sanity check flag
parser.add_argument('--shuffle_egg', action='store_true',
                    help="Permute EGG embeddings across speakers (sanity check).")
parser.add_argument('--normalize', action='store_true',
                    help="Defines if the embedding inputs are to be L2-normalized before entering the fusion module.")

# Architecture
parser.add_argument('--audio_dim', type=int, default=192)
parser.add_argument('--egg_dim', type=int, default=192)
parser.add_argument('--hidden_dim', type=int, default=384)
parser.add_argument('--output_dim', type=int, default=192)
parser.add_argument('--dropout', type=float, default=0.3)

# Training
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--margin', type=float, default=0.2)
parser.add_argument('--scale', type=float, default=30.0)

parser.add_argument('--val_pairs_per_category', type=int, default=2000)
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()

if args.fold is None and not args.all_folds:
    parser.error("Specify either --fold or --all_folds.")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
embedding_dir = Path(args.embedding_dir)

# ==============================================================================
# Constants
# ==============================================================================

FOLD_SEED = 42
NOISE_TYPES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_LEVELS = [-10, -5, 0, 5, 10]
N_CLASSES = 20
ALL_SPEAKERS = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10',
                'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10']
SPEAKER_TO_LABEL = {s: i for i, s in enumerate(ALL_SPEAKERS)}

CONDITION_LABELS = ["clean"]
for noise in NOISE_TYPES:
    for snr in SNR_LEVELS:
        CONDITION_LABELS.append(f"{noise}_{snr}dB")
N_CONDITIONS = len(CONDITION_LABELS)   # 26

print(f"Running: Fusion_type: {args.fusion_type}   |   Gate_input_norm: {args.gate_input_norm}   |   Hidden_dim: {args.hidden_dim}")

# ==============================================================================
# Data
# ==============================================================================

class FusionTrainDataset(Dataset):
    """
    Cached-embedding dataset with random audio condition sampling.

    If shuffle_egg=True, EGG embeddings are permuted across the split BEFORE
    training. The permutation is fixed for the lifetime of the dataset (not
    re-sampled per epoch) so that the same wrong-EGG pairing is used at every
    epoch and during validation -- this is the cleanest control: the model
    learns whatever it can from consistently-wrong EGG.
    """
    def __init__(self, embedding_dir: Path, fold_idx: int, split: str,
                 seed: int, shuffle_egg: bool = False):
        a_clean = np.load(embedding_dir / f"audio_clean_fold{fold_idx}.npz", allow_pickle=True)
        a_noisy = np.load(embedding_dir / f"audio_noisy_fold{fold_idx}.npz", allow_pickle=True)
        e_clean = np.load(embedding_dir / f"egg_clean_fold{fold_idx}.npz",   allow_pickle=True)

        assert np.array_equal(a_clean['utterance_ids'], a_noisy['utterance_ids'])
        assert np.array_equal(a_clean['utterance_ids'], e_clean['utterance_ids'])

        splits = a_clean['split']
        speakers = a_clean['speaker_ids']
        mask = (splits == split)
        self.indices = np.where(mask)[0]
        self.speakers = speakers[mask]
        self.utt_ids = a_clean['utterance_ids'][mask]

        self.a_clean = torch.from_numpy(a_clean['embeddings'][mask]).float()
        self.a_noisy = torch.from_numpy(a_noisy['embeddings'][mask]).float()
        e_clean_t = torch.from_numpy(e_clean['embeddings'][mask]).float()

        # Sanity-check shuffling: permute EGG across utterances.
        # Cross-speaker permutation: reject permutations where EGG remains
        # paired with the same speaker (would invalidate the control).
        self.shuffle_egg = shuffle_egg
        if shuffle_egg:
            rng = np.random.default_rng(seed + 9999)
            N = len(self.speakers)
            # Generate permutations until none of the pairs are same-speaker.
            # In practice this succeeds within a few tries for split sizes here.
            for attempt in range(100):
                perm = rng.permutation(N)
                same_speaker = sum(self.speakers[i] == self.speakers[perm[i]]
                                   for i in range(N))
                if same_speaker == 0:
                    break
            else:
                # Fall back: force-fix collisions deterministically
                perm = rng.permutation(N)
                for i in range(N):
                    if self.speakers[i] == self.speakers[perm[i]]:
                        # swap with next non-colliding index
                        for j in range(i + 1, N):
                            if (self.speakers[j] != self.speakers[i]
                                    and self.speakers[perm[j]] != self.speakers[i]):
                                perm[i], perm[j] = perm[j], perm[i]
                                break
            self.egg_perm = perm
            residual = int(np.sum(self.speakers == self.speakers[self.egg_perm]))
            print(f"  [shuffle_egg] {split} fold{fold_idx}: "f"{residual}/{len(self.speakers)} same-speaker residual")
            self.e_clean = e_clean_t[perm]
            print(f"  [shuffle_egg] {split}: permuted EGG across {N} utterances "
                  f"(same-speaker collisions: 0)")
        else:
            self.egg_perm = None
            self.e_clean = e_clean_t

        self.n_noisy = self.a_noisy.shape[1]
        self.n_conditions = self.n_noisy + 1
        assert self.n_conditions == N_CONDITIONS

        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        cond = self.rng.randrange(self.n_conditions)
        if cond == 0:
            e_audio = self.a_clean[i]
        else:
            e_audio = self.a_noisy[i, cond - 1]
        e_egg = self.e_clean[i]
        label = SPEAKER_TO_LABEL[self.speakers[i]]
        return e_audio, e_egg, label


def build_val_pairs(speakers, pairs_per_category, seed):
    """Stratified pair generation. Identical to evaluation scripts."""
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

    pairs = (
        [(1, 1, i, j) for i, j in same_pairs]
        + [(0, 1, i, j) for i, j in neg(sg)]
        + [(0, 0, i, j) for i, j in neg(cg)]
    )
    return pairs

# ==============================================================================
# Validation
# ==============================================================================

def validate(model, val_dataset, val_pairs, device):
    """Returns mean SG-EER across 26 conditions + per-condition breakdown."""
    model.eval()
    pair_labels = torch.tensor([p[0] for p in val_pairs])
    pair_gm     = torch.tensor([p[1] for p in val_pairs])
    pair_a = torch.tensor([p[2] for p in val_pairs], dtype=torch.long)
    pair_b = torch.tensor([p[3] for p in val_pairs], dtype=torch.long)

    e_egg_all = val_dataset.e_clean.to(device)

    sg_eers, sg_mdcfs, cg_fars = [], [], []

    with torch.no_grad():
        for cond in range(val_dataset.n_conditions):
            if cond == 0:
                e_audio_all = val_dataset.a_clean.to(device)
            else:
                e_audio_all = val_dataset.a_noisy[:, cond - 1, :].to(device)

            if args.normalize:
                e_audio_all = F.normalize(e_audio_all, dim=-1)
                e_egg_all = F.normalize(e_egg_all, dim=-1)

            fused = model(e_audio_all, e_egg_all)
            fused = F.normalize(fused, dim=-1)

            f_a = fused[pair_a].cpu()
            f_b = fused[pair_b].cpu()

            results = helper_functions.get_scores_stratified(
                emb=[(f_a, f_b)],
                true_labels=pair_labels,
                gender_match_labels=pair_gm,
                verbose=False,
            )
            sg_eers.append(results['sg_eer'])
            sg_mdcfs.append(results['sg_min_dcf'])
            cg_fars.append(results['cg_far'])

    model.train()
    return {
        'mean_sg_eer': float(np.mean(sg_eers)),
        'mean_sg_mdcf': float(np.mean(sg_mdcfs)),
        'mean_cg_far': float(np.mean(cg_fars)),
        'sg_eers_per_condition': sg_eers,
        'sg_mdcfs_per_condition': sg_mdcfs,
        'cg_fars_per_condition': cg_fars,
    }

# ==============================================================================
# Per-fold training
# ==============================================================================

def train_fold(fold_idx):
    tag = args.fusion_type + ("_shuffled" if args.shuffle_egg else "") + (f"_{args.gate_input_norm}" if args.gate_input_norm != 'none' else "")
    print(f"\n{'='*72}\nFOLD {fold_idx} | fusion={args.fusion_type} | "
          f"shuffle_egg={args.shuffle_egg}\n{'='*72}")

    seed = FOLD_SEED + fold_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = FusionTrainDataset(
        embedding_dir, fold_idx, "train",
        seed=seed, shuffle_egg=args.shuffle_egg,
    )
    val_ds = FusionTrainDataset(
        embedding_dir, fold_idx, "val",
        seed=seed + 100, shuffle_egg=args.shuffle_egg,
    )
    print(f"Train: {len(train_ds)} utts | Val: {len(val_ds)} utts")

    val_pair_seed = FOLD_SEED + 500 + fold_idx
    val_pairs = build_val_pairs(val_ds.speakers, args.val_pairs_per_category, val_pair_seed)
    print(f"Val pairs: {len(val_pairs)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )

    model = get_fusion(
        name=args.fusion_type,
        audio_dim=args.audio_dim, egg_dim=args.egg_dim,
        hidden_dim=args.hidden_dim, output_dim=args.output_dim,
        dropout=args.dropout, gate_input_norm=args.gate_input_norm,
    ).to(device)

    classifier = AAMSoftmax(
        embed_size=args.output_dim, num_classes=N_CLASSES,
        scale=args.scale, margin=args.margin,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Fusion params: {n_params/1e3:.1f}K | "
          f"AAM params: {sum(p.numel() for p in classifier.parameters())/1e3:.1f}K")

    metrics_history = []
    best_val_eer = float('inf')
    best_min_dcf = float('inf')
    best_epoch = 0
    patience_counter = 0
    ckpt_path = output_dir / f"{tag}_fold_{fold_idx}.pth"

    for epoch in range(args.epochs):
        model.train(); classifier.train()
        train_loss_total = 0.0
        n_batches = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}", ncols=100, mininterval=2)
        for e_audio, e_egg, labels in loop:
            e_audio = e_audio.to(device); e_egg = e_egg.to(device)

            if args.normalize:
                e_audio = F.normalize(e_audio, dim=-1)
                e_egg = F.normalize(e_egg, dim=-1)

            labels = labels.to(device)

            optimizer.zero_grad()
            fused = model(e_audio, e_egg)
            loss = classifier(fused, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            n_batches += 1
            loop.set_postfix_str(f"loss={loss.item():.3f}")
        avg_loss = train_loss_total / max(1, n_batches)

        val_results = validate(model, val_ds, val_pairs, device)
        mean_val_eer = val_results['mean_sg_eer']
        per_cond_eers = val_results['sg_eers_per_condition']
        mean_sg_mindcf = val_results['mean_sg_mdcf']
        mean_cg_far = val_results['mean_cg_far']

        metrics_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "mean_val_eer": mean_val_eer,
            **{f"val_eer_cond_{i}": e for i, e in enumerate(per_cond_eers)},
            "mean_min_dcf": mean_sg_mindcf,
            "mean_cg_far": mean_cg_far,
        })

        clean_eer = per_cond_eers[0]
        print(f"  ep {epoch+1}: loss {avg_loss:.3f} | "
              f"mean val EER {mean_val_eer:.4%} "
              f"(clean {clean_eer:.4%}, min {min(per_cond_eers):.4%}, "
              f"max {max(per_cond_eers):.4%})")

        is_best = (mean_val_eer < best_val_eer
                   or (mean_val_eer == best_val_eer and mean_sg_mindcf < best_min_dcf))
        if is_best:
            best_val_eer = mean_val_eer
            best_min_dcf = mean_sg_mindcf
            best_epoch = epoch + 1
            torch.save({
                "model": model.state_dict(),
                "classifier": classifier.state_dict(),
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_mean_val_eer": best_val_eer,
                "best_mean_min_dcf": best_min_dcf,
                "fusion_type": args.fusion_type,
                "shuffle_egg": args.shuffle_egg,
                "egg_perm_train": train_ds.egg_perm,
                "egg_perm_val": val_ds.egg_perm,
            }, ckpt_path)
            patience_counter = 0
            print(f"    -> new best (saved)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    csv_path = output_dir / f"{tag}_train_metrics_fold_{fold_idx}.csv"
    if metrics_history:
        fieldnames = list(metrics_history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_history)
        print(f"Saved metrics CSV: {csv_path}")

    print(f"Best epoch: {best_epoch} | Best mean val EER: {best_val_eer:.4%}")
    return ckpt_path, best_val_eer

# ==============================================================================
# Main
# ==============================================================================

folds_to_run = [args.fold] if args.fold is not None else [0, 1, 2, 3, 4]
results = []
for f in folds_to_run:
    ckpt, best = train_fold(f)
    results.append((f, ckpt, best))

print(f"\n{'='*72}\nSUMMARY | fusion={args.fusion_type} shuffle_egg={args.shuffle_egg}\n{'='*72}")
for f, ckpt, best in results:
    print(f"  fold {f}: best mean val EER = {best:.4%} | checkpoint: {ckpt}")
