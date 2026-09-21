from sklearn.model_selection import StratifiedKFold
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np
from tqdm import tqdm
import os
import torch
import argparse

import src.common as helper_functions
import src.encoders.egg_cnn as helper_classes
from src import config

p = argparse.ArgumentParser(description="Computing embeddings for using frozen encoders in fusion.")
p.add_argument("--output_dir", type=str, default=str(config.EMBEDDING_DIR))
p.add_argument("--folds", type=int, nargs="+", choices=[0, 1, 2, 3, 4], default=[0, 1, 2, 3, 4])
p.add_argument("--state_dict_dir", type=str, default=str(config.CHECKPOINT_DIR / "egg_encoder"))

args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
class Condition:
    rec_type: str # MIC or LAR
    noise_type: str | None  # None for clean
    snr_db: int | None  # None for clean

    def __post_init__(self):
        if self.rec_type not in ("MIC", "LAR"):
            raise ValueError(f"rec_type must be 'MIC' or 'LAR', got {self.rec_type}")
        if self.rec_type == "LAR" and (self.noise_type is not None or self.snr_db is not None):
            raise ValueError(f"LAR (EGG) cannot have noise - keep noise_type and snr_db as 'None'")
        if (self.noise_type is None) != (self.snr_db is None):
            raise ValueError(f"noise_type and snr_db must both be None or both be set") 

    def snr_tag(self) -> str:
        """Encode SNR for path: -5 -> 'snr_n5dB', 5 -> 'snr_5dB'."""
        if self.snr_db < 0:
            return f"snr_n{abs(self.snr_db)}dB"
        return f"snr_{self.snr_db}dB"

    def resolve_path(self, roots: dict[str, Path], relative_path: str) -> Path:
        if self.rec_type == "LAR":
            relative_path = relative_path.replace("mic_", "lar_", 1)
            relative_path = relative_path.replace("MIC", "LAR")
            return roots["audio_clean"] / relative_path
    
        if self.noise_type is None:
            return roots["audio_clean"] / relative_path
    
        return roots["noisy"] / self.snr_tag() / self.noise_type / relative_path

class UtteranceDataset(Dataset):
        def __init__(self, items, condition, roots, name:str):
            super().__init__()
            self.items = items
            self.condition = condition
            self.roots = roots
            self.name = name
        
        def __str__(self):
            return self.name

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            speaker, abs_path = self.items[index]
            rel_path = Path(abs_path).relative_to(self.roots["audio_clean"])
            full_path = self.condition.resolve_path(self.roots, str(rel_path))

            audio, _ = sf.read(full_path)
            audio = torch.tensor(audio, dtype=torch.float32)

            if self.condition.rec_type == "LAR": 
                audio = (audio - audio.mean()) / (audio.std() + 1e-8)
                degg = torch.diff(audio, prepend=audio[0:1])
                degg = (degg - degg.mean()) / (degg.std() + 1e-8)

                return torch.stack([audio, degg], dim=0)

            return audio

def collect_all_paths(data_dir, speaker_list, rec_type):
    """Returns list of (speaker_id, full_path) for every utterance."""
    paths = []
    for gender in ["FEMALE", "MALE"]:
        for root, _, files in os.walk(data_dir / gender / rec_type):
            speaker = os.path.basename(root)
            if speaker in speaker_list:
                for f in files:
                    paths.append((speaker, os.path.join(root, f)))
    paths.sort(key=lambda x: (x[0], x[1]))
    return paths

#all_paths = collect_all_paths(data_dir, ALL_SPEAKERS, REC_TYPE)

def build_fold_splits(all_paths, fold_idx, fold_seed):
    speakers_arr = np.array([p[0] for p in all_paths])
    paths_arr = np.array([p[1] for p in all_paths])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    fold_indices = list(skf.split(paths_arr, speakers_arr))
    test_idx = fold_indices[fold_idx][1]
    remaining = fold_indices[fold_idx][0]
    rng = np.random.default_rng(fold_seed + fold_idx)
    remaining_shuffled = rng.permutation(remaining)
    n_val = len(remaining_shuffled) // 8
    val_idx = remaining_shuffled[:n_val]
    train_idx = remaining_shuffled[n_val:]
    def to_items(idx):
        return [(speakers_arr[i], paths_arr[i]) for i in idx]
    return to_items(train_idx), to_items(val_idx), to_items(test_idx)


def make_utterance_dataset(
        fold_items: list[tuple[str,str]],   # (speaker, abs_path) pairs
        condition: Condition,               # MIC or LAR
        roots: dict[str, Path],             # Roots to audio_clean, audio_noisy and egg 
        name: str,                          # Name to print for debugging
) -> Dataset:
    
    return UtteranceDataset(items=fold_items, condition=condition, roots=roots, name=name)
    


def get_embeddings(
        model: torch.nn,
        dataset: Dataset,
        device: torch.device,
        batch_size: int = 1
) -> np.ndarray:                # Shape (N, emb_dim)
    
    print(f"\nComputing embeddings for {dataset}\n")
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    all_embs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding", mininterval=30):
            batch = batch.to(device)
            emb = model(batch)
            all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)

# === Setup (once) ===
roots = {
    "audio_clean": config.SPEECH_ROOT,
    "noisy": config.NOISY_ROOT,
}

NOISE_TYPES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_LEVELS = [-10, -5, 0, 5, 10]
FOLD_SEED = 42
ALL_SPEAKERS = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10',
                'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10']

data_dir = Path(roots["audio_clean"])
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

audio_model = helper_functions.load_ecapa2(str(config.STATE_DICT_DIR), device=device)  # ECAPA2

all_paths = collect_all_paths(data_dir, ALL_SPEAKERS, "MIC")

# === Per-fold loop ===
for fold_idx in args.folds:
    print(f"Starting fold {fold_idx}\n")
    train_items, val_items, test_items = build_fold_splits(all_paths, fold_idx, FOLD_SEED)
    items = train_items + val_items + test_items
    splits = (['train']*len(train_items)
              + ['val']*len(val_items)
              + ['test']*len(test_items))
    
    speakers = [s for s, _ in items]
    utt_ids  = [str(Path(p).relative_to(roots["audio_clean"])) for _, p in items]
    N = len(items)

    '''
    # --- 1. Audio clean ---
    cond = Condition("MIC", None, None)
    ds = make_utterance_dataset(items, cond, roots, f"MIC_clean_fold_{fold_idx}")
    emb = get_embeddings(audio_model, ds, device, batch_size=1)   # (N, D)
    np.savez(output_dir / f"audio_clean_fold{fold_idx}.npz",
             embeddings=emb, utterance_ids=utt_ids, speaker_ids=speakers, split=splits)
    
    # --- 2. Audio noisy (all 25 conditions stacked) ---
    D = emb.shape[1]
    noisy_emb = np.zeros((N, 25, D), dtype=np.float32)
    noise_axis = []
    snr_axis = []
    cond_idx = 0
    for noise in NOISE_TYPES:
        for snr in SNR_LEVELS:
            cond = Condition("MIC", noise, snr)
            ds = make_utterance_dataset(items, cond, roots, f"MIC_noisy_{noise}_{snr} ")
            noisy_emb[:, cond_idx, :] = get_embeddings(audio_model, ds, device, batch_size=1)
            noise_axis.append(noise)
            snr_axis.append(snr)
            cond_idx += 1
    np.savez(output_dir / f"audio_noisy_fold{fold_idx}.npz",
             embeddings=noisy_emb, utterance_ids=utt_ids, speaker_ids=speakers, split=splits,
             noise_types=noise_axis, snr_levels=snr_axis)
    '''
    egg_state_dict = Path(args.state_dict_dir) /f"EGG_encoder_custom_fold_{fold_idx}.pth"

    egg_model = helper_classes.EGGEncoder(
            in_channels=2,
            embedding_dim=192,
            dropout=0.3,
            win_length=160,
            hop_length=80,
        ).to(device)

    egg_model.load_state_dict(torch.load(egg_state_dict, map_location=device))
    egg_model.eval()
    print(f"Loaded checkpoint: {egg_state_dict.name}")

    # --- 3. EGG clean ---
    cond = Condition("LAR", None, None)
    ds = make_utterance_dataset(items, cond, roots, f"EGG_fold_{fold_idx}")
    emb = get_embeddings(egg_model, ds, device, batch_size=1)     # (N, D_egg)
    np.savez(output_dir / f"egg_clean_fold_new{fold_idx}.npz",
             embeddings=emb, utterance_ids=utt_ids, speaker_ids=speakers, split=splits)
    
print(f"Done.")