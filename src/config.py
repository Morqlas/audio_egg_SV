"""Central path + hyperparameter config. Replaces the machine-specific paths
that were hardcoded in the original scripts.

Resolution order for every path (first hit wins):
    1. the script's own --flag
    2. the matching environment variable (e.g. EGG_EMBEDDING_DIR)
    3. configs/paths.yaml, if present
    4. the repo-relative default below

Nothing here changes what any script computes; it only decides where files
are read from and written to.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_YAML: dict = {}
_yaml_path = REPO_ROOT / "configs" / "paths.yaml"
if _yaml_path.exists():
    try:
        import yaml
        _YAML = yaml.safe_load(_yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:  # missing PyYAML or malformed file -> fall through to defaults
        _YAML = {}


def _resolve(key: str, default: Path) -> Path:
    env = os.environ.get(f"EGG_{key.upper()}")
    if env:
        return Path(env)
    if key in _YAML and _YAML[key]:
        return Path(str(_YAML[key]))
    return default


# --- data (not redistributable; see data/README.md) -------------------------
DATA_ROOT     = _resolve("data_root",     REPO_ROOT / "data")
SPEECH_ROOT   = _resolve("speech_root",   DATA_ROOT / "PTDB-TUG")
NOISE_ROOT    = _resolve("noise_root",    DATA_ROOT / "DEMAND")
NOISY_ROOT    = _resolve("noisy_root",    DATA_ROOT / "NOISY_DATA")

# --- third-party weights (not vendored; see README) -------------------------
STATE_DICT_DIR = _resolve("state_dict_dir", DATA_ROOT / "StateDicts")

# --- generated artifacts (gitignored; regenerate from scripts) --------------
EMBEDDING_DIR  = _resolve("embedding_dir",  REPO_ROOT / "artifacts" / "embeddings")
CHECKPOINT_DIR = _resolve("checkpoint_dir", REPO_ROOT / "artifacts" / "checkpoints")

# --- committed outputs ------------------------------------------------------
RESULTS_DIR = _resolve("results_dir", REPO_ROOT / "results")
FIGURES_DIR = _resolve("figures_dir", REPO_ROOT / "figures")

# --- 5-fold split + training seeds (unchanged from the paper runs) ----------
FOLD_SEED = 42          # StratifiedKFold(n_splits=5, shuffle=True, random_state=FOLD_SEED)
N_FOLDS = 5
TEST_PAIR_SEED_BASE = 1000   # test pair pool uses FOLD_SEED + 1000 + fold_idx

NOISE_TYPES = ["cafeteria", "metro", "office", "park", "traffic"]
SNR_LEVELS = [-10, -5, 0, 5, 10]
