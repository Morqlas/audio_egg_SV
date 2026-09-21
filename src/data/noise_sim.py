"""
Paper artifact: builds the 25 noisy test conditions every reported number is evaluated on.

Pre-generate noisy MIC dataset for all 20 PTDB-TUG speakers.
 
For each MIC file, produces 25 noisy versions (5 noise types × 5 SNR levels).
Output structure mirrors input under OUTPUT_ROOT:
    OUTPUT_ROOT/snr_X/noise_type/{FEMALE,MALE}/MIC/{speaker}/{file}.wav
 
EGG (LAR) files are NOT touched — they remain clean throughout.
Reproducible via SEED.
"""
 
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm
from src import config


 
# ---------- CONFIG ----------
BASE_ROOT = config.DATA_ROOT
SPEECH_ROOT = BASE_ROOT/Path("SPEECH_DATA")
DEMAND_ROOT = BASE_ROOT/Path("DEMAND")
OUTPUT_ROOT = BASE_ROOT/Path("NOISY_MIC")
 
NOISE_TYPES = ["cafeteria", "traffic", "metro", "park", "office"]
SNR_LEVELS_DB = [10, 5, 0, -5, -10]
SR = 16000
SEED = 42
# ----------------------------
 
 
def snr_tag(snr_db: int) -> str:
    """Filesystem-safe SNR folder name: +10 -> snr_p10dB, -5 -> snr_n5dB."""
    sign = "p" if snr_db >= 0 else "n"
    return f"snr_{sign}{abs(snr_db)}dB"
 
 
def find_mic_files(speech_root: Path) -> list[Path]:
    """Find all MIC .wav files for every speaker."""
    files = []
    for gender in ("FEMALE", "MALE"):
        mic_dir = speech_root / gender / "MIC"
        if not mic_dir.exists():
            print(f"WARNING: {mic_dir} does not exist, skipping.")
            continue
        files.extend(mic_dir.rglob("*.wav"))
    return sorted(files)
 
 
def load_noise_files(demand_root: Path, noise_types: list[str]) -> dict[str, np.ndarray]:
    """Load each noise file into memory once."""
    noise = {}
    for nt in noise_types:
        path = demand_root / f"{nt}.wav"
        if not path.exists():
            raise FileNotFoundError(f"Missing DEMAND file: {path}")
        audio, fs = sf.read(str(path), dtype="float32")
        if fs != SR:
            raise ValueError(f"{path}: sample rate {fs} != expected {SR}")
        if audio.ndim > 1:
            audio = audio[:, 0]  # take channel 0 if multichannel
        noise[nt] = audio
        print(f"Loaded {nt}: {len(audio)/fs:.1f} seconds")
    return noise
 
 
def mix_at_snr(speech: np.ndarray, noise: np.ndarray, target_snr_db: float,
               meter: pyln.Meter, rng: np.random.Generator) -> np.ndarray:
    """Mix pre-loaded speech and noise arrays at target SNR."""
    speech_len = len(speech)
    if len(noise) < speech_len:
        noise = np.tile(noise, int(np.ceil(speech_len / len(noise))))
 
    start = rng.integers(0, len(noise) - speech_len + 1)
    segment = noise[start : start + speech_len]
 
    speech_loudness = meter.integrated_loudness(speech)
    noise_loudness = meter.integrated_loudness(segment)
 
    if np.isinf(speech_loudness):
        raise ValueError("Speech has zero loudness")
    if np.isinf(noise_loudness):
        raise ValueError("Noise segment has zero loudness")
 
    gain_db = (speech_loudness - target_snr_db) - noise_loudness
    scale = 10.0 ** (gain_db / 20.0)
    mixed = speech + segment * scale
 
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
 
    return mixed
 
 
def output_path(input_file: Path, noise_type: str, snr_db: int) -> Path:
    rel = input_file.relative_to(SPEECH_ROOT)
    return OUTPUT_ROOT / snr_tag(snr_db) / noise_type / rel
 
 
def main():
    rng = np.random.default_rng(SEED)
    meter = pyln.Meter(SR)
 
    speech_files = find_mic_files(SPEECH_ROOT)
    if not speech_files:
        raise RuntimeError(f"No MIC files found under {SPEECH_ROOT}")
    print(f"Found {len(speech_files)} clean MIC files.")
 
    noise_data = load_noise_files(DEMAND_ROOT, NOISE_TYPES)
 
    total = len(speech_files) * len(NOISE_TYPES) * len(SNR_LEVELS_DB)
    print(f"Generating {total} noisy files...")
 
    failures = []
    with tqdm(total=total) as pbar:
        for speech_file in speech_files:
            try:
                speech, _ = sf.read(str(speech_file), dtype="float32")
                if speech.ndim > 1:
                    speech = speech[:, 0]
            except Exception as e:
                failures.append((speech_file, "-", 0, f"failed to load: {e}"))
                pbar.update(len(NOISE_TYPES) * len(SNR_LEVELS_DB))
                continue
 
            for noise_type, noise in noise_data.items():
                for snr in SNR_LEVELS_DB:
                    out_path = output_path(speech_file, noise_type, snr)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
 
                    if out_path.exists():
                        pbar.update(1)
                        continue
 
                    try:
                        mixed = mix_at_snr(speech, noise, snr, meter, rng)
                        sf.write(str(out_path), mixed, SR, subtype="PCM_16")
                    except Exception as e:
                        failures.append((speech_file, noise_type, snr, str(e)))
 
                    pbar.update(1)
 
    print(f"\nDone. {len(failures)} failures.")
    if failures:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        log = OUTPUT_ROOT / "failures.log"
        with open(log, "w") as f:
            for sp, nt, snr, err in failures:
                f.write(f"{sp}\t{nt}\t{snr}\t{err}\n")
        print(f"Failure details written to {log}")
 
 
if __name__ == "__main__":
    main()