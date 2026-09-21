"""
Figure: EGG signal before vs. after the Kaiser band-pass filter.
Top:    time-domain, ~0.5 s window (shows baseline-drift removal)
Middle: time-domain zoom, ~60 ms (shows glottal-cycle shape preserved)
Bottom: magnitude spectrum, raw vs filtered (shows the band-pass action; cutoffs marked)

Run on a real file:
    python -m src.figures.egg_filter --egg /path/to/lar_xxx.wav --gender male
Or preview on synthetic data:
    python -m src.figures.egg_filter            (no --egg => synthetic)
"""
import argparse
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "mathtext.fontset": "dejavuserif",
})

RAW_C  = "#999999"
FILT_C = "#1f3b73"


def filter_egg_kaiser(egg, sr, gender):
    hp = 15.0 if gender.lower() == "male" else 25.0
    lp = 3000.0
    nyq = 0.5 * sr
    width = 10.0 / nyq
    N, beta = signal.kaiserord(60.0, width)
    if N % 2 == 0:
        N += 1
    taps = signal.firwin(N, [hp / nyq, lp / nyq], pass_zero=False,
                         window=("kaiser", beta), scale=False)
    # zero-phase application of the same taps (equiv. magnitude to filtfilt), via FFT conv
    y = signal.fftconvolve(egg, taps, mode="same")
    y = signal.fftconvolve(y[::-1], taps, mode="same")[::-1]
    return y, hp, lp


def make_synthetic(sr=48000, dur=3.0, f0=120.0, seed=0):
    """Quasi-periodic EGG-like wave + baseline drift + HF static. Illustration only."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    egg = np.zeros_like(t)
    for k, a in enumerate([1.0, 0.6, 0.35, 0.2, 0.12, 0.07], start=1):
        egg += a * np.sin(2 * np.pi * k * f0 * t)
    egg += 0.25 * np.sin(2 * np.pi * 2 * f0 * t + 0.8) ** 3      # sharpen closing
    drift = 0.9 * np.sin(2 * np.pi * 4 * t) + 0.6                 # <5 Hz wander + DC
    hf = 0.15 * np.sin(2 * np.pi * 6500 * t) + 0.1 * rng.standard_normal(t.size)
    return egg + drift + hf, sr


def spectrum(x, sr, nfft=8192):
    f, P = signal.welch(x, sr, nperseg=min(nfft, len(x)))
    P = 10 * np.log10(P + 1e-12)
    return f, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--egg", default=None)
    ap.add_argument("--gender", default="male", choices=["male", "female"])
    ap.add_argument("--out", default="egg_filter.pdf")
    args = ap.parse_args()

    if args.egg:
        import soundfile as sf
        raw, sr = sf.read(args.egg, dtype="float32")
        if raw.ndim > 1:
            raw = raw.mean(axis=1)
        title_tag = ""
    else:
        raw, sr = make_synthetic()
        title_tag = "  (synthetic illustration — replace with real EGG)"

    filt, hp, lp = filter_egg_kaiser(raw, sr, args.gender)

    # pick the most energetic ~0.5 s window (voiced region)
    win = int(0.5 * sr)
    if len(raw) > win:
        energy = np.convolve(filt ** 2, np.ones(win), mode="valid")
        s = int(np.argmax(energy))
    else:
        s = 0
    e = min(len(raw), s + win)
    t = np.arange(s, e) / sr

    zw = int(0.06 * sr)
    zs = s + (e - s) // 2
    ze = min(len(raw), zs + zw)
    tz = np.arange(zs, ze) / sr

    fig, ax = plt.subplots(3, 1, figsize=(6.0, 6.2))

    ax[0].plot(t, raw[s:e], color=RAW_C, lw=0.7, label="raw")
    ax[0].plot(t, filt[s:e], color=FILT_C, lw=0.8, label="filtered")
    ax[0].set_ylabel("amplitude")
    ax[0].set_title("(a) Time domain — baseline-drift removal" + title_tag, loc="left")
    ax[0].legend(frameon=False, ncol=2, loc="upper right")

    ax[1].plot(tz, raw[zs:ze], color=RAW_C, lw=0.8, label="raw")
    ax[1].plot(tz, filt[zs:ze], color=FILT_C, lw=1.0, label="filtered")
    ax[1].set_ylabel("amplitude")
    ax[1].set_xlabel("time (s)")
    ax[1].set_title("(b) Zoom — glottal-cycle shape", loc="left")

    fr, Pr = spectrum(raw, sr)
    ff, Pf = spectrum(filt, sr)
    ax[2].semilogx(fr, Pr, color=RAW_C, lw=0.8, label="raw")
    ax[2].semilogx(ff, Pf, color=FILT_C, lw=1.0, label="filtered")
    for c in (hp, lp):
        ax[2].axvline(c, color="k", ls="--", lw=0.6)
    ax[2].text(hp, ax[2].get_ylim()[1], f" {hp:.0f} Hz", va="top", fontsize=7)
    ax[2].text(lp, ax[2].get_ylim()[1], f" {lp/1000:.0f} kHz", va="top", fontsize=7)
    ax[2].set_xlim(1, sr / 2)
    ax[2].set_ylabel("PSD (dB)")
    ax[2].set_xlabel("frequency (Hz)")
    ax[2].set_title("(c) Magnitude spectrum — band-pass action", loc="left")
    ax[2].legend(frameon=False, loc="lower left")

    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()