# Condition-Adaptive Fusion of Electroglottography and Audio for Speaker Verification Under Extreme Noise

**Authors:** Manuel Deus¹ ², Simon Dahl Jepsen², Jesper Rindom Jensen²
¹Department of Physics, Faculdade de Ciências da Universidade de Lisboa, Lisbon, Portugal
²Department of Electronic Systems, Aalborg University, Aalborg, Denmark

**Venue:** Submitted to ICASSP 2027 — not yet accepted. Update the BibTeX entry
below once a decision lands (accepted → drop `note`, add proceedings
pages/DOI).

## Abstract

Speaker verification degrades under acoustic noise, the setting where it's
often most needed. Electroglottography (EGG), a skin-contact measure of
vocal-fold contact, is immune to airborne noise and complements audio as
conditions worsen, while acoustic-only verification collapses. We fuse a
frozen ECAPA2 audio encoder with a trained EGG encoder through an untied,
no-trunk, per-dimension gate whose weights are a direct, inspectable readout
of the joint embedding. On PTDB-TUG (20 speakers, 26 conditions), the gate
reaches near-parity with an unconstrained MLP system at 1/3 fewer parameters.
Its value is interpretability: three converging controls show the adaptive
weighting is genuine — content-driven, identity-grounded, and not a norm
artifact. We present this as a closed-set proof of concept.

**Index Terms:** Speaker verification, electroglottography, multimodal
fusion, noise robustness, interpretability

```bibtex
@inproceedings{deus2027eggfusion,
  author    = {Deus, Manuel and Jepsen, Simon Dahl and Jensen, Jesper Rindom},
  title     = {Condition-Adaptive Fusion of Electroglottography and Audio for
               Speaker Verification Under Extreme Noise},
  booktitle = {Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)},
  year      = {2027},
  note      = {Submitted, under review}
}
```

---

## What this is

Speaker verification under acoustic noise, fusing a **frozen ECAPA2** audio
embedding with a **from-scratch EGG (electroglottograph) encoder** embedding.
The fusion module is an **untied, no-trunk, per-dimension gate**: two
independent sigmoid gate vectors, one per modality, applied element-wise to
un-normalised embeddings.

```
g_a  = sigmoid(W_a · [e_audio ; e_egg])         in (0,1)^192
g_e  = sigmoid(W_e · [e_audio ; e_egg])         in (0,1)^192
fused = g_a ⊙ e_audio  +  g_e ⊙ e_egg
```

The contribution is **interpretability**, not a raw accuracy win — the MLP
baseline is in fact slightly more accurate (0.30% vs 0.44% mean SG-EER). The
claim is that the gate's behaviour is *legible*: it demonstrably down-weights
audio as SNR falls, and it does so by reading content rather than magnitude,
and it depends on genuine EGG speaker identity. Three controls establish this.

## Headline numbers

Mean SG-EER over the 26 test conditions (clean + 5 noise types x 5 SNRs),
5-fold cross-validation. Regenerate with `python -m src.figures.table1_main_results`.

| System | mean SG-EER | minDCF | fusion params |
|---|---:|---:|---:|
| Audio only (ECAPA2, frozen) | 9.09%¹ | — | — |
| Late fusion (score, α=0.5) | 1.01% | — | — |
| MLP fusion | **0.30%** | 0.013 | 222,528 |
| Gate + trunk (ablation) | 0.50% | 0.031 | 221,952 |
| Gate, tied (ablation) | 0.69% | 0.053 | 73,920 |
| Gate, scalar (ablation) | 3.56% | 0.310 | 770 |
| **Gate (primary)** | **0.44%** | 0.033 | 147,840 |

¹ audio-only is averaged over the 25 noisy conditions; its clean result is in
`results/unimodal_audio/clean_summary.csv`. Every other row is a 26-condition mean.

## The three interpretability controls

| Control | Question | Result |
|---|---|---|
| **Gate-vs-norm decomposition** | Is the falling audio share the *gate* acting, or just the audio embedding norm shrinking? | Gate-driven share **81–96%** across noise types — the gate, not a passive norm artifact |
| **L2-normalised-input retrain** | Does the gate decide on embedding *magnitude* or *content*? | Retrained with L2-normalised gate input: **0.79%** mean SG-EER, gate still tracks SNR → content |
| **Cross-speaker shuffled EGG** | Does the gate use genuine EGG *identity*, or just an EGG-shaped signal? | Shuffled EGG degrades to **4.88%** mean; real beats shuffled in **10/10** low-SNR cells |

Supporting statistic: the raw gate's SNR trend across the five noise types is
significant at **Wilcoxon p = 0.031** (n=5). No significance test is run on the
effective share itself — it is a confounded (gate × norm) quantity, which is
exactly what the decomposition resolves.

## Layout

```
configs/      default.yaml (hyperparams, seeds, split), paths.yaml (machine paths)
src/
  config.py           path + seed resolution (flag > env > yaml > repo default)
  encoders/           egg_cnn.py (from-scratch EGG), ecapa2*.py (frozen audio), blocks.py
  fusion/             modules.py (gate + ablations + MLP), score_level.py (late fusion)
  data/               noise_sim.py (DEMAND mixing), pairs.py (folds, pairs, embedding cache)
  train.py            shared training recipe for every fusion variant
  evaluate.py         SG-EER + minDCF over the 26 conditions
  controls/           decomposition.py, norm_stats.py, shuffled_egg_ttest.py, aggregate_eff_share.py
  figures/            table1_main_results.py + figure scripts
results/      final aggregated per-condition CSVs (committed)
data/         empty — see data/README.md
```

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

Then follow [data/README.md](data/README.md) to obtain PTDB-TUG, DEMAND and the
ECAPA2 weights, and point `configs/paths.yaml` at them.

All commands are run as modules from the repo root, e.g. `python -m src.train`.

## Reproducing

### From the committed results (no data, no GPU)

Every number in the table above and all three controls regenerate from the
CSVs in `results/`:

```bash
python -m src.figures.table1_main_results          # Table 1
python -m src.controls.decomposition               # decomposition: 81-96%
python -m src.controls.aggregate_eff_share         # effective-share + audio-norm-vs-SNR figure
python -m src.controls.shuffled_egg_ttest          # identity control: 10/10
python -m src.figures.shuffled_egg_control         # identity-control figure
python -m src.figures.fusion_results --gate results/gate_primary/per_condition.csv \
    --mlp results/mlp/per_condition.csv \
    --audio results/unimodal_audio/per_condition.csv \
    --egg results/unimodal_egg/test_summary.csv \
    --late "results/score_level_late_fusion/late_fusion_aggregated_audio_weight_0.5.csv"
python -m src.figures.noise_type_effect --system results/gate_primary/per_condition.csv
python -m src.figures.late_fusion_sweep
```

Most figures are written to `figures/` (gitignored). `decomposition.py` and
`aggregate_eff_share.py` write theirs alongside their CSVs in
`results/control_decomposition/` instead (also gitignored — only the CSVs
there are committed).

### From raw data (full pipeline)

```bash
# 1. build the 25 noisy conditions (seed 42, loudness-normalised SNR)
python -m src.data.noise_sim

# 2. cache frozen audio + EGG embeddings for all 5 folds
python -m src.data.pairs --folds 0 1 2 3 4

# 3. train each fusion variant (repeat per --fusion_type)
python -m src.train --fusion_type no_trunk_vector_gate --all_folds     # primary
python -m src.train --fusion_type mlp --all_folds                      # baseline
python -m src.train --fusion_type vector_gate --all_folds              # +trunk ablation
python -m src.train --fusion_type no_trunk_tied_vector_gate --all_folds
python -m src.train --fusion_type no_trunk_scalar_gate --all_folds

# 4. controls are the same recipe with one flag changed
python -m src.train --fusion_type no_trunk_vector_gate --all_folds --gate_input_norm l2
python -m src.train --fusion_type no_trunk_vector_gate --all_folds --shuffle_egg

# 5. evaluate (mirror the flags used at training time)
python -m src.evaluate --fusion_type no_trunk_vector_gate --all_folds

# 6. gate/norm statistics feeding the decomposition control
python -m src.controls.norm_stats --fusion_type no_trunk_vector_gate \
    --gate_input_norm none --fold 0 --split test
```

Step 3 requires the trained EGG encoder weights, which are **not** published —
see data/README.md §5.

> The L2 and shuffled-EGG controls are **flags on the shared recipe**, not
> separate scripts. Train and eval must use the same flags. The shuffled-EGG
> model is a **diagnostic control, never a system** — do not report its EER as
> a result of the proposed method.

## What is NOT here, and why

- **PTDB-TUG and DEMAND data** — licensed, non-redistributable. Linked in
  `data/README.md` instead.
- **ECAPA2 weights** — third party (Jenthe/ECAPA2 on HuggingFace). Linked, not vendored.
- **Trained EGG-encoder weights** — not published; would suit a GitHub release.
- **Cached embeddings, checkpoints, large arrays** — gitignored under
  `artifacts/`; they regenerate from steps 1–3.
- **The open-set / data-scarcity study** — deliberately cut from the paper.
  Its code is not here, and that is intentional.
- **Superseded result sets** — the source project carried two conflicting sets.
  Only the final one is here. The stale set lived under `Logs/Closed-set/Old/`
  and in two files suffixed `_WRONG` in the original project; none of it was copied.
- **Exploratory notebooks and dead experiments** — not reproduction-relevant.

## Protocol notes

- **Splits.** 5-fold `StratifiedKFold(shuffle=True, random_state=42)` stratified
  by speaker. Per fold: held-out fifth = test; of the remainder, 1/8 = val
  (`default_rng(42 + fold)`), rest = train. Closed-set — every speaker appears
  in all three splits.
- **Pairing.** Noisy conditions use **clean-enroll / noisy-test**. Test pair pool
  seed is `42 + 1000 + fold`, distinct from the val seed so pairs cannot overlap.
- **Noise.** Loudness-normalised (ITU-R BS.1770 via `pyloudnorm`) rather than
  plain RMS SNR. EGG/LAR is never noised.
- **Framing.** Audio uses 25 ms / 10 ms; the EGG encoder uses 10 ms / 5 ms. Both
  encoder builders are kept (`ecapa2.py`, `ecapa2_egg.py`) — they are not duplicates.
- **Metrics.** SG-EER and SG-minDCF are same-gender trials; CG-FAR is the
  cross-gender false-accept rate, reported alongside.
