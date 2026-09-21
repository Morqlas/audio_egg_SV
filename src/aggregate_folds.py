"""
Aggregate per-fold training metrics into a single summary.

For each fold's training_metrics_*.csv:
  - Identifies the best epoch (lowest val_sg_eer; tie-break on val_sg_mindcf).
  - Extracts metrics at that epoch.

Outputs:
  - aggregated_per_fold.csv: one row per fold, metrics at best epoch.
  - aggregated_summary.csv: mean and std across folds for each metric.
  - Pretty-printed console summary.

Usage:
    python -m src.aggregate_folds --input_dir /path/to/CV_Results_Custom/EGG
    python -m src.aggregate_folds --input_dir ./results --pattern "training_metrics_custom_fold_*.csv"
"""
import pandas as pd
import argparse
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from src import config



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=str(config.RESULTS_DIR),
                   help="Directory containing per-fold training_metrics CSV files.")
    p.add_argument("--audio_weight", type=float, default=0.5,
                   help="Audio modality weight in Late Fusion.")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: same as input_dir).")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    #files = sorted(glob.glob(str(input_dir / f"*_aggregated_*{args.audio_weight}.csv")))
    #files = sorted(Path(input_dir / f"**/*_aggregated_*{args.audio_weight}.csv"))

    # 1. Use pure Pathlib globbing instead of os/glob mixed paradigms
    # 2. Check if the directory actually exists first
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Use input_dir.glob() to search relative to the base path
    pattern = f"*_results_*{args.audio_weight}.csv"
    files = sorted(input_dir.glob(pattern))

    # Debugging check
    if not files:
        print(f"DEBUG: No files found matching pattern: {pattern}")
        print(f"DEBUG: Files currently in directory:")
        for f in list(input_dir.iterdir())[:5]: # Print first 5 files to inspect naming
            print(f"  - {f.name}")

    print(f"Found {len(files)} fold files in {input_dir}:")
    for f in files:
        print(f)

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    metric_cols = ['sg_eer', 'sg_min_dcf', 'sg_threshold', 'cg_far', 'overall_eer', 'overall_min_dcf', 'overall_threshold']    
    agg_dict = {col: ['mean', 'std'] for col in metric_cols}
    agg_dict['n_pairs'] = 'first'

    result = df.groupby(['noise_type', 'snr_db'], dropna=False).agg(agg_dict)

    result.columns = [f"{metric}_{agg}" for metric, agg in result.columns]
   
    result.rename({"n_pairs_first":"n_pairs"}, inplace=True, axis=1)
    result.reset_index(inplace=True)

    result.to_csv(output_dir / f"late_fusion_aggregated_audio_weight_{args.audio_weight}.csv", index=False)

    result = result[result['noise_type'] != 'clean']
    
    markers = {
        'cafeteria': 'o',
        'metro': 's',
        'office': '^',
        'park': 'D',
        'traffic': 'v',
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for noise_type, group in result.groupby('noise_type'):
        ax.errorbar(group['snr_db'], group['sg_eer_mean'] * 100,
                    yerr=group['sg_eer_std'] * 100,
                    label=noise_type, marker=markers[noise_type], capsize=3)

   
    ticks = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 40]
    
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xlabel("Acoustic SNR level (dB)")
    ax.set_yscale('log')
    #ax.set_ylim((0.1,5))
    ax.set_ylabel("EER (%)")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))  

    egg_eer_mean = 2.45  # <--- INSERT YOUR EGG EER MEAN (%) HERE
    egg_eer_std = 0.83  # <--- INSERT YOUR EGG EER STD DEV (%) HERE
    
    # 1. Draw the shaded error region (Std Dev)
    ax.axhspan(ymin=egg_eer_mean - egg_eer_std, 
               ymax=egg_eer_mean + egg_eer_std, 
               color='gray', alpha=0.3, zorder=1, label='EGG Std Dev')
               
    # 2. Draw the mean dashed line on top of it
    ax.axhline(y=egg_eer_mean, color='black', linestyle='--', 
               linewidth=1.5, zorder=2, label='EGG Baseline')

    ax.legend(title="Noise Type")
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.15)

    plt.tight_layout
    plt.savefig(output_dir/f"late_fusion_audio_weight_{args.audio_weight}.png", dpi=300)
    print("Done.")

if __name__ == "__main__":
    main()