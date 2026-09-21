# This is a helper function file to define the building blocks of the ECAPA2 model. It is not meant to be run on its own, but rather imported into the ECAPA2.py file.

import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pyloudnorm as pyln
import soundfile as sf

from src.encoders.blocks import Net
import src.encoders.ecapa2
from src.encoders.ecapa2_egg import build_ecapa2_from_state_dict
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']
from src.metrics import EER, minDCF
from sklearn.metrics import roc_curve


np.random.seed(42)

### Loading the ecapa2 model ###

def load_ecapa2_egg(state_dict_path: str | Path, device: torch.device | None = None, function : str | None = "test", state_dict_name = "ORIGINAL_ecapa2_state_dict.pth") -> Net:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict_path = Path(state_dict_path)

    state_dict = torch.load(state_dict_path / state_dict_name, map_location=device)

    #local_model_path = state_dict_path / "ecapa2.pt"
    '''
    if local_model_path.exists():
        print(f"Loading model from local path: {local_model_path}")
        model_file = local_model_path
    else:
        print("Model not found locally. Downloading from Hugging Face...")
        model_file = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
    '''
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('pre_processing')}

    model = build_ecapa2_from_state_dict(state_dict)
    model.to(device)

    if function == "test":
        model.eval()

    # Free GPU memory from the scripted model
    #del scripted
    torch.cuda.empty_cache()

    return model

def load_ecapa2(state_dict_path: str | Path, device: torch.device | None = None, function : str | None = "test", state_dict_name = "ORIGINAL_ecapa2_state_dict.pth") -> Net:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict_path = Path(state_dict_path)

    state_dict = torch.load(state_dict_path / state_dict_name, map_location=device)

    #local_model_path = state_dict_path / "ecapa2.pt"
    '''
    if local_model_path.exists():
        print(f"Loading model from local path: {local_model_path}")
        model_file = local_model_path
    else:
        print("Model not found locally. Downloading from Hugging Face...")
        model_file = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
    '''
    #scripted = torch.jit.load(model_file, map_location="cpu")
    model = src.encoders.ecapa2.build_ecapa2_from_state_dict(state_dict)
    model.to(device)
    
    if function == "test":
        model.eval()

    # Free GPU memory from the scripted model
    #del scripted
    torch.cuda.empty_cache()

    return model

### Computing the embeddings from the ECAPA2 model ###

def get_embeddings_from_ecapa2(model, loader, device, test=False):
    emb = []
    true_labels = []
    gender_match_labels = []
    pbar = tqdm(loader, desc=f"Evaluating {loader.dataset}", mininterval=30)

    print(f"Starting inference on {len(loader.dataset)} pairs \n")
        
    with torch.no_grad(): #torch.autocast(device_type="cuda", dtype=torch.float16):
        for audio_1, audio_2, label, gender_match in pbar:
            out_1 = model(audio_1.to(device)).cpu()
            out_2 = model(audio_2.to(device)).cpu()
            emb.append((out_1, out_2))
            true_labels.extend(label.tolist())
            gender_match_labels.extend(gender_match.tolist())
    
    return emb, torch.tensor(true_labels), torch.tensor(gender_match_labels)


def get_scores_stratified(emb, true_labels, gender_match_labels, verbose=True):
    """
    Compute EER stratified by gender pairing.
    
    Args:
        emb: list of (embedding_1, embedding_2) tuples
        true_labels: tensor of 0/1 (same-speaker labels)
        gender_match_labels: tensor of 0/1 (1 = same gender, 0 = cross gender)
    
    Returns:
        dict with sg_eer, sg_threshold, sg_min_dcf, cg_far, overall_eer, cos_sim_list
    """
    cos_sim_list = []
    
    total_pairs = sum(e[0].shape[0] for e in emb)

    if verbose:
        print(f"Computing cosine similarity on {total_pairs} pairs...")
    
    for embedding_pair in emb:
        cos_similar = torch.nn.functional.cosine_similarity(embedding_pair[0], embedding_pair[1])
        cos_sim_list.extend(cos_similar.tolist())
    
    cos_sim_tensor = torch.tensor(cos_sim_list, dtype=torch.float32)
    
    # --- Same-gender EER (primary metric) ---
    sg_mask = gender_match_labels == 1
    sg_scores = cos_sim_tensor[sg_mask]
    sg_labels = true_labels[sg_mask]
    sg_pos = sg_scores[sg_labels == 1]
    sg_neg = sg_scores[sg_labels == 0]
    
    sg_eer, sg_threshold = EER(sg_pos, sg_neg)
    sg_min_dcf, _ = minDCF(sg_pos, sg_neg, p_target=0.01)
    
    # --- Cross-gender FAR at same-gender threshold ---
    cg_mask = gender_match_labels == 0
    cg_scores = cos_sim_tensor[cg_mask]
    # All cross-gender pairs are negatives by construction
    cg_far = float((cg_scores > sg_threshold).float().mean()) # Uses the same gender EER threshold, not the minDCF one.
    
    # --- Overall EER (reference only, not headline) ---
    all_pos = cos_sim_tensor[true_labels == 1]
    all_neg = cos_sim_tensor[true_labels == 0]
    overall_eer, overall_threshold = EER(all_pos, all_neg)
    overall_min_dcf, _ = minDCF(all_pos, all_neg, p_target=0.01)
    
    if verbose:
        print(f"Same-gender EER (%)       = {sg_eer:.4%}")
        print(f"Same-gender minDCF        = {sg_min_dcf:.4f}")
        print(f"Cross-gender FAR (%)      = {cg_far:.4%}")
        print(f"Overall EER (reference)   = {overall_eer:.4%}")
    
    return {
        'sg_eer': sg_eer,
        'sg_threshold': sg_threshold,
        'sg_min_dcf': sg_min_dcf,
        'cg_far': cg_far,
        'overall_eer': overall_eer,
        'overall_min_dcf': overall_min_dcf,
        'overall_threshold': overall_threshold,
        'cos_sim_list': cos_sim_list,
    }


def training_loop(model, classifier, train_loader, val_loader, train_eval_loader, loss_function, optimizer, epochs, device, fold, output_path):
    val_sg_eers = []
    val_sg_mindcfs = []
    val_sg_thresholds = []
    val_cg_fars = []
    val_overall_eer = []
    train_sg_eers = []
    train_sg_mindcfs = []
    train_sg_thresholds = []
    train_cg_fars = []
    train_losses = []
    train_overall_eer = []


    best_sg_eer = float('inf')
    best_sg_mindcf = float('inf')
    best_epoch = 0
    best_cos_sim = []
    patience = 10
    patience_counter = 0
    print(f"Beginning training on {train_loader.dataset}")

    #Checking the memory available
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Memory allocated before training: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    scaler = torch.amp.GradScaler()

    output_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):

        train_loss = 0.0

        #Wrapping loader with tqdm for progress tracking
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", mininterval=30, ncols=100)
        model.train()
        for audio, labels in loop:         
            #Zeroing the optimizer so it doesnt start with previous values (meaningful for batches != 1)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
            #Forward pass
                embeddings = model(audio.to(device))
                labels = labels.to(device)
                output = classifier(embeddings, labels)
                loss = loss_function(output, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix_str(s=f"Batch training loss = {loss.item():.3f}")

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        #Validation phase
        model.eval()
        print(f"Validation phase, epoch {epoch+1}")
        embeddings, labels, gender_match = get_embeddings_from_ecapa2(model, val_loader, device)
        if epoch == 0:
            saved_labels = labels
            saved_gender_match = gender_match
        val_results = get_scores_stratified(embeddings, labels, gender_match) 

        val_sg_eers.append(val_results['sg_eer'])
        val_sg_mindcfs.append(val_results['sg_min_dcf'])
        val_sg_thresholds.append(val_results['sg_threshold'])
        val_cg_fars.append(val_results['cg_far'])
        val_overall_eer.append(val_results['overall_eer'])

        #Validation on training speakers
        print(f"Validation on train speakers phase, epoch {epoch+1}")

        embeddings_t, labels_t, gender_match_t = get_embeddings_from_ecapa2(model, train_eval_loader, device)
        train_results = get_scores_stratified(embeddings_t, labels_t, gender_match_t)

        train_sg_eers.append(train_results['sg_eer'])
        train_sg_mindcfs.append(train_results['sg_min_dcf'])
        train_sg_thresholds.append(train_results['sg_threshold'])
        train_cg_fars.append(train_results['cg_far'])
        train_overall_eer.append(train_results['overall_eer'])

        print("")
        print(
            f"Epoch {epoch+1} | "
            f"Loss: {avg_train_loss:.3f} | "
            f"Train SG-EER: {train_results['sg_eer']:.3%} | "
            f"Val SG-EER: {val_results['sg_eer']:.3%} | "
            f"Val CG-FAR: {val_results['cg_far']:.3%} | "
            f"Val SG-minDCF: {val_results['sg_min_dcf']:.3f}"
            )
        print("")

        # Checkpoint + early stopping
        if val_results['sg_eer'] < best_sg_eer or (val_results['sg_eer'] == best_sg_eer and val_results['sg_min_dcf'] < best_sg_mindcf):

            best_sg_eer = val_results['sg_eer']
            best_sg_mindcf = val_results['sg_min_dcf']
            best_epoch = epoch + 1
            best_cos_sim = val_results['cos_sim_list']
            
            torch.save(model.state_dict(), output_path/f"EGG_encoder_best_fold_{fold}.pth")
            torch.save(classifier.state_dict(), output_path/f"EGG_classifier_best_fold_{fold}.pth")
            print(f"New best at epoch {best_epoch}: val SG-EER {best_sg_eer:.3%}\n")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs)")
                break

    print(f"Best epoch = {best_epoch} | Best SG-EER = {best_sg_eer:.3%} | Best SG-minDCF = {best_sg_mindcf:.4f}")

    model.load_state_dict(torch.load(output_path/f"EGG_encoder_best_fold_{fold}.pth"))
    
    return {
    'model': model,
    'train_losses': train_losses,
    'val_sg_eers': val_sg_eers,
    'val_sg_mindcfs': val_sg_mindcfs,
    'val_sg_thresholds': val_sg_thresholds,
    'val_cg_fars': val_cg_fars,
    'val_overall_eer': val_overall_eer,
    'train_sg_eers': train_sg_eers,
    'train_sg_mindcfs': train_sg_mindcfs,
    'train_sg_thresholds': train_sg_thresholds,
    'train_cg_fars': train_cg_fars,
    'train_overall_eer': train_overall_eer,
    'best_sg_eer': best_sg_eer,
    'best_sg_mindcf': best_sg_mindcf,
    'best_epoch': best_epoch,
    'cos_sim_list': best_cos_sim,
    'labels': saved_labels,
    'gender_match_labels': saved_gender_match
}


def plot_gender_score_distributions(test_dataset, cos_sim_list, true_labels,
                                     fig_name: str, threshold: float = None):
    # Use the renamed attribute. Fallback for backwards-compatibility.
    pairs = getattr(test_dataset, 'pairs', None) or getattr(test_dataset, 'paths_list', None)
    if pairs is None:
        raise AttributeError("Dataset has neither 'pairs' nor 'paths_list' attribute.")

    male_indices, female_indices, cross_indices = [], [], []
    for i, (label, gender_match, path1, path2) in enumerate(pairs):
        if gender_match == 0:
            cross_indices.append(i)
        else:
            spk = os.path.basename(os.path.dirname(path1))
            (male_indices if spk.startswith('M') else female_indices).append(i)

    cos_tensor = torch.tensor(cos_sim_list)
    labels_tensor = true_labels

    same_gender_idx = torch.tensor(male_indices + female_indices, dtype=torch.long)
    diff_gender_idx = torch.tensor(cross_indices, dtype=torch.long)

    genuine         = cos_tensor[labels_tensor == 1]
    impostor_same_g = cos_tensor[same_gender_idx][labels_tensor[same_gender_idx] == 0]
    impostor_diff_g = cos_tensor[diff_gender_idx][labels_tensor[diff_gender_idx] == 0]

    plt.figure(figsize=(10, 6))
    plt.hist(genuine.numpy(),         bins=50, alpha=0.5, color="tab:green",
             label=f'Genuine (n={len(genuine)})')
    plt.hist(impostor_same_g.numpy(), bins=50, alpha=0.5, color="tab:orange",
             label=f'Impostor same-gender (n={len(impostor_same_g)})')
    plt.hist(impostor_diff_g.numpy(), bins=50, alpha=0.5, color="tab:blue",
             label=f'Impostor diff-gender (n={len(impostor_diff_g)})')
    if threshold is not None:
        plt.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                    label=f'SG-EER threshold = {threshold:.2f}')
    plt.xlabel('Cosine similarity')
    plt.ylabel('Counts')
    plt.title('EGG: Score distributions by gender pairing')
    plt.legend()
    plt.savefig(f'{fig_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    for name, indices in [("Male-Male", male_indices),
                           ("Female-Female", female_indices),
                           ("Cross-Gender", cross_indices)]:
        if not indices:
            continue
        idx = torch.tensor(indices, dtype=torch.long)
        scores = cos_tensor[idx]
        lbls = labels_tensor[idx]
        pos = scores[lbls == 1]
        neg = scores[lbls == 0]
        if len(pos) > 0 and len(neg) > 0:
            eer, thr = EER(pos, neg)
            print(f"{name}: EER = {eer:.4%} ({len(pos)} pos, {len(neg)} neg)")


def plot_det_eer_curves(cos_sim_list, true_labels, eer, min_dcf, gender_match_labels, fig_name:str, gender:bool = True):

    """Plot DET curve for same-gender pairs only."""
    cos_sim_arr = np.array(cos_sim_list)
    labels_arr = np.array(true_labels)
    gm_arr = np.array(gender_match_labels)
    if gender:
        sg_mask = gm_arr == 1

        fpr, tpr, thresholds = roc_curve(y_true=labels_arr[sg_mask], y_score=cos_sim_arr[sg_mask])
    else:
        fpr, tpr, thresholds = roc_curve(y_true=labels_arr, y_score=cos_sim_arr)
    frr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - frr))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # DET Curve
    ax1.plot(fpr, frr, label=f'DET Curve (EER: {eer:.2%})', color='blue', lw=2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.scatter(fpr[eer_idx], frr[eer_idx], color='black', zorder=5, label='EER Point')
    ax1.plot([1e-4, 1], [1e-4, 1], color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('False Acceptance Rate (FAR)')
    ax1.set_ylabel('False Rejection Rate (FRR)')
    ax1.set_title('DET Curve (Log-Log)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Error vs Threshold
    ax2.plot(thresholds[1:], fpr[1:], label='FAR (False Acceptance)', color='green')
    ax2.plot(thresholds[1:], frr[1:], label='FRR (False Rejection)', color='orange')
    ax2.axvline(thresholds[eer_idx], color='black', linestyle=':', label=f'EER Threshold: {thresholds[eer_idx]:.2f}')
    ax2.set_xlabel('Decision Threshold (Cosine Similarity)')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('Error Rates vs. Threshold')
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([0, 0.5])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{fig_name}.png', dpi=300)
    print(f"Results saved. EER: {eer:.4%}, minDCF: {min_dcf:.4f}")

'''
def plot_score_distributions(cos_sim_list, true_labels, fig_name:str):
    cos_sim = np.array(cos_sim_list)
    labels = np.array(true_labels)
    
    positive_scores = cos_sim[labels == 1]
    negative_scores = cos_sim[labels == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(positive_scores, bins=100, alpha=0.6, label=f'Same speaker (n={len(positive_scores)})', color='green')
    plt.hist(negative_scores, bins=100, alpha=0.6, label=f'Different speaker (n={len(negative_scores)})', color='red')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{fig_name}.png', dpi=300)
'''

def mix_at_snr(speech_path, noise_path, target_snr_db, sr=16000, rng=None):
    """
    Mix a speech file with a noise file at a target SNR.
 
    SNR is measured via ITU-R BS.1770-4 integrated loudness (pyloudnorm).
    Note: this is not ITU-T P.56 active speech level, and for highly
    non-stationary noise the instantaneous SNR may deviate from the target.
 
    Parameters
    ----------
    speech_path : str
        Path to clean speech .wav file.
    noise_path : str
        Path to noise .wav file (must match `sr`).
    target_snr_db : float
        Desired SNR in dB.
    sr : int
        Expected sampling rate of both files. Raises if noise file differs.
    rng : np.random.Generator, optional
        Random generator for the noise slice offset. Pass a seeded generator
        for reproducibility (recommended when pre-generating eval sets).
 
    Returns
    -------
    np.ndarray
        1D mixed signal, same length as the speech file.
 
    Raises
    ------
    ValueError
        If sample rates mismatch, or either signal has zero loudness
        (which would make SNR undefined).
    """
    if rng is None:
        rng = np.random.default_rng()
 
    # Load
    speech, fs_speech = sf.read(speech_path, dtype='float32')
    noise, fs_noise = sf.read(noise_path, dtype='float32')
 
    if fs_speech != sr:
        raise ValueError(f"Speech sample rate {fs_speech}Hz != expected {sr}Hz")
    if fs_noise != sr:
        raise ValueError(f"Noise sample rate {fs_noise}Hz != expected {sr}Hz")
 
    # Stochastic noise slice (loop if shorter than speech)
    speech_len = len(speech)
    if len(noise) < speech_len:
        repeats = int(np.ceil(speech_len / len(noise)))
        noise = np.tile(noise, repeats)
 
    max_start = len(noise) - speech_len
    start_idx = rng.integers(0, max_start + 1)
    noise_segment = noise[start_idx : start_idx + speech_len]
 
    # BS.1770 integrated loudness
    meter = pyln.Meter(sr)
    speech_loudness = meter.integrated_loudness(speech)
    noise_loudness = meter.integrated_loudness(noise_segment)
 
    if np.isinf(speech_loudness):
        raise ValueError(f"Speech file {speech_path} has zero loudness; cannot set SNR.")
    if np.isinf(noise_loudness):
        raise ValueError(f"Noise segment from {noise_path} has zero loudness; cannot set SNR.")
 
    # Scale noise to hit target SNR
    target_noise_loudness = speech_loudness - target_snr_db
    gain_db = target_noise_loudness - noise_loudness
    scale_factor = 10.0 ** (gain_db / 20.0)
 
    mixed = speech + noise_segment * scale_factor
 
    # Prevent digital clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
 
    return mixed




