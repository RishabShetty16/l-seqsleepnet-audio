"""
sonify_eeg_demo.py
==================
Demonstrates the EEG sonification pipeline from datagenerator_from_list_v3.py.

Generates a synthetic sleep-like EEG epoch (or loads a real one from a .mat
file if you supply a path), upsamples it from 100 Hz -> 8000 Hz (sonification),
and saves the result as a playable .wav file.

Requirements:
    pip install librosa scipy numpy h5py soundfile

Usage:
    # With synthetic EEG (no data needed):
    python sonify_eeg_demo.py

    # With a real .mat file (uses the first epoch):
    python sonify_eeg_demo.py --mat_file path/to/your/file.mat

    # Save multiple epochs as separate wav files:
    python sonify_eeg_demo.py --mat_file path/to/your/file.mat --n_epochs 5
"""

import argparse
import numpy as np
import librosa
import soundfile as sf
import os

# ── Sonification parameters (must match datagenerator_from_list_v3.py) ────────
SR_EEG      = 100        # original EEG sampling rate (Hz)
SR_AUDIO    = 8000       # sonification target rate (Hz)
N_MELS      = 64         # mel bins (matches config.n_mels)
MEL_FRAMES  = 29         # time frames (matches config.mel_frame_seq_len)
EPOCH_SAMPLES = 3000     # samples per 30-second epoch at 100 Hz


def make_synthetic_eeg(n_epochs=1, seed=42):
    """
    Generate realistic-looking sleep EEG epochs with mixed frequency content:
      - Delta (0.5–4 Hz)   — dominant in deep sleep
      - Theta (4–8 Hz)
      - Alpha (8–13 Hz)
      - Beta  (13–30 Hz)
    Returns shape (n_epochs, 3000).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 30, EPOCH_SAMPLES, endpoint=False)  # 30-second epoch

    epochs = []
    for _ in range(n_epochs):
        # Random amplitudes for each band
        delta = rng.uniform(10, 50) * np.sin(2 * np.pi * rng.uniform(0.5, 4) * t)
        theta = rng.uniform(5, 20)  * np.sin(2 * np.pi * rng.uniform(4, 8)   * t)
        alpha = rng.uniform(2, 15)  * np.sin(2 * np.pi * rng.uniform(8, 13)  * t)
        beta  = rng.uniform(1, 8)   * np.sin(2 * np.pi * rng.uniform(13, 30) * t)
        noise = rng.normal(0, 3, EPOCH_SAMPLES)

        eeg = delta + theta + alpha + beta + noise
        epochs.append(eeg.astype(np.float32))

    return np.array(epochs)


def load_real_eeg(mat_path, n_epochs=1):
    """
    Load raw EEG (X1) from a .mat file used by the sleep staging pipeline.
    Returns shape (n_epochs, 3000).
    """
    import h5py
    with h5py.File(mat_path, 'r') as f:
        X1 = np.array(f['X1'])          # shape: (3000, total_epochs)
        X1 = X1.T                        # -> (total_epochs, 3000)
    n_epochs = min(n_epochs, len(X1))
    print(f"  Loaded {n_epochs} epoch(s) from {mat_path}  "
          f"[total in file: {len(X1)}]")
    return X1[:n_epochs].astype(np.float32)


def sonify_epoch(eeg_epoch):
    """
    Replicate the exact pipeline in DataGenerator3._extract_mel_spectrogram():
      1. Upsample 100 Hz -> 8000 Hz
      2. Return the upsampled audio waveform (for saving as .wav)

    Also returns the mel spectrogram for reference.
    """
    # Step 1 — upsample (sonification)
    audio = librosa.resample(eeg_epoch, orig_sr=SR_EEG, target_sr=SR_AUDIO)

    # Step 2 — mel spectrogram (same parameters as the datagenerator)
    signal_len = int(EPOCH_SAMPLES * (SR_AUDIO / SR_EEG))  # 240,000
    hop_length = signal_len // MEL_FRAMES                   # ~8275

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR_AUDIO,
        n_mels=N_MELS,
        hop_length=hop_length,
        fmax=SR_AUDIO // 2)

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Trim/pad to exactly MEL_FRAMES columns
    if mel_db.shape[1] >= MEL_FRAMES:
        mel_db = mel_db[:, :MEL_FRAMES]
    else:
        pad = MEL_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='edge')

    mel_feat = mel_db.T  # (29, 64)

    return audio, mel_feat


def save_wav(audio, path, sr=SR_AUDIO):
    """Normalise to [-1, 1] and save as 16-bit PCM wav."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    sf.write(path, audio, sr, subtype='PCM_16')
    print(f"  Saved: {path}  ({len(audio)/sr:.1f} seconds @ {sr} Hz)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EEG Sonification Demo")
    parser.add_argument("--mat_file",  type=str, default=None,
                        help="Path to a .mat data file (optional). "
                             "If not supplied, synthetic EEG is used.")
    parser.add_argument("--n_epochs",  type=int, default=1,
                        help="Number of 30-second epochs to sonify (default: 1)")
    parser.add_argument("--out_dir",   type=str, default="./sonified_output",
                        help="Output directory for .wav files (default: ./sonified_output)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load EEG ──────────────────────────────────────────────────────────────
    if args.mat_file:
        print(f"\nLoading real EEG from: {args.mat_file}")
        epochs = load_real_eeg(args.mat_file, n_epochs=args.n_epochs)
    else:
        print("\nNo .mat file supplied — generating synthetic sleep EEG...")
        epochs = make_synthetic_eeg(n_epochs=args.n_epochs)
        print(f"  Generated {len(epochs)} synthetic epoch(s), "
              f"each = 30 s @ {SR_EEG} Hz")

    # ── Sonify & save ─────────────────────────────────────────────────────────
    print(f"\nSonification pipeline: {SR_EEG} Hz  ->  {SR_AUDIO} Hz")
    print(f"Output: {os.path.abspath(args.out_dir)}\n")

    for i, eeg in enumerate(epochs):
        print(f"Epoch {i+1}/{len(epochs)}:")
        audio, mel_feat = sonify_epoch(eeg)

        wav_path = os.path.join(args.out_dir, f"epoch_{i+1:03d}_sonified.wav")
        save_wav(audio, wav_path)

        print(f"  Mel spectrogram shape: {mel_feat.shape}  "
              f"(time_frames={MEL_FRAMES}, n_mels={N_MELS})")

    print("\nDone! Open the .wav files in any media player "
          "(VLC, QuickTime, Windows Media Player, etc.)")
    print("Each file is one 30-second EEG epoch upsampled to 8 kHz audio.")


if __name__ == "__main__":
    main()
