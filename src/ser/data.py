"""RAVDESS loading with a speaker-independent split.

RAVDESS file name: 03-01-06-01-02-01-12.wav
  modality-channel-EMOTION-intensity-statement-repetition-ACTOR
Emotion codes: 01 neutral, 02 calm, 03 happy, 04 sad, 05 angry,
               06 fearful, 07 disgust, 08 surprised.

Split by ACTOR, never by file: the same person must not appear in both train
and test, otherwise the model can memorise voices and the scores look better
than they really are.
"""
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample_poly

SR = 16000
DURATION = 3.0
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

TRAIN_ACTORS = range(1, 17)    # 8 male + 8 female
VAL_ACTORS = range(17, 21)     # 2 + 2
TEST_ACTORS = range(21, 25)    # 2 + 2


def parse_ravdess(root: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(root).rglob("*.wav")):
        parts = p.stem.split("-")
        if len(parts) != 7:
            continue
        emo, actor = int(parts[2]), int(parts[6])
        if 1 <= emo <= 8:
            rows.append(dict(path=str(p), label=emo - 1, emotion=EMOTIONS[emo - 1], actor=actor))
    if not rows:
        raise FileNotFoundError(f"No RAVDESS-style wav files found under '{root}'")
    return pd.DataFrame(rows)


def split_by_actor(df: pd.DataFrame) -> dict:
    return {
        "train": df[df.actor.isin(TRAIN_ACTORS)].reset_index(drop=True),
        "val": df[df.actor.isin(VAL_ACTORS)].reset_index(drop=True),
        "test": df[df.actor.isin(TEST_ACTORS)].reset_index(drop=True),
    }


def trim_silence(x: np.ndarray, top_db: float = 40.0, frame: int = 400, hop: int = 160) -> np.ndarray:
    """Cut leading/trailing frames quieter than (max - top_db) dB."""
    if x.size < frame:
        return x
    frames = sliding_window_view(x, frame)[::hop]
    energy_db = 10.0 * np.log10((frames.astype(np.float64) ** 2).mean(1) + 1e-10)
    keep = np.where(energy_db > energy_db.max() - top_db)[0]
    return x[keep[0] * hop: min(x.size, keep[-1] * hop + frame)]


def fix_length(x: np.ndarray, n: int) -> np.ndarray:
    """Center-crop or zero-pad (centered) to exactly n samples."""
    if x.size >= n:
        start = (x.size - n) // 2
        return x[start:start + n]
    pad = n - x.size
    return np.pad(x, (pad // 2, pad - pad // 2))


def load_wave(path: str, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    x, file_sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if file_sr != sr:
        g = gcd(sr, file_sr)
        x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
    x = fix_length(trim_silence(x), int(sr * duration))
    peak = np.abs(x).max()
    return (x / peak * 0.9).astype(np.float32) if peak > 0 else x


def load_split(split_df: pd.DataFrame, sr: int = SR, duration: float = DURATION):
    """-> waves (N, samples) float32, labels (N,) int64"""
    waves = np.stack([load_wave(p, sr, duration) for p in split_df.path])
    return waves, split_df.label.to_numpy(dtype=np.int64)
