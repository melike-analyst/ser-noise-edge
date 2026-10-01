"""Tiny synthetic 'RAVDESS-like' dataset for smoke tests and CI.

It only checks that the PIPELINE runs (loading -> training -> evaluation ->
export -> benchmark). Scores on it mean nothing about real speech emotion.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

# emotion -> (f0 multiplier, syllable rate in Hz, spectral tilt)
PARAMS = {
    1: (1.00, 3.0, 1.5), 2: (0.95, 2.5, 1.8), 3: (1.25, 4.0, 1.0), 4: (0.85, 2.0, 2.0),
    5: (1.10, 4.5, 0.6), 6: (1.30, 5.0, 1.2), 7: (0.90, 3.0, 1.4), 8: (1.40, 3.5, 0.9),
}


def synth_utterance(f0, rate, tilt, rng, sr=16000, dur=3.0):
    n = int(sr * dur)
    t = np.arange(n) / sr
    vib = 1 + 0.05 * np.sin(2 * np.pi * rng.uniform(3, 6) * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(f0 * vib) / sr
    sig = sum((k ** -tilt) * np.sin(k * phase) for k in range(1, 16))
    env = (0.5 * (1 + np.sin(2 * np.pi * rate * t + rng.uniform(0, 6.28)))) ** 1.5
    x = sig * env + 0.002 * rng.standard_normal(n)
    return (0.5 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def make_dataset(out_dir: str, reps: int = 3, seed: int = 0, sr: int = 16000):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for actor in range(1, 25):
        base = 110.0 if actor % 2 == 1 else 210.0      # odd = male, even = female (as in RAVDESS)
        for emo, (mult, rate, tilt) in PARAMS.items():
            for rep in range(1, reps + 1):
                f0 = base * mult * rng.uniform(0.95, 1.05)
                x = synth_utterance(f0, rate * rng.uniform(0.9, 1.1), tilt, rng, sr)
                name = f"03-01-{emo:02d}-01-01-{rep:02d}-{actor:02d}.wav"
                sf.write(str(out / name), x, sr, subtype="PCM_16")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    print("wrote", make_dataset(a.out, a.reps))
