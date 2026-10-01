"""Noise generation and SNR mixing.

Four synthetic noise types are built in, plus optional real recordings from a
folder of wav files (e.g. ESC-50, DEMAND, MUSAN).

IMPORTANT: "cabin" is a *synthetic approximation* of in-car noise (low-frequency
rumble + engine-like harmonics + light wind hiss). It is NOT a real recording.
For a stronger study, pass real noise with --noise_dir.
"""
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

NOISE_KINDS = ["white", "pink", "brown", "cabin"]


def colored_noise(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Noise with power spectrum ~ 1 / f**beta (0=white, 1=pink, 2=brown)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    x = np.fft.irfft(spec * f ** (-beta / 2.0), n)
    return x / (x.std() + 1e-12)


def cabin_noise(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / sr
    rumble = colored_noise(n, 2.0, rng)
    f0 = rng.uniform(30.0, 60.0)
    wobble = 1.0 + 0.02 * np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * t)
    phase = 2 * np.pi * np.cumsum(f0 * wobble) / sr
    hum = sum((1.0 / k) * np.sin(k * phase + rng.uniform(0, 2 * np.pi)) for k in range(1, 7))
    hiss = colored_noise(n, 1.0, rng)
    x = rumble + 0.6 * hum / (np.std(hum) + 1e-12) + 0.15 * hiss
    return x / (x.std() + 1e-12)


def load_noise_files(directory: str, sr: int) -> list:
    """Load all wav files of a folder as mono float32 arrays at `sr`."""
    bank = []
    for p in sorted(Path(directory).rglob("*.wav")):
        x, file_sr = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if file_sr != sr:
            g = gcd(sr, file_sr)
            x = resample_poly(x, sr // g, file_sr // g).astype(np.float32)
        if x.size > sr // 2 and x.std() > 0:
            bank.append(x)
    if not bank:
        raise FileNotFoundError(f"no usable wav files under {directory}")
    return bank


def make_noise(kind: str, n: int, sr: int, rng: np.random.Generator, bank: list = None) -> np.ndarray:
    if kind == "white":
        return colored_noise(n, 0.0, rng)
    if kind == "pink":
        return colored_noise(n, 1.0, rng)
    if kind == "brown":
        return colored_noise(n, 2.0, rng)
    if kind == "cabin":
        return cabin_noise(n, sr, rng)
    if kind == "real":
        if not bank:
            raise ValueError("kind='real' needs a noise bank (--noise_dir)")
        x = bank[rng.integers(len(bank))]
        if x.size < n:                       # loop short recordings
            x = np.tile(x, int(np.ceil(n / x.size)))
        start = rng.integers(0, x.size - n + 1)
        seg = x[start:start + n]
        return (seg / (seg.std() + 1e-12)).astype(np.float32)
    raise ValueError(f"unknown noise kind: {kind}")


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix so that 10*log10(P_clean / P_noise_scaled) == snr_db.

    Powers are averaged over the whole clip (including any zero padding).
    """
    p_clean = np.mean(clean.astype(np.float64) ** 2)
    p_noise = np.mean(noise.astype(np.float64) ** 2) + 1e-12
    scale = np.sqrt(p_clean / (p_noise * 10.0 ** (snr_db / 10.0)))
    return (clean + scale * noise).astype(np.float32)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    resid = noisy.astype(np.float64) - clean.astype(np.float64)
    return 10.0 * np.log10(np.mean(clean.astype(np.float64) ** 2) / (np.mean(resid ** 2) + 1e-12))
