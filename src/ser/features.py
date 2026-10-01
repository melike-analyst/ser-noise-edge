"""Log-mel features in plain NumPy.

Why NumPy and not torchaudio? The same function is used for training AND for
edge inference, so the deployed pipeline does not need PyTorch at all
(only onnxruntime + numpy). That also keeps the memory benchmark honest.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 16000
N_FFT = 512
WIN = 400      # 25 ms
HOP = 160      # 10 ms
N_MELS = 64
FMIN, FMAX = 50.0, 7600.0


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Triangular mel filterbank, shape (n_mels, n_fft // 2 + 1)."""
    mel_pts = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mel_pts)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, bins.size), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - left) / (center - left)
        down = (right - bins) / (right - center)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


_FB = mel_filterbank()
_WINDOW = np.hanning(WIN).astype(np.float32)


def log_mel(wave: np.ndarray, normalize: bool = True) -> np.ndarray:
    """waveform (n,) -> log-mel spectrogram (N_MELS, T).

    With normalize=True each utterance is standardised (mean 0, std 1).
    This removes recording-level differences and needs no dataset statistics.
    """
    x = np.pad(wave.astype(np.float32), (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = sliding_window_view(x, WIN)[::HOP] * _WINDOW       # (T, WIN)
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2   # (T, n_fft/2+1)
    mel = power @ _FB.T                                          # (T, N_MELS)
    out = np.log(mel + 1e-6).T.astype(np.float32)                # (N_MELS, T)
    if normalize:
        out = (out - out.mean()) / (out.std() + 1e-5)
    return out


def batch_log_mel(waves: np.ndarray) -> np.ndarray:
    """(B, n) waveforms -> (B, 1, N_MELS, T) float32, the model input layout."""
    return np.stack([log_mel(w) for w in waves])[:, None].astype(np.float32)


def mfcc_stats(wave: np.ndarray, n_mfcc: int = 20) -> np.ndarray:
    """Classical baseline features: mean and std of MFCCs over time."""
    from scipy.fft import dct

    mel = log_mel(wave, normalize=False)                         # (N_MELS, T)
    mfcc = dct(mel, type=2, axis=0, norm="ortho")[:n_mfcc]       # (n_mfcc, T)
    return np.concatenate([mfcc.mean(1), mfcc.std(1)]).astype(np.float32)
