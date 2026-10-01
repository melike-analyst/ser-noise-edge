import numpy as np
import pytest

from ser import features
from ser.noise import NOISE_KINDS, add_noise, make_noise, measured_snr_db
from ser.models import build_model

SR = 16000


@pytest.mark.parametrize("kind", NOISE_KINDS)
@pytest.mark.parametrize("snr", [20, 5, -5])
def test_add_noise_hits_target_snr(kind, snr):
    rng = np.random.default_rng(0)
    clean = (rng.standard_normal(SR * 3) * 0.1).astype(np.float32)
    noisy = add_noise(clean, make_noise(kind, clean.size, SR, rng), snr)
    assert abs(measured_snr_db(clean, noisy) - snr) < 0.1


def test_log_mel_shape_and_normalization():
    x = (np.random.default_rng(0).standard_normal(SR * 3)).astype(np.float32)
    m = features.log_mel(x)
    assert m.shape == (features.N_MELS, 301)
    assert abs(m.mean()) < 1e-3 and abs(m.std() - 1) < 1e-2
    assert features.batch_log_mel(np.stack([x, x])).shape == (2, 1, features.N_MELS, 301)


@pytest.mark.parametrize("name", ["cnn", "crnn"])
def test_models_forward(name):
    import torch
    model = build_model(name, 8, features.N_MELS).eval()
    assert model(torch.randn(3, 1, features.N_MELS, 301)).shape == (3, 8)
    assert model(torch.randn(2, 1, features.N_MELS, 150)).shape == (2, 8)   # other clip length
