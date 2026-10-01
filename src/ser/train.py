"""Train a model on the speaker-independent RAVDESS split.

    python -m ser.train --data_root data/ravdess --model cnn --out runs/cnn
    python -m ser.train --data_root data/ravdess --model cnn --aug_noise --out runs/cnn_aug
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score

from . import features
from .data import DURATION, EMOTIONS, SR, load_split, parse_ravdess, split_by_actor
from .models import build_model, count_params
from .noise import NOISE_KINDS, add_noise, make_noise


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment_batch(waves, rng, p_noise=0.5, snr_range=(0.0, 20.0)):
    """Randomly add noise (random type, random SNR) to a fraction of the clips."""
    out = waves.copy()
    for i in range(len(out)):
        if rng.random() < p_noise:
            kind = NOISE_KINDS[rng.integers(len(NOISE_KINDS))]
            noise = make_noise(kind, out.shape[1], SR, rng)
            out[i] = add_noise(out[i], noise, rng.uniform(*snr_range))
    return out


def spec_mask(feats, rng, max_freq=8, max_time=30):
    """Light SpecAugment: one frequency mask and one time mask per clip."""
    for f in feats:
        fw = rng.integers(0, max_freq + 1)
        f0 = rng.integers(0, f.shape[1] - fw + 1)
        f[:, f0:f0 + fw, :] = 0.0
        tw = rng.integers(0, max_time + 1)
        t0 = rng.integers(0, f.shape[2] - tw + 1)
        f[:, :, t0:t0 + tw] = 0.0
    return feats


@torch.no_grad()
def predict(model, waves, batch_size=64):
    model.eval()
    preds = []
    for i in range(0, len(waves), batch_size):
        x = torch.from_numpy(features.batch_log_mel(waves[i:i + batch_size]))
        preds.append(model(x).argmax(1).numpy())
    return np.concatenate(preds)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model", default="cnn", choices=["cnn", "crnn"])
    ap.add_argument("--out", default=None, help="output dir (default: runs/<model>)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug_noise", action="store_true", help="noise augmentation during training")
    args = ap.parse_args(argv)

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out or f"runs/{args.model}")
    out.mkdir(parents=True, exist_ok=True)

    splits = split_by_actor(parse_ravdess(args.data_root))
    print({k: len(v) for k, v in splits.items()}, "clips per split")
    x_tr, y_tr = load_split(splits["train"])
    x_va, y_va = load_split(splits["val"])
    x_te, y_te = load_split(splits["test"])

    model = build_model(args.model, len(EMOTIONS), features.N_MELS)
    print(f"model={args.model} params={count_params(model):,} noise_aug={args.aug_noise}")

    counts = np.bincount(y_tr, minlength=len(EMOTIONS)).astype(np.float32)
    weights = torch.tensor(counts.sum() / (len(counts) * np.maximum(counts, 1)))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_uar, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(x_tr))
        losses = []
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            wb = x_tr[idx]
            if args.aug_noise:
                wb = augment_batch(wb, rng)
            fb = spec_mask(features.batch_log_mel(wb), rng)
            logits = model(torch.from_numpy(fb))
            loss = loss_fn(logits, torch.from_numpy(y_tr[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        val_pred = predict(model, x_va)
        val_uar = balanced_accuracy_score(y_va, val_pred)   # = unweighted average recall
        history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_uar=val_uar))
        flag = ""
        if val_uar > best_uar:
            best_uar = val_uar
            torch.save(model.state_dict(), out / "best.pt")
            flag = "  <- best"
        print(f"epoch {epoch:3d}  loss {np.mean(losses):.3f}  val UAR {val_uar:.3f}  ({time.time() - t0:.1f}s){flag}")

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    test_pred = predict(model, x_te)
    metrics = dict(
        test_uar=float(balanced_accuracy_score(y_te, test_pred)),
        test_macro_f1=float(f1_score(y_te, test_pred, average="macro")),
        test_accuracy=float((test_pred == y_te).mean()),
        best_val_uar=float(best_uar),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    meta = dict(model=args.model, classes=EMOTIONS, sr=SR, duration=DURATION, n_mels=features.N_MELS,
                n_params=count_params(model), noise_aug=args.aug_noise, seed=args.seed)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("clean test:", metrics)


if __name__ == "__main__":
    main()
