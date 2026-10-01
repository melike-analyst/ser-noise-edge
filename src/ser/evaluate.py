"""Evaluate a trained model on the clean test set and on noisy versions of it.

The noisy test clips are generated deterministically (fixed seed), so every model
variant (PyTorch, ONNX FP32, ONNX INT8, SVM) sees exactly the same noisy inputs.

    python -m ser.evaluate --data_root data/ravdess --run runs/cnn --variant torch
    python -m ser.evaluate --data_root data/ravdess --run runs/cnn --variant onnx_int8
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from . import features
from .data import EMOTIONS, SR, load_split, parse_ravdess, split_by_actor
from .noise import NOISE_KINDS, add_noise, load_noise_files, make_noise

KIND_ID = {k: i for i, k in enumerate(NOISE_KINDS + ["real"])}


def noisy_version(waves, kind, snr_db, seed=0, bank=None):
    out = np.empty_like(waves)
    for i, w in enumerate(waves):
        rng = np.random.default_rng([seed, i, KIND_ID[kind], int(round(snr_db * 10)) + 1000])
        out[i] = add_noise(w, make_noise(kind, w.size, SR, rng, bank), snr_db)
    return out


class TorchPredictor:
    def __init__(self, run_dir):
        import torch
        from .models import build_model
        meta = json.loads((Path(run_dir) / "meta.json").read_text())
        self.torch = torch
        self.model = build_model(meta["model"], len(meta["classes"]), meta["n_mels"])
        self.model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location="cpu"))
        self.model.eval()

    def __call__(self, feats):
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(feats)).numpy()


class OrtPredictor:
    def __init__(self, onnx_path, threads=1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

    def __call__(self, feats):
        return self.sess.run(None, {self.input_name: feats})[0]


def get_predictor(run_dir, variant):
    run_dir = Path(run_dir)
    if variant == "torch":
        return TorchPredictor(run_dir)
    if variant == "onnx_fp32":
        return OrtPredictor(run_dir / "model.onnx")
    if variant == "onnx_int8":
        return OrtPredictor(run_dir / "model.int8.onnx")
    raise ValueError(variant)


def predict_labels(predictor, waves, batch_size=32):
    preds = []
    for i in range(0, len(waves), batch_size):
        preds.append(predictor(features.batch_log_mel(waves[i:i + batch_size])).argmax(1))
    return np.concatenate(preds)


def evaluate(predictor, waves, labels, name, kinds, snrs, seed=0, bank=None):
    rows, cm_clean = [], None
    conditions = [("clean", None)] + [(k, s) for k in kinds for s in snrs]
    for kind, snr in conditions:
        xs = waves if snr is None else noisy_version(waves, kind, snr, seed, bank)
        pred = predict_labels(predictor, xs)
        if snr is None:
            cm_clean = confusion_matrix(labels, pred, labels=range(len(EMOTIONS)))
        rows.append(dict(model=name, noise=kind, snr_db=np.nan if snr is None else snr,
                         uar=balanced_accuracy_score(labels, pred),
                         macro_f1=f1_score(labels, pred, average="macro"),
                         accuracy=float((pred == labels).mean())))
    return pd.DataFrame(rows), cm_clean


def save_confusion_png(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run", required=True, help="run dir, e.g. runs/cnn")
    ap.add_argument("--variant", default="torch", choices=["torch", "onnx_fp32", "onnx_int8"])
    ap.add_argument("--snrs", type=float, nargs="+", default=[20, 15, 10, 5, 0, -5])
    ap.add_argument("--noise_kinds", nargs="+", default=NOISE_KINDS)
    ap.add_argument("--noise_dir", default=None, help="folder with real noise wavs (ESC-50, DEMAND, MUSAN...)")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    splits = split_by_actor(parse_ravdess(args.data_root))
    waves, labels = load_split(splits["test"])
    bank = load_noise_files(args.noise_dir, SR) if args.noise_dir else None
    kinds = list(args.noise_kinds) + (["real"] if bank else [])

    name = f"{Path(args.run).name}/{args.variant}"
    df, cm = evaluate(get_predictor(args.run, args.variant), waves, labels, name, kinds, args.snrs, args.seed, bank)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = name.replace("/", "_")
    df.to_csv(out / f"robustness_{tag}.csv", index=False)
    save_confusion_png(cm, out / f"confusion_{tag}.png", f"{name} (clean test)")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
