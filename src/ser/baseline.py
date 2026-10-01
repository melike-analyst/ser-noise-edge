"""Classical baseline: MFCC mean/std features + RBF-SVM.

A deep model only means something if it beats a simple reference. This runs the
same noise sweep as the neural models and writes the same CSV format.

    python -m ser.baseline --data_root data/ravdess
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import features
from .data import SR, load_split, parse_ravdess, split_by_actor
from .evaluate import noisy_version
from .noise import NOISE_KINDS, load_noise_files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--snrs", type=float, nargs="+", default=[20, 15, 10, 5, 0, -5])
    ap.add_argument("--noise_kinds", nargs="+", default=NOISE_KINDS)
    ap.add_argument("--noise_dir", default=None)
    ap.add_argument("--out", default="results/robustness_svm_mfcc.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    splits = split_by_actor(parse_ravdess(args.data_root))
    x_tr, y_tr = load_split(splits["train"])
    x_te, y_te = load_split(splits["test"])

    clf = make_pipeline(StandardScaler(), SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced"))
    clf.fit(np.stack([features.mfcc_stats(w) for w in x_tr]), y_tr)

    bank = load_noise_files(args.noise_dir, SR) if args.noise_dir else None
    kinds = list(args.noise_kinds) + (["real"] if bank else [])
    rows = []
    for kind, snr in [("clean", None)] + [(k, s) for k in kinds for s in args.snrs]:
        xs = x_te if snr is None else noisy_version(x_te, kind, snr, args.seed, bank)
        pred = clf.predict(np.stack([features.mfcc_stats(w) for w in xs]))
        rows.append(dict(model="svm_mfcc", noise=kind, snr_db=np.nan if snr is None else snr,
                         uar=balanced_accuracy_score(y_te, pred),
                         macro_f1=f1_score(y_te, pred, average="macro"),
                         accuracy=float((pred == y_te).mean())))
        print(rows[-1])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
