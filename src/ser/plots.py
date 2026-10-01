"""Make the two figures used in the README.

    python -m ser.plots --results_dir results

  robustness.png : UAR vs SNR, one panel per noise type, one line per model
  tradeoff.png   : clean UAR vs latency (accuracy / speed trade-off) per machine
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_robustness(results_dir: Path):
    files = sorted(results_dir.glob("robustness_*.csv"))
    if not files:
        print("no robustness_*.csv files found")
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    kinds = [k for k in df.noise.unique() if k != "clean"]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.8), sharey=True, squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        for model, g in df.groupby("model"):
            clean = g[g.noise == "clean"].uar.iloc[0]
            g = g[g.noise == kind].sort_values("snr_db")
            ax.plot(list(g.snr_db) + [max(g.snr_db) + 5], list(g.uar) + [clean], marker="o", ms=3, label=model)
        ticks = sorted(df[df.noise == kind].snr_db.unique())
        ax.set_xticks(ticks + [max(ticks) + 5], [f"{t:g}" for t in ticks] + ["clean"])
        ax.axhline(1 / 8, color="gray", ls=":", lw=1)
        ax.set_title(f"{kind} noise")
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("UAR")
    axes[0][-1].legend(fontsize=7, loc="best")
    fig.suptitle("Robustness to noise (dotted line = chance)")
    fig.tight_layout()
    fig.savefig(results_dir / "robustness.png", dpi=150)
    print("saved", results_dir / "robustness.png")


def plot_tradeoff(results_dir: Path):
    bench_files = sorted(results_dir.glob("benchmark_*.csv"))
    rob_files = sorted(results_dir.glob("robustness_*.csv"))
    if not bench_files or not rob_files:
        print("need both benchmark_*.csv and robustness_*.csv for the trade-off plot")
        return
    bench = pd.concat([pd.read_csv(f) for f in bench_files])
    rob = pd.concat([pd.read_csv(f) for f in rob_files])
    clean = rob[rob.noise == "clean"][["model", "uar"]]
    df = bench.merge(clean, on="model")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for machine, g in df.groupby("machine"):
        ax.scatter(g.total_ms_p50, g.uar, label=f"{machine} ({g.arch.iloc[0]})")
        for _, r in g.iterrows():
            ax.annotate(r.model, (r.total_ms_p50, r.uar), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("latency per 3 s clip, p50 (ms)  [features + model]")
    ax.set_ylabel("clean test UAR")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results_dir / "tradeoff.png", dpi=150)
    print("saved", results_dir / "tradeoff.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    a = ap.parse_args()
    plot_robustness(Path(a.results_dir))
    plot_tradeoff(Path(a.results_dir))
