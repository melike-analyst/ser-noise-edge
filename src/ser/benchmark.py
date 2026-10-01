"""Latency / memory benchmark of ONNX models on the current machine.

Run the SAME command on your laptop (x86_64) and on an ARM64 device
(Raspberry Pi, Apple Silicon, cloud Graviton...) and compare the CSVs.

    python -m ser.benchmark --models runs/cnn/model.onnx runs/cnn/model.int8.onnx --tag laptop

Each model runs in its own process so the peak-RSS number is not polluted by
other models. What is measured for a single 3 s clip (batch = 1):
  feature_ms : NumPy log-mel extraction
  model_ms   : ONNX Runtime inference
  total_ms   : feature + model,  p50 / p95 over many runs
  rtf        : real-time factor = total p50 / clip duration (< 1 means faster than real time)
  peak_rss_mb: peak resident memory of the process
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from . import features

CLIP_SECONDS = 3.0


def model_label(path: str) -> str:
    p = Path(path)
    variant = "onnx_int8" if ".int8" in p.name else "onnx_fp32"
    return f"{p.parent.name}/{variant}"


def peak_rss_mb() -> float:
    """Peak resident memory of THIS process in MB."""
    try:   # Linux: VmHWM is the per-process high-water mark (not inherited from the parent)
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1e3
    except OSError:
        pass
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1e6 if sys.platform == "darwin" else 1e3)   # macOS: bytes, Linux: KB
    except ImportError:                                            # Windows
        import psutil
        return psutil.Process().memory_info().rss / 1e6


def cpu_name() -> str:
    """Best-effort CPU / board name (x86: 'model name', Raspberry Pi: 'Model')."""
    try:
        info = {}
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                info.setdefault(k.strip(), v.strip())
        for key in ("model name", "Model", "Hardware"):
            if info.get(key):
                return info[key]
    except OSError:
        pass
    return platform.processor() or platform.machine()


def bench_one(onnx_path: str, runs: int, warmup: int, threads: int) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    wave = (np.random.default_rng(0).standard_normal(int(features.SR * CLIP_SECONDS)) * 0.1).astype(np.float32)

    def once():
        t0 = perf_counter()
        x = features.log_mel(wave)[None, None]
        t1 = perf_counter()
        sess.run(None, {name: x})
        t2 = perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3

    for _ in range(warmup):
        once()
    feat, mod = map(np.array, zip(*[once() for _ in range(runs)]))
    tot = feat + mod
    return dict(
        model=model_label(onnx_path), size_mb=Path(onnx_path).stat().st_size / 1e6,
        feature_ms_p50=np.percentile(feat, 50), model_ms_p50=np.percentile(mod, 50),
        total_ms_p50=np.percentile(tot, 50), total_ms_p95=np.percentile(tot, 95),
        rtf=np.percentile(tot, 50) / 1e3 / CLIP_SECONDS, peak_rss_mb=peak_rss_mb(),
        runs=runs, warmup=warmup, threads=threads,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", help="ONNX files to benchmark")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default=platform.node() or "machine", help="label for this machine, used in the file name")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--single", default=None, help="(internal) benchmark one model and print JSON")
    args = ap.parse_args(argv)

    if args.single:
        print("JSON:" + json.dumps(bench_one(args.single, args.runs, args.warmup, args.threads)))
        return

    import onnxruntime as ort
    import pandas as pd

    rows = []
    for m in args.models:
        cmd = [sys.executable, "-m", "ser.benchmark", "--single", m, "--runs", str(args.runs),
               "--warmup", str(args.warmup), "--threads", str(args.threads)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in res.stdout.splitlines() if l.startswith("JSON:")]
        if res.returncode != 0 or not line:
            raise RuntimeError(f"benchmark failed for {m}:\n{res.stderr}")
        rows.append(json.loads(line[0][5:]))

    df = pd.DataFrame(rows)
    df.insert(0, "machine", args.tag)
    df["arch"] = platform.machine()
    df["cpu"] = cpu_name()
    df["ort_version"] = ort.__version__
    df["python"] = platform.python_version()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"benchmark_{args.tag}.csv"
    df.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df.drop(columns=["cpu", "python", "runs", "warmup"]).round(3).to_string(index=False))
    print(f"\nsaved {path}   (arch={platform.machine()}, cpu={cpu_name()})")


if __name__ == "__main__":
    main()
