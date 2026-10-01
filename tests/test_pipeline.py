"""End-to-end smoke test on a tiny synthetic dataset (checks plumbing, not accuracy)."""
import json

import pytest

from ser import baseline, benchmark, evaluate, export_onnx, synthetic, train


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws")
    synthetic.make_dataset(str(root / "data"), reps=1)
    return root


def test_full_pipeline(workspace):
    data, run, res = str(workspace / "data"), str(workspace / "run"), str(workspace / "res")
    train.main(["--data_root", data, "--model", "cnn", "--out", run, "--epochs", "2", "--aug_noise"])
    assert (workspace / "run" / "best.pt").exists()

    export_onnx.main(["--run", run])
    report = json.loads((workspace / "run" / "export_report.json").read_text())
    assert report["max_abs_diff_torch_vs_onnx"] < 1e-3

    for variant in ["torch", "onnx_fp32", "onnx_int8"]:
        evaluate.main(["--data_root", data, "--run", run, "--variant", variant,
                       "--snrs", "10", "0", "--noise_kinds", "white", "cabin", "--out_dir", res])
    baseline.main(["--data_root", data, "--snrs", "10", "--noise_kinds", "white",
                   "--out", str(workspace / "res" / "robustness_svm_mfcc.csv")])
    assert len(list((workspace / "res").glob("robustness_*.csv"))) == 4

    benchmark.main(["--models", run + "/model.onnx", run + "/model.int8.onnx",
                    "--runs", "5", "--warmup", "1", "--tag", "test", "--out_dir", res])
    assert (workspace / "res" / "benchmark_test.csv").exists()
