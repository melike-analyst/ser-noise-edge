"""Export a trained model to ONNX and create a dynamically quantized INT8 copy.

    python -m ser.export_onnx --run runs/cnn

Writes runs/cnn/model.onnx (FP32) and runs/cnn/model.int8.onnx, checks that the
ONNX output matches PyTorch, and prints file sizes and which ops were quantized.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .models import build_model


def op_counts(path):
    import onnx
    return Counter(n.op_type for n in onnx.load(str(path)).graph.node)


def export(run_dir: str, opset: int = 17, n_frames: int = 301):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    run = Path(run_dir)
    meta = json.loads((run / "meta.json").read_text())
    model = build_model(meta["model"], len(meta["classes"]), meta["n_mels"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()

    fp32, int8 = run / "model.onnx", run / "model.int8.onnx"
    dummy = torch.randn(1, 1, meta["n_mels"], n_frames)
    torch.onnx.export(
        model, dummy, str(fp32), input_names=["logmel"], output_names=["logits"],
        dynamic_axes={"logmel": {0: "batch", 3: "time"}, "logits": {0: "batch"}},
        opset_version=opset, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(fp32)))

    # parity check on two different lengths and batch sizes
    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    max_diff = 0.0
    for b, t in [(1, n_frames), (4, 200)]:
        x = torch.randn(b, 1, meta["n_mels"], t)
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"logmel": x.numpy()})[0]
        max_diff = max(max_diff, float(np.abs(ref - out).max()))

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QUInt8)

    report = dict(
        fp32_mb=fp32.stat().st_size / 1e6, int8_mb=int8.stat().st_size / 1e6,
        max_abs_diff_torch_vs_onnx=max_diff,
        ops_fp32=dict(op_counts(fp32)), ops_int8=dict(op_counts(int8)),
    )
    (run / "export_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)
    r = export(args.run, args.opset)
    print(f"FP32 {r['fp32_mb']:.2f} MB | INT8 {r['int8_mb']:.2f} MB | max |torch - onnx| = {r['max_abs_diff_torch_vs_onnx']:.2e}")
    print("ops FP32:", r["ops_fp32"])
    print("ops INT8:", r["ops_int8"])


if __name__ == "__main__":
    main()
