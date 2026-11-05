"""Export voc2vec to ONNX for the server and the browser, and refuse a build that drifted from PyTorch.

    uv run --group export python -m scripts.export_onnx --layers 12 --clips path/to/wavs

--layers keeps only that many transformer layers, so unused weights never ship. evals/run.py measures which depth is
worth keeping. The script exits with an error unless every gate passes; a failed gate means there is nothing to ship.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from tink import audio
from tink.embed import MODEL_ID, _load, preprocess

MIN_COSINE = 0.99  # mean quantized vs PyTorch on real clips
MIN_WORST_COSINE = 0.98  # and the single worst clip


class Encoder(torch.nn.Module):
    """Waveform to one unit vector, from the first `depth` transformer layers."""

    def __init__(self, model: torch.nn.Module, depth: int):
        super().__init__()
        self.model = model
        self.model.encoder.layers = self.model.encoder.layers[:depth]

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        pooled = self.model(input_values).last_hidden_state.mean(dim=1)
        return torch.nn.functional.normalize(pooled, dim=-1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--clips", type=Path, required=True, help="folder of real WAVs to check agreement on")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("models"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)

    model = _load.__wrapped__(MODEL_ID).cpu()
    encoder = Encoder(model, args.layers).eval()
    full = args.out / f"voc2vec-l{args.layers}.float.onnx"
    small = args.out / f"voc2vec-l{args.layers}.int8.onnx"
    torch.onnx.export(encoder, (torch.randn(1, audio.SR),), full, input_names=["input_values"],
                      output_names=["embedding"], dynamic_axes={"input_values": {1: "samples"}}, opset_version=17,
                      dynamo=False)
    quantize_dynamic(full, small, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])

    files = sorted(args.clips.rglob("*.wav"))
    picks = np.random.default_rng(0).choice(len(files), min(args.n, len(files)), replace=False)
    sessions = {p.name: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]) for p in (full, small)}
    cos: dict[str, list[float]] = {name: [] for name in sessions}
    ms: dict[str, list[float]] = {name: [] for name in sessions}
    seconds = 0.0
    for i in picks:
        wav = audio.load(files[i])[: 8 * audio.SR]
        seconds += len(wav) / audio.SR
        x = preprocess(wav)[None]
        with torch.no_grad():
            ref = encoder(torch.from_numpy(x)).numpy()[0]
        for name, session in sessions.items():
            t0 = time.perf_counter()
            vec = session.run(None, {"input_values": x})[0][0]
            ms[name].append((time.perf_counter() - t0) * 1000)
            cos[name].append(float(vec @ ref))

    report = {
        "model": MODEL_ID, "layers": args.layers, "clips_checked": len(picks),
        "mean_clip_seconds": seconds / len(picks),
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p), "cosine_mean": float(np.mean(cos[p.name])),
                     "cosine_worst": float(np.min(cos[p.name])), "cpu_ms_mean": float(np.mean(ms[p.name])),
                     "cpu_ms_p95": float(np.percentile(ms[p.name], 95))}
            for p in (full, small)
        },
    }
    manifest = args.out / f"voc2vec-l{args.layers}.json"
    manifest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    q = report["files"][small.name]
    if q["cosine_mean"] < MIN_COSINE or q["cosine_worst"] < MIN_WORST_COSINE:
        sys.exit(f"GATE FAILED: quantized model drifted from PyTorch ({q['cosine_mean']:.4f}, {q['cosine_worst']:.4f})")
    print("gates passed")


if __name__ == "__main__":
    main()
