"""voc2vec embeddings: one mean-pooled, L2-normalized vector per clip.

Paper: Koudounas et al., "voc2vec: A Foundation Model for Non-Verbal Vocalization", ICASSP 2025.
The released checkpoints are pretrained only (no classifier head), so we use frozen embeddings.

Two backends with the same output (scripts/export_onnx.py checks that they agree):
    torch   the default, for development and the layer-by-layer evaluations
    onnx    TINK_ONNX=models/voc2vec.onnx: no torch or transformers at runtime, so the server image stays small
"""
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from .audio import SR  # noqa: F401  (re-exported: this is the rate every embedding expects)

MODEL_ID = os.environ.get("TINK_MODEL", "alkiskoudounas/voc2vec-hubert-ls-pt")
LAYER = int(os.environ.get("TINK_LAYER", "-1"))  # pick with evals/run.py
ONNX_PATH = os.environ.get("TINK_ONNX")
_lock = threading.Lock()  # one model call at a time: MPS isn't thread-safe, and parallel calls just fight over cores


def preprocess(wav: np.ndarray) -> np.ndarray:
    """What HuBERT's feature extractor does: zero mean, unit variance."""
    return ((wav - wav.mean()) / np.sqrt(wav.var() + 1e-7)).astype(np.float32)


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=4)
def _load(model_id: str):
    from transformers import AutoModel
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # checkpoint-loading notes drown the evaluation output
    return AutoModel.from_pretrained(model_id).to(_device()).eval()


def layers(wav: np.ndarray, model_id: str = MODEL_ID) -> np.ndarray:
    """Mean-pooled hidden state of every layer, shape (n_layers + 1, dim), rows L2-normalized."""
    import torch

    model = _load(model_id)
    inputs = torch.from_numpy(preprocess(wav))[None].to(_device())
    with _lock, torch.no_grad():
        out = model(inputs, output_hidden_states=True)
        h = torch.stack(out.hidden_states)[:, 0].mean(dim=1)
        return torch.nn.functional.normalize(h, dim=-1).cpu().numpy()


@lru_cache(maxsize=1)
def _onnx_session(path: str):
    import onnxruntime as ort

    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def embed(wav: np.ndarray) -> np.ndarray:
    if ONNX_PATH:
        return _onnx_session(ONNX_PATH).run(None, {"input_values": preprocess(wav)[None]})[0][0]
    return layers(wav)[LAYER]


def model_tag() -> str:
    """Names what produced the vectors, for caches and for every prediction's response."""
    return Path(ONNX_PATH).name if ONNX_PATH else f"{MODEL_ID}@{LAYER}"
