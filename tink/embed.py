"""voc2vec embeddings: one mean-pooled, L2-normalized vector per clip.

Paper: Koudounas et al., "voc2vec: A Foundation Model for Non-Verbal Vocalization", ICASSP 2025.
The released checkpoints are pretrained only (no classifier head), so we use frozen embeddings.
"""
import os
from functools import lru_cache

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

from .audio import SR

MODEL_ID = os.environ.get("TINK_MODEL", "alkiskoudounas/voc2vec-hubert-ls-pt")
LAYER = int(os.environ.get("TINK_LAYER", "-1"))  # pick with risk_test.py
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load():
    fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    return fe, model


def layers(wav: np.ndarray) -> np.ndarray:
    """Mean-pooled hidden state of every layer, shape (n_layers + 1, dim), rows L2-normalized."""
    fe, model = _load()
    inputs = fe(wav, sampling_rate=SR, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    h = torch.stack(out.hidden_states)[:, 0].mean(dim=1)
    return torch.nn.functional.normalize(h, dim=-1).cpu().numpy()


def embed(wav: np.ndarray) -> np.ndarray:
    return layers(wav)[LAYER]
