"""Model loading, windowed prediction and output smoothing (no webcam code here)."""
import json
from pathlib import Path

import numpy as np
from tensorflow import keras

import signlang.models  # noqa: F401  (registers the custom Transformer layer)
from signlang.features import normalize_sequence, resample_sequence
from signlang.smoothing import PredictionSmoother  # noqa: F401  (re-exported)


class SignPredictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.classes = self.meta["classes"]
        self.seq_len = self.meta["seq_len"]
        self.model = keras.models.load_model(model_dir / "best.keras", compile=False)

    def predict(self, raw_seq: np.ndarray) -> np.ndarray:
        """raw_seq: (T, 144) raw landmarks. Returns class probabilities."""
        seq = resample_sequence(normalize_sequence(raw_seq), self.seq_len)
        return self.model.predict(seq[None], verbose=0)[0]

    def top_k(self, probs: np.ndarray, k: int = 3):
        idx = np.argsort(-probs)[:k]
        return [(self.classes[i], float(probs[i])) for i in idx]
