"""Load extracted landmark files and build train/val/test splits."""
from pathlib import Path

import numpy as np
import pandas as pd

from signlang.features import normalize_sequence, resample_sequence

SPLIT_ALIASES = {"train": "train", "val": "val", "validation": "val", "test": "test"}


def assign_splits(df: pd.DataFrame, seed: int = 42, fracs=(0.7, 0.15, 0.15)) -> pd.DataFrame:
    """Use the dataset's own split column if every row has one; otherwise split randomly per class.

    A random per-sample split is fine for your own recordings, but results are optimistic because
    the same person / background appears in train and test. Prefer signer-disjoint splits.
    """
    df = df.copy()
    if "split" in df.columns:
        mapped = df["split"].astype(str).str.lower().map(SPLIT_ALIASES)
        if mapped.notna().all():
            df["split"] = mapped
            return df

    rng = np.random.default_rng(seed)
    df["split"] = "train"
    for _, idx in df.groupby("label").groups.items():
        idx = np.array(list(idx))
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(fracs[2] * n))) if n >= 3 else 0
        n_val = max(1, int(round(fracs[1] * n))) if n >= 5 else 0
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"
    return df


def load_sequences(df: pd.DataFrame, base_dir: str | Path = "."):
    """Load and normalize every sample. Returns a list of (T, 144) arrays."""
    base = Path(base_dir)
    seqs = []
    for p in df["npy_path"]:
        path = Path(p) if Path(p).is_absolute() else base / p
        seqs.append(normalize_sequence(np.load(path)))
    return seqs


def to_array(seqs, seq_len: int) -> np.ndarray:
    return np.stack([resample_sequence(s, seq_len) for s in seqs]).astype(np.float32)


def load_dataset(csv_path: str | Path, seed: int = 42, base_dir: str | Path = "."):
    """Return dict with splits {name: (seqs, y)} and the sorted class-name list."""
    df = pd.read_csv(csv_path)
    df = assign_splits(df, seed)
    classes = sorted(df["label"].unique())
    to_idx = {c: i for i, c in enumerate(classes)}
    out = {}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name]
        seqs = load_sequences(part, base_dir)
        y = part["label"].map(to_idx).to_numpy(dtype=np.int32)
        out[name] = (seqs, y)
    return out, classes
