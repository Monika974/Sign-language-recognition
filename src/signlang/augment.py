"""Landmark-space data augmentation (applied to normalized sequences)."""
import numpy as np

from signlang.features import N_FEATURES, N_POINTS, resample_sequence


def augment_sequence(seq: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Random temporal crop + small rotation / scale / shift / noise, then resample to `length`.

    Horizontal mirroring is deliberately NOT used: many signs are not left/right symmetric.
    """
    T = len(seq)
    if T > 8:
        keep = int(T * rng.uniform(0.8, 1.0))
        start = int(rng.integers(0, T - keep + 1))
        seq = seq[start:start + keep]
    seq = resample_sequence(seq, length)

    pts = seq.reshape(length, N_POINTS, 3).copy()
    present = np.abs(pts).sum(axis=-1) > 0

    angle = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts[..., :2] = pts[..., :2] @ rot.T
    pts *= rng.uniform(0.9, 1.1)
    pts[..., :2] += rng.uniform(-0.05, 0.05, size=2).astype(np.float32)
    pts += rng.normal(0, 0.01, size=pts.shape).astype(np.float32)

    pts[~present] = 0.0
    return pts.reshape(length, N_FEATURES).astype(np.float32)
