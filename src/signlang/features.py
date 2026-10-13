"""Pure-NumPy landmark feature utilities (no MediaPipe / TensorFlow needed).

One frame = 48 points x (x, y, z) = 144 numbers:
    points  0-20  : left hand   (21 landmarks)
    points 21-41  : right hand  (21 landmarks)
    points 42-47  : upper body  (L shoulder, R shoulder, L elbow, R elbow, L wrist, R wrist)
A point that was not detected is stored as exactly (0, 0, 0).
"""
import numpy as np

N_HAND = 21
N_POSE = 6
N_POINTS = 2 * N_HAND + N_POSE      # 48
N_FEATURES = N_POINTS * 3           # 144
LEFT = slice(0, N_HAND)
RIGHT = slice(N_HAND, 2 * N_HAND)
POSE = slice(2 * N_HAND, N_POINTS)
POSE_IDS = (11, 12, 13, 14, 15, 16)  # MediaPipe pose indices for shoulders, elbows, wrists
L_SHOULDER, R_SHOULDER = 2 * N_HAND, 2 * N_HAND + 1


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Make landmarks independent of where the signer stands and how far they are.

    Every frame is shifted so the shoulder midpoint is the origin and scaled by the shoulder
    width. Frames where the shoulders were missed reuse the last good frame's reference.
    Missing points stay exactly zero.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != N_FEATURES:
        raise ValueError(f"Expected (T, {N_FEATURES}) array, got {seq.shape}")
    T = len(seq)
    pts = seq.reshape(T, N_POINTS, 3).copy()
    present = np.abs(pts).sum(axis=-1) > 0

    left, right = pts[:, L_SHOULDER], pts[:, R_SHOULDER]
    centers = (left + right) / 2
    widths = np.linalg.norm((left - right)[:, :2], axis=1)
    ok = present[:, L_SHOULDER] & present[:, R_SHOULDER] & (widths > 1e-3)

    if ok.any():
        idx = np.maximum.accumulate(np.where(ok, np.arange(T), -1))  # forward-fill last good frame
        idx = np.where(idx < 0, int(np.argmax(ok)), idx)             # back-fill the start
        centers, widths = centers[idx], widths[idx]
    else:
        centers = np.tile(np.array([0.5, 0.5, 0.0], dtype=np.float32), (T, 1))
        widths = np.full(T, 0.25, dtype=np.float32)

    out = (pts - centers[:, None, :]) / widths[:, None, None]
    out[~present] = 0.0
    return out.reshape(T, -1).astype(np.float32)


def resample_sequence(seq: np.ndarray, length: int) -> np.ndarray:
    """Pick `length` frames evenly across the sequence (nearest frame, keeps zeros intact)."""
    if len(seq) == 0:
        raise ValueError("Cannot resample an empty sequence")
    idx = np.round(np.linspace(0, len(seq) - 1, length)).astype(int)
    return seq[idx]


def hand_presence(seq: np.ndarray) -> float:
    """Fraction of frames in which at least one hand was detected (works on raw or normalized)."""
    seq = np.asarray(seq)
    return float((np.abs(seq[:, : 2 * N_HAND * 3]).sum(axis=1) > 0).mean())
