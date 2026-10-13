import numpy as np
import pytest

from signlang.features import (L_SHOULDER, N_FEATURES, N_POINTS, R_SHOULDER, hand_presence,
                               normalize_sequence, resample_sequence)


def make_seq(T=20, hands=True, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.zeros((T, N_POINTS, 3), dtype=np.float32)
    pts[:, L_SHOULDER] = [0.6, 0.5, 0.0]
    pts[:, R_SHOULDER] = [0.4, 0.5, 0.0]
    if hands:
        pts[:, 21:42] = rng.uniform(0.2, 0.8, size=(T, 21, 3)) + 0.01
    return pts.reshape(T, -1)


def test_normalize_is_translation_and_scale_invariant():
    seq = make_seq()
    moved = (seq.reshape(-1, N_POINTS, 3) * 1.7 + [0.1, 0.05, 0.0]).reshape(len(seq), -1)
    moved = moved * (seq != 0)  # missing points stay missing
    a, b = normalize_sequence(seq), normalize_sequence(moved)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_missing_points_stay_zero():
    seq = make_seq(hands=False)
    out = normalize_sequence(seq).reshape(len(seq), N_POINTS, 3)
    assert np.all(out[:, :42] == 0)


def test_shoulder_width_becomes_one():
    out = normalize_sequence(make_seq()).reshape(-1, N_POINTS, 3)
    width = np.linalg.norm(out[:, L_SHOULDER, :2] - out[:, R_SHOULDER, :2], axis=1)
    np.testing.assert_allclose(width, 1.0, atol=1e-4)


def test_missed_shoulders_reuse_last_good_frame():
    seq = make_seq()
    seq[5:10, L_SHOULDER * 3: R_SHOULDER * 3 + 3] = 0
    out = normalize_sequence(seq)
    assert np.isfinite(out).all()


def test_no_pose_at_all_does_not_crash():
    seq = make_seq()
    seq[:, 42 * 3:] = 0
    assert np.isfinite(normalize_sequence(seq)).all()


def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        normalize_sequence(np.zeros((10, 5)))


def test_resample_length_and_endpoints():
    seq = np.arange(50 * N_FEATURES, dtype=np.float32).reshape(50, N_FEATURES)
    out = resample_sequence(seq, 32)
    assert out.shape == (32, N_FEATURES)
    np.testing.assert_array_equal(out[0], seq[0])
    np.testing.assert_array_equal(out[-1], seq[-1])
    assert resample_sequence(seq[:5], 32).shape == (32, N_FEATURES)


def test_hand_presence():
    seq = make_seq(T=10)
    seq[:5, :126] = 0
    assert hand_presence(seq) == 0.5
