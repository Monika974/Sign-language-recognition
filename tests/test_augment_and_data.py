import numpy as np
import pandas as pd

from signlang.augment import augment_sequence
from signlang.data import assign_splits, to_array
from signlang.features import N_FEATURES, N_POINTS


def test_augment_shape_and_missing_preserved():
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(40, N_FEATURES)).astype("float32")
    pts = seq.reshape(40, N_POINTS, 3)
    pts[:, :21] = 0   # left hand never detected
    seq = pts.reshape(40, -1)
    out = augment_sequence(seq, 32, rng)
    assert out.shape == (32, N_FEATURES)
    assert np.all(out.reshape(32, N_POINTS, 3)[:, :21] == 0)
    assert not np.allclose(out, seq[:32])


def test_assign_splits_uses_existing_column():
    df = pd.DataFrame({"label": list("aabb"), "split": ["train", "val", "test", "train"]})
    assert assign_splits(df)["split"].tolist() == ["train", "val", "test", "train"]


def test_assign_splits_random_is_stratified():
    df = pd.DataFrame({"label": ["a"] * 20 + ["b"] * 20, "split": ""})
    out = assign_splits(df, seed=1)
    for label in ("a", "b"):
        counts = out[out.label == label]["split"].value_counts()
        assert counts["test"] >= 1 and counts["val"] >= 1 and counts["train"] > counts["test"]


def test_to_array():
    seqs = [np.zeros((30, N_FEATURES), "float32"), np.zeros((50, N_FEATURES), "float32")]
    assert to_array(seqs, 32).shape == (2, 32, N_FEATURES)
