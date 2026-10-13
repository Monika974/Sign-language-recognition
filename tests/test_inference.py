import numpy as np

from signlang.smoothing import PredictionSmoother


def test_smoother_needs_consensus():
    s = PredictionSmoother(threshold=0.6, consensus=3)
    assert s.update(2, 0.9) is None
    assert s.update(2, 0.9) is None
    assert s.update(2, 0.9) == 2


def test_smoother_ignores_low_confidence_and_flicker():
    s = PredictionSmoother(threshold=0.6, consensus=3)
    for idx, p in [(1, 0.9), (2, 0.9), (1, 0.9), (1, 0.3)]:
        assert s.update(idx, p) is None


def test_smoother_clears_after_silence():
    s = PredictionSmoother(threshold=0.6, consensus=2)
    s.update(4, 0.9)
    assert s.update(4, 0.9) == 4
    s.update(None)
    assert s.update(None) is None
