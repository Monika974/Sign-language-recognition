import numpy as np
import pytest

pytest.importorskip("tensorflow")

from signlang.models import BUILDERS, build_model


@pytest.mark.parametrize("name", list(BUILDERS))
def test_model_forward_pass(name):
    model = build_model(name, seq_len=16, n_features=144, n_classes=7)
    out = model.predict(np.zeros((2, 16, 144), dtype="float32"), verbose=0)
    assert out.shape == (2, 7)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-4)
