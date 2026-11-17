import numpy as np
import torch

from hlw.model import WaterNet, load_weights, predict_tiled, s1_to_input, s2_to_input, save_weights


def test_s2_input_inserts_b10():
    refl = np.full((12, 4, 4), 5000, np.uint16)
    x = s2_to_input(refl)
    assert x.shape == (13, 4, 4)
    assert np.all(x[10] == 0) and np.allclose(x[0], 0.5)


def test_s1_input_nan():
    db = np.full((2, 3, 3), np.nan, np.float32)
    assert np.all(s1_to_input(db) == 0)


def test_forward_and_tiled_shape(tmp_path):
    m = WaterNet("s2", init="none")
    x = np.random.default_rng(0).random((13, 300, 200)).astype(np.float32)
    mean, std = predict_tiled(m, x, tile=128, overlap=32, device="cpu")
    assert mean.shape == (300, 200) and np.all((mean >= 0) & (mean <= 1))
    assert np.allclose(std, 0)  # deterministic single model
    _, std_mc = predict_tiled(m, x, tile=128, overlap=32, mc=3, device="cpu")
    assert std_mc.max() > 0
    save_weights(m, tmp_path / "w.pt", {"a": 1})
    m2 = load_weights(tmp_path / "w.pt")
    xb = torch.from_numpy(x[None, :, :128, :128])
    m.eval()
    assert torch.allclose(m(xb), m2(xb), atol=5e-2)
    assert m2.meta == {"a": 1}
