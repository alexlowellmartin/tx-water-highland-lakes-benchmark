import numpy as np
import pandas as pd
import pytest

from hlw.metrics import ece, estimate, iou_dense


def test_census_equals_dense_iou():
    rng = np.random.default_rng(0)
    y = rng.random(400) > 0.5
    p = y.copy()
    p[:40] = ~p[:40]
    df = pd.DataFrame({"scene": "s", "stratum": "A", "stratum_pixels": 400,
                       "label": y.astype(int), "pred": p.astype(int)})
    assert estimate(df, "pred")["iou"] == pytest.approx(iou_dense(p, y))


def test_stratum_weighting():
    # stratum A: 1000 px, all water, predicted right. stratum B: 10 px land, 1 of 2 points wrong
    df = pd.DataFrame({
        "scene": ["s"] * 4, "stratum": ["A", "A", "B", "B"],
        "stratum_pixels": [1000, 1000, 10, 10],
        "label": [1, 1, 0, 0], "pred": [1, 1, 1, 0]})
    e = estimate(df, "pred")
    # TP = 1000, FP = 10 * 1/2 = 5
    assert e["iou"] == pytest.approx(1000 / 1005)


def test_ece_perfect_and_bad():
    p = np.r_[np.full(100, 0.05), np.full(100, 0.95)]
    y = np.r_[np.zeros(95), np.ones(5), np.ones(95), np.zeros(5)]
    e, _ = ece(p, y)
    assert e == pytest.approx(0.0, abs=1e-9)
    e2, _ = ece(np.full(200, 0.99), y)
    assert e2 > 0.4
