import threading
import time

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from hlw import transfer


def test_box_is_square_and_exclusion_cuts_corner():
    ring = transfer.box((-100.0, 31.0), 2)
    full = Polygon(ring)
    assert len(ring) == 5
    cut = Polygon(transfer.box((-100.0, 31.0), 2, exclude_km=[1, -3, 3, -1]))
    # the excluded 1 km x 1 km corner removes about a sixteenth of the 4 km x 4 km box
    assert 0.92 < cut.area / full.area < 0.96
    assert cut.is_valid


def test_lakes_parse_and_never_overlap_training_lakes():
    from hlw import config as C

    tl = transfer.lakes()
    assert len(tl) == 10
    assert not set(tl) & set(C.lakes())
    for lake in tl.values():
        assert Polygon(lake.clip).contains(Polygon(lake.clip).centroid)


def test_bounded_map_keeps_order_and_limits_in_flight():
    active, peak, lock = [0], [0], threading.Lock()

    def job(x):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        with lock:
            active[0] -= 1
        return x * 2

    out = []
    for r in transfer.bounded_map(job, range(20), 3):
        time.sleep(0.02)  # slow consumer: an unbounded map would run far ahead
        out.append(r)
    assert out == [2 * i for i in range(20)]
    assert peak[0] <= 3


def test_score_series_perfect_and_biased():
    ref = pd.Series(np.linspace(100, 300, 30))
    s = transfer.score_series(ref, ref)
    assert s["median_abs_pct_err"] == 0 and abs(s["r"] - 1) < 1e-12
    s = transfer.score_series(ref * 0.9, ref)
    assert abs(s["median_bias_pct"] + 10) < 1e-9
    assert s["rmse_after_fit_pct"] < 1e-9  # a pure scale bias is removed by the linear fit
