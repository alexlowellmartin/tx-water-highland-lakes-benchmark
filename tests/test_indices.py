import numpy as np

from hlw.config import S2_BANDS
from hlw.indices import mndwi, ndwi, otsu_cutoff, tune_cutoff


def _refl(green, nir, swir):
    r = np.full((len(S2_BANDS), 1, 3), 1000, np.uint16)
    r[S2_BANDS.index("B03")] = green
    r[S2_BANDS.index("B08")] = nir
    r[S2_BANDS.index("B11")] = swir
    return r


def test_index_signs():
    # water: green > nir, swir; land: nir/swir > green
    r = _refl([[800, 500, 0]], [[200, 2500, 0]], [[100, 3000, 0]])
    n, m = ndwi(r)[0], mndwi(r)[0]
    assert n[0] > 0 and m[0] > 0
    assert n[1] < 0 and m[1] < 0
    assert np.isnan(n[2]) and np.isnan(m[2])


def test_tune_cutoff_separates():
    s = np.r_[np.linspace(-0.5, -0.1, 50), np.linspace(0.2, 0.6, 50)]
    y = np.r_[np.zeros(50), np.ones(50)]
    t, f1 = tune_cutoff(s, y)
    assert -0.1 <= t < 0.2 and f1 == 1.0


def test_otsu_bimodal():
    v = np.r_[np.random.default_rng(0).normal(-0.4, 0.05, 500),
              np.random.default_rng(1).normal(0.4, 0.05, 500)]
    assert -0.3 < otsu_cutoff(v) < 0.3  # any cut in the empty gap is optimal
