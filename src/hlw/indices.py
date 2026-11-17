"""Water-index baselines: NDWI (McFeeters 1996) and MNDWI (Xu 2006) thresholds."""

from __future__ import annotations

import numpy as np
from skimage.filters import threshold_otsu

from .config import S2_BANDS

B = {b: i for i, b in enumerate(S2_BANDS)}


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out


def ndwi(refl: np.ndarray) -> np.ndarray:
    """(Green - NIR) / (Green + NIR) using B03, B08. refl: (12,H,W) in S2_BANDS order."""
    return _nd(refl[B["B03"]], refl[B["B08"]])


def mndwi(refl: np.ndarray) -> np.ndarray:
    """(Green - SWIR1) / (Green + SWIR1) using B03, B11."""
    return _nd(refl[B["B03"]], refl[B["B11"]])


INDEX_FUNCS = {"ndwi": ndwi, "mndwi": mndwi}


def otsu_cutoff(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    return float(threshold_otsu(v)) if v.size else 0.0


def tune_cutoff(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None,
                grid: np.ndarray | None = None) -> tuple[float, float]:
    """Cutoff maximising (weighted) F1 for `scores > cutoff` -> water. Returns (cutoff, f1)."""
    grid = np.linspace(-0.6, 0.6, 241) if grid is None else grid
    w = np.ones_like(scores, dtype=float) if weights is None else weights
    ok = np.isfinite(scores)
    s, y, w = scores[ok], labels[ok].astype(bool), w[ok]
    best = (0.0, -1.0)
    for t in grid:
        p = s > t
        tp = w[p & y].sum()
        fp = w[p & ~y].sum()
        fn = w[~p & y].sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best
