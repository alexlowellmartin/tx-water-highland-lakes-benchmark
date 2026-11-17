"""Accuracy estimation from stratified random point labels.

Hand labels are points, not dense masks. Each point belongs to a (scene, stratum) cell with a
known number of pixels N_h; n_h points were labelled there. The estimated pixel count of each
confusion-matrix cell is sum_h N_h * (count_h / n_h) (Olofsson et al. 2014, Remote Sens.
Environ. 148:42-57). IoU and F1 of the water class follow from the estimated TP, FP, FN.
Confidence intervals come from a stratified bootstrap over points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cells(df: pd.DataFrame, pred_col: str):
    g = df.groupby(["scene", "stratum"])
    out = []
    for (_, _), d in g:
        n = len(d)
        N = d["stratum_pixels"].iloc[0]
        y = d["label"].values.astype(bool)
        p = d[pred_col].values.astype(bool)
        out.append((N / n, (p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()))
    return np.array(out, float)


def estimate(df: pd.DataFrame, pred_col: str) -> dict:
    """df columns: scene, stratum, stratum_pixels, label (0/1), <pred_col> (0/1)."""
    c = _cells(df, pred_col)
    if len(c) == 0:
        return {"iou": np.nan, "f1": np.nan, "precision": np.nan, "recall": np.nan,
                "accuracy": np.nan, "n": 0}
    w = c[:, 0:1]
    tp, fp, fn, tn = (w * c[:, 1:]).sum(0)
    return {
        "iou": tp / max(tp + fp + fn, 1e-9),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1e-9),
        "precision": tp / max(tp + fp, 1e-9),
        "recall": tp / max(tp + fn, 1e-9),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1e-9),
        "n": len(df),
    }


def _boot_cells(df: pd.DataFrame, cols, n_boot: int, seed: int) -> np.ndarray:
    """Weighted confusion counts (n_boot, len(cols), 4: tp, fp, fn, tn) from a stratified
    bootstrap over (scene, stratum) cells. Draws in the same order as the original pandas loop,
    so results are identical, ~50x faster."""
    rng = np.random.default_rng(seed)
    cells = []
    for _, d in df.groupby(["scene", "stratum"]):
        y = d["label"].values.astype(bool)
        p = np.stack([d[c].values.astype(bool) for c in cols], 1)
        cells.append((d["stratum_pixels"].iloc[0] / len(d), y, p))
    out = np.zeros((n_boot, len(cols), 4))
    for b in range(n_boot):
        acc = np.zeros((len(cols), 4))
        for w, y, p in cells:
            i = rng.integers(0, len(y), len(y))
            yy, pp = y[i, None], p[i]
            acc += w * np.stack([(pp & yy).sum(0), (pp & ~yy).sum(0), (~pp & yy).sum(0),
                                 (~pp & ~yy).sum(0)], 1)
        out[b] = acc
    return out


def _metric(c: np.ndarray, key: str) -> np.ndarray:
    tp, fp, fn, _tn = (c[..., k] for k in range(4))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"iou": tp / np.maximum(tp + fp + fn, 1e-9),
                "f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9),
                "precision": tp / np.maximum(tp + fp, 1e-9),
                "recall": tp / np.maximum(tp + fn, 1e-9)}[key]


def bootstrap(df: pd.DataFrame, pred_col: str, n_boot: int = 500, seed: int = 0,
              keys=("iou", "f1")) -> dict:
    """Point estimate plus 95 % stratified-bootstrap interval for each key."""
    est = estimate(df, pred_col)
    c = _boot_cells(df, [pred_col], n_boot, seed)[:, 0]
    for k in keys:
        lo, hi = np.nanpercentile(_metric(c, k), [2.5, 97.5])
        est[f"{k}_lo"], est[f"{k}_hi"] = float(lo), float(hi)
    return est


def ece(prob: np.ndarray, label: np.ndarray, weight: np.ndarray | None = None,
        bins: int = 10) -> tuple[float, pd.DataFrame]:
    """Expected calibration error (weighted) and the reliability table."""
    w = np.ones_like(prob) if weight is None else weight
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    rows, total, err = [], w.sum(), 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf = np.average(prob[m], weights=w[m])
        acc = np.average(label[m], weights=w[m])
        err += w[m].sum() / total * abs(conf - acc)
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "mean_prob": conf,
                     "frac_water": acc, "weight": w[m].sum() / total, "n": int(m.sum())})
    return float(err), pd.DataFrame(rows)


def iou_dense(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> float:
    v = np.ones_like(pred, bool) if valid is None else valid
    p, r = pred[v].astype(bool), ref[v].astype(bool)
    return float((p & r).sum() / max((p | r).sum(), 1))


def paired_bootstrap(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int = 1000,
                     seed: int = 0, key: str = "iou") -> dict:
    """Difference key(a) - key(b) on the same points, with a 95 % stratified-bootstrap interval
    from shared resamples (much tighter than comparing two separate intervals)."""
    df = df[df[col_a].notna() & df[col_b].notna()]
    d0 = estimate(df, col_a)[key] - estimate(df, col_b)[key]
    c = _boot_cells(df, [col_a, col_b], n_boot, seed)
    diffs = _metric(c[:, 0], key) - _metric(c[:, 1], key)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {f"d_{key}": float(d0), f"d_{key}_lo": float(lo), f"d_{key}_hi": float(hi),
            "p_a_better": float(np.mean(diffs > 0)), "n": len(df)}
