"""Evaluate every method on the hand-labelled points.

Evaluation groups (training uses only 2018-2023 scenes from lakes other than the fold's test lake):
  heldout_lake_heldout_time : Travis/Buchanan points in 2024-2026 (strictest)
  heldout_lake_train_time   : Travis/Buchanan points in 2018-2023 (unseen lake)
  train_lake_heldout_time   : Inks/LBJ/Marble Falls points in 2024-2026 (unseen period)
Small-lake points from 2018-2023 sit on training scenes; they are used only to tune index
cutoffs and to measure weak-label noise, never to score the models.

Index cutoffs for a lake are tuned (weighted F1) on training-year points from *other* lakes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
import torch

from . import aoi, indices, labels
from . import config as C
from . import selection as S
from .dataset import s1_path, s2_path, s2_scl_path
from .download import fetch_s1
from .io import write_raster
from .metrics import bootstrap, ece, paired_bootstrap
from .model import load_weights, predict_tiled, s1_to_input, s2_to_input
from .train import run_name

PRED_DIR = C.OUT_DIR / "label_scenes"
# primary reference labels: one person's labels of the 660 points (first pass + blind second
# look at the conflicts, see hlw.handlabel); the AI labels in labels/hand_points.csv are kept and
# scored into results/ai_labels/
PRIMARY_LABELS = C.LABELS_DIR / "hand_points_human_final.csv"
POINT_PRED = C.OUT_DIR / "point_predictions.csv"
SEEDS = (0, 1, 2)
MC_PASSES = 10
GROUPS = ["heldout_lake_heldout_time", "heldout_lake_train_time", "train_lake_heldout_time"]


def fold_for_lake(lake: str) -> str:
    cfg = C.load_config()
    for f, v in cfg["folds"].items():
        if v["test_lake"] == lake:
            return f
    return "holdout_travis"  # small lakes are training lakes in every fold


def model_path(sensor, fold, label="scl_jrc", init="moco", seed=0):
    return C.RUNS_DIR / run_name(sensor, fold, label if sensor == "s2" else "teacher", init,
                                 seed) / "model.pt"


def load_models(sensor, fold, label="scl_jrc", init="moco", seeds=SEEDS):
    return [load_weights(model_path(sensor, fold, label, init, s)) for s in seeds
            if model_path(sensor, fold, label, init, s).exists()]


def group_of(lake: str, year: int) -> str | None:
    dyn = C.lakes()[lake].dynamic
    test = year in C.load_config()["test_years"]
    if dyn:
        return "heldout_lake_heldout_time" if test else "heldout_lake_train_time"
    return "train_lake_heldout_time" if test else None


def s1_pair_for(lake: str, date: str) -> str | None:
    df = S.clear_s2(lake)
    row = df[df.date == pd.Timestamp(date)]
    if not len(row):
        return None
    p = S.s1_pairs(lake, row)
    return p[0][0] if p else None


def _q(p):  # store probabilities compactly
    return np.round(np.clip(p, 0, 1) * 250).astype(np.uint8)


def predict_label_scenes(force=False):
    sel = S.selected()
    pts = pd.read_csv(C.LABELS_DIR / "hand_points.csv")
    rows = []
    for lake, dates in sel["label"].items():
        fold = fold_for_lake(lake)
        zone, transform, crs = aoi.load_zone(lake)
        static = labels.static_layers(lake)
        ens = load_models("s2", fold)
        abl = {"rand_init": load_models("s2", fold, init="none", seeds=(0,)),
               "jrc50_label": load_models("s2", fold, label="jrc50", seeds=(0,))}
        s1m = load_models("s1", fold, seeds=(0,))
        for date in dates:
            p = pts[(pts.lake == lake) & (pts.date == date)]
            if not len(p):
                continue
            r, c = p.row.values, p.col.values
            with rasterio.open(s2_path(lake, date)) as src:
                refl = src.read()
            with rasterio.open(s2_scl_path(lake, date)) as src:
                scl = src.read(1)
            out = p[["pid"]].copy()
            nd, mn = indices.ndwi(refl), indices.mndwi(refl)
            valid = zone & (refl[2] > 0)
            out["ndwi"], out["mndwi"] = nd[r, c], mn[r, c]
            out["ndwi_otsu_cut"] = indices.otsu_cutoff(nd[valid])
            out["mndwi_otsu_cut"] = indices.otsu_cutoff(mn[valid])
            for k in labels.LABEL_KINDS:
                wl = labels.weak_label(k, scl, zone, static, int(date[:4]))
                out[f"weak_{k}"] = np.where(wl[r, c] == labels.IGNORE, np.nan, wl[r, c])
            x = s2_to_input(refl)
            maps = {}
            if ens:
                members = [predict_tiled(m, x)[0] for m in ens]
                maps["ens_mean"] = np.mean(members, 0)
                maps["ens_std"] = np.std(members, 0)
                maps["single"] = members[0]
                mc_mean, mc_std = predict_tiled(ens[0], x, mc=MC_PASSES)
                maps["mc_mean"], maps["mc_std"] = mc_mean, mc_std
            for name, ms in abl.items():
                if ms:
                    maps[name] = predict_tiled(ms[0], x)[0]
            for name, m in maps.items():
                out[f"p_{name}"] = m[r, c]
            d1 = s1_pair_for(lake, date)
            out["s1_date"] = d1
            if d1:
                fetch_s1(lake, d1, zone, transform, crs)
                with rasterio.open(s1_path(lake, d1)) as src:
                    db = src.read().astype(np.float32) / 100
                db[db <= -300] = np.nan
                out["vv_db"], out["vh_db"] = db[0][r, c], db[1][r, c]
                v = db[0][valid & np.isfinite(db[0])]
                out["vv_otsu_cut"] = indices.otsu_cutoff(-v)
                if s1m:
                    ps1 = predict_tiled(s1m[0], s1_to_input(db))[0]
                    out["p_s1"] = ps1[r, c]
                    maps["s1"] = ps1
            for name in ("ens_mean", "ens_std", "mc_std", "s1"):
                if name in maps and (force or not (PRED_DIR / f"{lake}_{date}_{name}.tif").exists()):
                    write_raster(PRED_DIR / f"{lake}_{date}_{name}.tif",
                                 _q(np.where(zone, maps[name], 0))[None], transform, crs)
            rows.append(out)
            print("predicted", lake, date, flush=True)
        torch.cuda.empty_cache()
    df = pts.merge(pd.concat(rows), on="pid", how="left")
    df.to_csv(POINT_PRED, index=False)
    return df


def _weights(df):
    n = df.groupby(["scene", "stratum"]).pid.transform("count")
    return df.stratum_pixels / n


def score(df: pd.DataFrame | None = None, n_boot: int = 500, out_dir=None) -> dict:
    out_dir = C.RESULTS_DIR if out_dir is None else out_dir
    df = pd.read_csv(POINT_PRED) if df is None else df
    df = df[df.label.isin([0, 1])].copy()
    df["label"] = df.label.astype(int)
    df["year"] = df.date.str[:4].astype(int)
    df["group"] = [group_of(lk, y) for lk, y in zip(df.lake, df.year)]
    df["w"] = _weights(df)
    train_years = C.load_config()["train_years"]
    cut_log = {}
    # tuned index cutoffs: per evaluated lake, from other lakes' training-year points
    for idx, sign, col in (("ndwi", 1, "ndwi"), ("mndwi", 1, "mndwi"), ("vv", -1, "vv_db")):
        if col not in df:
            continue
        df[f"pred_{idx}_tuned"] = np.nan
        for lake in df.lake.unique():
            dev = df[(df.lake != lake) & df.year.isin(train_years) & df[col].notna()]
            t, f1 = indices.tune_cutoff(sign * dev[col].values, dev.label.values, dev.w.values,
                                        grid=np.linspace(5, 35, 301) if idx == "vv" else None)  # grid over -VV dB
            m = df.lake == lake
            df.loc[m, f"pred_{idx}_tuned"] = (sign * df.loc[m, col] > t).astype(float)
            df.loc[m & df[col].isna(), f"pred_{idx}_tuned"] = np.nan
            cut_log[f"{idx}:{lake}"] = {"cutoff": sign * t, "dev_f1": f1, "n_dev": len(dev)}
        ocol = f"{idx}_otsu_cut"
        if ocol in df:
            df[f"pred_{idx}_otsu"] = (sign * df[col] > df[ocol]).astype(float)
            df.loc[df[col].isna(), f"pred_{idx}_otsu"] = np.nan
    for pcol in [c for c in df.columns if c.startswith("p_") and not c.endswith("_std")]:
        df[f"pred_{pcol[2:]}"] = (df[pcol] > 0.5).astype(float)
        df.loc[df[pcol].isna(), f"pred_{pcol[2:]}"] = np.nan
    for k in labels.LABEL_KINDS:
        df[f"pred_weak_{k}"] = df[f"weak_{k}"]
    methods = [c[5:] for c in df.columns if c.startswith("pred_")]
    rows = []
    groups = {g: df[df.group == g] for g in GROUPS}
    groups["all_heldout"] = df[df.group.notna()]
    for g, d in groups.items():
        for m in methods:
            dd = d[d[f"pred_{m}"].notna()]
            if len(dd) < 20:
                continue
            e = bootstrap(dd, f"pred_{m}", n_boot=n_boot)
            e.update({"group": g, "method": m, "coverage": len(dd) / max(len(d), 1)})
            rows.append(e)
    res = pd.DataFrame(rows)
    # weak-label noise: all points (including training scenes)
    noise = []
    for k in labels.LABEL_KINDS:
        dd = df[df[f"weak_{k}"].notna()]
        e = bootstrap(dd, f"weak_{k}", n_boot=n_boot)
        e.update({"label": k, "coverage": len(dd) / len(df)})
        noise.append(e)
    noise = pd.DataFrame(noise)
    # calibration on held-out points
    cal = {}
    h = df[df.group.notna()]
    for name in ("p_ens_mean", "p_mc_mean", "p_single", "p_s1"):
        if name in h:
            hh = h[h[name].notna()]
            e, table = ece(hh[name].values, hh.label.values, hh.w.values)
            cal[name] = {"ece": e, "table": table.to_dict("records"), "n": len(hh)}
            if name != "p_s1":
                std = hh[name.replace("_mean", "_std")] if name.endswith("_mean") else None
                if std is not None:
                    err = (hh[name] > 0.5).astype(int) != hh.label
                    cal[name]["auroc_std_vs_error"] = _auroc(std.values, err.values)
                    cal[name]["mean_std_correct"] = float(std[~err].mean())
                    cal[name]["mean_std_wrong"] = float(std[err].mean())
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "point_metrics.csv", index=False)
    noise.to_csv(out_dir / "weak_label_noise.csv", index=False)
    (out_dir / "calibration.json").write_text(json.dumps(cal, indent=1, default=float))
    (out_dir / "cutoffs.json").write_text(json.dumps(cut_log, indent=1))
    df.to_csv(out_dir / "points_scored.csv", index=False)
    return {"metrics": res, "noise": noise, "calibration": cal, "cutoffs": cut_log}


def _auroc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


PAIRS = [  # (a, b, question)
    ("ens_mean", "mndwi_tuned", "S2 U-Net ensemble vs tuned MNDWI"),
    ("ens_mean", "ndwi_tuned", "S2 U-Net ensemble vs tuned NDWI"),
    ("ens_mean", "ndwi_otsu", "S2 U-Net ensemble vs per-scene Otsu on NDWI"),
    ("single", "mndwi_tuned", "S2 U-Net (one seed) vs tuned MNDWI"),
    ("ens_mean", "weak_scl_jrc", "S2 U-Net vs its own training label"),
    ("single", "rand_init", "MoCo pretraining vs random init (Travis fold)"),
    ("single", "jrc50_label", "per-scene SCL labels vs static JRC labels (Travis fold)"),
    ("s1", "vv_tuned", "S1 U-Net vs tuned VV threshold"),
    ("s1", "vv_otsu", "S1 U-Net vs per-scene Otsu on VV"),
    ("s1", "ens_mean", "S1 U-Net vs S2 U-Net (same points)"),
]


def paired_table(n_boot: int = 1000, out_dir=None) -> pd.DataFrame:
    out_dir = C.RESULTS_DIR if out_dir is None else out_dir
    df = pd.read_csv(out_dir / "points_scored.csv")
    rows = []
    for g in ["all_heldout", *GROUPS]:
        d = df[df.group.notna()] if g == "all_heldout" else df[df.group == g]
        for a, b, q in PAIRS:
            ca, cb = f"pred_{a}", f"pred_{b}"
            if ca not in d or cb not in d or (d[ca].notna() & d[cb].notna()).sum() < 20:
                continue
            rows.append({"group": g, "a": a, "b": b, "question": q,
                         **paired_bootstrap(d, ca, cb, n_boot=n_boot)})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "paired_differences.csv", index=False)
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="CSV with pid,label (1 water, 0 land, -1 unsure) that "
                    "replaces the shipped labels, e.g. labels/hand_points_human.csv")
    ap.add_argument("--rescore", action="store_true",
                    help="reuse data/outputs/point_predictions.csv (no GPU, no downloads)")
    ap.add_argument("--fill-shipped", action="store_true",
                    help="with --labels: points missing from the file keep the shipped label")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    if a.rescore or a.labels:
        if not POINT_PRED.exists():
            predict_label_scenes()
        df = pd.read_csv(POINT_PRED)
    else:
        df = predict_label_scenes()
    out_dir = C.RESULTS_DIR
    labels_csv = a.labels or (str(PRIMARY_LABELS) if PRIMARY_LABELS.exists() else None)
    if labels_csv and not a.labels:
        # primary reference: the human labels; the shipped AI labels are scored alongside
        ai = score(df.copy(), n_boot=a.n_boot, out_dir=C.RESULTS_DIR / "ai_labels")
        paired_table(out_dir=C.RESULTS_DIR / "ai_labels")
        print("AI labels ->", C.RESULTS_DIR / "ai_labels", f"({len(ai['metrics'])} rows)")
    if labels_csv:
        lab = pd.read_csv(labels_csv)[["pid", "label"]]
        n_h = len(lab)
        if a.fill_shipped or not a.labels:
            # partial file: points the human skipped keep the shipped label, so every stratum
            # stays in the estimate (a shoreline-only file would otherwise score only stratum B)
            df = df.merge(lab.rename(columns={"label": "label_h"}), on="pid", how="left")
            df["label"] = df["label_h"].fillna(df["label"]).astype(int)
            df = df.drop(columns=["label_h"])
        else:
            df = df.drop(columns=["label"]).merge(lab, on="pid", how="inner")
        out_dir = C.RESULTS_DIR / "human_labels" if a.labels else C.RESULTS_DIR
        print(f"scoring against {labels_csv} ({n_h} human labels"
              f"{', rest shipped' if a.fill_shipped else ''}): {len(df)} points -> {out_dir}")
    out = score(df, n_boot=a.n_boot, out_dir=out_dir)
    paired_table(out_dir=out_dir)
    m = out["metrics"]
    print(m[m.group == "all_heldout"][["method", "iou", "iou_lo", "iou_hi", "f1", "n"]]
          .sort_values("iou", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
