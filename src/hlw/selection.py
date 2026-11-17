"""Choose which scenes to download for training, testing, hand labelling and S1 pairing.

All choices are deterministic functions of the scene inventory and TWDB levels and are
written to data/scenes/selected.json.

- train/test S2 scenes: per lake and year, K windows across the year, the clearest day in
  each (>=99 % of the lake zone free of cloud/shadow per SCL). Dynamic lakes also get the
  days of minimum and maximum level within each split.
- S1 pairs: an S1 day within +/-2 days of a clear S2 day where the official level changed
  by less than 0.15 ft, so the S2 shoreline is valid for the S1 date.
- label scenes: subset of the above spanning low/median/high water in each split.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from . import config as C
from . import levels
from .scenes import SCENES_DIR, load

SEL_PATH = SCENES_DIR / "selected.json"
K_PER_YEAR = 4
MIN_CLEAR = 0.99
PAIR_DAYS = 2
PAIR_DLEVEL = 0.15


def clear_s2(lake: str) -> pd.DataFrame:
    df = load("s2", lake)
    df = df[df.get("clear_frac", 0) >= MIN_CLEAR].copy()
    df["date"] = pd.to_datetime(df["date"])
    lv = levels.load(lake)["water_level"]
    df["level"] = lv.reindex(df["date"]).values
    return df.dropna(subset=["level"])


def _pick_year(df: pd.DataFrame, year: int, k: int) -> list[str]:
    d = df[df.date.dt.year == year]
    picks = []
    edges = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", periods=k + 1)
    for a, b in itertools.pairwise(edges):
        w = d[(d.date >= a) & (d.date < b)]
        if len(w):
            mid = a + (b - a) / 2
            w = w.assign(score=w.clear_frac - (w.date - mid).abs().dt.days / 1e4)
            picks.append(w.sort_values("score").iloc[-1].date.strftime("%Y-%m-%d"))
    return picks


def s1_pairs(lake: str, s2_days: pd.DataFrame) -> list[tuple[str, str, float]]:
    s1 = load("s1", lake)
    s1d = pd.to_datetime(s1["date"])
    lv = levels.load(lake)["water_level"]
    out = []
    for _, r in s2_days.iterrows():
        dd = (s1d - r.date).dt.days.abs()
        cand = s1d[dd <= PAIR_DAYS]
        best = None
        for d1 in cand:
            dl = abs(lv.get(d1, np.nan) - r.level)
            if np.isfinite(dl) and dl < PAIR_DLEVEL and (best is None or dl < best[2]):
                best = (d1.strftime("%Y-%m-%d"), r.date.strftime("%Y-%m-%d"), float(dl))
        if best:
            out.append(best)
    return out


def build() -> dict:
    cfg = C.load_config()
    sel = {"s2": {}, "label": {}, "s1_pairs": {}}
    for lake, L in C.lakes().items():
        df = clear_s2(lake)
        days = set()
        for split, years in (("train", cfg["train_years"]), ("test", cfg["test_years"])):
            for y in years:
                days |= set(_pick_year(df, y, K_PER_YEAR))
            if L.dynamic:
                part = df[df.date.dt.year.isin(years)]
                days |= {part.loc[part.level.idxmin()].date.strftime("%Y-%m-%d"),
                         part.loc[part.level.idxmax()].date.strftime("%Y-%m-%d")}
        # S1 pairing on all clear S2 days, capped at 6 per year, preferring smallest dlevel
        pairs = s1_pairs(lake, df)
        pdf = pd.DataFrame(pairs, columns=["s1", "s2", "dlevel"])
        if len(pdf):
            pdf["year"] = pdf.s2.str[:4]
            pdf = pdf.sort_values("dlevel").groupby("year").head(6).sort_values("s2")
        sel["s1_pairs"][lake] = pdf[["s1", "s2", "dlevel"]].values.tolist() if len(pdf) else []
        days |= set(pdf.s2) if len(pdf) else set()
        sel["s2"][lake] = sorted(days)
        # label scenes: low / median / high water in each split, prefer days with an S1 pair
        paired = set(pdf.s2) if len(pdf) else set()
        cand = df[df.date.dt.strftime("%Y-%m-%d").isin(days)].copy()
        cand["paired"] = cand.date.dt.strftime("%Y-%m-%d").isin(paired)
        labs = []
        for years in (cfg["train_years"], cfg["test_years"]):
            part = cand[cand.date.dt.year.isin(years)]
            if not len(part):
                continue
            qs = (0.0, 0.5, 1.0) if L.dynamic else (0.5,)
            for q in qs:
                target = part.level.quantile(q)
                p2 = part.assign(score=(part.level - target).abs() - 0.5 * part.paired)
                p2 = p2[~p2.date.dt.strftime("%Y-%m-%d").isin(labs)]
                labs.append(p2.sort_values("score").iloc[0].date.strftime("%Y-%m-%d"))
        sel["label"][lake] = sorted(labs)
    SEL_PATH.write_text(json.dumps(sel, indent=1))
    return sel


def selected() -> dict:
    return json.loads(SEL_PATH.read_text())


def training_scenes(fold: str) -> list[tuple[str, str]]:
    cfg = C.load_config()
    test_lake = cfg["folds"][fold]["test_lake"]
    sel = selected()
    return [(lk, d) for lk, ds in sel["s2"].items() if lk != test_lake
            for d in ds if int(d[:4]) in cfg["train_years"]]


def s1_training_pairs(fold: str) -> list[tuple[str, str, str]]:
    cfg = C.load_config()
    test_lake = cfg["folds"][fold]["test_lake"]
    sel = selected()
    return [(lk, a, b) for lk, ps in sel["s1_pairs"].items() if lk != test_lake
            for a, b, _ in ps if int(b[:4]) in cfg["train_years"]]


def main():
    sel = build()
    for lk in sel["s2"]:
        print(lk, len(sel["s2"][lk]), "S2 scenes,", len(sel["s1_pairs"][lk]), "S1 pairs, label:",
              sel["label"][lk])


if __name__ == "__main__":
    main()
