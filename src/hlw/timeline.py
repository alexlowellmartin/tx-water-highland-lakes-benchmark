"""Map every usable date and compare mapped area with the official record.

S2: clear days (>=99 % of zone valid per SCL), at most 2 per month for dynamic lakes and 1 per
    month for pass-through lakes. Also a sample of cloudy days (20-90 % valid) to show what
    optical mapping does under cloud.
S1: every S1 RTC day.
Each lake is mapped only by the fold whose model never saw that lake (small lakes: the
Travis-held-out fold). Imagery is streamed and discarded; probabilities are kept as uint8 for the
web app. Areas are in acres to match TWDB.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from . import aoi, indices, stac
from . import config as C
from .evaluate import fold_for_lake, load_models
from .io import write_raster
from .model import predict_tiled, s1_to_input, s2_to_input
from .scenes import load as load_scenes

TL_DIR = C.OUT_DIR / "timeline"


def _thin(days: pd.DataFrame, per_month: int) -> pd.DataFrame:
    days = days.assign(month=days.date.str[:7])
    out = []
    for _, g in days.groupby("month"):
        g = g.sort_values("date")
        if len(g) <= per_month:
            out.append(g)
        else:
            idx = np.linspace(0, len(g) - 1, per_month).round().astype(int)
            out.append(g.iloc[idx])
    return pd.concat(out)


def s2_dates(lake: str) -> pd.DataFrame:
    df = load_scenes("s2", lake).dropna(subset=["clear_frac"])
    dyn = C.lakes()[lake].dynamic
    clear = _thin(df[df.clear_frac >= 0.99], 2 if dyn else 1).assign(kind="clear")
    cloudy = df[(df.clear_frac >= 0.2) & (df.clear_frac < 0.9)]
    cloudy = cloudy.sample(min(len(cloudy), 30 if dyn else 6), random_state=0).assign(kind="cloudy")
    return pd.concat([clear, cloudy]).sort_values("date")


def _fetch_day(collection, lake, date):
    d = pd.Timestamp(date)
    items = stac.search(collection, aoi.zone_bbox_ll(lake), date,
                        (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    return stac.group_by_day(items).get(date)


def run_lake(lake: str, sensors=("s2", "s1"), mndwi_cut: float = 0.0, vv_cut: float = -18.0,
             prefetch: int = 3):
    fold = fold_for_lake(lake)
    zone, transform, crs = aoi.load_zone(lake)
    gb = stac.geobox(transform, zone.shape, crs)
    px_acres = (transform.a * -transform.e) / C.ACRE_M2
    out_csv = TL_DIR / f"{lake}.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["sensor", "date"])
    have = set(zip(done.sensor, done.date))
    rows = done.to_dict("records")
    for sensor in sensors:
        models = load_models(sensor, fold, seeds=(0, 1, 2) if sensor == "s2" else (0,))
        if not models:
            print("no models for", sensor, fold)
            continue
        if sensor == "s2":
            dates = s2_dates(lake)
            todo = [(r.date, r.kind, r.clear_frac) for r in dates.itertuples()
                    if ("s2", r.date) not in have]
        else:
            s1 = load_scenes("s1", lake)
            todo = [(d, "all", np.nan) for d in s1.date if ("s1", d) not in have]

        def load(job, sensor=sensor):
            date = job[0]
            try:
                items = _fetch_day("sentinel-2-l2a" if sensor == "s2" else "sentinel-1-rtc",
                                   lake, date)
                return job, (stac.load_s2(items, gb) if sensor == "s2" else stac.load_s1(items, gb))
            except Exception as e:  # noqa: BLE001
                return job, e

        with ThreadPoolExecutor(prefetch) as ex:
            for (date, kind, clear), data in ex.map(load, todo):
                if isinstance(data, Exception):
                    print("FAILED", sensor, lake, date, str(data)[:120], flush=True)
                    continue
                row = {"sensor": sensor, "date": date, "kind": kind, "clear_frac": clear}
                if sensor == "s2":
                    refl, scl = data["refl"], data["scl"]
                    valid = zone & (refl[2] > 0)
                    cloud = np.isin(scl, C.SCL_INVALID) & zone
                    probs = [predict_tiled(m, s2_to_input(refl))[0] for m in models]
                    mn = indices.mndwi(refl)
                    row["area_mndwi"] = float(((mn > mndwi_cut) & valid & ~cloud).sum() * px_acres)
                    row["area_scl"] = float(((scl == C.SCL_WATER) & zone).sum() * px_acres)
                    row["valid_frac"] = float((valid & ~cloud).sum() / zone.sum())
                else:
                    valid = zone & np.isfinite(data[0])
                    probs = [predict_tiled(m, s1_to_input(data))[0] for m in models]
                    row["area_vv"] = float(((data[0] < vv_cut) & valid).sum() * px_acres)
                    row["valid_frac"] = float(valid.sum() / zone.sum())
                p = np.mean(probs, 0)
                member_areas = [float(((q > 0.5) & valid).sum() * px_acres) for q in probs]
                row["area_model"] = float(((p > 0.5) & valid).sum() * px_acres)
                row["area_model_expected"] = float((p * valid).sum() * px_acres)
                row["area_model_member_std"] = float(np.std(member_areas))
                if sensor == "s2":
                    # what a naive optical map reports with clouds left in place vs masked
                    row["area_model_cloudmasked"] = float(((p > 0.5) & valid & ~cloud).sum()
                                                          * px_acres)
                rows.append(row)
                std = np.std(probs, 0) if len(probs) > 1 else np.zeros_like(p)
                q = np.stack([np.round(p * 250), np.round(np.clip(std * 2, 0, 1) * 250)])
                q[:, ~valid] = 255
                write_raster(TL_DIR / lake / f"{sensor}_{date}.tif", q.astype(np.uint8),
                             transform, crs, nodata=255)
                if len(rows) % 10 == 0:
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(sensor, lake, date, kind, round(row["area_model"]), flush=True)
        torch.cuda.empty_cache()
    pd.DataFrame(rows).sort_values(["sensor", "date"]).to_csv(out_csv, index=False)
    return out_csv


def fix_vv(lake: str, vv_cut: float, prefetch: int = 3):
    """Recompute only the VV-threshold area for S1 rows already in the timeline (used when the
    VV cutoff changes; the model areas do not depend on it)."""
    zone, transform, crs = aoi.load_zone(lake)
    gb = stac.geobox(transform, zone.shape, crs)
    px_acres = (transform.a * -transform.e) / C.ACRE_M2
    out_csv = TL_DIR / f"{lake}.csv"
    df = pd.read_csv(out_csv)
    dates = list(df[df.sensor == "s1"].date)

    def load(date):
        try:
            return date, stac.load_s1(_fetch_day("sentinel-1-rtc", lake, date), gb)
        except Exception as e:  # noqa: BLE001
            return date, e

    vals = {}
    with ThreadPoolExecutor(prefetch) as ex:
        for date, data in ex.map(load, dates):
            if isinstance(data, Exception):
                print("FAILED vv", lake, date, str(data)[:120], flush=True)
                continue
            valid = zone & np.isfinite(data[0])
            vals[date] = float(((data[0] < vv_cut) & valid).sum() * px_acres)
    m = (df.sensor == "s1")
    df.loc[m, "area_vv"] = df.loc[m, "date"].map(vals)
    df["vv_cut"] = np.where(m, vv_cut, np.nan)
    df.to_csv(out_csv, index=False)
    print("fixed vv", lake, len(vals), "/", len(dates), "cut", vv_cut, flush=True)


INDEX_BANDS = ["B03", "B08", "B11"]


def fix_index(lake: str, cuts: dict, prefetch: int = 4):
    """Add NDWI areas and recompute the MNDWI area for the S2 rows already in the timeline, with
    the current cutoffs (results/cutoffs.json). Only B03/B08/B11 + SCL are read; the model areas
    do not depend on any cutoff. The previous MNDWI area is kept as ``area_mndwi_ai`` (cutoff
    tuned on the AI labels)."""
    from .transfer import bounded_map

    zone, transform, crs = aoi.load_zone(lake)
    gb = stac.geobox(transform, zone.shape, crs)
    px_acres = (transform.a * -transform.e) / C.ACRE_M2
    out_csv = TL_DIR / f"{lake}.csv"
    df = pd.read_csv(out_csv)
    nd_cut, mn_cut = cuts[f"ndwi:{lake}"]["cutoff"], cuts[f"mndwi:{lake}"]["cutoff"]
    dates = list(df[df.sensor == "s2"].date)

    def load(date):
        try:
            items = _fetch_day("sentinel-2-l2a", lake, date)
            return date, stac.load_s2(items, gb, bands=INDEX_BANDS)
        except Exception as e:  # noqa: BLE001
            return date, e

    vals = {}
    for date, data in bounded_map(load, dates, prefetch):
        if isinstance(data, Exception):
            print("FAILED index", lake, date, str(data)[:120], flush=True)
            continue
        r, scl = data["refl"], data["scl"]
        ok = zone & (r[0] > 0) & ~np.isin(scl, C.SCL_INVALID)
        with np.errstate(divide="ignore", invalid="ignore"):
            g, n, sw = (r[i].astype(np.float32) for i in range(3))
            nd, mn = (g - n) / (g + n), (g - sw) / (g + sw)
        v = {"area_ndwi": float(((nd > nd_cut) & ok).sum() * px_acres),
             "area_ndwi_zero": float(((nd > 0) & ok).sum() * px_acres),
             "area_mndwi": float(((mn > mn_cut) & ok).sum() * px_acres)}
        fin = ok & np.isfinite(nd)
        v["area_ndwi_otsu"] = (float(((nd > indices.otsu_cutoff(nd[fin])) & ok).sum() * px_acres)
                               if fin.sum() > 100 else np.nan)
        vals[date] = v
    m = df.sensor == "s2"
    if "area_mndwi_ai" not in df:
        df["area_mndwi_ai"] = df["area_mndwi"]
    for col in ("area_ndwi", "area_ndwi_zero", "area_ndwi_otsu", "area_mndwi"):
        df.loc[m, col] = df.loc[m, "date"].map(lambda d, c=col: vals.get(d, {}).get(c, np.nan))
    df["ndwi_cut"] = np.where(m, nd_cut, np.nan)
    df["mndwi_cut"] = np.where(m, mn_cut, np.nan)
    df.to_csv(out_csv, index=False)
    print("fixed index", lake, len(vals), "/", len(dates), "ndwi", nd_cut, "mndwi", mn_cut,
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lakes", nargs="*", default=list(C.lakes()))
    ap.add_argument("--sensors", nargs="*", default=["s2", "s1"])
    ap.add_argument("--mndwi-cut", type=float, default=None)
    ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--fix-vv", action="store_true", help="only recompute S1 VV-threshold areas")
    ap.add_argument("--fix-index", action="store_true",
                    help="only add NDWI and recompute MNDWI areas with the current cutoffs")
    a = ap.parse_args()
    import json

    cuts_path = C.RESULTS_DIR / "cutoffs.json"
    cuts = json.loads(cuts_path.read_text()) if cuts_path.exists() else {}
    TL_DIR.mkdir(parents=True, exist_ok=True)
    for lake in a.lakes:
        c = a.mndwi_cut if a.mndwi_cut is not None else             cuts.get(f"mndwi:{lake}", {}).get("cutoff", 0.0)
        v = cuts.get(f"vv:{lake}", {}).get("cutoff", -18.0)
        if a.fix_index:
            fix_index(lake, cuts, prefetch=a.prefetch)
        elif a.fix_vv:
            fix_vv(lake, v, prefetch=a.prefetch)
        else:
            run_lake(lake, a.sensors, mndwi_cut=c, vv_cut=v, prefetch=a.prefetch)


if __name__ == "__main__":
    main()
