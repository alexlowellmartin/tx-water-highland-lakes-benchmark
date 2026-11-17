"""Transfer test on small "pseudo-ungauged" reservoirs (configs/transfer.yaml).

The shipped models (Travis-held-out fold: 3-seed S2 ensemble, S1 U-Net) were trained only on
Highland Lakes imagery. Here they map ten small West/Central Texas reservoirs they have never
seen. Every baseline is also configured without touching the new lakes:

- MNDWI, fixed cutoff: the median of the per-lake cutoffs tuned on the Highland Lakes
  (``mndwi_hl``), plus the textbook 0 (``mndwi_zero``) and a per-scene Otsu cutoff
  (``mndwi_otsu``, the usual unsupervised choice).
- SCL: the Sentinel-2 L2A scene classification "water" class (free with the imagery).
- VV: median Highland Lakes VV cutoff (``vv_hl``) and per-scene Otsu (``vv_otsu``).

The TWDB record is used only by ``analyze``, to score the maps afterwards.

Stages: ``zones levels scan timeline analyze`` (default: all, resumable).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml
from affine import Affine
from pyproj import Transformer

from . import aoi, indices, levels, scenes, stac
from . import config as C
from .io import write_raster

CFG_PATH = C.ROOT / "configs" / "transfer.yaml"
OUT = C.RESULTS_DIR / "transfer"
TL_DIR = C.OUT_DIR / "timeline_transfer"
FOLD = "holdout_travis"  # the shipped models


@lru_cache
def tcfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def box(point, radius_km, exclude_km=None):
    """Square lon/lat ring of half-width radius_km around point, optionally minus a rectangle
    given as [dx0, dy0, dx1, dy1] km offsets from point."""
    from shapely.geometry import box as sbox

    to_utm = Transformer.from_crs("EPSG:4326", C.load_config()["crs"], always_xy=True)
    x, y = to_utm.transform(*point)
    r = radius_km * 1000
    poly = sbox(x - r, y - r, x + r, y + r)
    if exclude_km:
        dx0, dy0, dx1, dy1 = (v * 1000 for v in exclude_km)
        poly = poly.difference(sbox(x + dx0, y + dy0, x + dx1, y + dy1))
    return [list(to_utm.transform(a, b, direction="INVERSE")) for a, b in poly.exterior.coords]


def lakes() -> dict[str, C.Lake]:
    out = {}
    for k, v in tcfg()["lakes"].items():
        clip = v.get("clip") or box(v["point"], v["radius_km"], v.get("exclude_km"))
        out[k] = C.Lake(key=k, name=v["name"], twdb=v["twdb"], seed=tuple(v["point"]),
                        dynamic=True, clip=clip)
    return out


# ---- stages ------------------------------------------------------------------------------------
def build_zones():
    cfg = {**C.load_config(), "zone_buffer_m": tcfg()["zone_buffer_m"]}
    rows = []
    for lake in lakes().values():
        info = aoi.build_zone(lake, cfg)
        rows.append(info)
        print(info, flush=True)
    return pd.DataFrame(rows)


def fetch_levels():
    C.LEVELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = C.load_config()
    for lake in lakes().values():
        df = levels.clean_levels(levels.download(lake)).loc[cfg["study_start"]: cfg["study_end"]]
        df.to_csv(C.LEVELS_DIR / f"{lake.key}.csv")
        print(f"{lake.key}: {len(df)} days, area {df.surface_area.min():.0f}-"
              f"{df.surface_area.max():.0f} ac", flush=True)


def scan(keys=None, workers=8):
    scenes.SCENES_DIR.mkdir(parents=True, exist_ok=True)
    for key in keys or lakes():
        p1, p2 = scenes.SCENES_DIR / f"s1_{key}.csv", scenes.SCENES_DIR / f"s2_{key}.csv"
        if not p1.exists():
            scenes.scan_s1(key).to_csv(p1, index=False)
        if not p2.exists():
            s2 = scenes.scan_s2(key, workers=workers)
            s2.to_csv(p2, index=False)
        s2 = pd.read_csv(p2)
        print(f"{key}: {len(s2)} S2 days, {(s2.clear_frac >= 0.99).sum()} clear, "
              f"{len(pd.read_csv(p1))} S1 days", flush=True)


def hl_cutoffs() -> dict:
    cuts = json.loads((C.RESULTS_DIR / "cutoffs.json").read_text())
    med = {idx: float(np.median([v["cutoff"] for k, v in cuts.items() if k.startswith(idx + ":")]))
           for idx in ("ndwi", "mndwi", "vv")}
    return med


def s2_dates(key: str) -> pd.DataFrame:
    from .timeline import _thin

    df = scenes.load("s2", key).dropna(subset=["clear_frac"])
    clear = _thin(df[df.clear_frac >= 0.99], tcfg()["s2_per_month"]).assign(kind="clear")
    cloudy = df[(df.clear_frac >= 0.2) & (df.clear_frac < 0.9)]
    cloudy = cloudy.sample(min(len(cloudy), tcfg()["s2_cloudy"]), random_state=0)
    return pd.concat([clear, cloudy.assign(kind="cloudy")]).sort_values("date")


def bounded_map(fn, items, n):
    """Like ThreadPoolExecutor(n).map, but keeps at most n jobs in flight, so downloaded scenes
    can't pile up in memory when the consumer (GPU inference) is slower than the downloads."""
    from collections import deque
    from itertools import islice

    items = iter(items)
    with ThreadPoolExecutor(n) as ex:
        # islice, not zip(items, range(n)): zip would consume and drop one extra item
        q = deque(ex.submit(fn, it) for it in islice(items, n))
        while q:
            yield q.popleft().result()
            for it in items:
                q.append(ex.submit(fn, it))
                break


def s1_dates(key: str) -> list[str]:
    from .timeline import _thin

    s1 = scenes.load("s1", key)
    return list(_thin(s1[["date"]], tcfg()["s1_per_month"]).date)


def _otsu_area(x, valid, px, below=False):
    v = x[valid & np.isfinite(x)]
    if v.size < 100:
        return np.nan, np.nan
    t = indices.otsu_cutoff(v)
    m = (x < t) if below else (x > t)
    return float((m & valid).sum() * px), t


def _cropped(key: str):
    """Zone, transform, crs, geobox and acres per pixel on the zone's bounding box (the clip box
    is generous; at Medina the zone covers ~5 % of it)."""
    zone, transform, crs = aoi.load_zone(key)
    rows_, cols_ = np.nonzero(zone)
    pad = 16
    r0, c0 = max(rows_.min() - pad, 0), max(cols_.min() - pad, 0)
    r1, c1 = min(rows_.max() + pad + 1, zone.shape[0]), min(cols_.max() + pad + 1, zone.shape[1])
    zone = zone[r0:r1, c0:c1]
    transform = transform * Affine.translation(c0, r0)
    return zone, transform, crs, stac.geobox(transform, zone.shape, crs),         (transform.a * -transform.e) / C.ACRE_M2


def fix_cutoffs(key: str, prefetch: int = 4):
    """Recompute the cutoff-dependent baseline areas with the current Highland Lakes cutoffs
    (results/cutoffs.json, tuned on the primary hand labels) and add NDWI. Previous values are
    kept as ``*_ai`` (cutoffs tuned on the AI labels). Model areas are unaffected."""
    from .timeline import INDEX_BANDS, _fetch_day

    cuts = hl_cutoffs()
    zone, _, _, gb, px = _cropped(key)
    out_csv = TL_DIR / f"{key}.csv"
    df = pd.read_csv(out_csv)
    for col in ("area_mndwi_hl", "area_vv_hl"):
        if f"{col}_ai" not in df:
            df[f"{col}_ai"] = df[col]

    def load(job):
        sensor, date = job
        try:
            if sensor == "s2":
                items = _fetch_day("sentinel-2-l2a", key, date)
                return job, stac.load_s2(items, gb, bands=INDEX_BANDS)
            return job, stac.load_s1(_fetch_day("sentinel-1-rtc", key, date), gb)
        except Exception as e:  # noqa: BLE001
            return job, e

    vals = {}
    for (sensor, date), data in bounded_map(load, list(zip(df.sensor, df.date)), prefetch):
        if isinstance(data, Exception):
            print("FAILED", key, sensor, date, str(data)[:100], flush=True)
            continue
        if sensor == "s2":
            r, scl = data["refl"], data["scl"]
            ok = zone & (r[0] > 0) & ~np.isin(scl, C.SCL_INVALID)
            with np.errstate(divide="ignore", invalid="ignore"):
                g, n, sw = (r[i].astype(np.float32) for i in range(3))
                nd, mn = (g - n) / (g + n), (g - sw) / (g + sw)
            v = {"area_ndwi_hl": float(((nd > cuts["ndwi"]) & ok).sum() * px),
                 "area_ndwi_zero": float(((nd > 0) & ok).sum() * px),
                 "area_mndwi_hl": float(((mn > cuts["mndwi"]) & ok).sum() * px)}
            v["area_ndwi_otsu"], v["ndwi_otsu_cut"] = _otsu_area(nd, ok, px)
        else:
            valid = zone & np.isfinite(data[0])
            v = {"area_vv_hl": float(((data[0] < cuts["vv"]) & valid).sum() * px)}
        vals[(sensor, date)] = v
    for col in ("area_ndwi_hl", "area_ndwi_zero", "area_ndwi_otsu", "ndwi_otsu_cut",
                "area_mndwi_hl", "area_vv_hl"):
        # a failed reload leaves NaN, never the old value under a different cutoff
        df[col] = [vals.get((s_, d), {}).get(col, np.nan) for s_, d in zip(df.sensor, df.date)]
    df.to_csv(out_csv, index=False)
    print("fixed cutoffs", key, len(vals), "/", len(df), cuts, flush=True)


def run_lake(key: str, prefetch: int = 3):
    import torch

    from .evaluate import load_models
    from .model import predict_tiled, s1_to_input, s2_to_input
    from .timeline import _fetch_day

    cuts = hl_cutoffs()
    zone, transform, crs, gb, px = _cropped(key)
    out_csv = TL_DIR / f"{key}.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["sensor", "date"])
    have = set(zip(done.sensor, done.date))
    rows = done.to_dict("records")
    for sensor in ("s2", "s1"):
        models = load_models(sensor, FOLD, seeds=(0, 1, 2) if sensor == "s2" else (0,))
        if sensor == "s2":
            todo = [(r.date, r.kind, r.clear_frac) for r in s2_dates(key).itertuples()
                    if ("s2", r.date) not in have]
        else:
            todo = [(d, "all", np.nan) for d in s1_dates(key) if ("s1", d) not in have]

        def load(job, sensor=sensor):
            try:
                items = _fetch_day("sentinel-2-l2a" if sensor == "s2" else "sentinel-1-rtc",
                                   key, job[0])
                return job, (stac.load_s2(items, gb) if sensor == "s2" else stac.load_s1(items, gb))
            except Exception as e:  # noqa: BLE001
                return job, e

        for (date, kind, clear), data in bounded_map(load, todo, prefetch):
            if isinstance(data, Exception):
                print("FAILED", sensor, key, date, str(data)[:120], flush=True)
                continue
            row = {"sensor": sensor, "date": date, "kind": kind, "clear_frac": clear}
            if sensor == "s2":
                refl, scl = data["refl"], data["scl"]
                valid = zone & (refl[2] > 0)
                cloud = np.isin(scl, C.SCL_INVALID) & zone
                ok = valid & ~cloud
                probs = [predict_tiled(m, s2_to_input(refl))[0] for m in models]
                mn = indices.mndwi(refl)
                row["area_mndwi_hl"] = float(((mn > cuts["mndwi"]) & ok).sum() * px)
                row["area_mndwi_zero"] = float(((mn > 0) & ok).sum() * px)
                row["area_mndwi_otsu"], row["mndwi_otsu_cut"] = _otsu_area(mn, ok, px)
                row["area_scl"] = float(((scl == C.SCL_WATER) & zone).sum() * px)
                row["valid_frac"] = float(ok.sum() / zone.sum())
            else:
                valid = zone & np.isfinite(data[0])
                probs = [predict_tiled(m, s1_to_input(data))[0] for m in models]
                row["area_vv_hl"] = float(((data[0] < cuts["vv"]) & valid).sum() * px)
                row["area_vv_otsu"], row["vv_otsu_cut"] = _otsu_area(data[0], valid, px,
                                                                     below=True)
                row["valid_frac"] = float(valid.sum() / zone.sum())
            p = np.mean(probs, 0)
            row["area_model"] = float(((p > 0.5) & valid).sum() * px)
            row["area_model_expected"] = float((p * valid).sum() * px)
            row["area_model_member_std"] = float(np.std(
                [((q > 0.5) & valid).sum() * px for q in probs]))
            if sensor == "s2":
                row["area_model_cloudmasked"] = float(((p > 0.5) & ok).sum() * px)
            rows.append(row)
            std = np.std(probs, 0) if len(probs) > 1 else np.zeros_like(p)
            q = np.stack([np.round(p * 250), np.round(np.clip(std * 2, 0, 1) * 250)])
            q[:, ~valid] = 255
            write_raster(TL_DIR / key / f"{sensor}_{date}.tif", q.astype(np.uint8),
                         transform, crs, nodata=255)
            if len(rows) % 20 == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(sensor, key, date, kind, round(row["area_model"]), flush=True)
        torch.cuda.empty_cache()
    pd.DataFrame(rows).sort_values(["sensor", "date"]).to_csv(out_csv, index=False)


def timeline(keys=None, prefetch=3, wait=False):
    """Map each lake; with wait=True, wait for each lake's scene scan (run in parallel by another
    process) to appear before mapping it."""
    import time

    TL_DIR.mkdir(parents=True, exist_ok=True)
    for key in keys or lakes():
        while wait and not all((scenes.SCENES_DIR / f"{s}_{key}.csv").exists()
                               for s in ("s1", "s2")):
            time.sleep(30)
        run_lake(key, prefetch)


# ---- analysis ----------------------------------------------------------------------------------
S2_METHODS = {"area_model": "S2 U-Net", "area_ndwi_hl": "NDWI (HL cutoff)",
              "area_ndwi_otsu": "NDWI (Otsu)", "area_mndwi_hl": "MNDWI (HL cutoff)",
              "area_mndwi_zero": "MNDWI (0)", "area_mndwi_otsu": "MNDWI (Otsu)",
              "area_scl": "SCL water"}
S1_METHODS = {"area_model": "S1 U-Net", "area_vv_hl": "VV (HL cutoff)", "area_vv_otsu": "VV (Otsu)"}


def joined(key: str) -> pd.DataFrame:
    """Timeline rows with the same-day TWDB area and level (scoring only)."""
    tl = pd.read_csv(TL_DIR / f"{key}.csv")
    # keep exactly the current date selection (the timeline may hold extra dates mapped under a
    # denser earlier setting; the thinning rules make the current selection a subset)
    keep = {("s2", d) for d in s2_dates(key).date} | {("s1", d) for d in s1_dates(key)}
    tl = tl[[(s, d) in keep for s, d in zip(tl.sensor, tl.date)]].copy()
    tl["date"] = pd.to_datetime(tl.date)
    lv = levels.load(key)
    # S1 passes at ~00:30 UTC are the previous local evening; TWDB days are local
    local = tl.date - pd.to_timedelta(np.where(tl.sensor == "s1", 1, 0), unit="D")
    tl["twdb_area"] = lv["surface_area"].reindex(local).values
    tl["level"] = lv["water_level"].reindex(local).values
    return tl.dropna(subset=["twdb_area"])


def score_series(est: pd.Series, ref: pd.Series) -> dict:
    ok = est.notna() & ref.notna()
    e, r = est[ok].values, ref[ok].values
    if len(e) < 5:
        return {"n": len(e)}
    pct = 100 * (e - r) / r
    fit = np.polyfit(e, r, 1)
    resid = r - np.polyval(fit, e)
    return {"n": len(e), "r": float(np.corrcoef(e, r)[0, 1]),
            "median_abs_pct_err": float(np.median(np.abs(pct))),
            "median_bias_pct": float(np.median(pct)),
            "rmse_after_fit_pct": float(100 * np.sqrt(np.mean(resid ** 2)) / r.mean()),
            "swing_tracked": float((e.max() - e.min()) / max(r.max() - r.min(), 1e-9))}


def metrics_table() -> pd.DataFrame:
    rows = []
    for key, L in lakes().items():
        if not (TL_DIR / f"{key}.csv").exists():
            continue
        df = joined(key)
        amax = levels.load(key)["surface_area"].max()
        for sensor, methods in (("s2", S2_METHODS), ("s1", S1_METHODS)):
            d = df[df.sensor == sensor]
            if sensor == "s2":
                d = d[d.kind == "clear"]
            for col, name in methods.items():
                if col in d:
                    rows.append({"lake": key, "name": L.name, "twdb_max_acres": amax,
                                 "sensor": sensor, "method": name,
                                 **score_series(d[col], d.twdb_area)})
    return pd.DataFrame(rows)


def coverage_table() -> pd.DataFrame:
    """Share of study months with at least one usable map: clear S2 only vs S2 + S1."""
    cfg = C.load_config()
    months = pd.period_range(cfg["study_start"], cfg["study_end"], freq="M")
    rows = []
    for key in lakes():
        if not (TL_DIR / f"{key}.csv").exists():
            continue
        s2 = scenes.load("s2", key)
        s1 = scenes.load("s1", key)
        m2 = set(pd.to_datetime(s2[s2.clear_frac >= 0.99].date).dt.to_period("M"))
        m1 = set(pd.to_datetime(s1.date).dt.to_period("M"))
        rows.append({"lake": key, "months": len(months),
                     "s2_clear_pct": 100 * len(m2 & set(months)) / len(months),
                     "s2_or_s1_pct": 100 * len((m2 | m1) & set(months)) / len(months)})
    return pd.DataFrame(rows)


def cloudy_table() -> pd.DataFrame:
    """Error on the sampled cloudy S2 dates vs an S1 map within 3 days."""
    rows = []
    for key in lakes():
        if not (TL_DIR / f"{key}.csv").exists():
            continue
        df = joined(key)
        cl = df[(df.sensor == "s2") & (df.kind == "cloudy")]
        s1 = df[df.sensor == "s1"].set_index("date")
        for r in cl.itertuples():
            near = s1[(s1.index >= r.date - pd.Timedelta(days=3))
                      & (s1.index <= r.date + pd.Timedelta(days=3))]
            rows.append({"lake": key, "date": r.date, "twdb": r.twdb_area,
                         "s2_cloud_in_pct": 100 * (r.area_model - r.twdb_area) / r.twdb_area,
                         "s2_masked_pct": 100 * (r.area_model_cloudmasked - r.twdb_area)
                         / r.twdb_area,
                         "s1_pct": (100 * (near.area_model.iloc[0] - near.twdb_area.iloc[0])
                                    / near.twdb_area.iloc[0]) if len(near) else np.nan})
    return pd.DataFrame(rows)


def analyze():
    from . import transfer_figs

    OUT.mkdir(parents=True, exist_ok=True)
    m = metrics_table()
    m.to_csv(OUT / "metrics.csv", index=False)
    cov = coverage_table()
    cov.to_csv(OUT / "coverage.csv", index=False)
    cl = cloudy_table()
    cl.to_csv(OUT / "cloudy.csv", index=False)
    summary = {
        "cutoffs_from_highland_lakes": hl_cutoffs(),
        "median_by_method": m.groupby(["sensor", "method"])[
            ["r", "median_abs_pct_err", "median_bias_pct", "rmse_after_fit_pct",
             "swing_tracked"]].median().round(3).reset_index().to_dict("records"),
        "coverage_median": cov[["s2_clear_pct", "s2_or_s1_pct"]].median().round(1).to_dict(),
        "cloudy_median_abs": cl[["s2_cloud_in_pct", "s2_masked_pct", "s1_pct"]].abs().median()
        .round(1).to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    transfer_figs.main()
    pd.set_option("display.width", 200)
    print(m.pivot_table(index="lake", columns="method", values="median_abs_pct_err").round(1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*",
                    default=["zones", "levels", "scan", "timeline", "analyze"])
    ap.add_argument("--lakes", nargs="*")
    ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--wait", action="store_true", help="timeline: wait for parallel scans")
    a = ap.parse_args()
    for s in a.stages:
        if s == "zones":
            build_zones()
        elif s == "levels":
            fetch_levels()
        elif s == "scan":
            scan(a.lakes)
        elif s == "timeline":
            timeline(a.lakes, a.prefetch, a.wait)
        elif s == "analyze":
            analyze()
        elif s == "fix-cutoffs":
            for k in a.lakes or lakes():
                fix_cutoffs(k, a.prefetch)
        else:
            raise SystemExit(f"unknown stage {s}")


if __name__ == "__main__":
    main()
