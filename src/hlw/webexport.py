"""Build data for the static web app (web/index.html).

web/data/app.json            per-lake series (TWDB level/area, mapped areas) and metrics
web/data/<lake>/<date>_rgb.jpg   true-colour frame (12 per dynamic lake, 6 per small lake), cropped to the lake zone
web/data/<lake>/<date>_unc.png   overlay: water outline + shoreline uncertainty (ensemble std)

Also exports the ten small "pseudo-ungauged" transfer reservoirs (hlw.transfer) as a second group.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from scipy import ndimage

from . import aoi, levels, stac
from . import config as C
from .analyze import load_timeline
from .timeline import TL_DIR, _fetch_day

WEB_DATA = C.WEB_DIR / "data"
FRAMES_PER_LAKE = 12   # dynamic lakes
FRAMES_SMALL = 6       # pass-through lakes (near-constant level)
FRAMES_TRANSFER = 6    # small transfer reservoirs
MAX_W = 1400
WATER_RGB = (42, 120, 214)      # categorical slot 1 (blue)
UNC_RGB = (235, 104, 52)        # categorical slot 2 (orange)


def _crop_box(zone):
    ys, xs = np.nonzero(zone)
    pad = 20
    return (max(ys.min() - pad, 0), min(ys.max() + pad, zone.shape[0]),
            max(xs.min() - pad, 0), min(xs.max() + pad, zone.shape[1]))


def choose_frames(df: pd.DataFrame, n: int) -> list[str]:
    s2 = df[(df.sensor == "s2") & (df.kind == "clear")].dropna(subset=["level"]).sort_values("date")
    if not len(s2):
        return []
    picks = {s2.loc[s2.level.idxmin()].date, s2.loc[s2.level.idxmax()].date}
    for t in pd.date_range(s2.date.min(), s2.date.max(), periods=n - 2):
        picks.add(s2.iloc[(s2.date - t).abs().argmin()].date)
    return sorted(d.strftime("%Y-%m-%d") for d in picks)


def _rgb(refl, box, valid):
    y0, y1, x0, x1 = box
    rgb = refl[[3, 2, 1], y0:y1, x0:x1].astype(np.float32)  # B04, B03, B02
    v = valid[y0:y1, x0:x1]
    lo = np.array([np.percentile(c[v], 1) for c in rgb])
    hi = np.array([np.percentile(c[v], 99.5) for c in rgb])
    x = np.clip((rgb - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1) ** 0.85
    out = (x * 255).astype(np.uint8).transpose(1, 2, 0)
    out[~v] = (out[~v] * 0.35).astype(np.uint8)  # dim outside the counted zone
    return out


def _overlay(prob, std, box, zone):
    y0, y1, x0, x1 = box
    p, s, z = prob[y0:y1, x0:x1], std[y0:y1, x0:x1], zone[y0:y1, x0:x1]
    water = p > 0.5
    edge = water ^ ndimage.binary_erosion(water, iterations=1)
    rgba = np.zeros(p.shape + (4,), np.uint8)
    # uncertainty: alpha grows with ensemble std (0.05 -> faint, >=0.3 -> opaque)
    a = np.clip((s - 0.05) / 0.25, 0, 1) * z
    rgba[..., :3] = UNC_RGB
    rgba[..., 3] = (a * 230).astype(np.uint8)
    rgba[edge & z] = (*WATER_RGB, 255)
    return rgba


def _resize(img: Image.Image) -> Image.Image:
    if img.width <= MAX_W:
        return img
    h = round(img.height * MAX_W / img.width)
    return img.resize((MAX_W, h), Image.LANCZOS)


def _read_on_grid(tif, transform, shape):
    """Read a timeline raster onto the lake's full zone grid (transfer rasters may cover only
    the zone's bounding box)."""
    out = np.full((2,) + shape, 255, np.uint8)
    with rasterio.open(tif) as src:
        q = src.read()
        c0 = round((src.transform.c - transform.c) / transform.a)
        r0 = round((src.transform.f - transform.f) / transform.e)
    out[:, r0:r0 + q.shape[1], c0:c0 + q.shape[2]] = q
    return out


def export_lake(lake: str, df: pd.DataFrame, n_frames: int = FRAMES_PER_LAKE,
                tl_dir=TL_DIR) -> dict:
    zone, transform, crs = aoi.load_zone(lake)
    gb = stac.geobox(transform, zone.shape, crs)
    box = _crop_box(zone)
    out_dir = WEB_DATA / lake
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for date in choose_frames(df, n_frames):
        tif = tl_dir / lake / f"s2_{date}.tif"
        rgb_f, unc_f = out_dir / f"{date}_rgb.jpg", out_dir / f"{date}_unc.png"
        if not (rgb_f.exists() and unc_f.exists()):
            q = _read_on_grid(tif, transform, zone.shape)
            prob = np.where(q[0] == 255, 0, q[0] / 250.0)
            std = np.where(q[1] == 255, 0, q[1] / 250.0 / 2)
            s = stac.load_s2(_fetch_day("sentinel-2-l2a", lake, date), gb)
            valid = zone & (s["refl"][2] > 0)
            _resize(Image.fromarray(_rgb(s["refl"], box, valid))).save(rgb_f, quality=82)
            ov = Image.fromarray(_overlay(prob, std, box, zone), "RGBA")
            ov = ov.resize(_resize(Image.new("L", ov.size)).size, Image.NEAREST)
            ov.save(unc_f, optimize=True)
        row = df[(df.sensor == "s2") & (df.date == pd.Timestamp(date))].iloc[0]
        frames.append({"date": date, "level": _f(row.level), "twdb_area": _f(row.twdb_area),
                       "area": _f(row.area_model), "area_std": _f(row.area_model_member_std),
                       "rgb": f"data/{lake}/{date}_rgb.jpg", "unc": f"data/{lake}/{date}_unc.png"})
        print("frame", lake, date, flush=True)
    return {"frames": frames, "aspect": (box[1] - box[0]) / (box[3] - box[2])}


def _f(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def series(lake: str, df: pd.DataFrame) -> dict:
    lv = levels.load(lake)
    wk = lv[["water_level", "surface_area"]].resample("W").mean().dropna(how="all")
    s2 = df[(df.sensor == "s2") & (df.kind == "clear")].sort_values("date")
    s1 = df[(df.sensor == "s1") & (df.valid_frac > 0.95)].sort_values("date")
    return {
        "twdb": [[d.strftime("%Y-%m-%d"), _f(r.water_level, 2), _f(r.surface_area, 0)]
                 for d, r in wk.iterrows()],
        "s2": [[r.date.strftime("%Y-%m-%d"), _f(r.area_model, 0), _f(r.area_model_member_std, 0),
                _f(r.area_mndwi, 0), _f(r.level, 2), _f(getattr(r, "area_ndwi", np.nan), 0)]
               for r in s2.itertuples()],
        "s1": [[r.date.strftime("%Y-%m-%d"), _f(r.area_model, 0), _f(r.level, 2)]
               for r in s1.itertuples()],
    }


TRANSFER_NOTE = ("A small reservoir the models never saw in training. Neither the maps nor any "
                 "cutoff used this lake's gauge; the TWDB record is shown only to score them. "
                 "NDWI, MNDWI and VV use cutoffs tuned on the Highland Lakes.")


def transfer_frame(key: str) -> pd.DataFrame:
    """A transfer lake's timeline in the column layout of analyze.load_timeline."""
    from . import transfer

    df = transfer.joined(key).rename(columns={"area_mndwi_hl": "area_mndwi",
                                              "area_ndwi_hl": "area_ndwi",
                                              "area_vv_hl": "area_vv"})
    return df


def transfer_entries(area_rows: list) -> dict:
    from . import transfer
    from .analyze import area_table

    out = {}
    for key, L in transfer.lakes().items():
        if not (transfer.TL_DIR / f"{key}.csv").exists():
            continue
        df = transfer_frame(key)
        rows = area_table(key, df)
        area_rows += rows
        entry = {"name": L.name, "dynamic": True, "group": "transfer", "note": TRANSFER_NOTE,
                 **series(key, df), "metrics": pd.DataFrame(rows).round(3).to_dict("records")}
        entry.update(export_lake(key, df, FRAMES_TRANSFER, tl_dir=transfer.TL_DIR))
        out[key] = entry
    return out


def main():
    WEB_DATA.mkdir(parents=True, exist_ok=True)
    app = {"lakes": {}, "generated": pd.Timestamp.now().strftime("%Y-%m-%d")}
    area = pd.read_csv(C.RESULTS_DIR / "area_vs_level.csv") \
        if (C.RESULTS_DIR / "area_vs_level.csv").exists() else pd.DataFrame()
    pm = C.RESULTS_DIR / "point_metrics.csv"
    app["point_metrics"] = pd.read_csv(pm).round(3).to_dict("records") if pm.exists() else []
    for lake, L in C.lakes().items():
        if not (TL_DIR / f"{lake}.csv").exists():
            continue
        df = load_timeline(lake)
        entry = {"name": L.name, "dynamic": L.dynamic, "group": "highland", **series(lake, df),
                 "metrics": area[area.lake == lake].round(3).to_dict("records") if len(area) else []}
        entry.update(export_lake(lake, df, FRAMES_PER_LAKE if L.dynamic else FRAMES_SMALL))
        app["lakes"][lake] = entry
    app["lakes"].update(transfer_entries([]))
    geo = C.AOI_DIR / "lakes.geojson"
    if geo.exists():
        app["outlines"] = json.loads(geo.read_text())
    (WEB_DATA / "app.json").write_text(json.dumps(app, separators=(",", ":")))
    print("wrote", WEB_DATA / "app.json")


if __name__ == "__main__":
    main()
