"""Inference: map surface water for any area and date, or for a local GeoTIFF.

Examples:
  # Lake Travis on a given day, downloading S2 L2A from Planetary Computer
  hlw infer --lake travis --date 2023-08-02 --out travis_2023-08-02.tif
  # any bounding box (lon/lat) and date, with 10 MC-dropout passes for uncertainty
  hlw infer --bbox -98.0 30.38 -97.88 30.45 --date 2025-07-20 --mc 10 --out out.tif
  # Sentinel-1 (works through cloud) -- needs the S1 weights file
  hlw infer --lake buchanan --date 2023-08-15 --sensor s1 --weights weights/hlw_s1_unet_r50.pt
  # a local 12-band L2A GeoTIFF in S2_BANDS order (reflectance*10000, no +1000 offset)
  hlw infer --input scene.tif --out mask.tif

Output GeoTIFF bands: 1 = water probability (0-1), 2 = uncertainty (std of MC samples),
3 = binary water mask (p > 0.5). Area in acres and km2 is printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from . import aoi, stac
from . import config as C
from .io import write_raster
from .model import load_weights, predict_tiled, s1_to_input, s2_to_input

DEFAULT_WEIGHTS = {"s2": C.WEIGHTS_DIR / "hlw_s2_unet_r18.pt",
                   "s1": C.WEIGHTS_DIR / "hlw_s1_unet_r50.pt"}


def grid_for_bbox(bbox, crs="EPSG:32614", res=10.0):
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    xs, ys = zip(*[to_utm(x, y) for x in bbox[::2] for y in bbox[1::2]])
    x0, y1 = np.floor(min(xs) / res) * res, np.ceil(max(ys) / res) * res
    w = int(np.ceil((max(xs) - x0) / res))
    h = int(np.ceil((y1 - min(ys)) / res))
    return from_origin(x0, y1, res, res), (h, w), crs


def run(args) -> Path:
    weights = Path(args.weights or DEFAULT_WEIGHTS[args.sensor])
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    model = load_weights(weights)
    if model.sensor != args.sensor:
        raise SystemExit(f"{weights} is a {model.sensor} model, not {args.sensor}")
    zone = None
    if args.input:
        with rasterio.open(args.input) as src:
            data, transform, crs = src.read(), src.transform, src.crs
        x = s2_to_input(data) if args.sensor == "s2" else s1_to_input(data.astype(np.float32))
        valid = data[0] > 0 if args.sensor == "s2" else np.isfinite(data[0])
    else:
        if args.lake:
            zone, transform, crs = aoi.load_zone(args.lake)
            shape, bbox = zone.shape, aoi.zone_bbox_ll(args.lake)
        else:
            transform, shape, crs = grid_for_bbox(args.bbox)
            bbox = args.bbox
        coll = "sentinel-2-l2a" if args.sensor == "s2" else "sentinel-1-rtc"
        d = pd.Timestamp(args.date)
        items = stac.group_by_day(stac.search(coll, bbox, args.date,
                                              (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
        if args.date not in items:
            raise SystemExit(f"no {coll} acquisition on {args.date}; try a nearby date")
        gb = stac.geobox(transform, shape, crs)
        if args.sensor == "s2":
            s = stac.load_s2(items[args.date], gb)
            x, valid = s2_to_input(s["refl"]), s["refl"][2] > 0
            cloud = np.isin(s["scl"], C.SCL_INVALID)
            print(f"cloud/shadow fraction (SCL): {cloud[valid].mean():.1%}")
        else:
            db = stac.load_s1(items[args.date], gb)
            x, valid = s1_to_input(db), np.isfinite(db[0])
    prob, std = predict_tiled(model, x, mc=args.mc)
    mask = (prob > 0.5) & valid
    out = np.stack([prob, std, mask.astype(np.float32)]).astype(np.float32)
    out[:, ~valid] = np.nan
    write_raster(args.out, out, transform, crs, nodata=np.nan)
    px = abs(transform.a * transform.e)
    area_m2 = (mask & zone).sum() * px if zone is not None else mask.sum() * px
    print(f"water area{' in lake zone' if zone is not None else ''}: "
          f"{area_m2 / C.ACRE_M2:,.0f} acres ({area_m2 / 1e6:.2f} km2) -> {args.out}")
    return Path(args.out)


def add_args(ap: argparse.ArgumentParser):
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--lake", choices=list(C.lakes()))
    src.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    src.add_argument("--input", help="local GeoTIFF")
    ap.add_argument("--date", help="YYYY-MM-DD (required with --lake/--bbox)")
    ap.add_argument("--sensor", default="s2", choices=["s2", "s1"])
    ap.add_argument("--weights")
    ap.add_argument("--mc", type=int, default=0, help="MC-dropout passes (0 = deterministic)")
    ap.add_argument("--out", default="water.tif")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_args(ap)
    a = ap.parse_args(argv)
    if not a.input and not a.date:
        ap.error("--date is required with --lake or --bbox")
    run(a)


if __name__ == "__main__":
    main()
