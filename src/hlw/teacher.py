"""Soft targets for the Sentinel-1 model: the fold's S2 ensemble applied to the clear S2 scene
paired with each S1 acquisition (+/-2 days, level change < 0.15 ft). Cloud/shadow pixels per
SCL and pixels outside the lake zone are marked 255 (ignore)."""

from __future__ import annotations

import argparse

import numpy as np
import rasterio

from . import aoi
from . import config as C
from . import selection as S
from .dataset import s2_path, s2_scl_path, teacher_path
from .evaluate import load_models
from .io import write_raster
from .model import predict_tiled, s2_to_input


def build(fold: str):
    models = load_models("s2", fold)
    if not models:
        raise SystemExit(f"train the S2 ensemble for {fold} first")
    pairs = S.s1_training_pairs(fold)
    for lake, _, d2 in pairs:
        out = teacher_path(fold, lake, d2)
        if out.exists():
            continue
        zone, transform, crs = aoi.load_zone(lake)
        with rasterio.open(s2_path(lake, d2)) as src:
            refl = src.read()
        with rasterio.open(s2_scl_path(lake, d2)) as src:
            scl = src.read(1)
        p, _ = predict_tiled(models, s2_to_input(refl))
        t = np.round(p * 250).astype(np.uint8)
        t[~zone | np.isin(scl, C.SCL_INVALID) | (refl[2] == 0)] = 255
        write_raster(out, t[None], transform, crs, nodata=255)
        print("teacher", fold, lake, d2, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="holdout_travis")
    build(ap.parse_args().fold)


if __name__ == "__main__":
    main()
