"""Weak labels. No hand-drawn training masks exist, so we evaluate five public proxies:

- ``jrc50``      JRC Global Surface Water occurrence >= 50 % (Landsat 1984-2020, static).
- ``worldcover`` ESA WorldCover 2021 class 80 "permanent water bodies" (static).
- ``io_lulc``    Impact Observatory annual 10 m LULC v02, class 1 "water", matched by year
                 (2017-2023; later years reuse 2023).
- ``scl``        Sentinel-2 L2A Scene Classification class 6 "water", per scene.
- ``scl_jrc``    per-scene SCL water, restricted to pixels JRC has ever seen as water;
                 SCL water outside JRC history is set to ignore. This is the training label.

Label arrays use 1 = water, 0 = not water, 255 = ignore (cloud, nodata, outside zone).
"""

from __future__ import annotations

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.warp import Resampling, reproject

from . import aoi
from . import config as C
from .io import read_raster, write_raster

IGNORE = 255
LABEL_KINDS = ["jrc50", "worldcover", "io_lulc", "scl", "scl_jrc"]
IO_YEARS = list(range(2017, 2024))


def _mosaic_to_grid(collection, asset, transform, shp, crs, nodata, datetime=None, query=None):
    cat = pystac_client.Client.open(C.STAC_URL, modifier=pc.sign_inplace)
    from pyproj import Transformer

    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    h, w = shp
    (x0, y0), (x1, y1) = transform * (0, 0), transform * (w, h)
    lon0, lat0 = to_ll(x0, y1)
    lon1, lat1 = to_ll(x1, y0)
    items = list(cat.search(collections=[collection], bbox=[lon0, lat0, lon1, lat1],
                            datetime=datetime, query=query).items())
    if not items:
        raise RuntimeError(f"no {collection} items for {datetime}")
    out = np.full(shp, nodata, np.uint8)
    for it in items:
        with rasterio.open(it.assets[asset].href) as src:
            tmp = np.full(shp, nodata, np.uint8)
            reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest, src_nodata=src.nodata or nodata,
                      dst_nodata=nodata)
            fill = out == nodata
            out[fill] = tmp[fill]
    return out


def fetch_static_labels():
    """Download WorldCover 2021 and IO LULC 2017-2023 onto each lake grid."""
    for key in C.lakes():
        zone, transform, crs = aoi.load_zone(key)
        out = C.AOI_DIR / f"{key}_worldcover.tif"
        if not out.exists():
            wc = _mosaic_to_grid("esa-worldcover", "map", transform, zone.shape, crs, 0,
                                 datetime="2021-01-01/2021-12-31")
            write_raster(out, wc[None], transform, crs)
        out = C.AOI_DIR / f"{key}_io_lulc.tif"
        if not out.exists():
            stack = [_mosaic_to_grid("io-lulc-annual-v02", "data", transform, zone.shape, crs, 0,
                                     datetime=f"{y}-01-01/{y}-12-31") for y in IO_YEARS]
            write_raster(out, np.stack(stack), transform, crs)
        print("static labels ok:", key)


def static_layers(lake_key: str) -> dict[str, np.ndarray]:
    wc = read_raster(C.AOI_DIR / f"{lake_key}_worldcover.tif")[0][0]
    io = read_raster(C.AOI_DIR / f"{lake_key}_io_lulc.tif")[0]
    return {"jrc": aoi.load_jrc(lake_key), "worldcover": wc, "io_lulc": io}


def weak_label(kind: str, scl: np.ndarray, zone: np.ndarray, static: dict, year: int) -> np.ndarray:
    """Build a weak label for one scene."""
    jrc = static["jrc"]
    if kind == "jrc50":
        lab = (jrc >= 50).astype(np.uint8)
    elif kind == "worldcover":
        lab = (static["worldcover"] == 80).astype(np.uint8)
    elif kind == "io_lulc":
        idx = IO_YEARS.index(min(max(year, IO_YEARS[0]), IO_YEARS[-1]))
        lab = (static["io_lulc"][idx] == 1).astype(np.uint8)
    elif kind in ("scl", "scl_jrc"):
        lab = (scl == C.SCL_WATER).astype(np.uint8)
        if kind == "scl_jrc":
            lab[(lab == 1) & (jrc == 0)] = IGNORE
    else:
        raise ValueError(kind)
    lab = lab.copy()
    invalid = np.isin(scl, C.SCL_INVALID)
    lab[invalid | ~zone] = IGNORE
    return lab
