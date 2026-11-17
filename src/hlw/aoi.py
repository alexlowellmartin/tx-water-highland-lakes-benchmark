"""Lake zones: the region inside which surface water is counted for each lake.

Zone = (JRC GSW max water extent 1984-2020, clipped to the lake's dam-to-dam polygon,
restricted to the connected component(s) touching the lake seed) dilated by
`zone_buffer_m`, then clipped again. Each lake gets its own 10 m UTM grid.
Also stores JRC `occurrence` resampled to that grid (used as a weak label option).
"""

from __future__ import annotations

import json

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as shp_transform

from . import config as C
from .io import read_raster, write_raster


def lake_grid(lake: C.Lake, crs: str, res: float, pad_m: float = 600):
    """Grid (transform, shape) covering the lake clip polygon in UTM."""
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    poly = shp_transform(to_utm, Polygon(lake.clip))
    minx, miny, maxx, maxy = poly.bounds
    minx = np.floor((minx - pad_m) / res) * res
    maxy = np.ceil((maxy + pad_m) / res) * res
    maxx = np.ceil((maxx + pad_m) / res) * res
    miny = np.floor((miny - pad_m) / res) * res
    w, h = int((maxx - minx) / res), int((maxy - miny) / res)
    return from_origin(minx, maxy, res, res), (h, w), poly


def _jrc_on_grid(asset: str, transform, shp, crs) -> np.ndarray:
    cat = pystac_client.Client.open(C.STAC_URL, modifier=pc.sign_inplace)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    h, w = shp
    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (w, h)
    lon0, lat0 = to_ll(x0, y1)
    lon1, lat1 = to_ll(x1, y0)
    items = list(cat.search(collections=["jrc-gsw"], bbox=[lon0, lat0, lon1, lat1]).items())
    out = np.zeros(shp, np.uint8)
    for it in items:
        with rasterio.open(it.assets[asset].href) as src:
            # dst_nodata must not collide with valid values: GDAL shifts a valid 0 to 1
            # when dst_nodata=0, which silently turns all land into "water".
            tmp = np.full(shp, 255, np.uint8)
            reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
            tmp[tmp == 255] = 0
            out = np.maximum(out, tmp)
    return out


def build_zone(lake: C.Lake, cfg: dict) -> dict:
    crs, res = cfg["crs"], cfg["resolution"]
    transform, shp, poly = lake_grid(lake, crs, res)
    clip = rasterize([(mapping(poly), 1)], out_shape=shp, transform=transform).astype(bool)
    occ = _jrc_on_grid("occurrence", transform, shp, crs)
    extent = _jrc_on_grid("extent", transform, shp, crs).astype(bool) & clip
    lab, _n = ndimage.label(extent, structure=np.ones((3, 3)))
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    sx, sy = to_utm(*lake.seed)
    col, row = ~transform * (sx, sy)
    r, c = int(row), int(col)
    # component at the seed, or the largest component within 1 km of it
    win = lab[max(r - 100, 0): r + 100, max(c - 100, 0): c + 100]
    ids, counts = np.unique(win[win > 0], return_counts=True)
    if len(ids) == 0:
        raise RuntimeError(f"no JRC water near seed for {lake.key}")
    keep = ids[np.argmax(counts)]
    core = lab == keep
    rad = round(cfg["zone_buffer_m"] / res)
    yy, xx = np.ogrid[-rad: rad + 1, -rad: rad + 1]
    zone = ndimage.binary_dilation(core, structure=(xx**2 + yy**2) <= rad**2) & clip
    C.AOI_DIR.mkdir(parents=True, exist_ok=True)
    write_raster(C.AOI_DIR / f"{lake.key}_zone.tif", zone.astype(np.uint8)[None], transform, crs)
    write_raster(C.AOI_DIR / f"{lake.key}_jrc_occ.tif", np.where(clip, occ, 0)[None].astype(np.uint8),
                 transform, crs)
    return {
        "lake": lake.key,
        "shape": list(shp),
        "zone_km2": float(zone.sum() * res * res / 1e6),
        "jrc_core_km2": float(core.sum() * res * res / 1e6),
    }


def load_zone(lake_key: str):
    """Return (zone_bool, transform, crs) for a lake."""
    arr, transform, crs = read_raster(C.AOI_DIR / f"{lake_key}_zone.tif")
    return arr[0].astype(bool), transform, crs


def load_jrc(lake_key: str) -> np.ndarray:
    arr, _, _ = read_raster(C.AOI_DIR / f"{lake_key}_jrc_occ.tif")
    return arr[0]


def zone_bbox_ll(lake_key: str) -> list[float]:
    zone, transform, crs = load_zone(lake_key)
    h, w = zone.shape
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    xs, ys = [], []
    for cx, cy in [(0, 0), (w, 0), (0, h), (w, h)]:
        x, y = transform * (cx, cy)
        lon, lat = to_ll(x, y)
        xs.append(lon)
        ys.append(lat)
    return [min(xs), min(ys), max(xs), max(ys)]


def write_geojson():
    """Write lake zone outlines (lon/lat) for the web app and README map."""
    from rasterio.features import shapes

    feats = []
    for key, lake in C.lakes().items():
        zone, transform, crs = load_zone(key)
        to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
        polys = [shape(g) for g, v in shapes(zone.astype(np.uint8), transform=transform) if v == 1]
        geom = max(polys, key=lambda p: p.area).simplify(20)
        feats.append({"type": "Feature", "properties": {"lake": key, "name": lake.name},
                      "geometry": mapping(shp_transform(to_ll, geom))})
    out = C.AOI_DIR / "lakes.geojson"
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return out


def main():
    cfg = C.load_config()
    for lake in C.lakes().values():
        print(build_zone(lake, cfg))
    print("wrote", write_geojson())


if __name__ == "__main__":
    main()
