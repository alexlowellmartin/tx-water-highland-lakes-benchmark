"""Search and load Sentinel-2 L2A and Sentinel-1 RTC from Microsoft Planetary Computer."""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import odc.stac
import planetary_computer as pc
import pystac_client
from odc.geo.geobox import GeoBox

from . import config as C

warnings.filterwarnings("ignore", category=RuntimeWarning, module="odc")


POOL = 12


def catalog():
    return pystac_client.Client.open(C.STAC_URL, modifier=pc.sign_inplace)


def geobox(transform, shape, crs) -> GeoBox:
    return GeoBox(shape, transform, crs)


def search(collection: str, bbox, start: str, end: str, query=None):
    items = catalog().search(collections=[collection], bbox=bbox, datetime=f"{start}/{end}",
                             query=query).item_collection()
    return list(items)


def dedupe(items):
    """Planetary Computer holds both original and reprocessed (baseline 05.xx) copies of many
    S2 scenes. Keep only the most recently generated item per (tile, sensing time)."""
    best = {}
    for it in items:
        p = it.properties
        key = (p.get("s2:mgrs_tile", it.id), it.datetime.isoformat())
        gen = p.get("s2:generation_time", "")
        if key not in best or gen > best[key][0]:
            best[key] = (gen, it)
    return [v[1] for v in best.values()]


def group_by_day(items) -> dict[str, list]:
    """Group items by acquisition day (UTC date; passes here are ~17:00 UTC or ~00:30 UTC)."""
    g = defaultdict(list)
    for it in dedupe(items) if items and "s2:mgrs_tile" in items[0].properties else items:
        g[it.datetime.strftime("%Y-%m-%d")].append(it)
    return dict(sorted(g.items()))


def s2_offset(item) -> int:
    """Processing baseline >= 04.00 (from 2022-01-25) adds BOA_ADD_OFFSET = -1000 to DNs."""
    pb = item.properties.get("s2:processing_baseline", "00.00")
    try:
        return 1000 if float(pb) >= 4.0 else 0
    except ValueError:
        return 0


def load_s2(items, gbox: GeoBox, bands=None, resampling="bilinear") -> dict[str, np.ndarray]:
    """Load one day of S2 L2A onto `gbox`. Returns {"refl": (12,H,W) uint16 harmonised to the
    pre-2022 scale (reflectance*10000), "scl": (H,W) uint8}. 0 = nodata."""
    bands = bands or C.S2_BANDS
    refl = None
    scl = None
    # items of one day may differ in processing baseline -> harmonise each group separately
    for off in sorted({s2_offset(i) for i in items}):
        grp = [i for i in items if s2_offset(i) == off]
        ds = odc.stac.load(grp, bands=bands + ["SCL"], geobox=gbox, groupby="solar_day",
                           resampling={"SCL": "nearest", "*": resampling}, chunks=None,
                           fail_on_error=True, pool=POOL)
        r = np.stack([ds[b].values[0] for b in bands]).astype(np.int32)
        nod = r == 0
        r = np.clip(r - off, 1, 65535)
        r[nod] = 0
        s = ds["SCL"].values[0].astype(np.uint8)
        if refl is None:
            refl, scl = r, s
        else:
            fill = refl[2] == 0
            refl[:, fill] = r[:, fill]
            scl[fill] = s[fill]
    return {"refl": refl.astype(np.uint16), "scl": scl}


def load_scl(items, gbox: GeoBox) -> np.ndarray:
    ds = odc.stac.load(items, bands=["SCL"], geobox=gbox, groupby="solar_day",
                       resampling="nearest", chunks=None, pool=POOL)
    return ds["SCL"].values[0].astype(np.uint8)


def load_s1(items, gbox: GeoBox) -> np.ndarray:
    """Load one day of S1 RTC (gamma0, linear power) as dB, shape (2,H,W) float32, NaN nodata."""
    ds = odc.stac.load(items, bands=C.S1_BANDS, geobox=gbox, groupby="solar_day",
                       resampling="bilinear", chunks=None, nodata=0, pool=POOL)
    arr = np.stack([ds[b].values[0] for b in C.S1_BANDS]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(arr)
    db[~np.isfinite(db)] = np.nan
    return db
