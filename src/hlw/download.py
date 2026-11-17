"""Download the selected S2 (reflectance + SCL) and S1 (dB) scenes onto each lake grid.

Pixels outside the lake zone are zeroed (S2) / set to nodata (S1) to keep files small.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import aoi, stac
from . import selection as S
from .dataset import s1_path, s2_path, s2_scl_path
from .io import write_raster

S1_NODATA = -32768


def fetch_s2(lake: str, date: str, zone=None, transform=None, crs=None) -> bool:
    if s2_path(lake, date).exists() and s2_scl_path(lake, date).exists():
        return False
    if zone is None:
        zone, transform, crs = aoi.load_zone(lake)
    d = pd.Timestamp(date)
    items = stac.search("sentinel-2-l2a", aoi.zone_bbox_ll(lake), date,
                        (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    items = stac.group_by_day(items).get(date)
    if not items:
        raise RuntimeError(f"no S2 items {lake} {date}")
    x = stac.load_s2(items, stac.geobox(transform, zone.shape, crs))
    refl, scl = x["refl"], x["scl"]
    refl[:, ~zone] = 0
    scl[~zone] = 0
    write_raster(s2_scl_path(lake, date), scl[None], transform, crs, nodata=0)
    write_raster(s2_path(lake, date), refl, transform, crs, nodata=0)
    return True


def fetch_s1(lake: str, date: str, zone=None, transform=None, crs=None) -> bool:
    if s1_path(lake, date).exists():
        return False
    if zone is None:
        zone, transform, crs = aoi.load_zone(lake)
    d = pd.Timestamp(date)
    items = stac.search("sentinel-1-rtc", aoi.zone_bbox_ll(lake), date,
                        (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    items = stac.group_by_day(items).get(date)
    if not items:
        raise RuntimeError(f"no S1 items {lake} {date}")
    db = stac.load_s1(items, stac.geobox(transform, zone.shape, crs))
    q = np.where(np.isfinite(db), np.clip(np.round(db * 100), -30000, 30000), S1_NODATA)
    q[:, ~zone] = S1_NODATA
    write_raster(s1_path(lake, date), q.astype(np.int16), transform, crs, nodata=S1_NODATA)
    return True


def main(workers: int = 3):
    sel = S.selected()
    jobs = []
    for lake in sel["s2"]:
        jobs += [("s2", lake, d) for d in sel["s2"][lake]]
        jobs += [("s1", lake, p[0]) for p in sel["s1_pairs"][lake]]
    print(len(jobs), "scenes to check", flush=True)
    zones = {lk: aoi.load_zone(lk) for lk in sel["s2"]}

    def run(job):
        kind, lake, d = job
        try:
            f = fetch_s2 if kind == "s2" else fetch_s1
            new = f(lake, d, *zones[lake])
            return f"{kind} {lake} {d} {'ok' if new else 'cached'}"
        except Exception as e:  # noqa: BLE001
            return f"{kind} {lake} {d} FAILED {str(e)[:150]}"

    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
