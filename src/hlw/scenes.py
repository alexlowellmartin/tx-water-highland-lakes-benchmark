"""Scene inventory: per lake, every S2 day with its in-zone clear fraction, and every S1 day."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import aoi, stac
from . import config as C

SCENES_DIR = C.DATA / "scenes"


def scan_s2(lake_key: str, max_tile_cloud: float = 80, workers: int = 4) -> pd.DataFrame:
    cfg = C.load_config()
    zone, transform, crs = aoi.load_zone(lake_key)
    gb = stac.geobox(transform, zone.shape, crs)
    items = stac.search("sentinel-2-l2a", aoi.zone_bbox_ll(lake_key), cfg["study_start"],
                        cfg["study_end"], {"eo:cloud_cover": {"lt": max_tile_cloud}})
    days = stac.group_by_day(items)

    def one(day):
        try:
            scl = stac.load_scl(days[day], gb)[zone]
        except Exception as e:  # noqa: BLE001 - network hiccups: record and move on
            return {"date": day, "error": str(e)[:200]}
        invalid = np.isin(scl, C.SCL_INVALID)
        return {"date": day, "n_items": len(days[day]),
                "clear_frac": float(1 - invalid.mean()),
                "scl_water_frac": float((scl == C.SCL_WATER).mean()),
                "tile_cloud": float(np.mean([i.properties["eo:cloud_cover"] for i in days[day]]))}

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, days))
    return pd.DataFrame(rows).sort_values("date")


def scan_s1(lake_key: str) -> pd.DataFrame:
    cfg = C.load_config()
    items = stac.search("sentinel-1-rtc", aoi.zone_bbox_ll(lake_key), cfg["study_start"],
                        cfg["study_end"])
    rows = [{"date": d, "n_items": len(v),
             "orbit": v[0].properties.get("sat:orbit_state"),
             "platform": v[0].properties.get("platform")}
            for d, v in stac.group_by_day(items).items()]
    return pd.DataFrame(rows)


def load(sensor: str, lake_key: str) -> pd.DataFrame:
    return pd.read_csv(SCENES_DIR / f"{sensor}_{lake_key}.csv")


def main():
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    for key in C.lakes():
        s1 = scan_s1(key)
        s1.to_csv(SCENES_DIR / f"s1_{key}.csv", index=False)
        out = SCENES_DIR / f"s2_{key}.csv"
        if out.exists():
            print(key, "s2 scan exists, skipping")
            continue
        s2 = scan_s2(key)
        s2.to_csv(out, index=False)
        ok = (s2.get("clear_frac", pd.Series(dtype=float)) >= 0.99).sum()
        print(f"{key}: {len(s2)} S2 days ({ok} >=99% clear in zone), {len(s1)} S1 days", flush=True)


if __name__ == "__main__":
    main()
