"""Official lake level / storage records from the Texas Water Development Board.

Source: https://www.waterdatafortexas.org/reservoirs/individual/<slug>.csv
(daily estimated averages). Note TWDB `surface_area` (acres) is derived from water level
through an elevation-area rating curve from the latest bathymetric survey; it is not an
independent measurement of area.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from . import config as C

URL = "https://www.waterdatafortexas.org/reservoirs/individual/{slug}.csv"


def parse_twdb_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), comment="#", parse_dates=["date"])
    expected = {"date", "water_level", "surface_area", "reservoir_storage", "percent_full"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"TWDB CSV missing columns {missing}")
    return df.set_index("date").sort_index()


def clean_levels(df: pd.DataFrame, max_jump_ft: float = 5.0) -> pd.DataFrame:
    """Drop single-day spikes (e.g. Inks reads 764.8 ft for a day in 2023)."""
    lvl = df["water_level"]
    med = lvl.rolling(7, center=True, min_periods=3).median()
    bad = (lvl - med).abs() > max_jump_ft
    out = df.copy()
    out.loc[bad, ["water_level", "surface_area", "reservoir_storage"]] = float("nan")
    out["flag_spike"] = bad
    return out


def download(lake: C.Lake, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    r = s.get(URL.format(slug=lake.twdb), timeout=60)
    r.raise_for_status()
    return parse_twdb_csv(r.text)


def load(lake_key: str) -> pd.DataFrame:
    return pd.read_csv(C.LEVELS_DIR / f"{lake_key}.csv", parse_dates=["date"], index_col="date")


def main():
    C.LEVELS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = C.load_config()
    for lake in C.lakes().values():
        df = clean_levels(download(lake)).loc[cfg["study_start"]: cfg["study_end"]]
        df.to_csv(C.LEVELS_DIR / f"{lake.key}.csv")
        print(f"{lake.key}: {len(df)} days, level {df.water_level.min():.1f}-"
              f"{df.water_level.max():.1f} ft, {int(df.flag_spike.sum())} spikes removed")


if __name__ == "__main__":
    main()
