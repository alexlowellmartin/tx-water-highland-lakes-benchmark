"""Project configuration and paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "lakes.yaml"
DATA = Path(os.environ.get("HLW_DATA", ROOT / "data"))
AOI_DIR = DATA / "aoi"
RAW_DIR = DATA / "raw"
LEVELS_DIR = DATA / "levels"
CHIPS_DIR = DATA / "chips"
RUNS_DIR = DATA / "runs"
OUT_DIR = DATA / "outputs"
LABELS_DIR = ROOT / "labels"
WEIGHTS_DIR = ROOT / "weights"
WEB_DIR = ROOT / "web"
RESULTS_DIR = ROOT / "results"

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Sentinel-2 L2A bands we load, in the SSL4EO-S12 order minus B10 (absent from L2A).
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S1_BANDS = ["vv", "vh"]

# Sentinel-2 Scene Classification Layer codes.
SCL_NODATA, SCL_SATURATED, SCL_SHADOW_DARK, SCL_CLOUD_SHADOW = 0, 1, 2, 3
SCL_VEG, SCL_BARE, SCL_WATER, SCL_UNCLASSIFIED = 4, 5, 6, 7
SCL_CLOUD_MED, SCL_CLOUD_HIGH, SCL_CIRRUS, SCL_SNOW = 8, 9, 10, 11
SCL_INVALID = (SCL_NODATA, SCL_SATURATED, SCL_CLOUD_SHADOW, SCL_CLOUD_MED, SCL_CLOUD_HIGH,
               SCL_CIRRUS, SCL_SNOW)

ACRE_M2 = 4046.8564224


@dataclass(frozen=True)
class Lake:
    key: str
    name: str
    twdb: str
    seed: tuple[float, float]
    dynamic: bool
    clip: list[list[float]]


@lru_cache
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def lakes() -> dict[str, Lake]:
    cfg = load_config()
    return {
        k: Lake(key=k, name=v["name"], twdb=v["twdb"], seed=tuple(v["seed"]),
                dynamic=v["dynamic"], clip=v["clip"])
        for k, v in cfg["lakes"].items()
    }


def year_split(year: int) -> str:
    cfg = load_config()
    if year in cfg["train_years"]:
        return "train"
    if year in cfg["test_years"]:
        return "test"
    raise ValueError(f"year {year} outside study period")
