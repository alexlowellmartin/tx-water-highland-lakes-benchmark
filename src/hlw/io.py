"""Small raster I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def write_raster(path: Path, arr: np.ndarray, transform, crs, nodata=None, **kw) -> Path:
    """Write a (bands, H, W) array as a tiled, zstd-compressed GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr = arr[None]
    profile = {"driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2],
               "count": arr.shape[0], "dtype": arr.dtype, "crs": crs, "transform": transform,
               "compress": "zstd",
               "predictor": 2 if np.issubdtype(arr.dtype, np.integer) else 3,
               "tiled": True, "blockxsize": 256, "blockysize": 256, "nodata": nodata}
    profile.update(kw)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
    return path


def read_raster(path: Path):
    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs
