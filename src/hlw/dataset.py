"""Random-window training datasets over downloaded scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from . import aoi, labels
from . import config as C
from .model import s1_to_input, s2_to_input


def s2_path(lake, date) -> Path:
    return C.RAW_DIR / "s2" / lake / f"{date}.tif"


def s2_scl_path(lake, date) -> Path:
    return C.RAW_DIR / "s2" / lake / f"{date}_scl.tif"


def s1_path(lake, date) -> Path:
    return C.RAW_DIR / "s1" / lake / f"{date}.tif"


def teacher_path(fold, lake, date) -> Path:
    return C.DATA / "teacher" / fold / lake / f"{date}.tif"


class _LakeCache:
    """Per-lake static layers and candidate window centres (biased toward shorelines)."""

    def __init__(self, lake, chip, rng):
        self.zone, _, _ = aoi.load_zone(lake)
        self.static = labels.static_layers(lake)
        jrc = self.static["jrc"]
        H, W = self.zone.shape
        half = chip // 2
        ys, xs = np.nonzero(self.zone)
        keep = (ys >= half) & (ys < H - half) & (xs >= half) & (xs < W - half)
        ys, xs = ys[keep], xs[keep]
        fluct = (jrc[ys, xs] > 0) & (jrc[ys, xs] < 95)
        w = np.where(fluct, 4.0, 1.0)
        # with replacement: weighted sampling without replacement is O(n*k) in numpy
        idx = rng.choice(len(ys), size=200_000, replace=True, p=w / w.sum())
        self.centres = np.stack([ys[idx], xs[idx]], 1)


class S2WeakDataset(Dataset):
    """Yields (x[13,chip,chip], y[chip,chip] in {0,1,255}) using a weak-label kind."""

    def __init__(self, scenes: list[tuple[str, str]], label_kind="scl_jrc", chip=256,
                 length=48_000, seed=0, augment=True):
        self.scenes, self.kind, self.chip = scenes, label_kind, chip
        self.length, self.augment, self.seed = length, augment, seed
        rng = np.random.default_rng(seed)
        self.lakes = {lk: _LakeCache(lk, chip, rng) for lk in {s[0] for s in scenes}}

    def __len__(self):
        return self.length

    def _window(self, rng, lake):
        cy, cx = self.lakes[lake].centres[rng.integers(len(self.lakes[lake].centres))]
        h = self.chip // 2
        return int(cy - h), int(cx - h)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**31))
        lake, date = self.scenes[rng.integers(len(self.scenes))]
        y0, x0 = self._window(rng, lake)
        win = Window(x0, y0, self.chip, self.chip)
        with rasterio.open(s2_path(lake, date)) as src:
            refl = src.read(window=win)
        with rasterio.open(s2_scl_path(lake, date)) as src:
            scl = src.read(1, window=win)
        lc = self.lakes[lake]
        sl = (slice(y0, y0 + self.chip), slice(x0, x0 + self.chip))
        static = {"jrc": lc.static["jrc"][sl], "worldcover": lc.static["worldcover"][sl],
                  "io_lulc": lc.static["io_lulc"][:, sl[0], sl[1]]}
        lab = labels.weak_label(self.kind, scl, lc.zone[sl], static, int(date[:4]))
        lab[refl[2] == 0] = labels.IGNORE
        x = s2_to_input(refl)
        if self.augment:
            x, lab = _augment(x, lab, rng)
            x[:B10_SKIP] *= rng.uniform(0.9, 1.1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(lab))


B10_SKIP = 13  # brightness jitter applies to all channels (B10 is zero anyway)


class S1TeacherDataset(Dataset):
    """S1 windows with soft targets from the fold's S2 ensemble on a near-simultaneous,
    level-stable S2 scene. pairs: list of (lake, s1_date, s2_date)."""

    def __init__(self, pairs, fold, chip=256, length=48_000, seed=0, augment=True):
        self.pairs, self.fold, self.chip = pairs, fold, chip
        self.length, self.augment, self.seed = length, augment, seed
        rng = np.random.default_rng(seed)
        self.lakes = {lk: _LakeCache(lk, chip, rng) for lk in {p[0] for p in pairs}}

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**31))
        lake, d1, d2 = self.pairs[rng.integers(len(self.pairs))]
        c = self.lakes[lake].centres[rng.integers(len(self.lakes[lake].centres))]
        y0, x0 = int(c[0] - self.chip // 2), int(c[1] - self.chip // 2)
        win = Window(x0, y0, self.chip, self.chip)
        with rasterio.open(s1_path(lake, d1)) as src:
            db = src.read(window=win).astype(np.float32) / 100.0
            db[db <= -300] = np.nan
        with rasterio.open(teacher_path(self.fold, lake, d2)) as src:
            t = src.read(1, window=win)  # uint8 prob*250, 255 ignore
        t = t.astype(np.float32)
        ign = (t == 255) | np.isnan(db).any(0)
        target = np.where(ign, 255.0, t / 250.0).astype(np.float32)
        x = s1_to_input(db)
        if self.augment:
            x, target = _augment(x, target, rng)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(target))


def _augment(x, y, rng):
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    return x.copy(), y.copy()
