"""Truly ungauged test: stock tanks and farm ponds (configs/ponds.yaml).

No gauge exists, so the reference is USDA NAIP aerial imagery (0.6 m RGB+NIR, read at 1 m). On
each NAIP flight date, water is outlined from the aerial image and compared with maps from the
clear Sentinel-2 days (and the Sentinel-1 days) nearest that date. The shipped models and the
Highland Lakes cutoffs are used unchanged, exactly as in the reservoir transfer test.

Objects: every water body NAIP shows in any of the three years (union of the yearly masks,
connected components) is one "pond". Its reference area in a given year is the NAIP water inside
it, which may be zero (a dry tank: a test of false alarms on mud). Ponds touching the box edge
and water bodies above ``max_pond_acres`` are masked out, with a 50 m margin.

Stages: ``naip s2 s1 analyze`` (resumable).
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy import ndimage

from . import config as C
from . import indices, stac
from .io import read_raster, write_raster

CFG_PATH = C.ROOT / "configs" / "ponds.yaml"
OUT = C.RESULTS_DIR / "ponds"
P_DIR = C.OUT_DIR / "ponds"
CRS = "EPSG:32614"
FOLD = "holdout_travis"
NAIP_RES, RES = 1.0, 10.0
K = int(RES / NAIP_RES)  # NAIP cells per S2 cell along each axis


@lru_cache
def pcfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def sites() -> dict:
    return pcfg()["sites"]


def grid(site: str):
    """(transform at 10 m, shape at 10 m) of the site box, snapped to the 10 m S2 grid."""
    lon, lat = sites()[site]["point"]
    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    h = pcfg()["half_km"] * 1000
    x0, y1 = np.floor((x - h) / RES) * RES, np.floor((y + h) / RES) * RES
    n = int(2 * h / RES)
    return from_origin(x0, y1, RES, RES), (n, n)


def bbox_ll(site: str) -> list[float]:
    tr, (n, m) = grid(site)
    to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform
    (a, b), (c, d) = to_ll(*(tr * (0, n))), to_ll(*(tr * (m, 0)))
    return [a, b, c, d]


def block_mean(a: np.ndarray, k: int = K) -> np.ndarray:
    h, w = a.shape
    return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


# ---- NAIP reference ----------------------------------------------------------------------------
def load_naip(site: str, year: int):
    """NAIP (4,H,W) uint8 at 1 m on the site grid, plus a per-pixel flight date (yyyymmdd)."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    tr10, (n, m) = grid(site)
    tr = from_origin(tr10.c, tr10.f, NAIP_RES, NAIP_RES)
    H, W = n * K, m * K
    items = stac.search("naip", bbox_ll(site), f"{year}-01-01", f"{year}-12-31")
    img = np.zeros((4, H, W), np.uint8)
    dmap = np.zeros((H, W), np.int32)
    for it in sorted(items, key=lambda i: i.datetime):
        with rasterio.open(it.assets["image"].href) as src:
            tmp = np.zeros((4, H, W), np.uint8)
            reproject(rasterio.band(src, [1, 2, 3, 4]), tmp, dst_transform=tr, dst_crs=CRS,
                      resampling=Resampling.average, src_nodata=0, dst_nodata=0)
        new = (tmp[3] > 0) & (dmap == 0)
        img[:, new] = tmp[:, new]
        dmap[new] = int(it.datetime.strftime("%Y%m%d"))
    return img, dmap, tr


def naip_water(img: np.ndarray) -> np.ndarray:
    """Water from NAIP NDWI (green vs NIR) with hysteresis: seed pixels above `seed` grown through
    connected pixels above `grow`; holes filled, specks removed.

    NAIP is not radiometrically calibrated, and in some 2022 flights bright bare fields, roofs and
    roads reach NDWI 0.1-0.2. Water also absorbs NIR far more than red does, so both steps also
    require NIR below a fraction of red (bare ground sits at ~0.7, ponds at 0.2-0.6)."""
    c = pcfg()
    r, g, nir = (img[i].astype(np.float32) for i in (0, 1, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = (g - nir) / (g + nir)
    nd[img[3] == 0] = -1
    grow = (nd > c["naip_ndwi_grow"]) & (nir < c["naip_nir_red_grow"] * r)
    lab, _ = ndimage.label(grow)
    seeds = np.unique(lab[(nd > c["naip_ndwi_seed"]) & (nir < c["naip_nir_red_seed"] * r) & grow])
    w = np.isin(lab, seeds[seeds > 0])
    w = ndimage.binary_opening(w, iterations=1)
    w = ndimage.binary_fill_holes(w)
    lab, nlab = ndimage.label(w)
    if nlab:
        idx = np.arange(1, nlab + 1)
        size = ndimage.sum(w, lab, idx) * NAIP_RES ** 2
        # thin features (tree-line and creek-bank shadows, dark road edges) pass the NDWI test;
        # ponds have a compact core: keep only bodies with a point >= core_m from any edge
        core = ndimage.maximum(ndimage.distance_transform_edt(w), lab, idx) * NAIP_RES
        # ... and are not strips: area / (pi core^2) is ~1 for a disc, >> 10 for a line
        elong = size / (np.pi * np.maximum(core, 0.5) ** 2)
        keep = (size >= c["min_pond_m2"]) & (core >= c["naip_core_m"]) & (elong <= c["naip_max_elong"])
        w = np.isin(lab, idx[keep])
    return w


def cached_naip(site: str, year: int):
    """load_naip, cached as a local 1 m GeoTIFF (data/raw/naip, gitignored) so the water rule
    can be re-run without downloading again."""
    from rasterio.transform import Affine

    p = C.RAW_DIR / "naip" / f"{site}_{year}.tif"
    if not p.exists():
        img, dmap, tr = load_naip(site, year)
        write_raster(p, img, tr, CRS, nodata=0)
        tr10, _ = grid(site)
        write_raster(p.with_suffix(".date.tif"), dmap[K // 2::K, K // 2::K], tr10, CRS)
    img, tr, _ = read_raster(p)
    return img, read_raster(p.with_suffix(".date.tif"))[0][0], Affine(*tr[:6])


def stage_naip(keys=None, force=False):
    for site in keys or sites():
        d = P_DIR / site
        for year in pcfg()["naip_years"]:
            if (d / f"naip_{year}.tif").exists() and not force:
                continue
            img, dmap, tr = cached_naip(site, year)
            w = naip_water(img)
            # 1 m water mask (0/1, 255 = no NAIP), flight date per 10 m cell, 2 m RGB preview
            q = np.where(img[3] > 0, w, 255).astype(np.uint8)
            write_raster(d / f"naip_{year}.tif", q, tr, CRS, nodata=255)
            tr10, _ = grid(site)
            write_raster(d / f"naipdate_{year}.tif", dmap, tr10, CRS)
            rgb = np.stack([block_mean(b.astype(np.float32), 2) for b in img[:3]])
            write_raster(d / f"naiprgb_{year}.tif", rgb.round().astype(np.uint8),
                         from_origin(tr.c, tr.f, 2, 2), CRS, photometric="RGB")
            print(site, year, sorted(int(v) for v in set(np.unique(dmap)) - {0}),
                  f"water {w.sum() / 4046.86:.1f} ac", flush=True)
        build_ponds(site)


def worldcover(site: str) -> np.ndarray:
    from .labels import _mosaic_to_grid

    p = P_DIR / site / "worldcover.tif"
    if not p.exists():
        tr, shp = grid(site)
        write_raster(p, _mosaic_to_grid("esa-worldcover", "map", tr, shp, CRS, 0,
                                        datetime="2021-01-01/2021-12-31"), tr, CRS)
    return read_raster(p)[0][0]


def build_ponds(site: str):
    """Pond objects at 10 m from the union of NAIP water over all years (see module doc)."""
    c = pcfg()
    years = c["naip_years"]
    masks = {y: read_raster(P_DIR / site / f"naip_{y}.tif")[0][0] for y in years}
    union = np.zeros_like(masks[years[0]], bool)
    for m in masks.values():
        union |= m == 1
    lab1, n1 = ndimage.label(union, structure=np.ones((3, 3)))
    size_ac = ndimage.sum(union, lab1, np.arange(1, n1 + 1)) / C.ACRE_M2
    # object id per 10 m cell: the 1 m object covering most of the cell (any water wins over 0)
    H, W = union.shape
    blocks = lab1.reshape(H // K, K, W // K, K).transpose(0, 2, 1, 3).reshape(H // K, W // K, -1)
    obj = np.zeros((H // K, W // K), np.int32)
    has = blocks.max(-1) > 0
    for i, j in zip(*np.nonzero(has)):
        v = blocks[i, j]
        v = v[v > 0]
        obj[i, j] = np.bincount(v).argmax()
    # footprint: each object's cells plus up to 2 cells (20 m) around, split by nearest object,
    # so mixed edge pixels and small misregistration count toward the pond
    dist, (ii, jj) = ndimage.distance_transform_edt(obj == 0, return_indices=True)
    foot = np.where(dist <= 2, obj[ii, jj], 0)
    # mask: objects above max size or touching the edge, with a 50 m margin
    big = 1 + np.nonzero(size_ac > c["max_pond_acres"])[0]
    edge = np.unique(np.r_[lab1[0], lab1[-1], lab1[:, 0], lab1[:, -1]])
    bad = np.isin(foot, np.r_[big, edge[edge > 0]])
    # built-up land (ESA WorldCover 2021 class 50: towns, plants, yards): shadows and dark roofs
    # there pass the water test, and nobody's stock tank is in a refinery
    built = worldcover(site) == 50
    bad |= np.isin(foot, np.unique(foot[built & (foot > 0)])) | built
    bad = ndimage.binary_dilation(bad, iterations=5)
    # NAIP nodata anywhere in a year also masks
    nod = np.zeros_like(bad)
    for m in masks.values():
        nod |= block_mean((m == 255).astype(np.float32)) > 0
    # a pond whose footprint is partly masked is dropped whole (and masked), never kept with
    # only its fringe
    bad |= np.isin(foot, np.unique(foot[(bad | nod) & (foot > 0)]))
    foot[bad] = 0
    tr10, _ = grid(site)
    write_raster(P_DIR / site / "ponds.tif", foot.astype(np.int32), tr10, CRS)
    write_raster(P_DIR / site / "mask.tif", (bad | nod).astype(np.uint8), tr10, CRS)
    rows = []
    for oid in np.unique(foot[foot > 0]):
        r = {"site": site, "pond": int(oid), "cells": int((foot == oid).sum())}
        sel = lab1 == oid
        for y in years:
            r[f"naip_ac_{y}"] = float((sel & (masks[y] == 1)).sum() / C.ACRE_M2)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(P_DIR / site / "ponds.csv", index=False)
    print(site, len(df), "ponds;",
          {y: round(df[f"naip_ac_{y}"].sum(), 1) for y in years}, "ac", flush=True)
    return df


def naip_fraction(site: str, year: int) -> np.ndarray:
    m = read_raster(P_DIR / site / f"naip_{year}.tif")[0][0]
    return block_mean((m == 1).astype(np.float32))


def naip_dates(site: str, year: int) -> np.ndarray:
    return read_raster(P_DIR / site / f"naipdate_{year}.tif")[0][0]


# ---- satellite maps ----------------------------------------------------------------------------
def _days(collection: str, site: str, date: pd.Timestamp, window: int) -> dict:
    items = stac.search(collection, bbox_ll(site), (date - pd.Timedelta(days=window)).strftime(
        "%Y-%m-%d"), (date + pd.Timedelta(days=window + 1)).strftime("%Y-%m-%d"))
    return stac.group_by_day(items)


def flights(site: str) -> list[tuple[int, pd.Timestamp, np.ndarray]]:
    """(year, flight date, 10 m region mask) for every NAIP flight date covering the site."""
    out = []
    for y in pcfg()["naip_years"]:
        dm = naip_dates(site, y)
        for d in np.unique(dm[dm > 0]):
            region = dm == d
            if region.mean() > 0.02:
                out.append((y, pd.Timestamp(str(d)), region))
    return out


def stage_sat(sensor: str, keys=None):
    from .evaluate import load_models
    from .model import predict_tiled, s1_to_input, s2_to_input
    from .transfer import hl_cutoffs

    c = pcfg()
    cuts = hl_cutoffs()
    models = load_models(sensor, FOLD, seeds=(0, 1, 2) if sensor == "s2" else (0,))
    coll = "sentinel-2-l2a" if sensor == "s2" else "sentinel-1-rtc"
    win = c["s2_window_days"] if sensor == "s2" else c["s1_window_days"]
    for site in keys or sites():
        out_csv = P_DIR / site / f"{sensor}.csv"
        if out_csv.exists():
            continue
        tr, shp = grid(site)
        gb = stac.geobox(tr, shp, CRS)
        rows = []
        for year, fdate, region in flights(site):
            for day, items in _days(coll, site, fdate, win).items():
                try:
                    if sensor == "s2":
                        d = stac.load_s2(items, gb)
                        refl, scl = d["refl"], d["scl"]
                        valid = region & (refl[2] > 0)
                        clear = float((~np.isin(scl[valid], C.SCL_INVALID)).mean()) \
                            if valid.any() else 0.0
                        if valid.mean() < 0.95 * region.mean() or clear < 0.99:
                            print(site, fdate.date(), day, f"skip (clear {clear:.3f})",
                                  flush=True)
                            continue
                        p = np.mean([predict_tiled(m, s2_to_input(refl))[0] for m in models], 0)
                        mn, nd = indices.mndwi(refl), indices.ndwi(refl)
                        layers = {"unet": p, "mndwi_hl": (mn > cuts["mndwi"]).astype(float),
                                  "mndwi_zero": (mn > 0).astype(float),
                                  "ndwi_hl": (nd > cuts["ndwi"]).astype(float),
                                  "ndwi_zero": (nd > 0).astype(float),
                                  "scl": (scl == C.SCL_WATER).astype(float)}
                    else:
                        db = stac.load_s1(items, gb)
                        valid = region & np.isfinite(db[0])
                        if valid.mean() < 0.95 * region.mean():
                            continue
                        p = predict_tiled(models, s1_to_input(db))[0]
                        layers = {"unet": p, "vv_hl": (db[0] < cuts["vv"]).astype(float)}
                except Exception as e:  # noqa: BLE001
                    print("FAILED", sensor, site, day, str(e)[:120], flush=True)
                    continue
                q = np.stack([np.round(v * 250) for v in layers.values()]).astype(np.uint8)
                q[:, ~valid] = 255
                write_raster(P_DIR / site / sensor / f"{fdate:%Y%m%d}_{day}.tif", q, tr, CRS,
                             nodata=255)
                gap = (pd.Timestamp(day) - fdate).days
                rows.append({"site": site, "year": year, "flight": f"{fdate:%Y-%m-%d}",
                             "date": day, "gap_days": gap, "layers": ",".join(layers)})
                print(sensor, site, fdate.date(), day, gap, flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)


# ---- reference audit ---------------------------------------------------------------------------
AUDIT_CSV = C.LABELS_DIR / "ponds_naip_audit.csv"
AUDIT_DIR = P_DIR / "audit"


def audit_sheets(per_class: int = 20, seed: int = 0):
    """Numbered NAIP crops of a size-stratified random sample of reference ponds (pond-years with
    NAIP water), outlined, for a visual check of the reference. Writes the sample to
    AUDIT_DIR/sample.csv; judgements go in labels/ponds_naip_audit.csv (column `water`:
    yes / no / partly)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for site in sites():
        df = pd.read_csv(P_DIR / site / "ponds.csv")
        for y in pcfg()["naip_years"]:
            d = df[df[f"naip_ac_{y}"] > 0]
            rows.append(pd.DataFrame({"site": site, "pond": d.pond, "year": y,
                                      "naip_ac": d[f"naip_ac_{y}"]}))
    allr = pd.concat(rows, ignore_index=True)
    allr["cls"] = pd.cut(allr.naip_ac, [0, 0.25, 1, 1e9], labels=["<0.25", "0.25-1", ">=1"],
                         right=False)
    samp = (allr.groupby("cls", observed=True)
            .sample(per_class, random_state=seed).reset_index(drop=True))
    samp = samp.sample(frac=1, random_state=seed).reset_index(drop=True)  # hide size order
    samp.insert(0, "id", range(1, len(samp) + 1))
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    samp.to_csv(AUDIT_DIR / "sample.csv", index=False)
    for sheet in range(int(np.ceil(len(samp) / 12))):
        sub = samp.iloc[sheet * 12: (sheet + 1) * 12]
        fig, axes = plt.subplots(3, 4, figsize=(16, 12.5))
        for ax, r in zip(axes.flat, sub.itertuples()):
            foot = read_raster(P_DIR / r.site / "ponds.tif")[0][0]
            rgb = read_raster(P_DIR / r.site / f"naiprgb_{r.year}.tif")[0]
            m = read_raster(P_DIR / r.site / f"naip_{r.year}.tif")[0][0][::2, ::2] == 1
            ii, jj = np.nonzero(foot == r.pond)
            ci, cj, h = int(ii.mean() * 5), int(jj.mean() * 5), 50  # 2 m pixels, 200 m crop
            sl = (slice(max(ci - h, 0), ci + h), slice(max(cj - h, 0), cj + h))
            ax.imshow(np.moveaxis(rgb[:, sl[0], sl[1]], 0, -1))
            # outline only this pond's water (other ponds dimmer)
            lab = np.kron(foot[sl[0].start // 5: sl[0].stop // 5 + 1,
                               sl[1].start // 5: sl[1].stop // 5 + 1], np.ones((5, 5), int))
            lab = lab[sl[0].start % 5: sl[0].start % 5 + m[sl].shape[0],
                      sl[1].start % 5: sl[1].start % 5 + m[sl].shape[1]]
            mine = m[sl] & (lab == r.pond)
            edge = mine ^ ndimage.binary_erosion(mine)
            ov = np.zeros(edge.shape + (4,))
            ov[edge] = [0, 1, 1, 1]
            ax.imshow(ov)
            ax.set_title(f"{r.id}", fontsize=14)
            ax.axis("off")
        for ax in axes.flat[len(sub):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(AUDIT_DIR / f"sheet_{sheet + 1:02d}.png", dpi=55)
        plt.close(fig)
    print("wrote", AUDIT_DIR, len(samp), "crops", flush=True)


# ---- human audit: browser tool (labels/tool/ponds.html) ----------------------------------------
HUMAN_AUDIT_CSV = C.LABELS_DIR / "ponds_naip_audit_human.csv"
POND_TOOL = C.LABELS_DIR / "tool"


def _png(path, rgb: np.ndarray, scale: int = 1):
    from PIL import Image

    im = Image.fromarray(np.ascontiguousarray(rgb.astype(np.uint8)))
    if scale > 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    im.save(path, optimize=True)


def _stretch(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-band 2-98 % stretch (NAIP brightness varies a lot between flights)."""
    out = np.zeros(a.shape, np.float32)
    for b in range(a.shape[0]):
        v = a[b][valid]
        lo, hi = (np.percentile(v, [2, 98]) if v.size else (0, 255))
        out[b] = np.clip((a[b] - lo) / max(hi - lo, 1) * 255, 0, 255)
    return out


def audit_chips(half_m: int = 100, ctx_half_m: int = 400):
    """Images for the human pond audit, one set per pond in AUDIT_DIR/sample.csv (the same 60
    pond-years the AI checked): 1 m true colour (window sized to the pond, >= 200 m) with and without this pond's NAIP outline,
    colour infrared, and a wider 2 m context. Writes labels/tool/ponds.json (no AI answers,
    no sizes: the check is blind)."""
    samp = pd.read_csv(AUDIT_DIR / "sample.csv")
    out = POND_TOOL / "pondchips"
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for r in samp.itertuples():
        foot = read_raster(P_DIR / r.site / "ponds.tif")[0][0]
        img = cached_naip(r.site, r.year)[0]
        m = read_raster(P_DIR / r.site / f"naip_{r.year}.tif")[0][0] == 1
        ii, jj = np.nonzero(foot == r.pond)
        ci, cj = int(ii.mean() * K + K // 2), int(jj.mean() * K + K // 2)
        H, W = m.shape
        # big ponds get a bigger window, so the whole outline fits with a margin
        h = max(half_m, int(0.5 * K * max(np.ptp(ii), np.ptp(jj)) + 40))
        i0, j0 = max(ci - h, 0), max(cj - h, 0)
        i1, j1 = min(ci + h, H), min(cj + h, W)
        sl = (slice(i0, i1), slice(j0, j1))
        sub = img[:, sl[0], sl[1]].astype(np.float32)
        valid = sub[3] > 0
        lab = np.kron(foot, np.ones((K, K), foot.dtype))[sl]
        mine = m[sl] & (lab == r.pond)
        other = m[sl] & (lab != r.pond)
        edge = mine & ~ndimage.binary_erosion(mine)
        edge = ndimage.binary_dilation(edge) & ~ndimage.binary_erosion(mine, iterations=2)
        oedge = other & ~ndimage.binary_erosion(other)
        tc = np.moveaxis(_stretch(sub[:3], valid), 0, -1)
        cir = np.moveaxis(_stretch(sub[[3, 0, 1]], valid), 0, -1)
        ol = tc.copy()
        ol[oedge] = [255, 210, 0]
        ol[edge] = [0, 255, 255]
        _png(out / f"{r.id}_tc.png", tc)
        _png(out / f"{r.id}_ol.png", ol)
        _png(out / f"{r.id}_cir.png", cir)
        # context: 2 m preview, box marks the close-up
        rgb = read_raster(P_DIR / r.site / f"naiprgb_{r.year}.tif")[0].astype(np.float32)
        h2, c2i, c2j = max(ctx_half_m, 3 * h) // 2, ci // 2, cj // 2
        cs = (slice(max(c2i - h2, 0), c2i + h2), slice(max(c2j - h2, 0), c2j + h2))
        cx = rgb[:, cs[0], cs[1]]
        cx = np.moveaxis(_stretch(cx, cx.sum(0) > 0), 0, -1)
        bi0, bj0 = i0 // 2 - cs[0].start, j0 // 2 - cs[1].start
        bi1, bj1 = i1 // 2 - cs[0].start - 1, j1 // 2 - cs[1].start - 1
        for a, b in ((slice(bi0, bi0 + 2), slice(bj0, bj1 + 1)),
                     (slice(bi1 - 1, bi1 + 1), slice(bj0, bj1 + 1)),
                     (slice(bi0, bi1 + 1), slice(bj0, bj0 + 2)),
                     (slice(bi0, bi1 + 1), slice(bj1 - 1, bj1 + 1))):
            cx[a, b] = [255, 0, 255]
        _png(out / f"{r.id}_ctx.png", cx)
        dates = read_raster(P_DIR / r.site / f"naipdate_{r.year}.tif")[0][0]
        d = int(dates[int(ii.mean()), int(jj.mean())])
        meta.append({"id": int(r.id), "site": r.site, "year": int(r.year), "crop_m": 2 * h,
                     "date": f"{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}" if d else str(r.year)})
    (POND_TOOL / "ponds.json").write_text(json.dumps(meta), encoding="utf-8")
    print("wrote", out, len(meta), "ponds", flush=True)


def audit_compare() -> pd.DataFrame | None:
    """Human vs AI audit of the same ponds: agreement, and the per-size-class water rates under
    each. Writes results/ponds/audit_compare.csv."""
    if not HUMAN_AUDIT_CSV.exists():
        print("no", HUMAN_AUDIT_CSV.name, "yet: label with labels/tool/ponds.html", flush=True)
        return None
    ai = pd.read_csv(AUDIT_CSV)[["id", "site", "pond", "year", "naip_ac", "cls", "water"]]
    hu = pd.read_csv(HUMAN_AUDIT_CSV)
    df = ai.merge(hu[["id", "water", "misses_water"]], on="id", suffixes=("_ai", "_human"))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "audit_compare.csv", index=False)
    both = df[(df.water_human != "unsure") & (df.water_ai != "unsure")]
    print(f"{len(df)} ponds labelled by both; exact agreement {100 * (both.water_ai == both.water_human).mean():.0f} %"
          f" (n={len(both)}, human unsure {int((df.water_human == 'unsure').sum())})")
    print(pd.crosstab(df.water_ai, df.water_human, margins=True))
    for name, col in (("AI", "water_ai"), ("human", "water_human")):
        rates = (df[df[col] != "unsure"].groupby("cls")[col]
                 .apply(lambda s: round(100 * (s == "yes").mean())).to_dict())
        print(f"{name}: % outlines judged all water by size class (unsure dropped)", rates)
    print("human says the outline misses nearby water:", int(df.misses_water.sum()), "of", len(df))
    return df


def audit_summary() -> dict | None:
    """Water rates of the reference audit; the author's own check (HUMAN_AUDIT_CSV) is preferred
    over the AI's when it exists."""
    src = HUMAN_AUDIT_CSV if HUMAN_AUDIT_CSV.exists() else AUDIT_CSV
    if not src.exists():
        return None
    a = pd.read_csv(src)
    if src == HUMAN_AUDIT_CSV:  # sizes live in the AI file / sample
        a = a.merge(pd.read_csv(AUDIT_DIR / "sample.csv")[["id", "cls"]], on="id")
        a = a[a.water != "unsure"]
    out = {"n": len(a), "labeller": str(a.labeller.iloc[0]) if "labeller" in a else ""}
    for cls, g in a.groupby("cls"):
        out[f"water_yes_pct_{cls}"] = round(100 * (g.water == "yes").mean(), 1)
        out[f"water_yes_or_partly_pct_{cls}"] = round(100 * g.water.isin(["yes", "partly"])
                                                      .mean(), 1)
    out["water_yes_pct_all"] = round(100 * (a.water == "yes").mean(), 1)
    return out


# ---- analysis ----------------------------------------------------------------------------------
NAMES = {("s2", "unet"): "S2 U-Net", ("s2", "mndwi_hl"): "MNDWI (HL cutoff)",
         ("s2", "mndwi_zero"): "MNDWI (0)", ("s2", "ndwi_hl"): "NDWI (HL cutoff)",
         ("s2", "ndwi_zero"): "NDWI (0)", ("s2", "scl"): "SCL water",
         ("s1", "unet"): "S1 U-Net", ("s1", "vv_hl"): "VV (HL cutoff)"}
MIN_SCORED_AC = 0.25  # headline metrics: ponds at least this big on the flight date
NULL_SHIFT = (37, 53)  # cells (370 m south, 530 m east), wrapping round the box
SIZE_BINS = [0, 0.25, 0.5, 1, 2, 5, 50]
SIZE_LABELS = ["<0.25", "0.25-0.5", "0.5-1", "1-2", "2-5", "5-50"]


def pond_rows(site: str, sensor: str) -> pd.DataFrame:
    """One row per (pond, flight, satellite day, method): NAIP area vs mapped area (acres)."""
    foot = read_raster(P_DIR / site / "ponds.tif")[0][0]
    mask = read_raster(P_DIR / site / "mask.tif")[0][0].astype(bool)
    ponds = pd.read_csv(P_DIR / site / "ponds.csv").set_index("pond")
    days = pd.read_csv(P_DIR / site / f"{sensor}.csv")
    reg = {(y, f"{d:%Y-%m-%d}"): r for y, d, r in flights(site)}
    px = RES * RES / C.ACRE_M2
    ids = ponds.index.values
    out, fp = [], []
    for r in days.itertuples():
        region = reg[(r.year, r.flight)]
        q = read_raster(P_DIR / site / sensor / f"{r.flight.replace('-', '')}_{r.date}.tif")[0]
        valid = (q[0] != 255) & region & ~mask
        frac = naip_fraction(site, r.year)
        in_region = np.asarray(ndimage.maximum(region & valid, foot, ids), bool)
        ref_cells = np.asarray(ndimage.sum(frac * valid, foot, ids)) * px
        outside = valid & (foot == 0)
        # chance baseline: the same footprints moved ~650 m onto land (no NAIP water footprint)
        null = np.roll(foot, NULL_SHIFT, axis=(0, 1))
        null[~outside] = 0
        cells = np.bincount(foot.ravel(), minlength=ids.max() + 1)[ids]
        null_ok = np.bincount(null.ravel(), minlength=ids.max() + 1)[ids] >= 0.9 * cells
        variants = []
        for li, layer in enumerate(r.layers.split(",")):
            variants.append((NAMES[(sensor, layer)], (q[li] > 125) & valid))
            if layer == "unet":  # threshold-free: sum of probabilities
                variants.append((NAMES[(sensor, layer)] + " (expected area)",
                                 np.where(valid, q[li] / 250, 0)))
        for name, w in variants:
            mapped = np.asarray(ndimage.sum(w, foot, ids)) * px
            null_mapped = np.where(null_ok, np.asarray(ndimage.sum(w, null, ids)) * px, np.nan)
            out.append(pd.DataFrame({
                "site": site, "pond": ids, "year": r.year, "flight": r.flight, "date": r.date,
                "gap_days": r.gap_days, "sensor": sensor, "method": name,
                "naip_ac": ponds[f"naip_ac_{r.year}"].values, "naip_cells_ac": ref_cells,
                "mapped_ac": mapped, "null_mapped_ac": null_mapped,
                "in_region": in_region}))
            binary = w.dtype == bool
            fp.append({"site": site, "year": r.year, "flight": r.flight, "date": r.date,
                       "gap_days": r.gap_days, "sensor": sensor,
                       "method": name,
                       "outside_km2": outside.sum() * RES * RES / 1e6,
                       "false_ac": float((w * outside).sum() * px),
                       "false_blobs": ndimage.label(w & outside)[1] if binary else np.nan,
                       "naip_outside_ac": float((frac * outside).sum() * px)})
    df = pd.concat(out) if out else pd.DataFrame()
    return df[df.in_region].drop(columns="in_region") if len(df) else df, pd.DataFrame(fp)


def nearest_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep, per flight and sensor, only the satellite day closest to the flight."""
    if not len(df):
        return df
    d = df.assign(absgap=df.gap_days.abs())
    best = d.groupby(["site", "flight", "sensor"]).absgap.transform("min")
    # ties (a day either side): keep the earlier day
    d = d[d.absgap == best].sort_values("gap_days")
    return d.drop_duplicates(["site", "pond", "flight", "method"]).drop(columns="absgap")


def summarise(p: pd.DataFrame) -> pd.DataFrame:
    """By method and size class: detection of wet ponds, false alarms on dry ponds, area error."""
    p = p.copy()
    p["size"] = pd.cut(p.naip_ac, SIZE_BINS, labels=SIZE_LABELS, right=False)
    wet = p[p.naip_ac > 0]
    rows = []
    for (meth, size), g in wet.groupby(["method", "size"], observed=True):
        err = 100 * (g.mapped_ac - g.naip_ac) / g.naip_ac
        rows.append({"method": meth, "size": size, "n_wet": len(g),
                     "detected_pct": 100 * (g.mapped_ac > 0).mean(),
                     "chance_detected_pct": 100 * (g.null_mapped_ac.dropna() > 0).mean(),
                     "median_err_pct": float(err.median()),
                     "median_abs_err_pct": float(err.abs().median()),
                     "total_bias_pct": 100 * (g.mapped_ac.sum() / g.naip_ac.sum() - 1)})
    out = pd.DataFrame(rows)
    out.loc[out.method.str.contains("expected"), ["detected_pct", "chance_detected_pct"]] = np.nan  # soft: always > 0
    return out


def _sign(by_site, a, b):
    from scipy.stats import binomtest

    w = by_site.pivot(index="site", columns="method", values="detected_pct")[[a, b]]
    w = w[w.notna().all(1)]
    wins, losses = int((w[a] > w[b]).sum()), int((w[a] < w[b]).sum())
    return {"wins": wins, "losses": losses, "ties": len(w) - wins - losses,
            "sign_test_p": float(binomtest(wins, wins + losses).pvalue) if wins + losses else None}


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    allp, allfp = [], []
    for site in sites():
        for sensor in ("s2", "s1"):
            if (P_DIR / site / f"{sensor}.csv").exists():
                p, fp = pond_rows(site, sensor)
                allp.append(p)
                allfp.append(fp)
    p = pd.concat(allp, ignore_index=True)
    fp = pd.concat(allfp, ignore_index=True)
    p.to_csv(OUT / "pond_rows.csv.gz", index=False)
    near = nearest_only(p)
    s = summarise(near)
    s.to_csv(OUT / "by_size.csv", index=False)
    dry = near[near.naip_ac == 0]
    dry_tab = dry.groupby("method").apply(
        lambda g: pd.Series({"n_dry": len(g), "false_alarm_pct": 100 * (g.mapped_ac > 0).mean(),
                             "mean_false_ac": g.mapped_ac.mean()}), include_groups=False)
    fpn = nearest_only(fp.assign(pond=0))
    fp_tab = fpn.groupby("method").apply(
        lambda g: pd.Series({"false_ac_per_km2": g.false_ac.sum() / g.outside_km2.sum(),
                             "false_blobs_per_km2": g.false_blobs.sum() / g.outside_km2.sum()}),
        include_groups=False)
    wet = near[near.naip_ac > 0]
    def _overall(g):
        q = g[g.naip_ac >= MIN_SCORED_AC]  # the audit finds smaller reference ponds unreliable
        big = g[g.naip_ac >= 1]
        return pd.Series({
            "n_wet_ge025ac": len(q), "detected_pct_ge025ac": 100 * (q.mapped_ac > 0).mean(),
            "chance_detected_pct_ge025ac": 100 * (q.null_mapped_ac.dropna() > 0).mean(),
            "detected_pct_ge1ac": 100 * (big.mapped_ac > 0).mean(),
            "median_err_pct_ge1ac": float((100 * (big.mapped_ac - big.naip_ac)
                                           / big.naip_ac).median()),
            "total_bias_pct_ge025ac": 100 * (q.mapped_ac.sum() / q.naip_ac.sum() - 1),
            "detected_pct_lt025ac": 100 * (g[g.naip_ac < MIN_SCORED_AC].mapped_ac > 0).mean()})

    overall = wet.groupby("method").apply(_overall, include_groups=False)
    q = wet[wet.naip_ac >= MIN_SCORED_AC]
    by_site = q.groupby(["site", "method"]).apply(lambda g: pd.Series({
        "n": len(g), "detected_pct": 100 * (g.mapped_ac > 0).mean(),
        "total_bias_pct": 100 * (g.mapped_ac.sum() / g.naip_ac.sum() - 1)}),
        include_groups=False).reset_index()
    by_site.to_csv(OUT / "by_site.csv", index=False)
    table = overall.join(dry_tab).join(fp_tab).round(2)
    soft = table.index.str.contains("expected")
    table.loc[soft, [c for c in table if c.startswith(("detected", "chance"))]
              + ["false_alarm_pct"]] = np.nan
    table.to_csv(OUT / "summary_by_method.csv")
    summary = {
        "sites": len(sites()), "flights": int(near.flight.nunique()),
        "ponds": int(near.drop_duplicates(["site", "pond"]).shape[0]),
        "wet_pond_observations_s2": len(wet[wet.sensor == "s2"].drop_duplicates(
            ["site", "pond", "flight"])),
        "median_abs_gap_days": near.groupby("sensor").gap_days.apply(
            lambda g: float(g.abs().median())).to_dict(),
        "by_method": table.reset_index().to_dict("records"),
        "naip_reference_audit": audit_summary(),
        "sites_where_ndwi0_detects_more_than_s2_unet": _sign(by_site, "NDWI (0)", "S2 U-Net"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    from . import ponds_figs

    ponds_figs.main()
    pd.set_option("display.width", 220)
    print(table)
    print(s.pivot_table(index="size", columns="method", values="detected_pct", observed=True)
          .round(0))
    print(s.pivot_table(index="size", columns="method", values="median_err_pct", observed=True)
          .round(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", default=["naip", "s2", "s1", "analyze"])
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--force", action="store_true", help="naip: rebuild the water masks")
    a = ap.parse_args()
    for s in a.stages:
        if s == "naip":
            stage_naip(a.sites, a.force)
        elif s == "ponds":
            for site in a.sites or sites():
                build_ponds(site)
        elif s == "audit":
            audit_sheets()
        elif s == "audit-tool":
            audit_chips()
        elif s == "audit-compare":
            audit_compare()
        elif s in ("s2", "s1"):
            stage_sat(s, a.sites)
        elif s == "analyze":
            analyze()
        else:
            raise SystemExit(f"unknown stage {s}")


if __name__ == "__main__":
    main()
