"""Hand-label validation points.

1. ``sample``: per label scene, draw stratified random pixels inside the lake zone where S2 is
   valid. Strata use only JRC occurrence (independent of every method compared):
   A = occurrence >= 90 (stable water), B = 1..89 (fluctuation zone), C = 0 (dry in JRC record).
2. ``sheets``: render contact sheets (true colour B4/B3/B2 and false colour B12/B8A/B4, 40x40 px
   = 400 m crops, centre pixel boxed) for visual labelling. No model or index output is shown.
3. Labels are typed into labels/hand_points.csv (1 water, 0 not water, -1 unsure -> excluded).
4. ``chips``: per-point images + points.json for the browser labelling tool in labels/tool/
   (blind: it never shows the shipped labels). Its CSV export goes to
   labels/hand_points_human.csv; ``agree`` compares it with the shipped labels and
   ``python -m hlw.evaluate --labels labels/hand_points_human.csv`` re-scores every method.

5. ``recheck``: a blind second look at the hard conflicts (human and AI both committed, but
   disagree), shuffled with an equal number of agreed points as controls, stratum-matched so the
   two can't be told apart. Served by the tool at ``?set=recheck``; its export
   (labels/hand_points_recheck.csv) is merged by ``merge-recheck`` into
   labels/hand_points_human_final.csv, the first-pass file is kept unchanged.

Two answer keys are shipped: labels/hand_points.csv (the AI labeller's, column ``label``) and
labels/hand_points_human_final.csv (the author's, the primary reference). Re-run ``sheets`` to
audit either.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer

from . import aoi
from . import config as C
from . import selection as S
from .dataset import s2_path, s2_scl_path

POINTS_CSV = C.LABELS_DIR / "hand_points.csv"
SHEETS_DIR = C.DATA / "label_sheets"
ALLOC = {"A": 0.2, "B": 0.55, "C": 0.25}
N_DYNAMIC, N_SMALL = 45, 20
CROP, SCALE, PER_SHEET, COLS = 40, 4, 20, 5


def strata(jrc: np.ndarray) -> np.ndarray:
    s = np.full(jrc.shape, "", dtype="<U1")
    s[jrc >= 90] = "A"
    s[(jrc >= 1) & (jrc < 90)] = "B"
    s[jrc == 0] = "C"
    return s


def sample(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sel = S.selected()
    rows = []
    for lake, dates in sel["label"].items():
        L = C.lakes()[lake]
        zone, transform, crs = aoi.load_zone(lake)
        jrc = aoi.load_jrc(lake)
        st = strata(jrc)
        to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
        n_tot = N_DYNAMIC if L.dynamic else N_SMALL
        _H, _W = zone.shape
        for date in dates:
            with rasterio.open(s2_scl_path(lake, date)) as src:
                scl = src.read(1)
            valid = zone & ~np.isin(scl, C.SCL_INVALID)
            valid[:CROP, :] = valid[-CROP:, :] = False
            valid[:, :CROP] = valid[:, -CROP:] = False
            for h, frac in ALLOC.items():
                m = valid & (st == h)
                ys, xs = np.nonzero(m)
                n = min(round(n_tot * frac), len(ys))
                if n == 0:
                    continue
                pick = rng.choice(len(ys), n, replace=False)
                for i in pick:
                    r, c = int(ys[i]), int(xs[i])
                    x, y = transform * (c + 0.5, r + 0.5)
                    lon, lat = to_ll(x, y)
                    rows.append({"lake": lake, "date": date, "row": r, "col": c,
                                 "lon": round(lon, 6), "lat": round(lat, 6), "stratum": h,
                                 "stratum_pixels": int(m.sum()), "jrc_occ": int(jrc[r, c])})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle across scenes
    df.insert(0, "pid", np.arange(len(df)))
    df["scene"] = df.lake + "_" + df.date
    df["label"] = ""
    df["labeler"] = ""
    df["note"] = ""
    return df


def _stretch(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = (rgb.astype(np.float32) - lo[:, None, None]) / (hi - lo)[:, None, None]
    return (np.clip(x, 0, 1) ** 0.8 * 255).astype(np.uint8).transpose(1, 2, 0)


def sheets(df: pd.DataFrame | None = None):
    df = pd.read_csv(POINTS_CSV) if df is None else df
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    b = {n: i for i, n in enumerate(C.S2_BANDS)}
    tc_idx = [b["B04"], b["B03"], b["B02"]]
    fc_idx = [b["B12"], b["B8A"], b["B04"]]
    cache = {}

    def scene(lake, date):
        k = (lake, date)
        if k not in cache:
            with rasterio.open(s2_path(lake, date)) as src:
                a = src.read()
            m = a[2] > 0
            stats = {}
            for name, idx in (("tc", tc_idx), ("fc", fc_idx)):
                v = a[idx][:, m]
                stats[name] = (np.percentile(v, 1, axis=1), np.percentile(v, 99, axis=1))
            cache.clear()
            cache[k] = (a, stats)
        return cache[k]

    cell = CROP * SCALE
    font = ImageFont.load_default()
    order = df.sort_values(["scene", "pid"])  # group by scene for speed, sheets by pid order
    order = df.sort_values("pid")
    files = []
    for s in range(0, len(order), PER_SHEET):
        chunk = order.iloc[s: s + PER_SHEET]
        rows = int(np.ceil(len(chunk) / COLS))
        sheet = Image.new("RGB", (COLS * (2 * cell + 12), rows * (cell + 18)), (40, 40, 40))
        for j, (_, p) in enumerate(chunk.iterrows()):
            a, stats = scene(p.lake, p.date)
            r0, c0 = int(p.row) - CROP // 2, int(p.col) - CROP // 2
            crop = a[:, r0: r0 + CROP, c0: c0 + CROP]
            ox = (j % COLS) * (2 * cell + 12)
            oy = (j // COLS) * (cell + 18)
            for k, (name, idx) in enumerate((("tc", tc_idx), ("fc", fc_idx))):
                im = Image.fromarray(_stretch(crop[idx], *stats[name])).resize(
                    (cell, cell), Image.NEAREST)
                d = ImageDraw.Draw(im)
                cx = (CROP // 2) * SCALE
                d.rectangle([cx - 1, cx - 1, cx + SCALE, cx + SCALE], outline=(255, 0, 255))
                for a0, a1 in ((cx - 14, cx - 4), (cx + SCALE + 4, cx + SCALE + 14)):
                    d.line([a0, cx + SCALE // 2, a1, cx + SCALE // 2], fill=(255, 0, 255))
                    d.line([cx + SCALE // 2, a0, cx + SCALE // 2, a1], fill=(255, 0, 255))
                sheet.paste(im, (ox + k * (cell + 4), oy + 16))
            ImageDraw.Draw(sheet).text((ox + 2, oy + 2), f"#{int(p.pid)}", fill=(255, 255, 0),
                                       font=font)
        f = SHEETS_DIR / f"sheet_{s // PER_SHEET:03d}.png"
        sheet.save(f)
        files.append(f)
    return files


TOOL_DIR = C.LABELS_DIR / "tool"
HUMAN_CSV = C.LABELS_DIR / "hand_points_human.csv"
CTX = 120  # context window (1.2 km)


def _marked(img: Image.Image, cx: int, cy: int, px: int, color=(255, 0, 255)) -> Image.Image:
    """Corner brackets around the target pixel (leaves the pixel itself visible)."""
    d = ImageDraw.Draw(img)
    g, ln = 3, 10
    x0, y0, x1, y1 = cx - g, cy - g, cx + px + g, cy + px + g
    for (x, y, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([x, y, x + sx * ln, y], fill=color, width=2)
        d.line([x, y, x, y + sy * ln], fill=color, width=2)
    return img


def chips(df: pd.DataFrame | None = None):
    """Write labels/tool/chips/{pid}_{tc,fc,ctx}.png and labels/tool/points.json."""
    import json

    df = pd.read_csv(POINTS_CSV) if df is None else df
    out = TOOL_DIR / "chips"
    out.mkdir(parents=True, exist_ok=True)
    b = {n: i for i, n in enumerate(C.S2_BANDS)}
    tc_idx, fc_idx = [b["B04"], b["B03"], b["B02"]], [b["B12"], b["B8A"], b["B04"]]
    up, upc = 8, 3
    for (lake, date), g in df.groupby(["lake", "date"]):
        with rasterio.open(s2_path(lake, date)) as src:
            a = src.read()
        m = a[2] > 0
        st = {k: (np.percentile(a[i][:, m], 1, axis=1), np.percentile(a[i][:, m], 99, axis=1))
              for k, i in (("tc", tc_idx), ("fc", fc_idx))}
        pad = CTX
        ap_ = np.pad(a, ((0, 0), (pad, pad), (pad, pad)))
        for p in g.itertuples():
            r, c = int(p.row) + pad, int(p.col) + pad
            for k, idx in (("tc", tc_idx), ("fc", fc_idx)):
                crop = ap_[idx, r - CROP // 2: r + CROP // 2, c - CROP // 2: c + CROP // 2]
                im = Image.fromarray(_stretch(crop, *st[k])).resize((CROP * up, CROP * up),
                                                                    Image.NEAREST)
                _marked(im, CROP // 2 * up, CROP // 2 * up, up).save(out / f"{p.pid}_{k}.png")
            crop = ap_[tc_idx, r - CTX // 2: r + CTX // 2, c - CTX // 2: c + CTX // 2]
            im = Image.fromarray(_stretch(crop, *st["tc"])).resize((CTX * upc, CTX * upc),
                                                                   Image.BILINEAR)
            d = ImageDraw.Draw(im)
            o = (CTX // 2 - CROP // 2) * upc
            d.rectangle([o, o, o + CROP * upc, o + CROP * upc], outline=(255, 0, 255), width=2)
            _marked(im, CTX // 2 * upc, CTX // 2 * upc, upc).save(out / f"{p.pid}_ctx.png")
        print("chips", lake, date, len(g), flush=True)
    # blind: no labels, no notes; shoreline stratum first, random order within strata
    rng = np.random.default_rng(7)
    pts = df[["pid", "lake", "date", "stratum"]].copy()
    pts["r"] = rng.random(len(pts))
    pts["o"] = pts.stratum.map({"B": 0, "A": 1, "C": 1})
    pts = pts.sort_values(["o", "r"]).drop(columns=["r", "o"])
    (TOOL_DIR / "points.json").write_text(json.dumps(pts.to_dict("records")))
    print(len(pts), "points ->", TOOL_DIR)


def agree(human_csv=HUMAN_CSV) -> pd.DataFrame:
    """Agreement between a human label file and the shipped (AI) labels."""
    ai = pd.read_csv(POINTS_CSV)[["pid", "stratum", "label"]]
    hu = pd.read_csv(human_csv)[["pid", "label"]]
    m = ai.merge(hu, on="pid", suffixes=("_ai", "_human"))
    print(f"{len(m)} points labelled by both")
    print(pd.crosstab(m.label_ai.map({1: "water", 0: "not water", -1: "unsure"}).rename("AI"),
                      m.label_human.map({1: "water", 0: "not water", -1: "unsure"}).rename("human")))
    both = m[m.label_ai.isin([0, 1]) & m.label_human.isin([0, 1])]
    if len(both):
        po = (both.label_ai == both.label_human).mean()
        pa, ph = both.label_ai.mean(), both.label_human.mean()
        pe = pa * ph + (1 - pa) * (1 - ph)
        kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        print(f"agreement on points both called water/not water: {po:.3f} (n={len(both)}), "
              f"Cohen's kappa {kappa:.3f}")
        for s_, g in both.groupby("stratum"):
            print(f"  stratum {s_}: {(g.label_ai == g.label_human).mean():.3f} (n={len(g)})")
    return m


RECHECK_CSV = C.LABELS_DIR / "hand_points_recheck.csv"
RECHECK_MANIFEST = C.LABELS_DIR / "recheck_manifest.csv"
FINAL_CSV = C.LABELS_DIR / "hand_points_human_final.csv"


def recheck(seed: int = 11):
    """Write labels/tool/recheck.json (blind, shuffled) and labels/recheck_manifest.csv."""
    import json

    ai = pd.read_csv(POINTS_CSV)[["pid", "lake", "date", "stratum", "label"]]
    hu = pd.read_csv(HUMAN_CSV)[["pid", "label"]]
    m = ai.merge(hu, on="pid", suffixes=("_ai", "_human"))
    both = m[m.label_ai.isin([0, 1]) & m.label_human.isin([0, 1])]
    conf = both[both.label_ai != both.label_human]
    agreed = both[both.label_ai == both.label_human]
    rng = np.random.default_rng(seed)
    ctrl = []
    for st, n in conf.stratum.value_counts().items():  # same stratum mix as the conflicts
        pool = agreed[agreed.stratum == st]
        ctrl.append(pool.sample(min(n, len(pool)), random_state=int(rng.integers(1e9))))
    sel = pd.concat([conf.assign(kind="conflict"), pd.concat(ctrl).assign(kind="control")])
    sel = sel.sample(frac=1, random_state=seed)
    sel[["pid", "kind", "stratum", "label_ai", "label_human"]].to_csv(RECHECK_MANIFEST,
                                                                       index=False)
    (TOOL_DIR / "recheck.json").write_text(json.dumps(
        sel[["pid", "lake", "date", "stratum"]].to_dict("records")))
    print(f"{len(conf)} conflicts + {len(sel) - len(conf)} controls -> "
          f"{TOOL_DIR / 'recheck.json'} (serve the tool and open ?set=recheck)")


def merge_recheck():
    """Second-look labels replace first-pass labels; report flips and self-consistency."""
    first = pd.read_csv(HUMAN_CSV)
    second = pd.read_csv(RECHECK_CSV)[["pid", "label"]].rename(columns={"label": "label2"})
    man = pd.read_csv(RECHECK_MANIFEST)
    x = man.merge(second, on="pid", how="left")
    name = {1: "water", 0: "not water", -1: "unsure"}
    for kind, g in x.groupby("kind"):
        g = g.dropna(subset=["label2"])
        same = (g.label2 == g.label_human).mean() if len(g) else float("nan")
        print(f"{kind}: {len(g)} re-labelled, {same:.0%} unchanged from first pass")
        print(pd.crosstab(g.label_human.map(name).rename("first pass"),
                          g.label2.astype(int).map(name).rename("second look")))
        if kind == "conflict":
            ai = (g.label2 == g.label_ai).sum()
            print(f"  conflicts now matching the AI: {ai}, matching first pass: "
                  f"{(g.label2 == g.label_human).sum()}, now unsure: {(g.label2 == -1).sum()}")
    out = first.merge(second, on="pid", how="left")
    out["first_pass_label"] = out.label
    out["label"] = out.label2.fillna(out.label).astype(int)
    out = out.drop(columns="label2")
    out.to_csv(FINAL_CSV, index=False)
    print(f"{(out.label != out.first_pass_label).sum()} labels changed -> {FINAL_CSV}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "sheets", "chips", "agree", "recheck",
                                    "merge-recheck"])
    ap.add_argument("--human", default=str(HUMAN_CSV))
    a = ap.parse_args()
    if a.cmd == "chips":
        chips()
        return
    if a.cmd == "agree":
        agree(a.human)
        return
    if a.cmd == "recheck":
        recheck()
        return
    if a.cmd == "merge-recheck":
        merge_recheck()
        return
    if a.cmd == "sample":
        if POINTS_CSV.exists():
            raise SystemExit(f"{POINTS_CSV} exists; refusing to overwrite labels")
        df = sample()
        C.LABELS_DIR.mkdir(exist_ok=True)
        df.to_csv(POINTS_CSV, index=False)
        print(len(df), "points ->", POINTS_CSV)
        print(df.groupby(["lake", "stratum"]).size().unstack())
    else:
        print(len(sheets()), "sheets ->", SHEETS_DIR)


if __name__ == "__main__":
    main()
