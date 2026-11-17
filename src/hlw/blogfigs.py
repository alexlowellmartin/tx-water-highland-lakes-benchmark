"""Map figures for the README and notebooks (results/figures/blog_*.png).

Uses downloaded label scenes where possible and re-streams a few timeline dates from the
Planetary Computer. Colours follow the project palette (analyze.py): blue = water / S2 U-Net,
green = MNDWI, orange = radar or uncertainty.
"""

from __future__ import annotations

import json

import matplotlib
import matplotlib.patheffects
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from . import aoi, indices, levels, stac
from . import config as C
from .analyze import FIG, IDXC, INK, INK2, MUTED, S1C, S2C, SURF, _style, load_timeline
from .dataset import s2_path
from .timeline import TL_DIR, _fetch_day

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAND = (0.95, 0.94, 0.92)
OUTSIDE = (0.84, 0.84, 0.82)


def _hex(c):
    return tuple(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))


def best_window(score: np.ndarray, h: int, w: int, step: int = 25):
    """Top-left corner of the h x w window with the largest sum of score (integral image)."""
    ii = np.pad(score.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = score.shape
    best, arg = -1, (0, 0)
    for y in range(0, max(H - h, 0) + 1, step):
        for x in range(0, max(W - w, 0) + 1, step):
            s = ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x]
            if s > best:
                best, arg = s, (y, x)
    return arg


def rgb(refl: np.ndarray, valid: np.ndarray, sl=None, gamma=0.85) -> np.ndarray:
    x = refl[[3, 2, 1]].astype(np.float32)
    v = x[:, valid]
    lo, hi = np.percentile(v, 1, axis=1), np.percentile(v, 99.5, axis=1)
    if sl is not None:
        x = x[:, sl[0], sl[1]]
    out = np.clip((x - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1) ** gamma
    return out.transpose(1, 2, 0)


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _scalebar(ax, width_px, km=2, res=10):
    n = km * 1000 / res
    x0, y0 = width_px * 0.04, None
    ylim = ax.get_ylim()
    y0 = ylim[0] - (ylim[0] - ylim[1]) * 0.05
    ax.plot([x0, x0 + n], [y0, y0], color="white", lw=3, solid_capstyle="butt")
    ax.plot([x0, x0 + n], [y0, y0], color=INK, lw=1.2, solid_capstyle="butt")
    ax.text(x0 + n / 2, y0 - (ylim[0] - ylim[1]) * 0.02, f"{km} km", color="white",
            ha="center", va="bottom", fontsize=8.5,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground=INK)])


def _read_q(path):
    with rasterio.open(path) as src:
        q = src.read()
    return q


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=120, facecolor=SURF)
    plt.close(fig)
    print("wrote", FIG / name, flush=True)


# --------------------------------------------------------------------------------------------
LABEL_OFFSETS = {"buchanan": (-110, 30), "inks": (95, 25), "lbj": (-105, -35),
                 "marble_falls": (-30, -70), "travis": (90, 75)}


def study_area():
    g = json.loads((C.AOI_DIR / "lakes.geojson").read_text())
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    lv_txt = {"travis": "627–704 ft", "buchanan": "992–1020 ft", "lbj": "held ~825 ft",
              "inks": "held ~888 ft", "marble_falls": "held ~737 ft"}
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        k = f["properties"]["lake"]
        dyn = C.lakes()[k].dynamic
        for poly in polys:
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, fc=S2C if dyn else "#7fa9dc",
                                 ec="none"))
        xy = np.vstack([np.array(p[0]) for p in polys])
        anchor = xy[np.argmin(((xy - xy.mean(0)) ** 2).sum(1))]  # boundary point nearest centroid
        ax.annotate(f"{f['properties']['name']}\n{lv_txt.get(k, '')}", anchor,
                    xytext=LABEL_OFFSETS.get(k, (40, 20)), textcoords="offset points",
                    ha="center", va="center", fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax.autoscale_view()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.03, x1 + 0.03)
    ax.set_ylim(y0 - 0.03, y1 + 0.1)
    ax.set_aspect(1 / np.cos(np.deg2rad(30.6)))
    # 10 km scale bar
    km_deg = 10 / (111.32 * np.cos(np.deg2rad(30.6)))
    bx, by = x1 - km_deg - 0.02, y0 - 0.01
    ax.plot([bx, bx + km_deg], [by, by], color=INK, lw=2)
    ax.text(bx + km_deg / 2, by + 0.008, "10 km", ha="center", fontsize=8.5, color=INK2)
    ax.annotate("Colorado River flows downstream\nfrom Buchanan to Travis", (x1 - 0.05, y1 + 0.05),
                ha="right", fontsize=8.5, color=INK2)
    ax.legend(handles=[Patch(color=S2C, label="Level varies (tested for area vs level)"),
                       Patch(color="#7fa9dc", label="Pass-through, near-constant level")],
              frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2)
    ax.set_title("The five Highland Lakes studied (lake zones: JRC maximum extent + 300 m)",
                 loc="left", fontsize=11, color=INK)
    _clean(ax)
    fig.tight_layout()
    _save(fig, "blog_study_area.png")


# --------------------------------------------------------------------------------------------
def travis_before_after():
    """Flood (2018), drought low (2023), refilled (2026): same window, true colour + shoreline."""
    lake = "travis"
    zone, transform, _ = aoi.load_zone(lake)
    jrc = aoi.load_jrc(lake)
    fluct = zone & (jrc > 0) & (jrc < 90)
    h, w = 1500, 2100
    y, x = best_window(fluct, h, w, step=50)
    sl = (slice(y, y + h), slice(x, x + w))
    lv = levels.load(lake)
    pred = C.OUT_DIR / "label_scenes"
    dates = [("2018-10-27", "Flood crest week"), ("2023-10-16", "Drought low"),
             ("2026-07-27", "Refilled after July 2025 floods")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
    px_ac = (transform.a * -transform.e) / C.ACRE_M2
    for ax, (d, cap) in zip(axes, dates):
        with rasterio.open(s2_path(lake, d)) as src:
            refl = src.read()
        valid = zone & (refl[2] > 0)
        ax.imshow(rgb(refl, valid, sl))
        p = _read_q(pred / f"{lake}_{d}_ens_mean.tif")[0] / 250
        ax.contour(p[sl], levels=[0.5], colors=[S2C], linewidths=1.0)
        area = ((p > 0.5) & zone).sum() * px_ac
        lvl = lv.water_level.get(pd.Timestamp(d), np.nan)
        tw = lv.surface_area.get(pd.Timestamp(d), np.nan)
        ax.set_title(f"{cap}, {d}\nlevel {lvl:.1f} ft · mapped {area:,.0f} ac · TWDB {tw:,.0f} ac",
                     loc="left", fontsize=9.5, color=INK)
        _clean(ax)
    _scalebar(axes[0], w)
    axes[0].legend(handles=[Line2D([], [], color=S2C, lw=1.5, label="U-Net shoreline (ensemble)")],
                   loc="upper right", fontsize=8.5, framealpha=0.85)
    fig.suptitle("Lake Travis, lower basin: the same view at three levels (true colour)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "blog_travis_before_after.png")


# --------------------------------------------------------------------------------------------
def _s2_day(lake, date):
    zone, transform, crs = aoi.load_zone(lake)
    gb = stac.geobox(transform, zone.shape, crs)
    return stac.load_s2(_fetch_day("sentinel-2-l2a", lake, date), gb)


def _s1_day(lake, date):
    zone, transform, crs = aoi.load_zone(lake)
    gb = stac.geobox(transform, zone.shape, crs)
    return stac.load_s1(_fetch_day("sentinel-1-rtc", lake, date), gb)


def index_vs_unet(index="mndwi", cases=None):
    """Where a tuned index threshold and the U-Net disagree on Travis. By default shows the date
    with the largest area gap between them and a date with a median gap."""
    lake = "travis"
    NAME = index.upper()
    zone, _transform, _ = aoi.load_zone(lake)
    cut = json.loads((C.RESULTS_DIR / "cutoffs.json").read_text())[f"{index}:{lake}"]["cutoff"]
    tl = load_timeline(lake)
    if cases is None:
        c = tl[(tl.sensor == "s2") & (tl.kind == "clear")].dropna(subset=[f"area_{index}"])
        gap = (c[f"area_{index}"] - c.area_model).abs()
        worst = c.loc[gap.idxmax(), "date"]
        typ = c.loc[(gap - gap.median()).abs().idxmin(), "date"]
        cases = ((f"{worst:%Y-%m-%d}", f"Largest gap, {NAME} vs U-Net"),
                 (f"{typ:%Y-%m-%d}", "A typical date (gap near the median)"))
    fig, axes = plt.subplots(len(cases), 3, figsize=(14, 4.4 * len(cases)), facecolor=SURF)
    h, w = 700, 1000
    for i, (d, cap) in enumerate(cases):
        s = _s2_day(lake, d)
        refl, scl = s["refl"], s["scl"]
        valid = zone & (refl[2] > 0) & ~np.isin(scl, C.SCL_INVALID)
        q = _read_q(TL_DIR / lake / f"s2_{d}.tif")
        u = (q[0] != 255) & (q[0] / 250 > 0.5)
        f = indices.INDEX_FUNCS[index]
        m = f(refl) > cut
        dis = valid & (u ^ m)
        y, x = best_window(dis, h, w, step=25)
        sl = (slice(y, y + h), slice(x, x + w))
        cat = np.zeros((h, w, 3))
        cat[:] = LAND
        both, uo, mo = (u & m)[sl], (u & ~m)[sl], (~u & m)[sl]
        cat[both] = _hex(S2C)
        cat[uo] = _hex(S1C)
        cat[mo] = _hex(IDXC)
        cat[~zone[sl]] = OUTSIDE
        mn = f(refl)[sl]
        row = tl[(tl.sensor == "s2") & (tl.date == pd.Timestamp(d))].iloc[0]
        axes[i, 0].imshow(rgb(refl, zone & (refl[2] > 0), sl))
        axes[i, 0].set_title(f"{cap}: {d}, level {row.level:.1f} ft\nTrue colour", loc="left",
                             fontsize=9.5, color=INK)
        im = axes[i, 1].imshow(mn, cmap="BrBG", vmin=-0.6, vmax=0.6)
        axes[i, 1].set_title(f"{NAME} (water if > {cut:.3f})", loc="left", fontsize=9.5, color=INK)
        cb = fig.colorbar(im, ax=axes[i, 1], fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=7.5, colors=MUTED)
        axes[i, 2].imshow(cat)
        axes[i, 2].set_title(f"Whole lake: U-Net {row.area_model:,.0f} ac · {NAME} "
                             f"{row[f'area_{index}']:,.0f} ac · TWDB {row.twdb_area:,.0f} ac",
                             loc="left", fontsize=9.5, color=INK)
        for ax in axes[i]:
            _clean(ax)
    axes[0, 2].legend(handles=[Patch(color=S2C, label="Water in both"),
                               Patch(color=S1C, label="Water for U-Net only"),
                               Patch(color=IDXC, label=f"Water for {NAME} only"),
                               Patch(color=OUTSIDE, label="Outside lake zone")],
                      loc="lower right", fontsize=8, framealpha=0.9)
    _scalebar(axes[0, 0], w, km=1)
    fig.suptitle(f"Where a tuned {NAME} threshold and the U-Net disagree (Lake Travis; window "
                 "with the most disagreement)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, f"blog_{index}_vs_unet.png")


def ndwi_vs_unet():
    index_vs_unet("ndwi")




# --------------------------------------------------------------------------------------------
def radar_vs_cloud(lake="travis", d2="2023-02-03", d1="2023-02-04"):
    """A cloudy optical day and the radar pass one day later."""
    zone, _transform, _ = aoi.load_zone(lake)
    tl = load_timeline(lake)
    s = _s2_day(lake, d2)
    refl = s["refl"]
    db = _s1_day(lake, d1)
    q2 = _read_q(TL_DIR / lake / f"s2_{d2}.tif")
    q1 = _read_q(TL_DIR / lake / f"s1_{d1}.tif")
    jrc = aoi.load_jrc(lake)
    h, w = 1500, 2100
    y, x = best_window(zone & (jrc > 0) & (jrc < 90), h, w, step=50)
    sl = (slice(y, y + h), slice(x, x + w))
    r2 = tl[(tl.sensor == "s2") & (tl.date == pd.Timestamp(d2))].iloc[0]
    r1 = tl[(tl.sensor == "s1") & (tl.date == pd.Timestamp(d1))].iloc[0]

    def wmap(q):
        c = np.zeros(q.shape[1:] + (3,))
        c[:] = LAND
        c[(q[0] != 255) & (q[0] / 250 > 0.5)] = _hex(S2C)
        c[~zone] = OUTSIDE
        return c[sl]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), facecolor=SURF)
    axes[0].imshow(rgb(refl, zone & (refl[2] > 0), sl, gamma=1.0))
    axes[0].set_title(f"Sentinel-2, {d2} ({r2.clear_frac:.0%} of lake cloud-free)", loc="left",
                      fontsize=9.5, color=INK)
    axes[1].imshow(wmap(q2))
    axes[1].set_title(f"S2 U-Net on the cloudy image: {r2.area_model:,.0f} ac", loc="left",
                      fontsize=9.5, color=INK)
    vv = db[0][sl]
    axes[2].imshow(vv, cmap="gray", vmin=-25, vmax=-3)
    axes[2].set_title(f"Sentinel-1 VV backscatter, {d1} (dark = smooth water)", loc="left",
                      fontsize=9.5, color=INK)
    axes[3].imshow(wmap(q1))
    axes[3].set_title(f"S1 U-Net: {r1.area_model:,.0f} ac · TWDB {r1.twdb_area:,.0f} ac",
                      loc="left", fontsize=9.5, color=INK)
    for ax in axes:
        _clean(ax)
    _scalebar(axes[0], w)
    axes[3].legend(handles=[Patch(color=S2C, label="Mapped water"),
                            Patch(color=OUTSIDE, label="Outside lake zone")],
                   loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"{C.lakes()[lake].name}: optical fails under cloud, radar one day later does "
                 "not (whole-lake areas in titles)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "blog_radar_vs_cloud.png")


# --------------------------------------------------------------------------------------------
def uncertainty_zoom(lake="buchanan", date="2026-07-27"):
    """True colour, ensemble probability, ensemble spread, with hand points marked right/wrong."""
    zone, _, _ = aoi.load_zone(lake)
    pred = C.OUT_DIR / "label_scenes"
    p = _read_q(pred / f"{lake}_{date}_ens_mean.tif")[0] / 250
    sd = _read_q(pred / f"{lake}_{date}_ens_std.tif")[0] / 250
    with rasterio.open(s2_path(lake, date)) as src:
        refl = src.read()
    pts = pd.read_csv(C.RESULTS_DIR / "points_scored.csv")
    pts = pts[(pts.lake == lake) & (pts.date == date) & pts.label.isin([0, 1])]
    h, w = 450, 650
    y, x = best_window(np.where(zone, sd, 0), h, w, step=25)
    sl = (slice(y, y + h), slice(x, x + w))
    inwin = pts[(pts.row >= y) & (pts.row < y + h) & (pts.col >= x) & (pts.col < x + w)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9), facecolor=SURF)
    axes[0].imshow(rgb(refl, zone & (refl[2] > 0), sl))
    axes[0].set_title(f"{C.lakes()[lake].name}, {date}: true colour (turbid water)", loc="left",
                      fontsize=9.5, color=INK)
    im1 = axes[1].imshow(np.where(zone[sl], p[sl], np.nan), cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Ensemble water probability", loc="left", fontsize=9.5, color=INK)
    im2 = axes[2].imshow(np.where(zone[sl], sd[sl], np.nan), cmap="Oranges", vmin=0, vmax=0.3)
    axes[2].set_title("Ensemble spread (std of 3 models)", loc="left", fontsize=9.5, color=INK)
    for im, ax in ((im1, axes[1]), (im2, axes[2])):
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
        cb.ax.tick_params(labelsize=7.5, colors=MUTED)
    for ax in axes[1:]:
        ax.set_facecolor(OUTSIDE)
        ok = inwin[inwin.pred_ens_mean == inwin.label]
        bad = inwin[inwin.pred_ens_mean != inwin.label]
        ax.scatter(ok.col - x, ok.row - y, s=26, facecolors="none", edgecolors=INK, lw=1)
        ax.scatter(bad.col - x, bad.row - y, s=50, marker="x", color="#d03b3b", lw=2)
    for ax in axes:
        _clean(ax)
    axes[2].legend(handles=[Line2D([], [], ls="", marker="o", mfc="none", mec=INK,
                                   label="hand point, model right"),
                            Line2D([], [], ls="", marker="x", color="#d03b3b", mew=2,
                                   label="hand point, model wrong")],
                   loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("Uncertainty concentrates on the shoreline, and so do the errors (window with "
                 "the largest spread)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "blog_uncertainty.png")


# --------------------------------------------------------------------------------------------
def noise_floor():
    lakes = ["lbj", "inks", "marble_falls"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.6), sharex=True, facecolor=SURF)
    for ax, lake in zip(axes, lakes):
        _style(ax)
        df = load_timeline(lake)
        lv = levels.load(lake)
        s2 = df[(df.sensor == "s2") & (df.kind == "clear")].sort_values("date")
        ax.plot(lv.index, lv.surface_area, color=INK, lw=1.2, label="TWDB area (from level)")
        ax.plot(s2.date, s2.area_mndwi, "s", ms=3, color=IDXC, alpha=.85, label="MNDWI threshold")
        if "area_ndwi" in s2:
            ax.plot(s2.date, s2.area_ndwi, "D", ms=2.8, color="#7a5af5", alpha=.85,
                    label="NDWI threshold")
        ax.plot(s2.date, s2.area_model, "o", ms=3.6, color=S2C, label="S2 U-Net",
                markeredgecolor=SURF, markeredgewidth=.5)
        st = s2[(s2.level - lv.water_level.median()).abs() < 0.5]
        cv_u = st.area_model.std() / st.area_model.mean() * 100
        cv_m = st.area_mndwi.std() / st.area_mndwi.mean() * 100
        cv_n = st.area_ndwi.std() / st.area_ndwi.mean() * 100
        ax.set_title(f"{C.lakes()[lake].name}: spread on steady-level days (CV) U-Net "
                     f"{cv_u:.1f} %, NDWI {cv_n:.1f} %, MNDWI {cv_m:.1f} %", loc="left",
                     fontsize=10, color=INK)
        ax.set_ylabel("acres", color=INK2, fontsize=9)
        med = lv.water_level.median()
        near = lv[(lv.water_level - med).abs() < 0.2].surface_area
        yr = near.groupby(near.index.year).median()
        if yr.iloc[-1] / yr.iloc[0] - 1 > 0.03:
            step = near[near > (yr.iloc[0] + yr.iloc[-1]) / 2].index.min()
            ax.annotate(f"TWDB area steps +{(yr.iloc[-1] / yr.iloc[0] - 1) * 100:.0f} % at the "
                        "same level\n(likely a rating-curve revision)",
                        (step, (yr.iloc[0] + yr.iloc[-1]) / 2),
                        xytext={"lbj": (25, -30), "inks": (40, 45), "marble_falls": (70, -115)}[lake],
                        textcoords="offset points", fontsize=8, color=INK2,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    axes[0].legend(frameon=False, ncol=4, fontsize=8.5, loc="lower right", labelcolor=INK2)
    fig.suptitle("Pass-through lakes: near-constant level, occasional drawdowns (the dips)", x=0.01,
                 ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "blog_noise_floor.png")


def lbj_drawdown():
    """LBJ was drawn down ~4 ft in early 2020; the exposed bars are visible in the imagery."""
    lake = "lbj"
    zone, _transform, _ = aoi.load_zone(lake)
    tl = load_timeline(lake)
    s2 = tl[(tl.sensor == "s2") & (tl.kind == "clear")].dropna(subset=["level"])
    lo = s2.loc[s2.level.idxmin()]
    hi = s2[(s2.date.dt.year == lo.date.year + 1) | (s2.date.dt.year == lo.date.year - 1)]
    hi = hi.loc[hi.level.idxmax()]
    imgs = []
    for r in (lo, hi):
        d = r.date.strftime("%Y-%m-%d")
        s = _s2_day(lake, d)
        imgs.append((r, s["refl"], _read_q(TL_DIR / lake / f"s2_{d}.tif")))
    q_lo, q_hi = imgs[0][2][0], imgs[1][2][0]
    change = zone & (q_hi != 255) & (q_lo != 255) & (q_hi / 250 > 0.5) & (q_lo / 250 <= 0.5)
    h, w = 500, 700
    y, x = best_window(change, h, w, step=20)
    sl = (slice(y, y + h), slice(x, x + w))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF)
    for ax, (r, refl, q) in zip(axes, imgs):
        ax.imshow(rgb(refl, zone & (refl[2] > 0), sl))
        ax.contour((q[0] / 250)[sl], levels=[0.5], colors=[S2C], linewidths=1.1)
        ax.set_title(f"{r.date:%Y-%m-%d}: level {r.level:.1f} ft · mapped {r.area_model:,.0f} ac",
                     loc="left", fontsize=9.5, color=INK)
        _clean(ax)
    _scalebar(axes[0], w, km=1)
    fig.suptitle("Lake LBJ is not quite constant: a ~4 ft drawdown exposes sandbars (U-Net "
                 "shoreline in blue)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    _save(fig, "blog_lbj_drawdown.png")


def label_examples(pids=None):
    """What the labeller saw: true-colour and SWIR false-colour chips around single points."""
    from .handlabel import CROP, _stretch
    pts = pd.read_csv(C.LABELS_DIR / "hand_points.csv")
    from .evaluate import PRIMARY_LABELS
    if PRIMARY_LABELS.exists():  # show the primary (human) labels
        h = pd.read_csv(PRIMARY_LABELS)[["pid", "label"]].rename(columns={"label": "label_h"})
        pts = pts.merge(h, on="pid", how="left")
        pts["label"] = pts.label_h.fillna(pts.label).astype(int)
    if pids is None:
        pick = []
        rng = np.random.default_rng(3)
        for lab, n in ((1, 2), (0, 2), (-1, 2)):
            cand = pts[(pts.label == lab) & (pts.stratum == "B")]
            pick += list(cand.pid.values[rng.choice(len(cand), n, replace=False)])
        pids = pick
    b = {n: i for i, n in enumerate(C.S2_BANDS)}
    tc, fc = [b["B04"], b["B03"], b["B02"]], [b["B12"], b["B8A"], b["B04"]]
    fig, axes = plt.subplots(2, len(pids), figsize=(2.3 * len(pids), 5), facecolor=SURF)
    names = {1: "water", 0: "not water", -1: "unsure (dropped)"}
    for j, pid in enumerate(pids):
        p = pts[pts.pid == pid].iloc[0]
        with rasterio.open(s2_path(p.lake, p.date)) as src:
            a = src.read()
        m = a[2] > 0
        r0, c0 = int(p.row) - CROP // 2, int(p.col) - CROP // 2
        crop = a[:, r0:r0 + CROP, c0:c0 + CROP]
        for i, idx in enumerate((tc, fc)):
            v = a[idx][:, m]
            lo, hi = np.percentile(v, 1, axis=1), np.percentile(v, 99, axis=1)
            axes[i, j].imshow(_stretch(crop[idx], lo, hi), interpolation="nearest")
            axes[i, j].add_patch(matplotlib.patches.Rectangle((CROP // 2 - .5, CROP // 2 - .5), 1, 1,
                                                              fill=False, ec="#ff00ff", lw=1.5))
            _clean(axes[i, j])
        axes[0, j].set_title(f"{names[int(p.label)]}\n{C.lakes()[p.lake].name.replace('Lake ', '')}"
                             f" {p.date}", fontsize=8.5, color=INK)
    axes[0, 0].set_ylabel("true colour", fontsize=9, color=INK2)
    axes[1, 0].set_ylabel("SWIR / NIR / red", fontsize=9, color=INK2)
    fig.suptitle("What the labeller saw: 400 m chips around each point (magenta square = the "
                 "10 m pixel being labelled)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, "blog_label_examples.png")


def main():
    for f in (study_area, travis_before_after, ndwi_vs_unet, radar_vs_cloud,
              uncertainty_zoom,
              noise_floor, lbj_drawdown, label_examples):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print("FAILED", f.__name__, repr(e)[:300], flush=True)


if __name__ == "__main__":
    main()
