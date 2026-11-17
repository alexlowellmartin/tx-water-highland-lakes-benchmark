"""Compare mapped area with the official TWDB record; summarise all results.

Outputs results/area_vs_level.csv, results/cloudy.csv, results/figures/*.png.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import levels
from .timeline import TL_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = C.RESULTS_DIR / "figures"
# validated reference palette (dataviz skill), light mode
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
S2C, S1C, IDXC, NDC = "#2a78d6", "#eb6834", "#1baf7a", "#7a5af5"  # categorical slots 1-4
METHODS = {  # column -> (label, color)
    "area_model": ("S2 U-Net (ensemble)", S2C),
    "area_ndwi": ("NDWI threshold", NDC),
    "area_ndwi_otsu": ("NDWI, per-scene Otsu", NDC),
    "area_mndwi": ("MNDWI threshold", IDXC),
    "area_s1": ("S1 U-Net", S1C),
    "area_vv": ("S1 VV threshold", MUTED),
    "area_scl": ("S2 SCL water class", INK2),
}


def load_timeline(lake: str) -> pd.DataFrame:
    df = pd.read_csv(TL_DIR / f"{lake}.csv")
    lv = levels.load(lake)
    df["date"] = pd.to_datetime(df["date"])
    # S1 passes at ~00:30 UTC are the previous local evening; TWDB days are local
    local = df.date - pd.to_timedelta(np.where(df.sensor == "s1", 1, 0), unit="D")
    df["level"] = lv.water_level.reindex(local).values
    df["twdb_area"] = lv.surface_area.reindex(local).values
    return df


def _fit_stats(area, level, twdb_area):
    ok = np.isfinite(area) & np.isfinite(level) & np.isfinite(twdb_area)
    a, lv, t = area[ok], level[ok], twdb_area[ok]
    if ok.sum() < 5:
        return {"n": int(ok.sum())}
    r_level = stats.pearsonr(a, lv)[0]
    rho = stats.spearmanr(a, lv)[0]
    r_area = stats.pearsonr(a, t)[0]
    bias = float(np.mean(a - t))
    rmse = float(np.sqrt(np.mean((a - t) ** 2)))
    # after a linear calibration (absorbs zone-definition bias, e.g. river reaches)
    slope, icpt = np.polyfit(a, t, 1)
    res = t - (slope * a + icpt)
    return {"n": int(ok.sum()), "r_level": r_level, "spearman_level": rho, "r_twdb_area": r_area,
            "bias_acres": bias, "rmse_acres": rmse, "mape_pct": float(np.mean(np.abs(a - t) / t) * 100),
            "rmse_after_linear_acres": float(np.sqrt(np.mean(res**2))),
            "mean_area": float(a.mean()), "std_area": float(a.std()),
            "cv_pct": float(a.std() / a.mean() * 100), "mean_twdb_area": float(t.mean()),
            "std_twdb_area": float(t.std())}


def area_table(lake: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    s2 = df[(df.sensor == "s2") & (df.kind == "clear")]
    s1 = df[df.sensor == "s1"]
    for col, sub, sensor in (("area_model", s2, "s2"), ("area_ndwi", s2, "s2"),
                             ("area_ndwi_otsu", s2, "s2"), ("area_mndwi", s2, "s2"),
                             ("area_scl", s2, "s2"), ("area_model", s1, "s1"),
                             ("area_vv", s1, "s1")):
        if col not in sub or not len(sub):
            continue
        sub = sub[sub.valid_frac > 0.95] if sensor == "s1" else sub
        name = "area_s1" if (sensor == "s1" and col == "area_model") else col
        st = _fit_stats(sub[col].values, sub.level.values, sub.twdb_area.values)
        rows.append({"lake": lake, "method": METHODS[name][0], "sensor": sensor, **st})
    return rows


def cloudy_table(lake: str, df: pd.DataFrame) -> list[dict]:
    """Relative area error vs TWDB on clear vs cloudy optical dates, and S1 on the dates
    with no clear optical image within +/-3 days."""
    out = []
    s2 = df[df.sensor == "s2"]
    s1 = df[(df.sensor == "s1") & (df.valid_frac > 0.95)]
    clear_days = pd.DatetimeIndex(pd.to_datetime(s2[s2.kind == "clear"].date))

    def rel(a, t):
        e = (a - t) / t * 100
        return {"n": int(np.isfinite(e).sum()), "median_abs_err_pct": float(np.nanmedian(np.abs(e))),
                "p90_abs_err_pct": float(np.nanpercentile(np.abs(e), 90)) if np.isfinite(e).any()
                else np.nan}

    c = s2[s2.kind == "clear"]
    out.append({"lake": lake, "case": "S2 U-Net, clear dates", **rel(c.area_model, c.twdb_area)})
    k = s2[s2.kind == "cloudy"]
    if len(k):
        out.append({"lake": lake, "case": "S2 U-Net, cloudy dates (clouds left in)",
                    **rel(k.area_model, k.twdb_area)})
        out.append({"lake": lake, "case": "S2 U-Net, cloudy dates (clouds masked)",
                    **rel(k.area_model_cloudmasked, k.twdb_area)})
    out.append({"lake": lake, "case": "S1 U-Net, all dates", **rel(s1.area_model, s1.twdb_area)})
    gap = s1[[np.min(np.abs((clear_days - d).days)) > 3 if len(clear_days) else True
              for d in s1.date]]
    out.append({"lake": lake, "case": "S1 U-Net, dates with no clear S2 within 3 days",
                **rel(gap.area_model, gap.twdb_area)})
    # revisit: months with at least one usable observation
    months = pd.period_range(df.date.min(), df.date.max(), freq="M")
    m2 = set(pd.to_datetime(c.date).dt.to_period("M"))
    m12 = m2 | set(pd.to_datetime(s1.date).dt.to_period("M"))
    out.append({"lake": lake, "case": "months with a usable observation: S2 clear only",
                "n": len(m2), "median_abs_err_pct": len(m2) / len(months) * 100})
    out.append({"lake": lake, "case": "months with a usable observation: S2 clear + S1",
                "n": len(m12), "median_abs_err_pct": len(m12) / len(months) * 100})
    return out


def noise_floor_table(lake: str, df: pd.DataFrame) -> list[dict]:
    """Pass-through lakes: spread of mapped area on days the level is within 0.5 ft of its median
    (the real noise floor), area change during drawdowns (level > 2 ft below median), and the
    TWDB area at the median level by year (reveals rating-curve revisions)."""
    lv = levels.load(lake)
    med = lv.water_level.median()
    near = lv[(lv.water_level - med).abs() < 0.2].surface_area
    by_year = near.groupby(near.index.year).median()
    out = []
    for sensor, cols in (("s2", ["area_model", "area_ndwi", "area_mndwi", "area_scl"]),
                         ("s1", ["area_model", "area_vv"])):
        sub = df[(df.sensor == sensor) & df.level.notna()]
        sub = sub[sub.kind == "clear"] if sensor == "s2" else sub[sub.valid_frac > 0.95]
        steady, down = sub[(sub.level - med).abs() < 0.5], sub[sub.level < med - 2]
        for c in cols:
            name = "area_s1" if (sensor == "s1" and c == "area_model") else c
            out.append({
                "lake": lake, "method": METHODS[name][0], "median_level_ft": med,
                "n_steady": len(steady), "cv_steady_pct": steady[c].std() / steady[c].mean() * 100,
                "n_drawdown": len(down),
                "drawdown_change_pct": (down[c].mean() / steady[c].mean() - 1) * 100
                if len(down) else np.nan,
                "twdb_drawdown_change_pct": (down.twdb_area.mean() / steady.twdb_area.mean() - 1)
                * 100 if len(down) else np.nan,
                "twdb_area_at_median_level_first_year": float(by_year.iloc[0]),
                "twdb_area_at_median_level_last_year": float(by_year.iloc[-1])})
    return out


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures(lake: str, df: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    lv = levels.load(lake)
    s2 = df[(df.sensor == "s2") & (df.kind == "clear")].sort_values("date")
    s1 = df[(df.sensor == "s1") & (df.valid_frac > 0.95)].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=SURF)
    _style(ax)
    ax.plot(lv.index, lv.surface_area, color=INK, lw=1.4, label="TWDB area (from level)")
    ax.plot(s1.date, s1.area_model, "o", ms=3.2, color=S1C, alpha=.75, label="S1 U-Net")
    ax.plot(s2.date, s2.area_mndwi, "s", ms=3.2, color=IDXC, alpha=.8, label="MNDWI threshold")
    if "area_ndwi" in s2:
        ax.plot(s2.date, s2.area_ndwi, "D", ms=3, color=NDC, alpha=.8, label="NDWI threshold")
    ax.plot(s2.date, s2.area_model, "o", ms=3.8, color=S2C, label="S2 U-Net",
            markeredgecolor=SURF, markeredgewidth=.6)
    ax.set_ylabel("Surface area (acres)", color=INK2)
    ax.set_title(f"{C.lakes()[lake].name}: mapped area vs official record", loc="left",
                 color=INK, fontsize=12)
    ax.legend(frameon=False, ncol=5, fontsize=9, loc="upper left", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(FIG / f"{lake}_timeseries.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURF)
    _style(ax)
    ax.plot(s1.level, s1.area_model, "o", ms=3.2, color=S1C, alpha=.6, label="S1 U-Net")
    ax.plot(s2.level, s2.area_model, "o", ms=4, color=S2C, label="S2 U-Net",
            markeredgecolor=SURF, markeredgewidth=.6)
    curve = lv.dropna(subset=["water_level", "surface_area"]).sort_values("water_level")
    ax.plot(curve.water_level, curve.surface_area, color=INK, lw=1.2, label="TWDB rating curve")
    ax.set_xlabel("Official water level (ft above MSL)", color=INK2)
    ax.set_ylabel("Surface area (acres)", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    ax.set_title(C.lakes()[lake].name, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"{lake}_area_vs_level.png", dpi=130)
    plt.close(fig)


REL_EDGES = [0, 0.05, 0.5, 0.95, 1.0000001]


def reliability_figure():
    """Coarse reliability diagram. Almost all held-out points get p < 0.05 or p > 0.95, so the
    usual 10 bins leave 2-8 points in each middle bin; four bins with counts and Wilson
    intervals are more honest. ECE (10 bins) is from evaluate.py."""
    p = C.RESULTS_DIR / "points_scored.csv"
    if not p.exists():
        return
    cal = json.loads((C.RESULTS_DIR / "calibration.json").read_text())
    df = pd.read_csv(p)
    df = df[df.group.notna()]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=SURF)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    names = {"p_ens_mean": ("S2 deep ensemble", S2C, -0.012),
             "p_mc_mean": ("S2 MC dropout", IDXC, 0.0), "p_s1": ("S1 U-Net", S1C, 0.012)}
    for k, (lab, col, dx) in names.items():
        if k not in df:
            continue
        d = df[df[k].notna()]
        b = np.digitize(d[k], REL_EDGES) - 1
        xs, ys, lo, hi, ns = [], [], [], [], []
        for i in range(len(REL_EDGES) - 1):
            m = b == i
            n = int(m.sum())
            if not n:
                continue
            k_ = d.label[m].sum()
            ph = k_ / n
            z = 1.96  # Wilson interval
            c = (ph + z * z / (2 * n)) / (1 + z * z / n)
            h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            xs.append(d[k][m].mean() + dx)
            ys.append(ph)
            lo.append(ph - (c - h))
            hi.append(c + h - ph)
            ns.append(n)
        e = cal.get(k, {}).get("ece", np.nan)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=col, ms=6, lw=1.6, capsize=3,
                    label=f"{lab} (ECE {e:.3f})", markeredgecolor=SURF, markeredgewidth=.8)
        if k == "p_ens_mean":
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(f"n={n}", (x, y), xytext=(6, -12), textcoords="offset points",
                            fontsize=8, color=INK2)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("Predicted water probability (bins: <.05, .05-.5, .5-.95, >.95)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("Observed water fraction (hand labels)", color=INK2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    ax.set_title("Calibration on held-out points (95 % intervals)", loc="left", color=INK,
                 fontsize=11)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "reliability.png", dpi=130)
    plt.close(fig)


WEAK_CASES = [  # (lake, date, row0, row1, col0, col1, caption)
    ("travis", "2023-10-16", 1500, 2300, 2700, 3700, "Drought low (627.7 ft)"),
    ("buchanan", "2026-07-27", 300, 1100, 300, 1300, "Turbid, refilled lake (Jul 2026)"),
    ("travis", "2018-10-27", 1500, 2300, 2700, 3700, "Flood crest week (Oct 2018)"),
]


def weak_label_figure():
    """RGB | static JRC>=50 % | per-scene SCL (training label) | U-Net, for three cases."""
    import rasterio

    from . import aoi
    from . import labels as WL
    from .dataset import s2_path, s2_scl_path

    pred_dir = C.OUT_DIR / "label_scenes"
    fig, axes = plt.subplots(len(WEAK_CASES), 4, figsize=(13, 3.3 * len(WEAK_CASES)),
                             facecolor=SURF)
    cols = ["True colour", "JRC occurrence >= 50 % (static)", "S2 SCL water (per scene)",
            "S2 U-Net ensemble"]
    for i, (lake, date, r0, r1, c0, c1, cap) in enumerate(WEAK_CASES):
        sl = (slice(r0, r1), slice(c0, c1))
        with rasterio.open(s2_path(lake, date)) as src:
            refl = src.read()
        with rasterio.open(s2_scl_path(lake, date)) as src:
            scl = src.read(1)
        zone, _, _ = aoi.load_zone(lake)
        static = WL.static_layers(lake)
        rgb = np.clip(refl[[3, 2, 1]][:, sl[0], sl[1]].transpose(1, 2, 0) / 2200, 0, 1) ** 0.85
        jrc = WL.weak_label("jrc50", scl, zone, static, int(date[:4]))[sl]
        sclw = WL.weak_label("scl", scl, zone, static, int(date[:4]))[sl]
        pf = pred_dir / f"{lake}_{date}_ens_mean.tif"
        ens = rasterio.open(pf).read(1)[sl] / 250 if pf.exists() else np.zeros_like(jrc, float)
        z = zone[sl]
        for j, img in enumerate([rgb, jrc, sclw, ens]):
            ax = axes[i, j]
            if j == 0:
                ax.imshow(img)
            else:
                m = np.zeros(img.shape + (3,))
                m[:] = (0.96, 0.95, 0.93)
                water = (img == 1) if j < 3 else (img > 0.5)
                m[water] = (0.165, 0.47, 0.84)
                m[~z] = (0.85, 0.85, 0.83)
                if j < 3:
                    m[img == WL.IGNORE] = (0.85, 0.85, 0.83)
                ax.imshow(m)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(cols[j], fontsize=10, color=INK, loc="left")
        axes[i, 0].set_ylabel(f"{C.lakes()[lake].name}\n{cap}", fontsize=9.5, color=INK2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "weak_labels.png", dpi=110)
    plt.close(fig)


def main():
    rows, cloudy, floor = [], [], []
    for lake in C.lakes():
        if not (TL_DIR / f"{lake}.csv").exists():
            print("no timeline for", lake)
            continue
        df = load_timeline(lake)
        rows += area_table(lake, df)
        cloudy += cloudy_table(lake, df)
        if not C.lakes()[lake].dynamic:
            floor += noise_floor_table(lake, df)
        figures(lake, df)
    a = pd.DataFrame(rows)
    a.to_csv(C.RESULTS_DIR / "area_vs_level.csv", index=False)
    pd.DataFrame(cloudy).to_csv(C.RESULTS_DIR / "cloudy.csv", index=False)
    pd.DataFrame(floor).to_csv(C.RESULTS_DIR / "noise_floor.csv", index=False)
    reliability_figure()
    weak_label_figure()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(a[["lake", "method", "n", "r_level", "r_twdb_area", "bias_acres", "rmse_acres",
                 "rmse_after_linear_acres", "cv_pct"]].round(3).to_string(index=False))
        print(pd.DataFrame(cloudy).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
