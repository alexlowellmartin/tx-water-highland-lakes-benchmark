"""Figures for the stock-tank test (results/ponds/*.png)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

from . import ponds as P
from . import stac
from .analyze import INK, INK2, MUTED, SURF, _style
from .io import read_raster

matplotlib.use("Agg")
import matplotlib.pyplot as plt

S2C, S1C, IDXC, NDC, VVC, SCLC = "#2a78d6", "#eb6834", "#1baf7a", "#7a5af5", "#898781", "#4a4a48"
STYLE = {"S2 U-Net": (S2C, "o", "-"), "S2 U-Net (expected area)": (S2C, "o", ":"),
         "NDWI (0)": (NDC, "D", "-"), "NDWI (HL cutoff)": (NDC, "s", ":"),
         "MNDWI (HL cutoff)": (IDXC, "s", "-"), "SCL water": (SCLC, "v", ":"),
         "S1 U-Net": (S1C, "o", "-"), "S1 U-Net (expected area)": (S1C, "o", ":"),
         "VV (HL cutoff)": (VVC, "s", "-")}
OPTICAL = ["S2 U-Net", "NDWI (0)", "NDWI (HL cutoff)", "MNDWI (HL cutoff)", "SCL water"]
RADAR = ["S1 U-Net", "VV (HL cutoff)"]


def detection():
    s = pd.read_csv(P.OUT / "by_size.csv")
    s["size"] = pd.Categorical(s["size"], P.SIZE_LABELS, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), facecolor=SURF, sharey=True)
    for ax, meths, title in ((axes[0], OPTICAL, "Optical (nearest clear Sentinel-2 day)"),
                             (axes[1], RADAR, "Radar (nearest Sentinel-1 day)")):
        _style(ax)
        for m in meths:
            g = s[s.method == m].sort_values("size")
            c, mk, ls = STYLE[m]
            x = g["size"].cat.codes
            ax.plot(x, g.detected_pct, marker=mk, ls=ls, color=c, lw=1.6, ms=6, label=m)
            if m == "VV (HL cutoff)":
                ax.plot(x, g.chance_detected_pct, ls="--", color=VVC, lw=1,
                        label="VV on the same footprints moved onto dry land (chance)")
        ax.set_xticks(range(len(P.SIZE_LABELS)), P.SIZE_LABELS)
        ax.set(xlabel="pond size on the NAIP flight date (acres)", ylim=(-3, 103), title=title)
        ax.title.set_color(INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("wet ponds detected (%)")
    fig.suptitle("Share of NAIP-visible ponds that each method finds at all "
                 "(at least one 10 m water pixel within 20 m)", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.tight_layout()
    fig.savefig(P.OUT / "ponds_detection.png", dpi=110, facecolor=SURF)
    plt.close(fig)


def area_error():
    s = pd.read_csv(P.OUT / "by_size.csv")
    s["size"] = pd.Categorical(s["size"], P.SIZE_LABELS, ordered=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.3), facecolor=SURF)
    _style(ax)
    for m in ["S2 U-Net", "S2 U-Net (expected area)", "NDWI (0)", "MNDWI (HL cutoff)",
              "S1 U-Net"]:
        g = s[s.method == m].sort_values("size")
        c, mk, ls = STYLE[m]
        ax.plot(g["size"].cat.codes, g.median_err_pct, marker=mk, ls=ls, color=c, lw=1.6, ms=6,
                label=m)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(P.SIZE_LABELS)), P.SIZE_LABELS)
    ax.set(xlabel="pond size on the NAIP flight date (acres)",
           ylabel="median area error vs NAIP (%)", ylim=(-105, 30))
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Mapped pond area vs the aerial-photo outline (-100 % = not found)", loc="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(P.OUT / "ponds_area_error.png", dpi=110, facecolor=SURF)
    plt.close(fig)


def _s2_rgb(site, day):
    p = P.P_DIR / site / "s2rgb" / f"{day}.npy"
    if not p.exists():
        tr, shp = P.grid(site)
        items = stac.group_by_day(stac.search("sentinel-2-l2a", P.bbox_ll(site), day,
                                              (pd.Timestamp(day) + pd.Timedelta(days=1))
                                              .strftime("%Y-%m-%d")))[day]
        refl = stac.load_s2(items, stac.geobox(tr, shp, P.CRS), bands=["B04", "B03", "B02"])
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, refl["refl"])
    r = np.load(p).astype(np.float32)
    return np.clip(np.moveaxis(r, 0, -1) / 2200, 0, 1) ** 0.8


def gallery(n=6, seed=3):
    """Ponds of 1-10 acres: NAIP photo, Sentinel-2 image, U-Net and NDWI (0) maps."""
    rows = pd.read_csv(P.OUT / "pond_rows.csv.gz")
    near = P.nearest_only(rows[rows.sensor == "s2"])
    u = near[(near.method == "S2 U-Net") & near.naip_ac.between(1, 10)]
    # one per site where possible, then any
    pick = u.groupby("site").sample(1, random_state=seed)
    pick = pick.sample(min(n, len(pick)), random_state=seed).sort_values("naip_ac")
    names = P.sites()
    fig, axes = plt.subplots(len(pick), 4, figsize=(9.2, 2.45 * len(pick)), facecolor=SURF,
                             squeeze=False)
    for row, r in zip(axes, pick.itertuples()):
        foot = read_raster(P.P_DIR / r.site / "ponds.tif")[0][0]
        rgb = read_raster(P.P_DIR / r.site / f"naiprgb_{r.year}.tif")[0]
        q = read_raster(P.P_DIR / r.site / "s2" / f"{r.flight.replace('-', '')}_{r.date}.tif")[0]
        layers = pd.read_csv(P.P_DIR / r.site / "s2.csv").layers.iloc[0].split(",")
        ii, jj = np.nonzero(foot == r.pond)
        ci, cj, h = int(ii.mean()), int(jj.mean()), 14
        s = (slice(max(ci - h, 0), ci + h), slice(max(cj - h, 0), cj + h))
        nd_val = q[layers.index("ndwi_zero")]
        nd_ac = near[(near.site == r.site) & (near.pond == r.pond) & (near.flight == r.flight)
                     & (near.method == "NDWI (0)")].mapped_ac.iloc[0]
        crop = rgb[:, s[0].start * 5: s[0].stop * 5, s[1].start * 5: s[1].stop * 5]
        panels = [
            (np.moveaxis(crop, 0, -1), f"NAIP {r.flight}\n{r.naip_ac:.1f} ac"),
            (_s2_rgb(r.site, r.date)[s], f"Sentinel-2 {r.date}\n({r.gap_days:+d} days)"),
            (q[0][s] / 250, f"S2 U-Net probability\nmapped {r.mapped_ac:.1f} ac"),
            (nd_val[s] / 250, f"NDWI > 0\nmapped {nd_ac:.1f} ac"),
        ]
        for k, (ax, (img, title)) in enumerate(zip(row, panels)):
            if k < 2:
                ax.imshow(img, interpolation="nearest")
            else:
                ax.imshow(np.where(img > 1, np.nan, img), vmin=0, vmax=1, cmap="Blues",
                          interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#d6d4cf")
            ax.set_title(title, fontsize=8, color=INK2)
        row[0].set_ylabel(names[r.site]["name"].split(" (")[0], fontsize=8.5, color=INK)
    fig.suptitle("Random ponds of 1-10 acres (one per site): aerial photo, Sentinel-2, and two maps",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(P.OUT / "ponds_gallery.png", dpi=110, facecolor=SURF)
    plt.close(fig)


def main():
    P.OUT.mkdir(parents=True, exist_ok=True)
    detection()
    area_error()
    gallery()
    print("wrote", P.OUT, flush=True)
