"""Figures for the small-reservoir transfer test (results/transfer/*.png)."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import rasterio

from . import levels, transfer
from .analyze import INK, INK2, MUTED, SURF, _style

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S2C, S1C, IDXC, VVC, NDC = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#7a5af5"
OUT = transfer.OUT


def _lakes_done():
    return {k: v for k, v in transfer.lakes().items()
            if (transfer.TL_DIR / f"{k}.csv").exists()}


def _order(lakes):
    """Smallest to largest by TWDB maximum area."""
    return sorted(lakes, key=lambda k: levels.load(k).surface_area.max())


def timeseries():
    L = _lakes_done()
    keys = _order(L)
    n = len(keys)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.35 * rows), facecolor=SURF, squeeze=False)
    for ax, k in zip(axes.flat, keys):
        _style(ax)
        df = transfer.joined(k)
        lv = levels.load(k)
        s2 = df[(df.sensor == "s2") & (df.kind == "clear")]
        s1 = df[(df.sensor == "s1") & (df.valid_frac > 0.95)]
        ax.plot(lv.index, lv.surface_area, color=INK, lw=1.2, zorder=1)
        ax.plot(s1.date, s1.area_vv_hl, ".", ms=2.5, color=VVC, alpha=0.5, zorder=2)
        ax.plot(s1.date, s1.area_model, "o", ms=2.3, color=S1C, alpha=0.75, zorder=3)
        ax.plot(s2.date, s2.area_mndwi_hl, "s", ms=2.6, color=IDXC, alpha=0.85, zorder=4)
        ax.plot(s2.date, s2.area_ndwi_hl, "D", ms=2.4, color=NDC, alpha=0.85, zorder=4)
        ax.plot(s2.date, s2.area_model, "o", ms=2.6, color=S2C, zorder=5)
        top = np.nanpercentile(np.r_[lv.surface_area.values, s2.area_model.values], 99.5) * 1.35
        ax.set_ylim(0, top)
        ax.set_title(f"{L[k].name}  (TWDB max {lv.surface_area.max():,.0f} ac)", loc="left",
                     fontsize=9.5, color=INK)
        ax.tick_params(labelsize=8)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=INK, lw=1.2, label="TWDB area (from level; not used by any map)"),
               Line2D([], [], marker="o", ls="", color=S2C, label="S2 U-Net"),
               Line2D([], [], marker="D", ls="", color=NDC, label="NDWI (Highland Lakes cutoff)"),
               Line2D([], [], marker="s", ls="", color=IDXC, label="MNDWI (Highland Lakes cutoff)"),
               Line2D([], [], marker="o", ls="", color=S1C, label="S1 U-Net (radar)"),
               Line2D([], [], marker=".", ls="", color=VVC, label="VV threshold")]
    fig.legend(handles=handles, loc="upper left", ncol=6, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.01, 1.0))
    fig.suptitle("Ten small reservoirs the models never saw: mapped area (acres) vs the TWDB record",
                 x=0.01, y=1.025, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / "transfer_timeseries.png", dpi=110, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


METHOD_STYLE = {"S2 U-Net": (S2C, "o"), "NDWI (HL cutoff)": (NDC, "D"), "NDWI (Otsu)": (NDC, "^"),
                "MNDWI (HL cutoff)": (IDXC, "s"),
                "MNDWI (Otsu)": (IDXC, "D"), "MNDWI (0)": (IDXC, "^"), "SCL water": (INK2, "v"),
                "S1 U-Net": (S1C, "o"), "VV (HL cutoff)": (VVC, "s"), "VV (Otsu)": (VVC, "D")}


def error_vs_size():
    m = pd.read_csv(OUT / "metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURF, sharey=True)
    for ax, sensor, title in ((axes[0], "s2", "Optical (clear Sentinel-2 dates)"),
                              (axes[1], "s1", "Radar (all Sentinel-1 dates)")):
        _style(ax)
        d = m[m.sensor == sensor]
        for meth, g in d.groupby("method"):
            c, mk = METHOD_STYLE.get(meth, (MUTED, "o"))
            filled = meth in ("S2 U-Net", "S1 U-Net")
            g = g.sort_values("twdb_max_acres")
            ax.plot(g.twdb_max_acres, g.median_abs_pct_err, marker=mk, ls="-" if filled else ":",
                    lw=1 if filled else 0.8, ms=6, color=c, mfc=c if filled else "white",
                    label=meth)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
        ax.set(xlabel="reservoir size: TWDB max area 2018-2026 (acres, log)",
               title=title)
        ax.title.set_color(INK)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("median |error| vs TWDB area (%, log)")
    fig.suptitle("Error against reservoir size (each point is one reservoir)", x=0.01,
                 ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_error_vs_size.png", dpi=110, facecolor=SURF)
    plt.close(fig)


def _prob(key, sensor, date):
    with rasterio.open(transfer.TL_DIR / key / f"{sensor}_{pd.Timestamp(date):%Y-%m-%d}.tif") as s:
        return s.read(1)


def maps():
    """Each lake at its lowest and highest mapped (clear S2) area, as mapped by the S2 U-Net."""
    L = _lakes_done()
    keys = _order(L)
    fig, axes = plt.subplots(2, len(keys), figsize=(2.1 * len(keys), 4.6), facecolor=SURF,
                             squeeze=False)
    for j, k in enumerate(keys):
        df = transfer.joined(k)
        s2 = df[(df.sensor == "s2") & (df.kind == "clear")].dropna(subset=["twdb_area"])
        for i, row in enumerate((s2.loc[s2.twdb_area.idxmin()], s2.loc[s2.twdb_area.idxmax()])):
            q = _prob(k, "s2", row.date)
            rr, cc = np.nonzero(q != 255)  # crop to the zone (older rasters cover the full box)
            q = q[max(rr.min() - 10, 0): rr.max() + 11, max(cc.min() - 10, 0): cc.max() + 11]
            img = np.ones(q.shape + (3,)) * 0.95
            img[q == 255] = 0.86
            w = (q != 255) & (q > 125)
            img[w] = matplotlib.colors.to_rgb(S2C)
            ax = axes[i, j]
            ax.imshow(img, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{row.date:%b %Y}: {row.area_model:,.0f} ac\n(TWDB {row.twdb_area:,.0f})",
                         fontsize=7.5, color=INK2)
        axes[0, j].text(0.5, 1.32, L[k].name.replace(" Reservoir", "").replace(" Lake", ""),
                        transform=axes[0, j].transAxes, ha="center", fontsize=9, color=INK)
    axes[0, 0].set_ylabel("lowest", color=MUTED)
    axes[1, 0].set_ylabel("highest", color=MUTED)
    fig.suptitle("S2 U-Net water (blue) at each reservoir's lowest and highest mapped level; "
                 "grey = outside the lake zone", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "transfer_maps.png", dpi=120, facecolor=SURF)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    timeseries()
    error_vs_size()
    maps()
    print("wrote", OUT, flush=True)



