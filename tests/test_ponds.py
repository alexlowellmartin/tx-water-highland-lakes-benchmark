import numpy as np
import pandas as pd

from hlw import ponds


def _naip(water_mask):
    """Synthetic NAIP: land has NIR > green (NDWI < 0), water the reverse (NDWI ~ 0.5)."""
    img = np.zeros((4,) + water_mask.shape, np.uint8)
    img[0], img[2] = 80, 60
    img[1] = np.where(water_mask, 90, 70)
    img[3] = np.where(water_mask, 30, 120)
    return img


def test_naip_water_keeps_ponds_drops_strips_and_specks():
    yy, xx = np.mgrid[:200, :200]
    pond = (yy - 60) ** 2 + (xx - 60) ** 2 <= 15 ** 2  # ~700 m2 disc
    strip = (yy >= 150) & (yy < 156) & (xx >= 20) & (xx < 180)  # 6 m x 160 m shadow line
    speck = (yy - 60) ** 2 + (xx - 150) ** 2 <= 5 ** 2  # ~80 m2: below the 200 m2 minimum
    w = ponds.naip_water(_naip(pond | strip | speck))
    assert w[60, 60] and abs(w.sum() - pond.sum()) < 0.1 * pond.sum()
    assert not w[153, 100]
    assert not w[60, 150]


def test_block_mean_aggregates_to_10m():
    a = np.zeros((20, 20))
    a[:10, :10] = 1
    a[10:, 10:15] = 1
    assert np.allclose(ponds.block_mean(a), [[1, 0], [0, 0.5]])


def test_nearest_only_per_sensor_and_earlier_on_ties():
    rows = []
    for sensor, gaps in (("s2", [-3, 3, 5]), ("s1", [-6, 2])):
        for g in gaps:
            rows.append({"site": "a", "pond": 1, "flight": "2020-09-29", "sensor": sensor,
                         "method": f"{sensor}-m", "gap_days": g})
    out = ponds.nearest_only(pd.DataFrame(rows)).set_index("sensor").gap_days
    assert out["s2"] == -3 and out["s1"] == 2


def test_audit_compare_and_summary_prefer_human(tmp_path, monkeypatch):
    import pandas as pd

    from hlw import ponds as P

    ai = pd.DataFrame({"id": [1, 2, 3, 4], "site": "s", "pond": [1, 2, 3, 4], "year": 2020,
                       "naip_ac": [0.1, 0.1, 0.5, 2.0], "cls": ["<0.25", "<0.25", "0.25-1", ">=1"],
                       "water": ["unsure", "no", "yes", "partly"], "labeller": "ai"})
    hu = pd.DataFrame({"id": [1, 2, 3, 4], "water": ["yes", "yes", "yes", "unsure"],
                       "misses_water": [0, 1, 0, 0], "labeller": "me"})
    (tmp_path / "audit").mkdir()
    ai.to_csv(tmp_path / "ai.csv", index=False)
    hu.to_csv(tmp_path / "hu.csv", index=False)
    ai[["id", "site", "pond", "year", "naip_ac", "cls"]].to_csv(tmp_path / "audit" / "sample.csv",
                                                               index=False)
    monkeypatch.setattr(P, "AUDIT_CSV", tmp_path / "ai.csv")
    monkeypatch.setattr(P, "HUMAN_AUDIT_CSV", tmp_path / "hu.csv")
    monkeypatch.setattr(P, "AUDIT_DIR", tmp_path / "audit")
    monkeypatch.setattr(P, "OUT", tmp_path / "out")
    df = P.audit_compare()
    assert len(df) == 4 and (tmp_path / "out" / "audit_compare.csv").exists()
    s = P.audit_summary()
    assert s["labeller"] == "me" and s["n"] == 3  # the human unsure is dropped
    assert s["water_yes_pct_<0.25"] == 100.0
