import pandas as pd

from hlw.timeline import _thin
from hlw.webexport import choose_frames


def test_thin_caps_per_month():
    days = pd.DataFrame({"date": [f"2020-01-{d:02d}" for d in range(1, 31)] + ["2020-02-03"]})
    out = _thin(days, 2)
    assert (out.groupby("month").size() <= 2).all()
    assert set(out.month) == {"2020-01", "2020-02"}


def test_choose_frames_includes_extremes():
    dates = pd.date_range("2020-01-01", periods=40, freq="15D")
    df = pd.DataFrame({"sensor": "s2", "kind": "clear", "date": dates,
                       "level": [600 + (i % 7) for i in range(40)]})
    df.loc[17, "level"] = 500
    df.loc[30, "level"] = 700
    fr = choose_frames(df, 8)
    assert dates[17].strftime("%Y-%m-%d") in fr and dates[30].strftime("%Y-%m-%d") in fr
    assert len(fr) <= 8 and fr == sorted(fr)
