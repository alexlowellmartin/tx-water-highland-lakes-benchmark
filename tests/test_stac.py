import datetime as dt
from types import SimpleNamespace

from hlw.stac import dedupe, group_by_day, s2_offset


def _item(tile, gen, pb="05.10", day=2):
    return SimpleNamespace(
        id=f"{tile}_{gen}", datetime=dt.datetime(2023, 8, day, 17, 8),
        properties={"s2:mgrs_tile": tile, "s2:generation_time": gen,
                    "s2:processing_baseline": pb})


def test_dedupe_keeps_latest_generation():
    items = [_item("14RPU", "2023-08-02"), _item("14RPU", "2024-10-16"),
             _item("14RNU", "2023-08-02")]
    ids = sorted(i.id for i in dedupe(items))
    assert ids == ["14RNU_2023-08-02", "14RPU_2024-10-16"]
    g = group_by_day(items + [_item("14RPU", "2023-08-07", day=7)])
    assert list(g) == ["2023-08-02", "2023-08-07"] and len(g["2023-08-02"]) == 2


def test_offset_by_baseline():
    assert s2_offset(_item("t", "g", pb="03.01")) == 0
    assert s2_offset(_item("t", "g", pb="04.00")) == 1000
