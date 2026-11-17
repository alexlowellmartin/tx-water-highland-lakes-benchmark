from pathlib import Path

import pytest

from hlw.levels import clean_levels, parse_twdb_csv

FIX = Path(__file__).parent / "fixtures" / "twdb_sample.csv"


def test_parse_and_clean_removes_spike():
    df = parse_twdb_csv(FIX.read_text())
    assert len(df) == 7 and df.index.is_monotonic_increasing
    c = clean_levels(df)
    assert c.flag_spike.sum() == 1
    assert c.water_level.isna().sum() == 1
    assert c.loc["2023-05-03", "water_level"] == pytest.approx(887.22)


def test_parse_rejects_wrong_columns():
    with pytest.raises(ValueError):
        parse_twdb_csv("date,foo\n2020-01-01,1\n")
