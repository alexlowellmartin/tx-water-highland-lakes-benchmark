import numpy as np

from hlw import config as C
from hlw.labels import IGNORE, weak_label


def test_scl_jrc_rules():
    scl = np.array([[C.SCL_WATER, C.SCL_WATER, C.SCL_VEG, C.SCL_CLOUD_HIGH]], np.uint8)
    zone = np.array([[True, True, True, True]])
    static = {"jrc": np.array([[80, 0, 0, 50]], np.uint8),
              "worldcover": np.zeros((1, 4), np.uint8),
              "io_lulc": np.zeros((7, 1, 4), np.uint8)}
    lab = weak_label("scl_jrc", scl, zone, static, 2020)
    assert lab.tolist() == [[1, IGNORE, 0, IGNORE]]  # water outside JRC history -> ignore
    lab = weak_label("scl", scl, zone, static, 2020)
    assert lab.tolist() == [[1, 1, 0, IGNORE]]
    lab = weak_label("jrc50", scl, ~zone, static, 2020)
    assert (lab == IGNORE).all()


def test_io_lulc_year_clamp():
    scl = np.full((1, 1), C.SCL_VEG, np.uint8)
    io = np.zeros((7, 1, 1), np.uint8)
    io[-1] = 1  # 2023 water
    static = {"jrc": np.zeros((1, 1), np.uint8), "worldcover": np.zeros((1, 1), np.uint8),
              "io_lulc": io}
    assert weak_label("io_lulc", scl, np.ones((1, 1), bool), static, 2025)[0, 0] == 1
    assert weak_label("io_lulc", scl, np.ones((1, 1), bool), static, 2018)[0, 0] == 0
