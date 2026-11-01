"""Deterministic synthetic fixtures (no network, no model weights)."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from quito_building_lab.inference import Predictor

# Synthetic scene in WGS 84 / UTM 17S (Quito). Origin is arbitrary but inside zone 17S.
ORIGIN = (780000.0, 9985000.0)
RES = 0.3
SQUARES = [(40, 40, 60), (40, 200, 40), (300, 300, 90), (500, 100, 50)]  # (row, col, size) in px


def make_scene(h: int = 640, w: int = 640, nodata_cols: int = 0) -> np.ndarray:
    img = np.full((3, h, w), 60, np.uint8)
    for r, c, s in SQUARES:
        img[:, r : r + s, c : c + s] = 220
    if nodata_cols:
        img[:, :, -nodata_cols:] = 0
    return img


class ThresholdPredictor(Predictor):
    """Fake model: 'building' where the normalised red band is high. Deterministic."""

    backend = "fake"
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def predict(self, batch: np.ndarray) -> np.ndarray:
        self.calls += 1
        return (1 / (1 + np.exp(-4 * (batch[:, 0] - 0.5)))).astype(np.float32)


@pytest.fixture
def predictor():
    return ThresholdPredictor()


@pytest.fixture
def scene_tif(tmp_path):
    path = tmp_path / "scene.tif"
    img = make_scene(nodata_cols=40)
    with rasterio.open(path, "w", driver="GTiff", width=img.shape[2], height=img.shape[1], count=3,
                       dtype="uint8", crs="EPSG:32717", transform=from_origin(*ORIGIN, RES, RES), nodata=0) as d:
        d.write(img)
    return path


@pytest.fixture
def nocrs_tif(tmp_path):
    path = tmp_path / "nocrs.tif"
    img = make_scene(128, 128)
    with rasterio.open(path, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8") as d:
        d.write(img)
    return path
