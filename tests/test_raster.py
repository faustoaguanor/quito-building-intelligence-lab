import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from quito_building_lab.preprocessing import normalize
from quito_building_lab.raster_io import (
    RasterValidationError,
    compute_scale,
    generate_tiles,
    inspect_raster,
    plan_tiles,
    read_tile,
)


def test_metadata_validation(scene_tif):
    info = inspect_raster(str(scene_tif))
    assert info.crs == "EPSG:32717" and info.georeferenced
    assert info.width == 640 and info.count == 3
    assert info.gsd_m == pytest.approx(0.3)
    assert info.nodata == 0


def test_rejects_single_band(tmp_path):
    p = tmp_path / "gray.tif"
    with rasterio.open(p, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint8",
                       crs="EPSG:32717", transform=from_origin(0, 0, 1, 1)) as d:
        d.write(np.zeros((1, 64, 64), np.uint8))
    with pytest.raises(RasterValidationError):
        inspect_raster(str(p))


def test_missing_crs_flagged(nocrs_tif):
    info = inspect_raster(str(nocrs_tif))
    assert not info.georeferenced and info.gsd_m is None
    assert any("No CRS" in w for w in info.warnings)
    assert compute_scale(info, 0.3) == 1.0


def test_geographic_gsd_estimate(tmp_path):
    p = tmp_path / "geo.tif"
    with rasterio.open(p, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(-78.45, -0.10, 2.7e-6, 2.7e-6)) as d:
        d.write(np.zeros((3, 64, 64), np.uint8))
    info = inspect_raster(str(p))
    assert info.gsd_m == pytest.approx(0.30, abs=0.01)


def test_window_creation_covers_grid():
    tiles = generate_tiles(1000, 700, 512, 64)
    covered = np.zeros((1000, 700), bool)
    for r, c in tiles:
        covered[r : r + 512, c : c + 512] = True
    assert covered.all()
    assert all(r + 512 <= 1000 and c + 512 <= 700 for r, c in tiles)


def test_window_overlap():
    tiles = sorted({c for _, c in generate_tiles(512, 1500, 512, 128)})
    assert tiles[0] == 0 and tiles[-1] == 1500 - 512
    for a, b in zip(tiles, tiles[1:], strict=False):
        assert 512 - (b - a) >= 128  # consecutive tiles overlap at least by `overlap`
    with pytest.raises(ValueError):
        generate_tiles(100, 100, 64, 64)


def test_small_raster_single_padded_tile():
    assert generate_tiles(100, 80, 512, 64) == [(0, 0)]


def test_nodata_handling(scene_tif):
    info = inspect_raster(str(scene_tif))
    _, _, specs = plan_tiles(info, 1.0, 512, 64)
    with rasterio.open(scene_tif) as src:
        last = [s for s in specs if s.col > 0][0]
        img, valid = read_tile(src, last, info.rgb_bands)
    assert img.shape == (512, 512, 3)
    # last 40 columns of the scene are NoData
    right_edge = 640 - last.col
    assert not valid[:, right_edge - 40 : right_edge].any()
    assert valid[:, : right_edge - 40].all()
    assert not valid[:, right_edge:].any()  # padding is invalid


def test_resampled_plan(scene_tif):
    info = inspect_raster(str(scene_tif))
    scale = compute_scale(info, 0.6)
    assert scale == pytest.approx(2.0)
    gh, gw, specs = plan_tiles(info, scale, 256, 32)
    assert (gh, gw) == (320, 320)
    assert specs[-1].src_window.col_off + specs[-1].src_window.width <= 640


def test_image_normalization():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (64, 64, 3)).astype(np.float32)
    out = normalize(img, "image_stats")
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.reshape(-1, 3).mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(out.reshape(-1, 3).std(0), 1, atol=1e-3)
    # linear radiometric scaling does not change image_stats normalisation
    np.testing.assert_allclose(normalize(img * 4 + 10, "image_stats"), out, atol=1e-4)
    valid = np.ones((64, 64), bool)
    valid[:10] = False
    assert (normalize(img, "image_stats", valid=valid)[:10] == 0).all()
    with pytest.raises(ValueError):
        normalize(img, "bogus")
