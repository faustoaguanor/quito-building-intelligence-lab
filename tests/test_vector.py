import json

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from quito_building_lab.config import PostprocessConfig
from quito_building_lab.crs_utils import metric_crs_for, reproject_geom, utm_crs_for_lonlat
from quito_building_lab.export import to_geojson_dict, write_geojson, write_gpkg, write_mask_geotiff
from quito_building_lab.vectorize import (
    FootprintSet,
    extract_footprints,
    fix_geometry,
    mask_to_polygons,
    suppress_duplicates,
)

TR = from_origin(780000.0, 9985000.0, 0.5, 0.5)


def _prob(h=200, w=200, rects=((20, 20, 40, 30), (100, 120, 60, 60))):
    p = np.zeros((h, w), np.float32)
    for r, c, hh, ww in rects:
        p[r : r + hh, c : c + ww] = 0.95
    return p


def test_mask_vectorization_pixel_space():
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 5:25] = 1
    polys = mask_to_polygons(m)
    assert len(polys) == 1
    assert polys[0].area == pytest.approx(200)
    assert polys[0].bounds == (5, 10, 25, 20)


def test_pixel_to_map_transform():
    m = np.zeros((50, 50), np.uint8)
    m[10:20, 5:25] = 1
    (p,) = mask_to_polygons(m, TR)
    minx, miny, maxx, maxy = p.bounds
    assert minx == pytest.approx(780000 + 5 * 0.5)
    assert maxy == pytest.approx(9985000 - 10 * 0.5)
    assert p.area == pytest.approx(200 * 0.25)


def test_blockwise_edge_reconciliation():
    m = np.zeros((100, 100), np.uint8)
    m[30:70, 30:70] = 1  # crosses block borders at 50
    polys = mask_to_polygons(m, block=50)
    assert len(polys) == 1
    assert polys[0].area == pytest.approx(1600)


def test_geometry_validity_fix():
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    assert not bowtie.is_valid
    parts = fix_geometry(bowtie)
    assert parts and all(p.is_valid for p in parts)
    assert sum(p.area for p in parts) == pytest.approx(50)


def test_polygon_area_metric_and_min_area_filter():
    prob = _prob(rects=((20, 20, 40, 30), (100, 120, 60, 60), (180, 180, 4, 4)))
    valid = np.ones_like(prob, bool)
    cfg = PostprocessConfig(min_area_m2=10.0, sieve_pixels=1, simplify_tolerance=0)
    raw, proc = extract_footprints(prob, valid, TR, "EPSG:32717", cfg)
    assert len(raw) == 3
    assert len(proc) == 2  # 4x4 px * 0.25 = 4 m2 < 10 m2 dropped
    areas = sorted(a["area"] for a in proc.attrs)
    assert areas == pytest.approx([300.0, 900.0])
    assert proc.crs == "EPSG:32717"
    assert all(0.9 < a["score"] <= 1.0 for a in proc.attrs)


def test_area_computed_in_metric_crs_for_geographic_input():
    tr = from_origin(-78.45, -0.10, 2.7e-6, 2.7e-6)
    prob = _prob()
    raw, proc = extract_footprints(prob, np.ones_like(prob, bool), tr, "EPSG:4326",
                                   PostprocessConfig(simplify_tolerance=0, sieve_pixels=1))
    assert proc.crs == "EPSG:32717"
    # 40x30 px at ~0.3 m -> ~108 m2
    assert min(a["area"] for a in proc.attrs) == pytest.approx(40 * 30 * 0.3 * 0.3, rel=0.02)


def test_nodata_excluded_from_footprints():
    prob = _prob()
    valid = np.ones_like(prob, bool)
    valid[:, :100] = False
    _, proc = extract_footprints(prob, valid, TR, "EPSG:32717", PostprocessConfig(sieve_pixels=1))
    assert len(proc) == 1


def test_hole_handling():
    prob = np.zeros((100, 100), np.float32)
    prob[10:90, 10:90] = 0.9
    prob[40:42, 40:42] = 0.0  # tiny hole (1 m2 at 0.5 m)
    prob[60:75, 60:75] = 0.0  # big hole
    _, proc = extract_footprints(prob, np.ones_like(prob, bool), TR, "EPSG:32717",
                                 PostprocessConfig(sieve_pixels=1, simplify_tolerance=0, fill_holes_smaller_than_m2=4))
    (g,) = proc.geoms
    assert len(g.interiors) == 1


def test_duplicate_suppression():
    a = box(0, 0, 10, 10)
    b = box(0.2, 0, 10.2, 10)  # near duplicate
    c = box(20, 20, 30, 30)
    kept, idx = suppress_duplicates([a, b, c], 0.8)
    assert len(kept) == 2 and 2 in idx


def test_utm_selection_quito():
    assert utm_crs_for_lonlat(-78.5, -0.2).to_epsg() == 32717
    assert utm_crs_for_lonlat(-78.5, 0.05).to_epsg() == 32617
    assert metric_crs_for("EPSG:32717", (0, 0, 1, 1)).to_epsg() == 32717


def test_crs_roundtrip():
    g = box(780000, 9985000, 780010, 9985010)
    back = reproject_geom(reproject_geom(g, "EPSG:32717", "EPSG:4326"), "EPSG:4326", "EPSG:32717")
    assert back.symmetric_difference(g).area < 1e-6


def test_geojson_export_wgs84(tmp_path):
    fs = FootprintSet([box(780000, 9985000, 780010, 9985010)], [{"cand_id": 1, "score": float("nan")}], "EPSG:32717")
    p = write_geojson(fs, tmp_path / "c.geojson")
    d = json.loads(p.read_text())
    assert "crs" not in d  # RFC 7946
    x, y = d["features"][0]["geometry"]["coordinates"][0][0]
    assert -79 < x < -78 and -0.2 < y < 0  # lon, lat order
    assert d["features"][0]["properties"]["score"] is None
    gdf = gpd.read_file(p)
    assert gdf.crs.to_epsg() == 4326 and len(gdf) == 1


def test_geojson_pixel_space_is_flagged():
    d = to_geojson_dict(FootprintSet([box(0, 0, 5, 5)], [{}], None))
    assert d["coordinate_space"] == "image_pixels"


def test_geopackage_export(tmp_path):
    fs = FootprintSet([box(780000, 9985000, 780010, 9985010)], [{"cand_id": 1}], "EPSG:32717")
    p = write_gpkg({"candidates": fs}, tmp_path / "r.gpkg")
    gdf = gpd.read_file(p, layer="candidates")
    assert gdf.crs.to_epsg() == 32717 and gdf.geometry.iloc[0].area == pytest.approx(100)


def test_mask_geotiff_export(tmp_path):
    import rasterio

    prob = _prob()
    p = write_mask_geotiff(prob, tmp_path / "m.tif", TR, "EPSG:32717", "mask", block=64)
    with rasterio.open(p) as src:
        a = src.read(1)
        assert src.crs.to_epsg() == 32717 and src.transform == TR
    assert a.sum() == 40 * 30 + 60 * 60
