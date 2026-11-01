import json

import numpy as np
import pytest
from conftest import ORIGIN, RES, SQUARES, make_scene
from shapely.geometry import box

from quito_building_lab.config import PipelineConfig
from quito_building_lab.crs_utils import CRSError
from quito_building_lab.inference import create_predictor, resolve_device
from quito_building_lab.models import ModelWeightsError, download_weights, get_spec, load_torch_model
from quito_building_lab.pipeline import run_image, run_raster
from quito_building_lab.reference import load_reference
from quito_building_lab.segmentation import blend_weights, segment_raster


def _cfg(tile=256, overlap=64):
    cfg = PipelineConfig()
    cfg.tiling.tile_size, cfg.tiling.overlap, cfg.tiling.target_gsd = tile, overlap, None
    cfg.inference.normalization = "image_stats"
    cfg.postprocess.min_area_m2 = 2.0
    cfg.postprocess.sieve_pixels = 4
    return cfg


def test_missing_model_weights(tmp_path):
    with pytest.raises(ModelWeightsError):
        create_predictor("xdxd_spacenet4_unet_vgg16", "onnx", "cpu", str(tmp_path))
    with pytest.raises(ModelWeightsError):
        load_torch_model("xdxd_spacenet4_unet_vgg16", str(tmp_path), download=False)
    with pytest.raises(ModelWeightsError):
        get_spec("unknown_model")


def test_corrupt_weights_rejected(tmp_path, monkeypatch):
    import quito_building_lab.models as m

    (tmp_path / "x").write_bytes(b"x")
    monkeypatch.setattr(m.urllib.request, "urlopen", lambda *a, **k: open(tmp_path / "x", "rb"))
    with pytest.raises(ModelWeightsError, match="Checksum"):
        download_weights("xdxd_spacenet4_unet_vgg16", str(tmp_path))


def test_cpu_configuration():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")
    cfg = PipelineConfig.from_dict({"inference": {"device": "cpu", "batch_size": 2}})
    assert cfg.inference.device == "cpu" and cfg.inference.batch_size == 2
    with pytest.raises(ValueError):
        PipelineConfig.from_dict({"tiling": {"nope": 1}})


def test_blend_weights_symmetric():
    w = blend_weights(128, 32)
    assert w.shape == (128, 128) and w[64, 64] == 1.0
    np.testing.assert_allclose(w, w[::-1, ::-1])


def test_tiled_equals_single_pass(scene_tif, predictor):
    """Per-pixel fake model => tiled stitching must reproduce the single-tile result."""
    a = segment_raster(str(scene_tif), predictor, _cfg(256, 64))
    b = segment_raster(str(scene_tif), predictor, _cfg(1024, 0))
    assert a.n_tiles > 1 and b.n_tiles == 1
    np.testing.assert_allclose(a.probability, b.probability, atol=1e-5)
    assert not a.valid[:, -40:].any()  # NoData columns stay invalid


def test_batching_same_result(scene_tif, predictor):
    cfg = _cfg(256, 64)
    cfg.inference.batch_size = 3
    a = segment_raster(str(scene_tif), predictor, cfg)
    b = segment_raster(str(scene_tif), predictor, _cfg(256, 64))
    np.testing.assert_allclose(a.probability, b.probability, atol=1e-6)


def test_complete_synthetic_pipeline(scene_tif, predictor, tmp_path):
    ref = [box(ORIGIN[0] + c * RES, ORIGIN[1] - (r + s) * RES, ORIGIN[0] + (c + s) * RES, ORIGIN[1] - r * RES)
           for r, c, s in SQUARES]
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"id": range(3)}, geometry=ref[:3], crs="EPSG:32717").to_crs(4326)
    gdf.to_file(tmp_path / "ref.geojson", driver="GeoJSON")
    out = run_raster(str(scene_tif), str(tmp_path / "out"), _cfg(), str(tmp_path / "ref.geojson"),
                     reviewed_labels=str(tmp_path / "ref.geojson"), predictor=predictor, reference_id_field="id")
    assert len(out.candidates) == 4
    cls = [a["match_class"] for a in out.matching.candidates.attrs]
    assert cls.count("strong_correspondence") == 3 and cls.count("candidate_without_reference") == 1
    assert out.report["detection_metrics"]["true_positives"] == 3
    for k in ("mask", "probability", "candidates_geojson", "gpkg", "report_json", "report_md"):
        assert (tmp_path / "out").joinpath(out.files[k].split("/")[-1]).exists()
    rep = json.loads((tmp_path / "out" / "report.json").read_text())
    assert rep["provenance"]["input"]["crs"] == "EPSG:32717"
    assert rep["limitations"] and rep["provenance"]["model"]["source_sha256"]


def test_image_without_crs_stays_in_pixels(predictor):
    img = np.moveaxis(make_scene(200, 200), 0, -1)
    out = run_image(img, _cfg(256, 32), predictor)
    assert out.candidates.crs is None and len(out.candidates) >= 1
    from quito_building_lab.export import to_geojson_dict

    assert to_geojson_dict(out.candidates)["coordinate_space"] == "image_pixels"


def test_reference_without_crs_requires_explicit_crs(tmp_path):
    p = tmp_path / "nocrs.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}]}))
    import geopandas as gpd

    # GeoJSON without CRS defaults to WGS84 in GDAL; use a GPKG with no CRS to test the guard
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)])
    g.to_file(tmp_path / "n.gpkg", driver="GPKG")
    with pytest.raises(CRSError):
        load_reference(str(tmp_path / "n.gpkg"))
    fs, rep = load_reference(str(tmp_path / "n.gpkg"), assume_crs="EPSG:32717")
    assert rep.crs_assigned_by_user and len(fs) == 1


def test_reference_cleaning(tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point, Polygon

    bow = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    g = gpd.GeoDataFrame({"id": list("abcdef")}, geometry=[box(0, 0, 5, 5), box(0, 0, 5, 5), bow, Polygon(), Point(1, 1), None],
                         crs="EPSG:32717")
    g.to_file(tmp_path / "r.gpkg", driver="GPKG")
    fs, rep = load_reference(str(tmp_path / "r.gpkg"), id_field="id")
    assert rep.duplicates_removed == 1 and rep.invalid_fixed == 1
    assert rep.non_polygonal + rep.empty_geometries + rep.null_geometries >= 2
    assert all(g.is_valid for g in fs.geoms)


@pytest.mark.model
def test_real_inference_smoke():
    """Optional: needs weights in $QBL_WEIGHTS_DIR (run `qbl download-model`)."""
    from quito_building_lab.models import weight_paths

    p = weight_paths("xdxd_spacenet4_unet_vgg16")
    if not (p["pth"].exists() or p["safetensors"].exists() or p["onnx"].exists()):
        pytest.skip("model weights not available")
    from pathlib import Path

    demo = Path(__file__).parents[1] / "examples/demo/vegas_sn2_img10_rgb8.tif"
    out = run_raster(str(demo), None, PipelineConfig())
    assert len(out.candidates) > 5


def test_ensure_onnx_downloads_and_verifies(tmp_path, monkeypatch):
    import hashlib
    import io

    import quito_building_lab.models as m

    payload = b"fake-onnx-bytes"
    digest = hashlib.sha256(payload).hexdigest()

    def fake_urlopen(url, timeout=0):
        return io.BytesIO(digest.encode() if url.endswith(".sha256") else payload)

    monkeypatch.setattr(m.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("QBL_ONNX_URL", raising=False)
    monkeypatch.delenv("QBL_ONNX_SHA256", raising=False)
    p = m.ensure_onnx("xdxd_spacenet4_unet_vgg16", str(tmp_path))
    assert p.read_bytes() == payload


def test_ensure_onnx_rejects_bad_checksum(tmp_path, monkeypatch):
    import io

    import quito_building_lab.models as m

    monkeypatch.setattr(m.urllib.request, "urlopen",
                        lambda url, timeout=0: io.BytesIO(b"0" * 64 if url.endswith(".sha256") else b"x"))
    with pytest.raises(ModelWeightsError, match="Checksum"):
        m.ensure_onnx("xdxd_spacenet4_unet_vgg16", str(tmp_path))
    assert not list(tmp_path.glob("*.onnx"))
