"""OGC client tests against a local stub server (no external network)."""

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import rasterio
from conftest import ThresholdPredictor
from PIL import Image

from quito_building_lab import ogc
from quito_building_lab.config import PipelineConfig
from quito_building_lab.crs_utils import reproject_geom
from quito_building_lab.pipeline import run_raster
from quito_building_lab.reference import reference_from_geojson

WMS_CAPS = b"""<?xml version="1.0"?>
<WMS_Capabilities version="1.3.0" xmlns="http://www.opengis.net/wms">
 <Service><Title>Stub WMS</Title></Service>
 <Capability><Request><GetMap><Format>image/png</Format><Format>image/jpeg</Format></GetMap></Request>
  <Layer><Title>root</Title><CRS>EPSG:4326</CRS><CRS>EPSG:3857</CRS>
   <Layer><Name>ortho</Name><Title>Orthophoto</Title><CRS>EPSG:32717</CRS>
    <MinScaleDenominator>1000</MinScaleDenominator><MaxScaleDenominator>200000</MaxScaleDenominator>
    <EX_GeographicBoundingBox><westBoundLongitude>-78.6</westBoundLongitude><eastBoundLongitude>-78.4</eastBoundLongitude>
    <southBoundLatitude>-0.3</southBoundLatitude><northBoundLatitude>-0.1</northBoundLatitude></EX_GeographicBoundingBox></Layer>
   <Layer><Name>only4326</Name><Title>Only 4326</Title></Layer>
  </Layer></Capability></WMS_Capabilities>"""
WFS_CAPS = b"""<?xml version="1.0"?>
<wfs:WFS_Capabilities version="2.0.0" xmlns:wfs="http://www.opengis.net/wfs/2.0" xmlns:xs="x">
 <wfs:FeatureTypeList><wfs:FeatureType><wfs:Name>sgpgeom:uc</wfs:Name><wfs:Title>UC</wfs:Title>
 <wfs:DefaultCRS>urn:ogc:def:crs:EPSG::32717</wfs:DefaultCRS></wfs:FeatureType></wfs:FeatureTypeList></wfs:WFS_Capabilities>"""
EXC = b"""<?xml version="1.0"?><ServiceExceptionReport><ServiceException>Layer not found</ServiceException></ServiceExceptionReport>"""

SEEN: list[dict] = []
ROOF_PX = (60, 60, 120, 120)  # x0, y0, x1, y1 pixel box that is "bright" in the stub image


def make_png(w, h):
    a = np.full((h, w, 3), 60, np.uint8)
    a[ROOF_PX[1]:ROOF_PX[3], ROOF_PX[0]:ROOF_PX[2]] = 230
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k.lower(): v[0] for k, v in parse_qs(u.query).items()}
        SEEN.append({"path": u.path, **q})
        if u.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:1/")
            self.end_headers()
        elif u.path == "/big":
            self.reply(b"x" * 5000, "application/octet-stream")
        elif u.path == "/wms":
            req = q.get("request", "").lower()
            if req == "getcapabilities":
                self.reply(WMS_CAPS, "text/xml")
            elif q.get("layers") == "missing":
                self.reply(EXC, "text/xml")
            elif q.get("layers") == "alwaysfail" or (q.get("layers") == "flaky" and q.get("crs") == "EPSG:3857"):
                self.reply(b'<?xml version="1.0"?><ServiceExceptionReport><ServiceException>'
                           b'java.lang.NullPointerException</ServiceException></ServiceExceptionReport>', "text/xml")
            elif q.get("layers") == "tileonly" and q.get("width") != "256":
                self.reply(b'<?xml version="1.0"?><ServiceExceptionReport><ServiceException>'
                           b'java.lang.NullPointerException</ServiceException></ServiceExceptionReport>', "text/xml")
            elif q.get("layers") == "blank":
                buf = io.BytesIO()
                Image.new("RGBA", (int(q["width"]), int(q["height"])), (0, 0, 0, 0)).save(buf, "PNG")
                self.reply(buf.getvalue(), "image/png")
            elif q.get("layers") == "white":
                buf = io.BytesIO()
                Image.new("RGB", (int(q["width"]), int(q["height"])), (255, 255, 255)).save(buf, "PNG")
                self.reply(buf.getvalue(), "image/png")
            else:
                self.reply(make_png(int(q["width"]), int(q["height"])), "image/png")
        elif u.path == "/wfs":
            if q.get("request", "").lower() == "getcapabilities":
                self.reply(WFS_CAPS, "text/xml")
            else:
                minx, miny, maxx, maxy = map(float, q["bbox"].split(",")[:4])
                cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
                ring = [[cx - 20, cy - 20], [cx + 20, cy - 20], [cx + 20, cy + 20], [cx - 20, cy + 20], [cx - 20, cy - 20]]
                fc = {"type": "FeatureCollection", "features": [
                    {"type": "Feature", "properties": {"clave": "UC-1"}, "geometry": {"type": "Polygon", "coordinates": [ring]}},
                    {"type": "Feature", "properties": {}, "geometry": None}]}
                self.reply(json.dumps(fc).encode(), "application/json")
        else:
            self.reply(b"nope", "text/plain", 404)


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


POL = ogc.UrlPolicy(allow_http=True, allow_private=True, timeout=5, max_bytes=2000)
POL_BIG = ogc.UrlPolicy(allow_http=True, allow_private=True, timeout=5)
QUITO_BBOX = (-78.5002, -0.1802, -78.4992, -0.1792)  # ~110 m square


def test_url_policy_blocks_unsafe_urls():
    for bad in ("http://example.com/wms", "https://127.0.0.1/wms", "https://localhost/wms",
                "https://169.254.169.254/latest", "https://user:pw@example.com/", "ftp://x/y"):
        with pytest.raises(ogc.OGCError):
            ogc.validate_url(bad)
    assert ogc.validate_url("http://127.0.0.1:9/x", POL)  # allowed only with the explicit test policy


def test_redirects_and_oversize_are_refused(server):
    with pytest.raises(ogc.OGCError):
        ogc._get(server + "/redirect", {}, POL)
    with pytest.raises(ogc.OGCError, match="larger"):
        ogc._get(server + "/big", {}, POL)


def test_wms_capabilities_parsing(server):
    caps = ogc.wms_capabilities(server + "/wms", POL)
    assert caps.version == "1.3.0" and caps.formats == ["image/png", "image/jpeg"]
    names = {layer.name: layer for layer in caps.layers}
    assert set(names) == {"ortho", "only4326"}
    assert set(names["ortho"].crs) == {"EPSG:4326", "EPSG:3857", "EPSG:32717"}  # inherited + own
    assert names["ortho"].bbox_lonlat == (-78.6, -0.3, -78.4, -0.1)
    assert (names["ortho"].min_scale, names["ortho"].max_scale) == (1000.0, 200000.0)
    zmin, zmax = names["ortho"].zoom_range()
    # web-map zoom 19 is about 1:1066 and every zoom level halves the scale denominator
    assert zmin == pytest.approx(11.45, abs=0.05) and zmax == pytest.approx(19.09, abs=0.05)
    assert names["only4326"].min_scale is None and names["only4326"].zoom_range() == (None, None)
    assert ogc.choose_crs(names["ortho"].crs) == "EPSG:3857"  # Web Mercator is the preferred request CRS


def test_get_map_is_georeferenced_and_requests_a_small_image(server, tmp_path):
    SEEN.clear()
    img = ogc.wms_get_map(server + "/wms", "ortho", QUITO_BBOX, ["EPSG:4326", "EPSG:32717"], policy=POL_BIG, tile_size=None)
    req = SEEN[-1]
    assert req["crs"] == "EPSG:32717" and int(req["width"]) <= 512 and int(req["height"]) <= 512
    assert img.array.shape == (int(req["height"]), int(req["width"]), 3)
    assert img.crs == "EPSG:32717" and 0.25 < img.meta["approx_gsd_m"] < 0.5
    # transform maps the requested bbox exactly
    minx, miny, maxx, maxy = img.bbox
    assert img.transform * (0, 0) == pytest.approx((minx, maxy))
    path = ogc.write_geotiff(img, str(tmp_path / "wms.tif"))
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32717 and src.count == 3


def test_get_map_axis_order_for_wms_130_geographic(server):
    SEEN.clear()
    ogc.wms_get_map(server + "/wms", "only4326", QUITO_BBOX, ["EPSG:4326"], policy=POL_BIG, tile_size=None)
    minlat, minlon, maxlat, maxlon = map(float, SEEN[-1]["bbox"].split(","))
    assert minlat == pytest.approx(QUITO_BBOX[1]) and minlon == pytest.approx(QUITO_BBOX[0])  # lat,lon order
    SEEN.clear()  # tiled: every tile keeps the lat,lon order and together they cover the same box
    ogc.wms_get_map(server + "/wms", "only4326", QUITO_BBOX, ["EPSG:4326"], policy=POL_BIG)
    tiles = [tuple(map(float, r["bbox"].split(","))) for r in SEEN if r.get("layers") == "only4326"]
    assert len(tiles) == 4
    assert min(t[0] for t in tiles) == pytest.approx(QUITO_BBOX[1]) and max(t[2] for t in tiles) == pytest.approx(QUITO_BBOX[3])
    assert min(t[1] for t in tiles) == pytest.approx(QUITO_BBOX[0]) and max(t[3] for t in tiles) == pytest.approx(QUITO_BBOX[2])
    assert all(t[0] < t[2] and t[1] < t[3] for t in tiles)
    SEEN.clear()
    ogc.wms_get_map(server + "/wms", "only4326", QUITO_BBOX, ["EPSG:4326"], version="1.1.1", policy=POL_BIG, tile_size=None)
    minx, miny, *_ = map(float, SEEN[-1]["bbox"].split(","))
    assert minx == pytest.approx(QUITO_BBOX[0]) and SEEN[-1]["srs"] == "EPSG:4326"  # x,y order in 1.1.1


def test_service_exception_becomes_error(server):
    with pytest.raises(ogc.OGCError, match="Layer not found"):
        ogc.wms_get_map(server + "/wms", "missing", QUITO_BBOX, ["EPSG:32717"], policy=POL_BIG)


def test_wfs_capabilities_and_features(server):
    caps = ogc.wfs_capabilities(server + "/wfs", POL)
    assert caps.feature_types[0].name == "sgpgeom:uc" and "32717" in caps.feature_types[0].default_crs
    data, info = ogc.wfs_get_features(server + "/wfs", "sgpgeom:uc", QUITO_BBOX, max_features=50, policy=POL_BIG)
    assert info["returned"] == 2 and not info["truncated"]
    ref, rep = reference_from_geojson(data, info["srs"], "uc", id_field="clave", source="stub")
    assert len(ref) == 1 and ref.attrs[0]["ref_id"] == "UC-1" and rep.null_geometries == 1
    with pytest.raises(ogc.OGCError):
        ogc.wfs_get_features(server + "/wfs", "x", QUITO_BBOX, srs="EPSG:4326", policy=POL_BIG)


def test_reference_from_geojson_requires_crs():
    from quito_building_lab.crs_utils import CRSError

    with pytest.raises(CRSError):
        reference_from_geojson({"features": []}, None)


def test_wms_image_plus_wfs_reference_end_to_end(server, tmp_path):
    img = ogc.wms_get_map(server + "/wms", "ortho", QUITO_BBOX, ["EPSG:32717"], policy=POL_BIG, tile_size=None)
    tif = ogc.write_geotiff(img, str(tmp_path / "w.tif"))
    # reference square centred exactly on the bright roof of the stub image
    r0, c0, r1, c1 = ROOF_PX[1], ROOF_PX[0], ROOF_PX[3], ROOF_PX[2]
    x0, y0 = img.transform * (c0, r0)
    x1, y1 = img.transform * (c1, r1)
    ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"clave": "A"}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}
    ref, rep = reference_from_geojson(fc, "EPSG:32717", "uc", "clave")
    cfg = PipelineConfig()
    cfg.tiling.target_gsd = None
    cfg.inference.normalization = "image_stats"
    cfg.postprocess.min_area_m2 = 2.0
    cfg.postprocess.sieve_pixels = 4
    out = run_raster(tif, None, cfg, reference_set=ref, reference_report=rep.to_dict(), predictor=ThresholdPredictor())
    assert len(out.candidates) == 1
    assert out.matching.candidates.attrs[0]["match_class"] == "strong_correspondence"
    assert reproject_geom(out.candidates.geoms[0], out.candidates.crs, "EPSG:4326").is_valid


def test_cli_fetch_area_then_run(server, tmp_path, monkeypatch, capsys):
    from quito_building_lab import cli

    monkeypatch.setattr(ogc, "validate_url", lambda u, p=None: u)  # stub server is http://127.0.0.1
    monkeypatch.setattr(ogc, "_get", lambda url, params, policy, _g=ogc._get: _g(url, params, POL_BIG))
    rc = cli.main(["fetch-area", "--wms-url", server + "/wms", "--layer", "ortho", "--bbox",
                   *map(str, QUITO_BBOX), "--wfs-url", server + "/wfs", "--wfs-type", "sgpgeom:uc",
                   "--out", str(tmp_path / "a")])
    assert rc == 0
    assert (tmp_path / "a" / "wms_area.tif").exists() and (tmp_path / "a" / "reference.geojson").exists()
    prov = json.loads((tmp_path / "a" / "fetch_provenance.json").read_text())
    assert prov["wms"]["layer"] == "ortho" and prov["wfs"]["returned"] == 2
    import geopandas as gpd

    assert gpd.read_file(tmp_path / "a" / "reference.geojson").crs.to_string() == prov["wms"]["request_crs"]
    assert cli.main(["fetch-area", "--wms-url", server + "/wms", "--layer", "nope", "--bbox",
                     *map(str, QUITO_BBOX), "--out", str(tmp_path / "b")]) == 2


def test_blank_images_are_explained(server):
    ok = ogc.wms_get_map(server + "/wms", "ortho", QUITO_BBOX, ["EPSG:32717"], policy=POL_BIG)
    assert ogc.blank_reason(ok) is None
    blank = ogc.wms_get_map(server + "/wms", "blank", QUITO_BBOX, ["EPSG:32717"], policy=POL_BIG)
    assert ogc.blank_reason(blank) == "totalmente transparente"
    white = ogc.wms_get_map(server + "/wms", "white", QUITO_BBOX, ["EPSG:32717"], policy=POL_BIG)
    assert ogc.blank_reason(white) == "casi toda blanca"


def test_bbox_intersects():
    layer = (-78.6, -0.3, -78.4, -0.1)
    assert ogc.bbox_intersects(QUITO_BBOX, layer)
    assert not ogc.bbox_intersects((-80.0, -2.0, -79.9, -1.9), layer)
    assert ogc.bbox_intersects((-78.41, -0.2, -78.3, -0.1), layer)  # partial overlap counts


def test_crs_chain_prefers_web_mercator_then_others():
    assert ogc.choose_crs_chain(["EPSG:4326", "EPSG:32717", "EPSG:3857"]) == ["EPSG:3857", "EPSG:32717", "EPSG:4326"]
    assert ogc.choose_crs_chain(["EPSG:4326"]) == ["EPSG:4326"]
    assert ogc.choose_crs_chain([]) == ["EPSG:3857", "EPSG:4326"]
    assert len(ogc.choose_crs_chain(["EPSG:3857", "EPSG:32717", "EPSG:4326", "CRS:84"])) == 3
    with pytest.raises(ogc.OGCError):
        ogc.choose_crs_chain(["EPSG:2154"])


def test_get_map_falls_back_to_next_crs_when_the_service_fails(server):
    SEEN.clear()
    img = ogc.wms_get_map(server + "/wms", "flaky", QUITO_BBOX, ["EPSG:3857", "EPSG:32717"], policy=POL_BIG, tile_size=None)
    assert img.crs == "EPSG:32717"
    assert [a["crs"] for a in img.meta["attempts"]] == ["EPSG:3857", "EPSG:32717"]
    assert img.meta["attempts"][0]["ok"] is False and "NullPointerException" in img.meta["attempts"][0]["error"]
    assert [r["crs"] for r in SEEN if r.get("layers") == "flaky"] == ["EPSG:3857", "EPSG:32717"]  # exactly 2 requests


def test_get_map_reports_every_attempt_when_all_fail(server):
    SEEN.clear()
    with pytest.raises(ogc.OGCServiceError) as exc:
        ogc.wms_get_map(server + "/wms", "alwaysfail", QUITO_BBOX, ["EPSG:3857", "EPSG:32717", "EPSG:4326"], policy=POL_BIG)
    assert "NullPointerException" in str(exc.value)
    assert [a["crs"] for a in exc.value.attempts] == ["EPSG:3857", "EPSG:32717", "EPSG:4326"]
    assert all("alwaysfail" in a["url"] for a in exc.value.attempts)
    assert len([r for r in SEEN if r.get("layers") == "alwaysfail"]) == 3  # bounded: never more than 3 requests


def test_unsafe_url_is_not_retried_with_other_crs():
    with pytest.raises(ogc.OGCError) as exc:
        ogc.wms_get_map("https://127.0.0.1/wms", "x", QUITO_BBOX, ["EPSG:3857", "EPSG:32717"])
    assert not isinstance(exc.value, ogc.OGCServiceError)


def test_web_mercator_image_is_georeferenced_and_measured_in_true_metres(server, tmp_path):
    from quito_building_lab.crs_utils import is_metric, metric_crs_for
    from quito_building_lab.raster_io import inspect_raster

    img = ogc.wms_get_map(server + "/wms", "ortho", QUITO_BBOX, ["EPSG:3857"], policy=POL_BIG, tile_size=None)
    assert img.crs == "EPSG:3857"
    info = inspect_raster(ogc.write_geotiff(img, str(tmp_path / "m.tif")))
    assert info.gsd_m == pytest.approx(img.meta["approx_gsd_m"], rel=0.02)
    assert not is_metric(__import__("pyproj").CRS.from_epsg(3857))
    assert metric_crs_for("EPSG:3857", (-8_700_000, -20_000, -8_699_900, -19_900)).to_epsg() == 32717
    # at 60 degrees north a Mercator "metre" is twice as long as a ground metre
    from pyproj import CRS
    from rasterio.transform import from_origin

    from quito_building_lab.raster_io import estimate_gsd_m

    y60 = 8_399_737.9
    assert estimate_gsd_m(CRS.from_epsg(3857), from_origin(0, y60, 1.0, 1.0), 10, 10) == pytest.approx(0.5, abs=0.01)


def test_web_mercator_image_gives_true_metre_areas_through_the_pipeline(server, tmp_path):
    # roof = 60x60 px of the stub image; its true area must be (60 * gsd)^2 whatever CRS was requested
    bbox = (-78.5002, -0.001, -78.4992, 0.0)  # just south of the equator (UTM 17S)
    for crs in ("EPSG:3857", "EPSG:32717"):
        img = ogc.wms_get_map(server + "/wms", "ortho", bbox, [crs], policy=POL_BIG, tile_size=None)
        tif = ogc.write_geotiff(img, str(tmp_path / f"{crs.replace(':', '_')}.tif"))
        cfg = PipelineConfig()
        cfg.tiling.target_gsd = None
        cfg.inference.normalization = "image_stats"
        cfg.postprocess.min_area_m2, cfg.postprocess.sieve_pixels, cfg.postprocess.simplify_tolerance = 2.0, 4, 0
        out = run_raster(tif, None, cfg, predictor=ThresholdPredictor())
        assert len(out.candidates) == 1 and out.candidates.crs == "EPSG:32717"
        expected = (60 * img.meta["approx_gsd_m"]) ** 2
        assert out.candidates.geoms[0].area == pytest.approx(expected, rel=0.08), crs


def test_default_request_is_made_of_256px_tiles_like_a_web_map(server):
    SEEN.clear()
    img = ogc.wms_get_map(server + "/wms", "ortho", QUITO_BBOX, ["EPSG:3857"], policy=POL_BIG)
    reqs = [r for r in SEEN if r.get("layers") == "ortho" and r.get("request") == "GetMap"]
    assert len(reqs) == img.meta["tiles"] == 4 and img.array.shape == (512, 512, 3)
    assert {(r["width"], r["height"]) for r in reqs} == {("256", "256")}
    # tiles are contiguous: each one starts where the previous ends, and together they span the whole bbox
    boxes = sorted(tuple(map(float, r["bbox"].split(","))) for r in reqs)
    xs = sorted({round(b[0], 3) for b in boxes} | {round(b[2], 3) for b in boxes})
    assert len(xs) == 3 and xs[0] == pytest.approx(img.bbox[0], abs=1e-3) and xs[-1] == pytest.approx(img.bbox[2], abs=1e-3)
    assert img.transform * (0, 0) == pytest.approx((img.bbox[0], img.bbox[3]))
    assert img.meta["approx_gsd_m"] < 0.35


def test_small_area_is_a_single_256px_request(server):
    SEEN.clear()
    tiny = (-78.50010, -0.18010, -78.49995, -0.17995)  # about 17 m
    img = ogc.wms_get_map(server + "/wms", "ortho", tiny, ["EPSG:3857"], policy=POL_BIG)
    assert img.meta["tiles"] == 1 and img.array.shape == (256, 256, 3)
    assert len([r for r in SEEN if r.get("layers") == "ortho" and r.get("request") == "GetMap"]) == 1


def test_service_that_only_accepts_256px_requests_works_in_tile_mode_but_not_single_mode(server):
    img = ogc.wms_get_map(server + "/wms", "tileonly", QUITO_BBOX, ["EPSG:3857"], policy=POL_BIG)
    assert img.array.shape == (512, 512, 3)
    with pytest.raises(ogc.OGCServiceError, match="NullPointerException"):
        ogc.wms_get_map(server + "/wms", "tileonly", QUITO_BBOX, ["EPSG:3857"], policy=POL_BIG, tile_size=None)


def test_tile_mode_failure_costs_one_request_per_crs(server):
    SEEN.clear()
    with pytest.raises(ogc.OGCServiceError) as exc:
        ogc.wms_get_map(server + "/wms", "alwaysfail", QUITO_BBOX, ["EPSG:3857", "EPSG:32717", "EPSG:4326"], policy=POL_BIG)
    assert len([r for r in SEEN if r.get("layers") == "alwaysfail"]) == 3  # not 4 tiles x 3 CRS
    assert [a["requests"] for a in exc.value.attempts] == [1, 1, 1]


def test_probe_reports_which_request_shapes_the_service_accepts(server):
    SEEN.clear()
    res = ogc.wms_probe(server + "/wms", "tileonly", QUITO_BBOX, ["EPSG:3857", "EPSG:32717"], policy=POL_BIG)
    assert len(res) <= 5 and len([r for r in SEEN if r.get("layers") == "tileonly"]) == len(res)
    by = {r["variant"].split(",")[0].split(" (")[0]: r["ok"] for r in res}
    assert any(r["ok"] for r in res if r["variant"].startswith("256")) and not by["512×512"]
    assert all("tileonly" in r["url"] for r in res)
    bad = ogc.wms_probe(server + "/wms", "alwaysfail", QUITO_BBOX, ["EPSG:3857"], policy=POL_BIG)
    assert not any(r["ok"] for r in bad) and all("NullPointerException" in r["detail"] for r in bad)


def test_browser_request_and_upload_round_trip(server):
    req = ogc.browser_getmap_request(server + "/wms", "ortho", QUITO_BBOX, ["EPSG:4326", "EPSG:3857"])
    # longer side 256 px; the other keeps the ground aspect (this test box is 111.3 x 110.6 m, not square)
    assert req["crs"] == "EPSG:3857" and (req["width"], req["height"]) == (256, 254)
    q = {k.lower(): v[0] for k, v in parse_qs(urlparse(req["url"]).query).items()}
    assert q["request"] == "GetMap" and q["width"] == "256" and q["crs"] == "EPSG:3857" and q["layers"] == "ortho"
    # what the user's browser would save: the bytes the service returns for that exact URL
    import urllib.request

    body = urllib.request.urlopen(req["url"], timeout=5).read()
    img = ogc.geoimage_from_bytes(body, req)
    assert img.array.shape == (254, 256, 3) and img.crs == "EPSG:3857"
    assert img.transform * (0, 0) == pytest.approx((req["bbox"][0], req["bbox"][3]))
    assert img.transform * (256, 254) == pytest.approx((req["bbox"][2], req["bbox"][1]))
    assert img.meta["downloaded_by"] == "user's browser"


def test_square_area_gives_a_square_browser_request():
    lat, lon, half = -0.18, -78.5, 75.0
    dlat, dlon = half / 110_574.0, half / (111_320.0 * __import__("math").cos(__import__("math").radians(lat)))
    req = ogc.browser_getmap_request("https://example.org/wms", "x", (lon - dlon, lat - dlat, lon + dlon, lat + dlat),
                                     ["EPSG:3857"])
    assert (req["width"], req["height"]) == (256, 256) and req["approx_gsd_m"] == pytest.approx(150 / 256, abs=0.01)


def test_uploaded_image_of_another_size_is_rejected():
    req = ogc.browser_getmap_request("https://example.org/wms", "x", QUITO_BBOX, ["EPSG:3857"])
    buf = io.BytesIO()
    Image.new("RGB", (300, 256), (90, 90, 90)).save(buf, "PNG")
    with pytest.raises(ogc.OGCError, match="300x256"):
        ogc.geoimage_from_bytes(buf.getvalue(), req)
