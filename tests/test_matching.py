import pytest
from shapely.geometry import box

from quito_building_lab.config import MatchingConfig
from quito_building_lab.matching import (
    AMBIGUOUS,
    CAND_UNMATCHED,
    NOT_EVALUATED,
    PARTIAL,
    REF_UNMATCHED,
    STRONG,
    iou,
    match,
)
from quito_building_lab.quality import detection_metrics, geometry_quality, iso19157_report
from quito_building_lab.vectorize import FootprintSet

X0, Y0 = 780000, 9985000
CRS = "EPSG:32717"


def B(x, y, w, h):
    return box(X0 + x, Y0 + y, X0 + x + w, Y0 + y + h)


def fs(geoms, kind="cand"):
    key = "cand_id" if kind == "cand" else "ref_id"
    return FootprintSet(geoms, [{key: i + 1} for i in range(len(geoms))], CRS)


def test_iou_values():
    assert iou(box(0, 0, 2, 2), box(0, 0, 2, 2)) == pytest.approx(1.0)
    assert iou(box(0, 0, 2, 2), box(1, 0, 3, 2)) == pytest.approx(1 / 3)
    assert iou(box(0, 0, 1, 1), box(5, 5, 6, 6)) == 0.0


def test_exact_match_is_strong():
    m = match(fs([B(0, 0, 10, 10)]), fs([B(0, 0, 10, 10)], "ref"))
    a = m.candidates.attrs[0]
    assert a["match_class"] == STRONG and a["iou"] == pytest.approx(1.0)
    assert a["area_diff_m2"] == pytest.approx(0) and a["relation"] == "one-to-one"


def test_partial_overlap():
    m = match(fs([B(0, 0, 10, 10)]), fs([B(6, 0, 10, 10)], "ref"))
    a = m.candidates.attrs[0]
    assert a["match_class"] == PARTIAL
    assert a["iou"] == pytest.approx(40 / 160)
    assert a["intersection_m2"] == pytest.approx(40)
    assert a["coverage_cand"] == pytest.approx(0.4)


def test_non_overlapping():
    m = match(fs([B(0, 0, 10, 10)]), fs([B(50, 50, 10, 10)], "ref"))
    assert m.candidates.attrs[0]["match_class"] == CAND_UNMATCHED
    assert m.references.attrs[0]["match_class"] == REF_UNMATCHED
    assert m.summary["n_pairs"] == 0


def test_ambiguous_one_to_many():
    # one roof candidate spanning two construction units
    m = match(fs([B(0, 0, 20, 10)]), fs([B(0, 0, 10, 10), B(10, 0, 10, 10)], "ref"))
    assert m.candidates.attrs[0]["match_class"] == AMBIGUOUS
    assert m.candidates.attrs[0]["relation"] == "one-to-many"
    assert m.candidates.attrs[0]["n_matches"] == 2
    assert {a["match_class"] for a in m.references.attrs} == {AMBIGUOUS}
    assert m.summary["groups"][0]["group_iou"] == pytest.approx(1.0)


def test_many_to_one():
    m = match(fs([B(0, 0, 10, 10), B(10, 0, 10, 10)]), fs([B(0, 0, 20, 10)], "ref"))
    assert m.references.attrs[0]["relation"] == "many-to-one"


def test_min_coverage_filters_slivers():
    m = match(fs([B(0, 0, 10, 10)]), fs([B(9.5, 0, 10, 10)], "ref"), MatchingConfig(min_coverage=0.1))
    assert m.candidates.attrs[0]["match_class"] == CAND_UNMATCHED


def test_not_evaluated_outside_area():
    aoi = B(0, 0, 30, 30)
    m = match(fs([B(0, 0, 10, 10)]), fs([B(0, 0, 10, 10), B(100, 100, 5, 5)], "ref"), evaluated_area=aoi,
              evaluated_area_crs=CRS)
    assert [a["match_class"] for a in m.references.attrs] == [STRONG, NOT_EVALUATED]


def test_matching_reprojects_geographic_reference():
    from quito_building_lab.crs_utils import reproject_geom

    ref = FootprintSet([reproject_geom(B(0, 0, 10, 10), CRS, "EPSG:4326")], [{"ref_id": "a"}], "EPSG:4326")
    m = match(fs([B(0, 0, 10, 10)]), ref)
    assert m.candidates.attrs[0]["iou"] == pytest.approx(1.0, abs=1e-4)


def test_detection_metrics():
    pred = fs([B(0, 0, 10, 10), B(20, 0, 10, 10), B(60, 60, 5, 5)])
    truth = fs([B(0, 0, 10, 10), B(21, 0, 10, 10), B(40, 40, 10, 10)], "ref")
    d = detection_metrics(pred, truth, 0.5)
    assert (d["true_positives"], d["false_positives"], d["false_negatives"]) == (2, 1, 1)
    assert d["precision"] == pytest.approx(2 / 3, abs=1e-4)


def test_geometry_quality_and_iso_report():
    g = geometry_quality(fs([B(0, 0, 10, 10), B(5, 5, 10, 10)]))
    assert g["overlapping_pairs"] == 1 and g["invalid_geometries"] == 0
    rep = iso19157_report(g)
    assert "Not assessed" in rep["positional_accuracy"]
    assert rep["logical_consistency"]["topological_consistency"]["overlapping_pairs"] == 1
