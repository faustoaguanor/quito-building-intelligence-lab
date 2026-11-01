"""Quality measures, organised along ISO 19157-1:2023 data-quality elements.

Three separate questions are kept apart:

1. Model detection performance  -> against *independently reviewed roof labels* only.
2. Cadastral correspondence      -> agreement with cadastral construction/UC/block geometry
                                     (a different concept: legal/registered units, not visible roofs).
3. Geometry & data quality       -> structural, geometric and topological validation of outputs.

No ISO certification, official conformity or positional accuracy claim is implied.
"""

from __future__ import annotations

from shapely.ops import unary_union
from shapely.strtree import STRtree

from .crs_utils import metric_crs_for, reproject_geom
from .matching import MatchResult, iou
from .vectorize import FootprintSet


def detection_metrics(pred: FootprintSet, truth: FootprintSet, iou_threshold: float = 0.5) -> dict:
    """Object (one-to-one greedy by IoU) and area-based metrics vs reviewed roof labels."""
    if pred.crs is None or truth.crs is None:
        if pred.crs != truth.crs:
            raise ValueError("Prediction and truth must both be georeferenced or both in pixel space.")
        common = None
    else:
        b = unary_union(truth.geoms or pred.geoms).bounds
        common = metric_crs_for(truth.crs, b)
    P = [reproject_geom(g, pred.crs, common) for g in pred.geoms] if common else list(pred.geoms)
    T = [reproject_geom(g, truth.crs, common) for g in truth.geoms] if common else list(truth.geoms)
    cand = []
    if P and T:
        tree = STRtree(T)
        for i, g in enumerate(P):
            for j in tree.query(g, predicate="intersects"):
                v = iou(g, T[int(j)])
                if v >= iou_threshold:
                    cand.append((v, i, int(j)))
    cand.sort(reverse=True)
    used_p, used_t, ious = set(), set(), []
    for v, i, j in cand:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        ious.append(v)
    tp = len(ious)
    fp, fn = len(P) - tp, len(T) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    up = unary_union(P) if P else None
    ut = unary_union(T) if T else None
    inter = up.intersection(ut).area if up is not None and ut is not None else 0.0
    a_p = up.area if up is not None else 0.0
    a_t = ut.area if ut is not None else 0.0
    union = a_p + a_t - inter
    return {
        "iou_threshold": iou_threshold,
        "n_predictions": len(P),
        "n_reference": len(T),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_iou_matched": round(sum(ious) / tp, 4) if tp else None,
        "area_iou": round(inter / union, 4) if union else 0.0,
        "area_precision": round(inter / a_p, 4) if a_p else 0.0,
        "area_recall": round(inter / a_t, 4) if a_t else 0.0,
        "units": "m2" if common else "px2",
    }


def geometry_quality(fs: FootprintSet) -> dict:
    """Logical-consistency checks (ISO 19157: topological & format consistency)."""
    n_invalid = sum(not g.is_valid for g in fs.geoms)
    n_empty = sum(g.is_empty for g in fs.geoms)
    n_non_poly = sum(g.geom_type != "Polygon" for g in fs.geoms)
    overlaps = 0
    if len(fs.geoms) > 1:
        tree = STRtree(fs.geoms)
        for i, g in enumerate(fs.geoms):
            for j in tree.query(g, predicate="overlaps"):
                if int(j) > i:
                    overlaps += 1
    areas = [g.area for g in fs.geoms]
    return {
        "n_features": len(fs),
        "invalid_geometries": n_invalid,
        "empty_geometries": n_empty,
        "non_polygon_geometries": n_non_poly,
        "overlapping_pairs": overlaps,
        "with_holes": sum(len(g.interiors) > 0 for g in fs.geoms if g.geom_type == "Polygon"),
        "crs": fs.crs,
        "has_crs": fs.crs is not None,
        "min_area": round(min(areas), 3) if areas else None,
        "max_area": round(max(areas), 3) if areas else None,
        "mean_area": round(sum(areas) / len(areas), 3) if areas else None,
    }


def iso19157_report(
    geometry: dict,
    detection: dict | None = None,
    correspondence: MatchResult | None = None,
    temporal_info: str | None = None,
) -> dict:
    """Group available measures by ISO 19157-1 data-quality element."""
    rep = {
        "standard_reference": "ISO 19157-1:2023 (conceptual organisation only; no conformity claim)",
        "logical_consistency": {
            "topological_consistency": {
                "invalid_geometries": geometry["invalid_geometries"],
                "overlapping_pairs": geometry["overlapping_pairs"],
            },
            "format_consistency": {
                "non_polygon_geometries": geometry["non_polygon_geometries"],
                "empty_geometries": geometry["empty_geometries"],
                "crs_declared": geometry["has_crs"],
            },
        },
        "completeness": None,
        "thematic_accuracy": None,
        "positional_accuracy": "Not assessed: requires an independent reference of higher accuracy.",
        "temporal_quality": temporal_info or "Not assessed: acquisition dates of imagery/reference unknown.",
    }
    if detection:
        rep["completeness"] = {
            "basis": "independently reviewed roof labels",
            "commission_rate": round(detection["false_positives"] / detection["n_predictions"], 4)
            if detection["n_predictions"] else None,
            "omission_rate": round(detection["false_negatives"] / detection["n_reference"], 4)
            if detection["n_reference"] else None,
        }
        rep["thematic_accuracy"] = {
            "basis": "classification correctness of 'roof' objects vs reviewed labels",
            "precision": detection["precision"], "recall": detection["recall"], "f1": detection["f1"],
        }
    if correspondence:
        s = correspondence.summary
        rep["cadastral_correspondence"] = {
            "note": "Agreement between visible-roof candidates and cadastral geometry. Discrepancies are "
                    "items for human review, not proven cadastral errors.",
            "reference_kind": s["reference_kind"],
            "candidates_by_class": s["candidates_by_class"],
            "references_by_class": s["references_by_class"],
            "area_iou": s["area_iou"],
        }
    return rep
