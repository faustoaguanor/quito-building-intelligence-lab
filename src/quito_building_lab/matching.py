"""Spatial matching between roof candidates and cadastral / reference polygons.

Matching model: **many-to-many**. Every candidate/reference pair whose intersection covers at least
``min_coverage`` of the smaller geometry is an edge of a bipartite graph. Connected components of
that graph define the relationship:

* 1 candidate - 1 reference  -> strong (IoU >= strong_iou) or partial correspondence
* any component with >1 candidate or >1 reference -> ambiguous (one-to-many / many-to-one / many-to-many)
* candidates/references without edges -> unmatched (or "not evaluated" when outside the evaluated area)

Thresholds are demonstration parameters, not official cadastral tolerances. A discrepancy is a
prompt for human review, never proof of a cadastral error or of an unregistered construction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .config import MatchingConfig
from .crs_utils import metric_crs_for, reproject_geom
from .vectorize import FootprintSet

STRONG = "strong_correspondence"
PARTIAL = "partial_correspondence"
CAND_UNMATCHED = "candidate_without_reference"
REF_UNMATCHED = "reference_without_candidate"
AMBIGUOUS = "ambiguous_match"
NOT_EVALUATED = "not_evaluated"

LABELS_ES = {
    STRONG: "Correspondencia fuerte",
    PARTIAL: "Correspondencia parcial",
    CAND_UNMATCHED: "Candidato sin referencia",
    REF_UNMATCHED: "Referencia sin candidato",
    AMBIGUOUS: "Coincidencia ambigua",
    NOT_EVALUATED: "No evaluado",
}


def iou(a: BaseGeometry, b: BaseGeometry) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    if inter == 0:
        return 0.0
    return float(inter / (a.area + b.area - inter))


@dataclass
class MatchResult:
    crs: str
    candidates: FootprintSet  # attrs enriched with match fields
    references: FootprintSet
    pairs: list[dict]
    summary: dict = field(default_factory=dict)


def _to_common(fs: FootprintSet, crs) -> FootprintSet:
    if fs.crs is None:
        raise ValueError(f"{fs.stage} has no CRS; cadastral comparison needs georeferenced data.")
    return FootprintSet([reproject_geom(g, fs.crs, crs) for g in fs.geoms], [dict(a) for a in fs.attrs], str(crs), fs.stage)


def match(
    candidates: FootprintSet,
    references: FootprintSet,
    cfg: MatchingConfig | None = None,
    evaluated_area: BaseGeometry | None = None,
    evaluated_area_crs: str | None = None,
) -> MatchResult:
    cfg = cfg or MatchingConfig()
    # common metric CRS: candidates' CRS if metric, else UTM of the references/candidates
    all_bounds_src = candidates if len(candidates) else references
    if not len(all_bounds_src):
        raise ValueError("Both candidate and reference sets are empty.")
    b = unary_union(all_bounds_src.geoms).bounds
    common = metric_crs_for(all_bounds_src.crs, b)
    cand = _to_common(candidates, common)
    ref = _to_common(references, common)
    aoi = reproject_geom(evaluated_area, evaluated_area_crs, common) if evaluated_area is not None else None

    tree = STRtree(ref.geoms) if ref.geoms else None
    pairs: list[dict] = []
    if tree is not None:
        for ci, cg in enumerate(cand.geoms):
            for rj in tree.query(cg, predicate="intersects"):
                rj = int(rj)
                rg = ref.geoms[rj]
                inter = cg.intersection(rg).area
                if inter <= 0:
                    continue
                cov = inter / min(cg.area, rg.area)
                if cov < cfg.min_coverage:
                    continue
                pairs.append({
                    "cand_idx": ci,
                    "ref_idx": rj,
                    "intersection_area": inter,
                    "iou": inter / (cg.area + rg.area - inter),
                    "cand_coverage": inter / cg.area,
                    "ref_coverage": inter / rg.area,
                })

    # connected components (union-find over 'c:i' / 'r:j' nodes)
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        a, bnode = find(f"c{p['cand_idx']}"), find(f"r{p['ref_idx']}")
        if a != bnode:
            parent[a] = bnode
    comps: dict[str, dict[str, set]] = defaultdict(lambda: {"c": set(), "r": set()})
    for p in pairs:
        root = find(f"c{p['cand_idx']}")
        comps[root]["c"].add(p["cand_idx"])
        comps[root]["r"].add(p["ref_idx"])

    def inside(g):
        return aoi is None or g.representative_point().within(aoi)

    c_cls: dict[int, str] = {}
    r_cls: dict[int, str] = {}
    comp_info: dict[str, dict] = {}
    for k, (root, comp) in enumerate(comps.items(), 1):
        nc, nr = len(comp["c"]), len(comp["r"])
        uc = unary_union([cand.geoms[i] for i in comp["c"]])
        ur = unary_union([ref.geoms[j] for j in comp["r"]])
        group_iou = iou(uc, ur)
        if nc == 1 and nr == 1:
            label = STRONG if group_iou >= cfg.strong_iou else PARTIAL
            relation = "one-to-one"
        else:
            label = AMBIGUOUS
            relation = "one-to-many" if nc == 1 else "many-to-one" if nr == 1 else "many-to-many"
        comp_info[root] = {"group_id": k, "relation": relation, "group_iou": round(group_iou, 4),
                           "n_candidates": nc, "n_references": nr,
                           "group_area_diff": ur.area - uc.area}
        for i in comp["c"]:
            c_cls[i] = label
        for j in comp["r"]:
            r_cls[j] = label

    by_cand: dict[int, list[dict]] = defaultdict(list)
    by_ref: dict[int, list[dict]] = defaultdict(list)
    for p in pairs:
        by_cand[p["cand_idx"]].append(p)
        by_ref[p["ref_idx"]].append(p)

    for i, (g, a) in enumerate(zip(cand.geoms, cand.attrs, strict=True)):
        ps = by_cand.get(i, [])
        if ps:
            best = max(ps, key=lambda p: p["iou"])
            rg = ref.geoms[best["ref_idx"]]
            info = comp_info[find(f"c{i}")]
            a.update({
                "match_class": c_cls[i],
                "relation": info["relation"],
                "group_id": info["group_id"],
                "n_matches": len(ps),
                "best_ref_id": ref.attrs[best["ref_idx"]].get("ref_id", str(best["ref_idx"])),
                "iou": round(best["iou"], 4),
                "intersection_m2": round(best["intersection_area"], 3),
                "coverage_cand": round(best["cand_coverage"], 4),
                "coverage_ref": round(best["ref_coverage"], 4),
                "area_diff_m2": round(g.area - rg.area, 3),
                "area_diff_pct": round(100.0 * (g.area - rg.area) / rg.area, 2) if rg.area else None,
            })
        else:
            a.update({"match_class": CAND_UNMATCHED if inside(g) else NOT_EVALUATED, "relation": None,
                      "group_id": None, "n_matches": 0, "best_ref_id": None, "iou": 0.0})
        a["match_class_es"] = LABELS_ES[a["match_class"]]
        a["area_m2"] = round(g.area, 3)

    for j, (g, a) in enumerate(zip(ref.geoms, ref.attrs, strict=True)):
        ps = by_ref.get(j, [])
        if ps:
            best = max(ps, key=lambda p: p["iou"])
            info = comp_info[find(f"r{j}")]
            a.update({"match_class": r_cls[j], "relation": info["relation"], "group_id": info["group_id"],
                      "n_matches": len(ps), "best_cand_id": cand.attrs[best["cand_idx"]].get("cand_id"),
                      "iou": round(best["iou"], 4), "coverage_ref": round(best["ref_coverage"], 4)})
        else:
            a.update({"match_class": REF_UNMATCHED if inside(g) else NOT_EVALUATED, "relation": None,
                      "group_id": None, "n_matches": 0, "iou": 0.0})
        a["match_class_es"] = LABELS_ES[a["match_class"]]
        a["area_m2"] = round(g.area, 3)

    summary = summarize(cand, ref, pairs, cfg)
    summary["crs"] = common.to_string()
    summary["groups"] = list(comp_info.values())
    return MatchResult(common.to_string(), cand, ref, pairs, summary)


def summarize(cand: FootprintSet, ref: FootprintSet, pairs: list[dict], cfg: MatchingConfig) -> dict:
    def counts(fs):
        out = {k: 0 for k in LABELS_ES}
        for a in fs.attrs:
            out[a["match_class"]] += 1
        return out

    uc = unary_union(cand.geoms) if cand.geoms else None
    ur = unary_union(ref.geoms) if ref.geoms else None
    inter = uc.intersection(ur).area if uc is not None and ur is not None else 0.0
    return {
        "matching_model": "many-to-many (connected components of the overlap graph)",
        "thresholds": {"strong_iou": cfg.strong_iou, "min_coverage": cfg.min_coverage,
                       "note": "Demonstration parameters, not official cadastral tolerances."},
        "reference_kind": cfg.reference_kind,
        "n_candidates": len(cand),
        "n_references": len(ref),
        "n_pairs": len(pairs),
        "candidates_by_class": counts(cand),
        "references_by_class": counts(ref),
        "candidate_area_m2": round(cand.total_area, 2),
        "reference_area_m2": round(ref.total_area, 2),
        "overlap_area_m2": round(inter, 2),
        "area_iou": round(inter / (uc.union(ur).area), 4) if inter else 0.0,
        "reference_area_covered_pct": round(100 * inter / ur.area, 2) if ur is not None and ur.area else None,
    }
