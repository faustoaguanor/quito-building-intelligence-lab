"""Mask postprocessing and polygon extraction (mask -> candidate roof footprints)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rasterio import features
from rasterio.transform import Affine
from shapely import make_valid
from shapely.affinity import affine_transform as shp_affine
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .config import PostprocessConfig
from .crs_utils import metric_crs_for, reproject_geom, to_crs


@dataclass
class FootprintSet:
    """Candidate roof polygons.

    ``geoms`` are in ``crs`` (a metric CRS when georeferenced) or in pixel coordinates when
    ``crs is None`` (x = column, y = row, origin top-left).
    """

    geoms: list[Polygon]
    attrs: list[dict]
    crs: str | None
    stage: str = "processed_prediction"  # raw_prediction | processed_prediction | human_reviewed
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.geoms)

    @property
    def total_area(self) -> float:
        return float(sum(g.area for g in self.geoms))


def binarize(prob: np.ndarray, valid: np.ndarray, threshold: float, sieve_pixels: int) -> np.ndarray:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    mask = ((prob >= threshold) & valid).astype(np.uint8)
    if sieve_pixels > 1 and mask.any():
        mask = features.sieve(mask, size=int(sieve_pixels), connectivity=8)
    return mask


def _explode(geom) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [g for sub in geom.geoms for g in _explode(sub)]
    return []


def fix_geometry(geom) -> list[Polygon]:
    """Return valid, non-empty polygon parts of ``geom``."""
    if geom is None or geom.is_empty:
        return []
    if not geom.is_valid:
        geom = make_valid(geom)
    return [g for g in _explode(geom) if g.is_valid and not g.is_empty and g.area > 0]


def mask_to_polygons(mask: np.ndarray, transform: Affine | None = None, block: int = 4096) -> list[Polygon]:
    """Vectorise a binary mask blockwise (bounded memory for large grids).

    Polygons touching an internal block border are merged with their neighbours by a union
    (edge reconciliation), so objects crossing block borders are not split.
    """
    transform = transform or Affine.identity()
    h, w = mask.shape
    interior: list[Polygon] = []
    border: list[Polygon] = []
    for r in range(0, h, block):
        for c in range(0, w, block):
            sub = np.ascontiguousarray(mask[r : r + block, c : c + block]).astype(np.uint8)
            if not sub.any():
                continue
            bh, bw = sub.shape
            edges = []  # internal edges of this block in local pixel coordinates (axis, value)
            if r > 0:
                edges.append((1, 0))
            if c > 0:
                edges.append((0, 0))
            if r + bh < h:
                edges.append((1, bh))
            if c + bw < w:
                edges.append((0, bw))
            for geom, _ in features.shapes(sub, mask=sub > 0, connectivity=8):
                poly_px = shape(geom)
                coords = np.asarray(poly_px.exterior.coords)
                touches = any(np.isclose(coords[:, ax], val).any() for ax, val in edges)
                tr = transform @ Affine.translation(c, r)
                poly = shp_affine(poly_px, [tr.a, tr.b, tr.d, tr.e, tr.xoff, tr.yoff])
                (border if touches else interior).append(poly)
    if border:
        interior.extend(_explode(unary_union(border)))
    return interior


def _fill_small_holes(poly: Polygon, max_hole_area: float, keep_holes: bool) -> Polygon:
    if not poly.interiors:
        return poly
    if not keep_holes:
        return Polygon(poly.exterior)
    holes = [h for h in poly.interiors if Polygon(h).area >= max_hole_area]
    return Polygon(poly.exterior, holes)


def suppress_duplicates(geoms: list[Polygon], iou_threshold: float = 0.8) -> tuple[list[Polygon], list[int]]:
    """Drop near-duplicate polygons (IoU >= threshold), keeping the larger one. Returns (kept, kept_idx)."""
    order = sorted(range(len(geoms)), key=lambda i: -geoms[i].area)
    tree = STRtree(geoms) if geoms else None
    removed: set[int] = set()
    kept: list[int] = []
    for i in order:
        if i in removed:
            continue
        kept.append(i)
        for j in tree.query(geoms[i]):
            j = int(j)
            if j == i or j in removed or j in kept:
                continue
            inter = geoms[i].intersection(geoms[j]).area
            union = geoms[i].union(geoms[j]).area
            if union > 0 and inter / union >= iou_threshold:
                removed.add(j)
    kept.sort()
    return [geoms[i] for i in kept], kept


def _polygon_score(prob: np.ndarray, transform: Affine, poly_src: Polygon) -> float:
    """Mean probability inside a polygon (works for map and pixel coordinates)."""
    inv = ~transform
    xs, ys = poly_src.bounds[0::2], poly_src.bounds[1::2]
    cols, rows = zip(*[inv @ (x, y) for x in xs for y in ys], strict=True)
    r0, c0 = max(0, int(np.floor(min(rows)))), max(0, int(np.floor(min(cols))))
    r1, c1 = min(prob.shape[0], int(np.ceil(max(rows)))), min(prob.shape[1], int(np.ceil(max(cols))))
    if r1 <= r0 or c1 <= c0:
        return float("nan")
    sub_tr = transform @ Affine.translation(c0, r0)
    m = features.rasterize([(poly_src, 1)], out_shape=(r1 - r0, c1 - c0), transform=sub_tr, dtype=np.uint8)
    vals = np.asarray(prob[r0:r1, c0:c1])[m > 0]
    return float(vals.mean()) if vals.size else float("nan")


def extract_footprints(
    prob: np.ndarray,
    valid: np.ndarray,
    transform: Affine | None,
    crs: str | None,
    cfg: PostprocessConfig | None = None,
) -> tuple[FootprintSet, FootprintSet]:
    """Return (raw_prediction, processed_prediction) footprint sets.

    raw: direct vectorisation of the thresholded, sieved mask (source CRS / pixels).
    processed: validated, hole-filtered, simplified, area-filtered and de-duplicated polygons in a
    metric CRS (or pixel space without CRS).
    """
    cfg = cfg or PostprocessConfig()
    mask = binarize(prob, valid, cfg.threshold, cfg.sieve_pixels)
    georef = transform is not None and crs is not None
    tr = transform if georef else Affine.identity()
    raw_polys = mask_to_polygons(mask, tr)
    raw = FootprintSet(raw_polys, [{} for _ in raw_polys], crs if georef else None, "raw_prediction")

    if georef:
        work_crs = metric_crs_for(crs, _bounds(raw_polys) if raw_polys else _grid_bounds(tr, mask.shape))
        min_area, simp, hole = cfg.min_area_m2, cfg.simplify_tolerance, cfg.fill_holes_smaller_than_m2
    else:
        work_crs, min_area, simp, hole = None, cfg.min_area_px, cfg.simplify_tolerance, cfg.min_area_px / 2

    out_geoms: list[Polygon] = []
    out_attrs: list[dict] = []
    stats = {"dropped_small": 0, "invalid_fixed": 0}
    for p_src in raw_polys:
        g = reproject_geom(p_src, crs, work_crs) if georef else p_src
        if not g.is_valid:
            stats["invalid_fixed"] += 1
        for part in fix_geometry(g):
            part = _fill_small_holes(part, hole, cfg.keep_holes)
            if simp > 0:
                part = part.simplify(simp, preserve_topology=True)
            for q in fix_geometry(part):
                if q.area < min_area:
                    stats["dropped_small"] += 1
                    continue
                out_geoms.append(q)
                out_attrs.append({"score": round(_polygon_score(prob, tr, p_src), 4)})
    out_geoms, kept = suppress_duplicates(out_geoms)
    out_attrs = [out_attrs[i] for i in kept]
    for i, (g, a) in enumerate(zip(out_geoms, out_attrs, strict=True)):
        a.update({"cand_id": i + 1, "area": round(g.area, 3), "perimeter": round(g.length, 3)})
    processed = FootprintSet(
        out_geoms, out_attrs, work_crs.to_string() if work_crs else None, "processed_prediction",
        notes=[f"{k}={v}" for k, v in stats.items()],
    )
    return raw, processed


def _bounds(polys: list[Polygon]) -> tuple[float, float, float, float]:
    b = np.array([p.bounds for p in polys])
    return float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())


def _grid_bounds(tr: Affine, shape_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    h, w = shape_hw
    xs, ys = zip(*[tr * (0, 0), tr * (w, h)], strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def footprints_to_crs(fs: FootprintSet, dst) -> FootprintSet:
    if fs.crs is None:
        raise ValueError("Footprints have no CRS (pixel coordinates); cannot reproject.")
    dst = to_crs(dst)
    return FootprintSet([reproject_geom(g, fs.crs, dst) for g in fs.geoms], fs.attrs, dst.to_string(), fs.stage, fs.notes)
