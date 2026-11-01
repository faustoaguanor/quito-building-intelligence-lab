"""Import of cadastral / reference polygon datasets (GeoJSON, GeoPackage) with cleaning report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from shapely import normalize as shp_normalize

from .crs_utils import CRSError, to_crs
from .vectorize import FootprintSet, fix_geometry

REFERENCE_KINDS = ("construction", "block", "uc", "reviewed_roofs", "other")


@dataclass
class ReferenceReport:
    source: str
    layer: str | None
    kind: str
    input_features: int = 0
    empty_geometries: int = 0
    null_geometries: int = 0
    non_polygonal: int = 0
    invalid_fixed: int = 0
    duplicates_removed: int = 0
    output_polygons: int = 0
    source_crs: str | None = None
    crs_assigned_by_user: bool = False
    missing_attributes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_reference(
    path: str,
    layer: str | None = None,
    kind: str = "construction",
    assume_crs: str | None = None,
    id_field: str | None = None,
    required_fields: list[str] | None = None,
) -> tuple[FootprintSet, ReferenceReport]:
    """Load reference polygons. Missing CRS is an error unless ``assume_crs`` is given explicitly."""
    import geopandas as gpd

    if kind not in REFERENCE_KINDS:
        raise ValueError(f"kind must be one of {REFERENCE_KINDS}")
    gdf = gpd.read_file(path, layer=layer)
    rep = ReferenceReport(source=str(path), layer=layer, kind=kind, input_features=len(gdf))
    if gdf.crs is None:
        if not assume_crs:
            raise CRSError(
                f"Reference dataset '{path}' has no CRS. Provide one explicitly "
                "(e.g. --reference-crs EPSG:32717); it will not be guessed."
            )
        gdf = gdf.set_crs(to_crs(assume_crs))
        rep.crs_assigned_by_user = True
    rep.source_crs = gdf.crs.to_string()
    for f in required_fields or []:
        if f not in gdf.columns:
            rep.missing_attributes.append(f)
    if id_field and id_field not in gdf.columns:
        rep.warnings.append(f"id field '{id_field}' not found; sequential ids are used.")
        id_field = None

    attr_cols = [c for c in gdf.columns if c != gdf.geometry.name]
    items = ((g, {c: gdf[c].iloc[i] for c in attr_cols}) for i, g in enumerate(gdf.geometry))
    geoms, attrs = _ingest(items, rep, kind, id_field)
    return FootprintSet(geoms, attrs, rep.source_crs, "reference"), rep


def _ingest(items, rep: ReferenceReport, kind: str, id_field: str | None):
    """Clean (geometry, properties) pairs: nulls, empties, non-polygons, invalid, duplicates."""
    geoms, attrs, seen = [], [], set()
    for idx, (g, row) in enumerate(items):
        if g is None or not hasattr(g, "is_empty"):
            rep.null_geometries += 1
            continue
        if g.is_empty:
            rep.empty_geometries += 1
            continue
        if g.geom_type not in ("Polygon", "MultiPolygon", "GeometryCollection"):
            rep.non_polygonal += 1
            continue
        if not g.is_valid:
            rep.invalid_fixed += 1
        for part in fix_geometry(g):
            key = shp_normalize(part).wkb
            if key in seen:
                rep.duplicates_removed += 1
                continue
            seen.add(key)
            geoms.append(part)
            ref_id = row.get(id_field) if id_field else None
            attrs.append({"ref_id": str(ref_id if ref_id is not None else idx), "ref_kind": kind})
    rep.output_polygons = len(geoms)
    if not geoms:
        rep.warnings.append("No usable polygons in reference dataset.")
    return geoms, attrs


def reference_from_geojson(
    data: dict,
    crs: str | None,
    kind: str = "construction",
    id_field: str | None = None,
    source: str = "geojson",
) -> tuple[FootprintSet, ReferenceReport]:
    """Build a reference set from an in-memory GeoJSON dict (no geopandas needed).

    ``crs`` is the CRS the coordinates are expressed in. It is mandatory (never guessed).
    """
    from shapely.geometry import shape

    if kind not in REFERENCE_KINDS:
        raise ValueError(f"kind must be one of {REFERENCE_KINDS}")
    if not crs:
        raise CRSError("The CRS of the GeoJSON coordinates must be given explicitly.")
    to_crs(crs)  # validates
    feats = data.get("features", [])
    rep = ReferenceReport(source=source, layer=None, kind=kind, input_features=len(feats), source_crs=crs)
    if id_field and feats and id_field not in (feats[0].get("properties") or {}):
        rep.warnings.append(f"id field '{id_field}' not found; sequential ids are used.")
        id_field = None

    def items():
        for f in feats:
            geom = f.get("geometry")
            try:
                g = shape(geom) if geom else None
            except (ValueError, TypeError, AttributeError):
                g = None
            yield g, (f.get("properties") or {})

    geoms, attrs = _ingest(items(), rep, kind, id_field)
    return FootprintSet(geoms, attrs, crs, "reference"), rep
