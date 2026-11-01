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

    geoms, attrs, seen = [], [], set()
    attr_cols = [c for c in gdf.columns if c != gdf.geometry.name]
    for idx, g in enumerate(gdf.geometry):
        row = {c: gdf[c].iloc[idx] for c in attr_cols}
        if g is None or not hasattr(g, 'is_empty'):
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
            ref_id = row[id_field] if id_field else idx
            attrs.append({"ref_id": str(ref_id), "ref_kind": kind})
    rep.output_polygons = len(geoms)
    if not geoms:
        rep.warnings.append("No usable polygons in reference dataset.")
    return FootprintSet(geoms, attrs, rep.source_crs, "reference"), rep
