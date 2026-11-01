"""CRS helpers: metric CRS selection and geometry reprojection."""

from __future__ import annotations

import numpy as np
import shapely
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry.base import BaseGeometry

WGS84 = CRS.from_epsg(4326)


class CRSError(ValueError):
    """Missing or unusable coordinate reference system."""


def to_crs(value) -> CRS:
    if value is None:
        raise CRSError("A CRS is required (none was found and none was provided).")
    return value if isinstance(value, CRS) else CRS.from_user_input(value)


def is_web_mercator(crs: CRS) -> bool:
    """EPSG:3857 'metres' are stretched by 1/cos(latitude): lengths and areas are NOT true metres."""
    crs = to_crs(crs)  # also accepts rasterio / string CRS definitions
    return crs.to_epsg() in (3857, 900913) or "pseudo-mercator" in crs.name.lower()


def is_metric(crs: CRS) -> bool:
    crs = to_crs(crs)
    if not crs.is_projected or is_web_mercator(crs):
        return False
    units = {ax.unit_name.lower() for ax in crs.axis_info}
    return units <= {"metre", "meter", "m"}


def utm_crs_for_lonlat(lon: float, lat: float) -> CRS:
    infos = query_utm_crs_info(
        datum_name="WGS 84", area_of_interest=AreaOfInterest(lon, lat, lon, lat)
    )
    if not infos:
        raise CRSError(f"No UTM zone found for lon={lon}, lat={lat}")
    # prefer the zone whose hemisphere matches the latitude
    for info in infos:
        if (lat < 0) == info.name.endswith("S"):
            return CRS.from_epsg(int(info.code))
    return CRS.from_epsg(int(infos[0].code))


def metric_crs_for(crs, bounds: tuple[float, float, float, float]) -> CRS:
    """Return ``crs`` if it is projected in metres, otherwise the WGS 84 / UTM zone of the bounds centre.

    For Quito this yields EPSG:32717 (WGS 84 / UTM 17S) unless the data are already projected
    (e.g. the municipal TM-Quito / SIRES-DMQ system), in which case that CRS is kept.
    """
    crs = to_crs(crs)
    if is_metric(crs):
        return crs
    minx, miny, maxx, maxy = bounds
    tr = Transformer.from_crs(crs, WGS84, always_xy=True)
    lon, lat = tr.transform((minx + maxx) / 2, (miny + maxy) / 2)
    return utm_crs_for_lonlat(lon, lat)


def reproject_geom(geom: BaseGeometry, src, dst) -> BaseGeometry:
    src, dst = to_crs(src), to_crs(dst)
    if src == dst:
        return geom
    tr = Transformer.from_crs(src, dst, always_xy=True)
    return shapely.transform(geom, lambda xy: np.column_stack(tr.transform(xy[:, 0], xy[:, 1])))
