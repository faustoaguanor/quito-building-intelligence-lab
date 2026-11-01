"""Small, polite OGC clients: WMS GetCapabilities/GetMap and WFS GetCapabilities/GetFeature.

Design rules
- One request per call, short timeouts, no retries, no crawling of layers or tiles.
- TLS verification is always on. URLs must be https (http only via an explicit test policy).
- SSRF protection for user-supplied URLs: hosts that resolve to private, loopback, link-local or
  otherwise non-public addresses are refused; redirects are never followed; responses are size-capped.
- A WMS image is a rendered picture, **not** original orthophoto pixels or cadastral vector data.
  Its georeferencing comes only from the requested bounding box + CRS.
"""

from __future__ import annotations

import ipaddress
import json
import math
import socket
import urllib.error
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from rasterio.transform import Affine, from_bounds

USER_AGENT = "quito-building-lab/0.1 (+https://github.com/faustoaguanor/quito-building-intelligence-lab)"

# Candidate services (from the project brief). NOT verified: layers, permissions and redistribution
# terms must be checked with the data owner.
CANDIDATE_WMS = {
    "Quito · usigc (ortofotos, candidato)": "https://geocatastro.quito.gob.ec:8443/geoserver/usigc/wms",
    "Quito · dmot (ortofotos, candidato)": "https://geocatastro.quito.gob.ec:8443/geoserver/dmot/wms",
    "Quito · sgpgeom (catastro, candidato)": "https://geomdmq.quito.gob.ec:8443/geoserver/sgpgeom/wms",
}
CANDIDATE_WFS = {
    "Quito · sgpgeom (UC / bloques, candidato)": "https://geomdmq.quito.gob.ec:8443/geoserver/sgpgeom/wfs",
}
PREFERRED_CRS = ("EPSG:32717", "EPSG:3857", "EPSG:4326", "CRS:84")


class OGCError(RuntimeError):
    """Remote service error, unsafe URL, or unusable response."""


@dataclass
class UrlPolicy:
    allow_http: bool = False
    allow_private: bool = False  # tests only
    timeout: float = 20.0
    max_bytes: int = 12 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - refuse every redirect
        return None


def validate_url(url: str, policy: UrlPolicy | None = None) -> str:
    policy = policy or UrlPolicy()
    u = urllib.parse.urlparse(url.strip())
    if u.scheme != "https" and not (policy.allow_http and u.scheme == "http"):
        raise OGCError("Only https:// service URLs are accepted.")
    if not u.hostname:
        raise OGCError("The URL has no host.")
    if u.username or u.password:
        raise OGCError("Credentials inside the URL are not accepted.")
    if not policy.allow_private:
        try:
            infos = socket.getaddrinfo(u.hostname, u.port or 443, proto=socket.IPPROTO_TCP)
        except OSError as exc:
            raise OGCError(f"Cannot resolve host '{u.hostname}': {exc}") from exc
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise OGCError("The host resolves to a non-public address; refused.")
    return url.strip()


def _get(base_url: str, params: dict, policy: UrlPolicy) -> tuple[bytes, str]:
    base = validate_url(base_url, policy)
    sep = "&" if urllib.parse.urlparse(base).query else "?"
    url = base + sep + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=policy.timeout) as resp:
            body = resp.read(policy.max_bytes + 1)
            ctype = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raise OGCError(f"The service answered HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OGCError(f"Could not reach the service: {exc}") from exc
    if len(body) > policy.max_bytes:
        raise OGCError("The response is larger than the allowed limit.")
    return body, ctype


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _text(el, name, default=""):
    for c in el:
        if _local(c.tag) == name and c.text:
            return c.text.strip()
    return default


def _parse_xml(body: bytes) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise OGCError("The service did not return valid XML (wrong URL?).") from exc


def _raise_if_exception(root: ET.Element) -> None:
    if _local(root.tag) in ("ServiceExceptionReport", "ExceptionReport"):
        msgs = [(e.text or "").strip() for e in root.iter() if _local(e.tag) in ("ServiceException", "ExceptionText")]
        raise OGCError("Service error: " + ("; ".join(m for m in msgs if m) or "unknown"))


# ----------------------------------------------------------------------------------------- WMS
@dataclass
class WmsLayer:
    name: str
    title: str
    crs: list[str]
    bbox_lonlat: tuple[float, float, float, float] | None = None


@dataclass
class WmsCapabilities:
    url: str
    version: str
    title: str
    formats: list[str]
    layers: list[WmsLayer] = field(default_factory=list)


def wms_capabilities(url: str, policy: UrlPolicy | None = None) -> WmsCapabilities:
    policy = policy or UrlPolicy()
    body, _ = _get(url, {"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"}, policy)
    root = _parse_xml(body)
    _raise_if_exception(root)
    if _local(root.tag) not in ("WMS_Capabilities", "WMT_MS_Capabilities"):
        raise OGCError("The URL does not look like a WMS service.")
    version = root.attrib.get("version", "1.3.0")
    service = (_children(root, "Service") or [None])[0]
    cap = (_children(root, "Capability") or [None])[0]
    if cap is None:
        raise OGCError("The WMS capabilities have no Capability section.")
    formats: list[str] = []
    for req in _children(cap, "Request"):
        for gm in _children(req, "GetMap"):
            formats = [f.text.strip() for f in _children(gm, "Format") if f.text]
    layers: list[WmsLayer] = []

    def walk(el, inherited_crs):
        crs = list(inherited_crs) + [c.text.strip() for c in el if _local(c.tag) in ("CRS", "SRS") and c.text]
        # SRS in 1.1.1 may hold a space-separated list
        crs = [x for item in crs for x in item.split()]
        bbox = None
        for c in el:
            if _local(c.tag) == "EX_GeographicBoundingBox":
                try:
                    bbox = tuple(float(_text(c, k)) for k in
                                 ("westBoundLongitude", "southBoundLatitude", "eastBoundLongitude", "northBoundLatitude"))
                except ValueError:
                    bbox = None
            elif _local(c.tag) == "LatLonBoundingBox" and bbox is None:
                try:
                    bbox = (float(c.attrib["minx"]), float(c.attrib["miny"]),
                            float(c.attrib["maxx"]), float(c.attrib["maxy"]))
                except (KeyError, ValueError):
                    bbox = None
        name = _text(el, "Name")
        if name:
            layers.append(WmsLayer(name, _text(el, "Title", name), sorted(set(crs)), bbox))
        for sub in _children(el, "Layer"):
            walk(sub, crs)

    for top in _children(cap, "Layer"):
        walk(top, [])
    return WmsCapabilities(url, version, _text(service, "Title") if service is not None else "", formats, layers)


def choose_crs(layer_crs: list[str], preferred=PREFERRED_CRS) -> str:
    upper = {c.upper(): c for c in layer_crs}
    for p in preferred:
        if p in upper:
            return p
    if not layer_crs:  # layer inherits nothing listed: assume the most common one
        return "EPSG:4326"
    raise OGCError("The layer offers none of the supported CRS: " + ", ".join(layer_crs[:6]))


def _swap_axes(crs: CRS, version: str, crs_code: str) -> bool:
    """WMS 1.3.0 + geographic EPSG codes use lat,lon axis order (CRS:84 stays lon,lat)."""
    if version != "1.3.0" or crs_code.upper().startswith("CRS:"):
        return False
    return bool(crs.axis_info) and crs.axis_info[0].direction.lower() == "north"


@dataclass
class GeoImage:
    array: np.ndarray  # HxWx3 uint8
    valid: np.ndarray  # HxW bool
    transform: Affine
    crs: str
    bbox: tuple[float, float, float, float]  # in `crs`
    meta: dict


def bbox_lonlat_to_crs(bbox_lonlat, crs_code: str) -> tuple[float, float, float, float]:
    tr = Transformer.from_crs("EPSG:4326", CRS.from_user_input(crs_code), always_xy=True)
    w, s, e, n = bbox_lonlat
    xs, ys = zip(*[tr.transform(x, y) for x in (w, e) for y in (s, n)], strict=True)
    return min(xs), min(ys), max(xs), max(ys)


def image_size_for(bbox_m: tuple[float, float, float, float], target_gsd: float, max_side: int = 512) -> tuple[int, int, float]:
    """Pixel size so that GSD ~ target_gsd, with the longer side capped at ``max_side``."""
    w_m, h_m = bbox_m[2] - bbox_m[0], bbox_m[3] - bbox_m[1]
    if w_m <= 0 or h_m <= 0:
        raise OGCError("Empty bounding box.")
    longer = max(w_m, h_m)
    side = min(max_side, max(64, math.ceil(longer / target_gsd)))
    scale = side / longer
    return max(32, round(w_m * scale)), max(32, round(h_m * scale)), longer / side


def wms_get_map(
    url: str,
    layer: str,
    bbox_lonlat: tuple[float, float, float, float],
    layer_crs: list[str] | None = None,
    version: str = "1.3.0",
    target_gsd: float = 0.3,
    fmt: str = "image/png",
    styles: str = "",
    max_side: int = 512,
    policy: UrlPolicy | None = None,
) -> GeoImage:
    """One GetMap request for a small area; georeferenced from the requested bbox + CRS."""
    policy = policy or UrlPolicy()
    crs_code = choose_crs(layer_crs or [])
    crs = CRS.from_user_input(crs_code)
    bbox = bbox_lonlat_to_crs(bbox_lonlat, crs_code)
    # metric extent (for pixel size): use UTM-like metres via a local transform when CRS is geographic
    if crs.is_geographic:
        lat = (bbox_lonlat[1] + bbox_lonlat[3]) / 2
        bbox_m = (0.0, 0.0, (bbox_lonlat[2] - bbox_lonlat[0]) * 111_320 * math.cos(math.radians(lat)),
                  (bbox_lonlat[3] - bbox_lonlat[1]) * 110_574)
    else:
        bbox_m = bbox
    width, height, gsd = image_size_for(bbox_m, target_gsd, max_side)
    bb = (bbox[1], bbox[0], bbox[3], bbox[2]) if _swap_axes(crs, version, crs_code) else bbox
    params = {
        "service": "WMS", "version": version, "request": "GetMap", "layers": layer, "styles": styles,
        "bbox": ",".join(f"{v:.8f}".rstrip("0").rstrip(".") for v in bb),
        "width": width, "height": height, "format": fmt, "transparent": "true",
        ("crs" if version == "1.3.0" else "srs"): crs_code,
    }
    body, ctype = _get(url, params, policy)
    if not ctype.lower().startswith("image/"):
        root = _parse_xml(body)
        _raise_if_exception(root)
        raise OGCError("The service did not return an image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # PNG/JPEG have no geotransform: expected, we georeference from bbox
            with MemoryFile(body) as mf, mf.open() as src:
                data = src.read()
    except Exception as exc:  # noqa: BLE001 - GDAL raises various errors
        raise OGCError("The returned image could not be decoded.") from exc
    if data.shape[0] < 3:
        data = np.repeat(data[:1], 3, axis=0)
    rgb = np.moveaxis(data[:3], 0, -1).astype(np.uint8)
    # alpha band (transparent = no data) when present, otherwise all-black pixels count as no data
    valid = data[3] > 0 if data.shape[0] >= 4 else rgb.any(axis=2)
    h, w = rgb.shape[:2]
    return GeoImage(
        rgb, valid, from_bounds(*bbox, w, h), crs_code, bbox,
        {"service_url": url, "layer": layer, "version": version, "request_crs": crs_code, "size_px": [w, h],
         "approx_gsd_m": round(gsd, 3), "format": fmt,
         "note": "Rendered WMS image; not original orthophoto pixels. GSD is the request grid, not the source GSD."},
    )


def write_geotiff(img: GeoImage, path: str) -> str:
    import rasterio

    h, w = img.array.shape[:2]
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=img.crs,
                       transform=img.transform, photometric="RGB", compress="deflate") as dst:
        arr = img.array.copy()
        arr[~img.valid] = 0
        dst.write(np.moveaxis(arr, -1, 0))
        dst.update_tags(**{k: str(v) for k, v in img.meta.items()})
    return path


# ----------------------------------------------------------------------------------------- WFS
@dataclass
class WfsFeatureType:
    name: str
    title: str
    default_crs: str


@dataclass
class WfsCapabilities:
    url: str
    version: str
    feature_types: list[WfsFeatureType]


def wfs_capabilities(url: str, policy: UrlPolicy | None = None) -> WfsCapabilities:
    policy = policy or UrlPolicy()
    body, _ = _get(url, {"service": "WFS", "request": "GetCapabilities", "version": "2.0.0"}, policy)
    root = _parse_xml(body)
    _raise_if_exception(root)
    if _local(root.tag) != "WFS_Capabilities":
        raise OGCError("The URL does not look like a WFS service.")
    fts = []
    for lst in _children(root, "FeatureTypeList"):
        for ft in _children(lst, "FeatureType"):
            name = _text(ft, "Name")
            if name:
                crs = _text(ft, "DefaultCRS") or _text(ft, "DefaultSRS")
                fts.append(WfsFeatureType(name, _text(ft, "Title", name), crs))
    return WfsCapabilities(url, root.attrib.get("version", "2.0.0"), fts)


def wfs_get_features(
    url: str,
    type_name: str,
    bbox_lonlat: tuple[float, float, float, float],
    srs: str = "EPSG:32717",
    version: str = "2.0.0",
    max_features: int = 2000,
    policy: UrlPolicy | None = None,
) -> tuple[dict, dict]:
    """One bounded GetFeature request (GeoJSON). Returns (geojson, info). Coordinates are in ``srs``."""
    policy = policy or UrlPolicy()
    crs = CRS.from_user_input(srs)
    if crs.is_geographic:
        raise OGCError("Use a projected srs (x,y axis order) for WFS requests, e.g. EPSG:32717.")
    minx, miny, maxx, maxy = bbox_lonlat_to_crs(bbox_lonlat, srs)
    params = {
        "service": "WFS", "version": version, "request": "GetFeature",
        ("typeNames" if version.startswith("2") else "typeName"): type_name,
        "outputFormat": "application/json", "srsName": srs,
        "bbox": f"{minx:.3f},{miny:.3f},{maxx:.3f},{maxy:.3f},{srs}",
        ("count" if version.startswith("2") else "maxFeatures"): max_features,
    }
    body, ctype = _get(url, params, policy)
    try:
        data = json.loads(body)
    except ValueError:
        root = _parse_xml(body)
        _raise_if_exception(root)
        raise OGCError("The service did not return GeoJSON (outputFormat unsupported?).") from None
    if not isinstance(data, dict) or "features" not in data:
        raise OGCError("The response is not a GeoJSON FeatureCollection.")
    n = len(data["features"])
    info = {"service_url": url, "type_name": type_name, "srs": srs, "version": version, "returned": n,
            "max_features": max_features, "truncated": n >= max_features,
            "note": "Vector data fetched live for the selected area; check the owner's terms before reuse."}
    return data, info
