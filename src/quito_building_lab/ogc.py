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

# Example service supplied by a user (not verified by the maintainers). Any WMS URL can be used instead.
EXAMPLE_WMS = {
    "IGM Ecuador · ortofotos (por defecto)": "https://www.geoportaligm.gob.ec/orto/igm/wms",
    "IGM Ecuador · ortofotos 2022": "https://www.geoportaligm.gob.ec/orto/toma2022/wms",
    "IGM Ecuador · ortofotos UAV": "https://www.geoportaligm.gob.ec/orto/uav_igm/wms",
}
# EPSG:3857 first: it is what web maps (and therefore the layer overlay the user sees) request, so it is the
# most widely exercised path of a WMS. The others are fallbacks, tried only if the service fails the request.
PREFERRED_CRS = ("EPSG:3857", "EPSG:32717", "EPSG:4326", "CRS:84")
MAX_CRS_ATTEMPTS = 3


class OGCError(RuntimeError):
    """Remote service error, unsafe URL, or unusable response."""


class OGCServiceError(OGCError):
    """The service answered but refused or failed the request (HTTP error or ServiceException).

    ``attempts`` lists what was tried (CRS and request URL) so the user can reproduce it in a browser.
    """

    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


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
        raise OGCServiceError(f"The service answered HTTP {exc.code}.") from exc
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
        raise OGCServiceError("Service error: " + ("; ".join(m for m in msgs if m) or "unknown"))


# ----------------------------------------------------------------------------------------- WMS
@dataclass
class WmsLayer:
    name: str
    title: str
    crs: list[str]
    bbox_lonlat: tuple[float, float, float, float] | None = None
    min_scale: float | None = None  # MinScaleDenominator: below it (zoomed in further) the server draws nothing
    max_scale: float | None = None  # MaxScaleDenominator: above it (zoomed out further) the server draws nothing

    def zoom_range(self) -> tuple[float | None, float | None]:
        """Approximate web-map zoom levels (at the equator) matching the declared scale limits."""
        def zoom(denom):
            return None if not denom else math.log2(156_543.03392 / (denom * 0.00028))
        return zoom(self.max_scale), zoom(self.min_scale)


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

    def num(el, name):
        try:
            return float(_text(el, name)) or None
        except ValueError:
            return None

    def walk(el, inherited_crs, scales=(None, None)):
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
        mn = num(el, "MinScaleDenominator") or scales[0]
        mx = num(el, "MaxScaleDenominator") or scales[1]
        name = _text(el, "Name")
        if name:
            layers.append(WmsLayer(name, _text(el, "Title", name), sorted(set(crs)), bbox, mn, mx))
        for sub in _children(el, "Layer"):
            walk(sub, crs, (mn, mx))

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


def choose_crs_chain(layer_crs: list[str], preferred=PREFERRED_CRS, limit: int = MAX_CRS_ATTEMPTS) -> list[str]:
    """Supported CRS in preference order (at most ``limit``); a layer that lists none gets the common ones."""
    upper = {c.upper(): c for c in layer_crs}
    chain = [p for p in preferred if p in upper]
    if not layer_crs:
        chain = ["EPSG:3857", "EPSG:4326"]
    if not chain:
        raise OGCError("The layer offers none of the supported CRS: " + ", ".join(layer_crs[:6]))
    return chain[:limit]


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


TILE_PX = 256  # what web maps request; some services fail on other request sizes


def _decode_image(body: bytes) -> tuple[np.ndarray, bool]:
    """Decode PNG/JPEG bytes into (4 x H x W uint8 array, had_alpha); alpha is 255 when the image has none."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # PNG/JPEG have no geotransform: expected, we georeference from bbox
            with MemoryFile(body) as mf, mf.open() as src:
                data = src.read()
    except Exception as exc:  # noqa: BLE001 - GDAL raises various errors
        raise OGCError("The returned image could not be decoded.") from exc
    had_alpha = data.shape[0] >= 4
    if data.shape[0] < 3:
        data = np.repeat(data[:1], 3, axis=0)
    out = np.full((4, data.shape[1], data.shape[2]), 255, np.uint8)
    out[:3] = data[:3]
    if had_alpha:
        out[3] = data[3]
    return out, had_alpha


def _getmap_params(layer, bbox, crs_code, version, width, height, fmt, styles, transparent=True) -> dict:
    crs = CRS.from_user_input(crs_code)
    bb = (bbox[1], bbox[0], bbox[3], bbox[2]) if _swap_axes(crs, version, crs_code) else bbox
    return {
        "service": "WMS", "version": version, "request": "GetMap", "layers": layer, "styles": styles,
        "bbox": ",".join(f"{v:.8f}".rstrip("0").rstrip(".") for v in bb),
        "width": width, "height": height, "format": fmt, "transparent": "true" if transparent else "false",
        ("crs" if version == "1.3.0" else "srs"): crs_code,
    }


def _request_url(url: str, params: dict) -> str:
    return url.rstrip("?&") + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)


def _fetch_image(url: str, params: dict, policy: UrlPolicy):
    """One GetMap request -> (4xHxW array, had_alpha). Service-side failures raise OGCServiceError."""
    body, ctype = _get(url, params, policy)
    if not ctype.lower().startswith("image/"):
        _raise_if_exception(_parse_xml(body))
        raise OGCServiceError("The service did not return an image.")
    return _decode_image(body)


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
    tile_size: int | None = TILE_PX,
) -> GeoImage:
    """Fetch a small area, georeferenced from the requested bbox + CRS.

    With ``tile_size`` (default 256) the area is requested as a few ``tile_size`` x ``tile_size`` GetMap tiles,
    exactly the request shape web maps use, and stitched (at most 4 requests for 512 px). ``tile_size=None``
    asks for the whole area in a single request. If a CRS fails, the next one is tried (at most 3 CRS); a CRS
    that fails on its first tile costs a single request.
    """
    policy = policy or UrlPolicy()
    # ground size in metres from the lon/lat box (independent of the request CRS, e.g. Web Mercator)
    lat = (bbox_lonlat[1] + bbox_lonlat[3]) / 2
    bbox_m = (0.0, 0.0, (bbox_lonlat[2] - bbox_lonlat[0]) * 111_320 * math.cos(math.radians(lat)),
              (bbox_lonlat[3] - bbox_lonlat[1]) * 110_574)
    width, height, _ = image_size_for(bbox_m, target_gsd, max_side)
    if tile_size:
        nx, ny = math.ceil(width / tile_size), math.ceil(height / tile_size)
        tw = th = tile_size
    else:
        nx = ny = 1
        tw, th = width, height
    W, H = nx * tw, ny * th
    gsd = max(bbox_m[2], bbox_m[3]) / max(W, H)
    attempts: list[dict] = []
    mosaic = None
    had_alpha = False
    crs_code = bbox = None
    for crs_code in choose_crs_chain(layer_crs or []):
        bbox = bbox_lonlat_to_crs(bbox_lonlat, crs_code)
        x0, y0, x1, y1 = bbox
        canvas = np.zeros((4, H, W), np.uint8)
        alpha_seen = False
        first_url = last_url = None
        try:
            for j in range(ny):
                for i in range(nx):
                    tb = (x0 + (x1 - x0) * i / nx, y1 - (y1 - y0) * (j + 1) / ny,
                          x0 + (x1 - x0) * (i + 1) / nx, y1 - (y1 - y0) * j / ny)
                    params = _getmap_params(layer, tb, crs_code, version, tw, th, fmt, styles)
                    first_url = first_url or _request_url(url, params)
                    last_url = _request_url(url, params)
                    tile, a = _fetch_image(url, params, policy)
                    if tile.shape[1:] != (th, tw):
                        raise OGCServiceError(f"The service returned a {tile.shape[2]}x{tile.shape[1]} image "
                                              f"instead of the requested {tw}x{th}.")
                    canvas[:, j * th:(j + 1) * th, i * tw:(i + 1) * tw] = tile
                    alpha_seen |= a
            attempts.append({"crs": crs_code, "ok": True, "url": first_url, "requests": nx * ny})
            mosaic, had_alpha = canvas, alpha_seen
            break
        except OGCServiceError as exc:
            attempts.append({"crs": crs_code, "ok": False, "error": str(exc), "url": last_url, "requests": 1})
    if mosaic is None:
        raise OGCServiceError(attempts[-1]["error"] + f" (tried {', '.join(a['crs'] for a in attempts)})", attempts)
    rgb = np.moveaxis(mosaic[:3], 0, -1)
    # alpha band (transparent = no data) when present, otherwise all-black pixels count as no data
    valid = mosaic[3] > 0 if had_alpha else rgb.any(axis=2)
    return GeoImage(
        rgb, valid, from_bounds(*bbox, W, H), crs_code, bbox,
        {"service_url": url, "layer": layer, "version": version, "request_crs": crs_code, "size_px": [W, H],
         "had_alpha": bool(had_alpha), "attempts": attempts, "tiles": nx * ny, "tile_px": tile_size,
         "approx_gsd_m": round(gsd, 3), "format": fmt,
         "note": "Rendered WMS image; not original orthophoto pixels. GSD is the request grid, not the source GSD."},
    )


def wms_probe(
    url: str,
    layer: str,
    bbox_lonlat: tuple[float, float, float, float],
    layer_crs: list[str] | None = None,
    version: str = "1.3.0",
    policy: UrlPolicy | None = None,
) -> list[dict]:
    """Diagnose a failing service: at most 5 single GetMap requests that vary one thing at a time
    (tile size, transparency, format, CRS) around the centre of ``bbox_lonlat``. Run on explicit request only."""
    policy = policy or UrlPolicy()
    chain = choose_crs_chain(layer_crs or [])
    lon, lat = (bbox_lonlat[0] + bbox_lonlat[2]) / 2, (bbox_lonlat[1] + bbox_lonlat[3]) / 2
    half = 38.0  # metres -> about 256 px at 0.3 m/px
    dlat, dlon = half / 110_574.0, half / (111_320.0 * math.cos(math.radians(lat)))
    small = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    first = chain[0]
    variants = [
        ("256×256 como el navegador (PNG transparente)", first, 256, "image/png", True),
        ("256×256, PNG sin transparencia", first, 256, "image/png", False),
        ("256×256, JPEG", first, 256, "image/jpeg", False),
        ("512×512, PNG transparente", first, 512, "image/png", True),
    ]
    if len(chain) > 1:
        variants.append((f"256×256 en {chain[1]}", chain[1], 256, "image/png", True))
    results = []
    for name, crs_code, size, fmt, transparent in variants:
        bbox = bbox_lonlat_to_crs(small, crs_code)
        if size == 512:  # same ground size at twice the pixels is not what we want: use twice the ground span
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            hw, hh = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            bbox = (cx - hw, cy - hh, cx + hw, cy + hh)
        params = _getmap_params(layer, bbox, crs_code, version, size, size, fmt, "", transparent)
        item = {"variant": name, "url": _request_url(url, params)}
        try:
            _fetch_image(url, params, policy)
            item.update(ok=True, detail="imagen recibida")
        except OGCServiceError as exc:
            item.update(ok=False, detail=str(exc))
        results.append(item)
    return results


def blank_reason(img: GeoImage) -> str | None:
    """Why a returned image is useless (``None`` if it has content). Typical cause: area outside the layer."""
    if not img.valid.any():
        return "totalmente transparente" if img.meta.get("had_alpha") else "totalmente negra"
    px = img.array[img.valid]
    if (px == 255).all(axis=1).mean() > 0.98:
        return "casi toda blanca"
    return None


def bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Overlap test for two lon/lat boxes (W, S, E, N)."""
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


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
