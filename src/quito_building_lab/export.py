"""Exports: GeoJSON (RFC 7946, WGS 84), GeoPackage (metric CRS), raster masks and reports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from shapely.geometry import mapping

from .crs_utils import WGS84, reproject_geom
from .vectorize import FootprintSet


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, np.generic):
        return v.item()
    return v


def to_geojson_dict(fs: FootprintSet, precision: int = 7) -> dict:
    """RFC 7946 FeatureCollection in WGS 84 lon/lat; pixel-space sets are flagged explicitly."""
    feats = []
    for g, a in zip(fs.geoms, fs.attrs, strict=True):
        if fs.crs is not None:
            g = reproject_geom(g, fs.crs, WGS84)
        geom = json.loads(json.dumps(mapping(g)), parse_float=lambda s: round(float(s), precision))
        feats.append({"type": "Feature", "properties": {k: _clean(v) for k, v in a.items()}, "geometry": geom})
    out = {"type": "FeatureCollection", "features": feats}
    if fs.crs is None:
        out["coordinate_space"] = "image_pixels"
        out["note"] = ("Coordinates are image pixel coordinates (x=column, y=row, origin top-left), "
                       "NOT geographic coordinates. The source image had no georeferencing.")
    out["stage"] = fs.stage
    return out


def write_geojson(fs: FootprintSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_geojson_dict(fs)), encoding="utf-8")
    return path


def to_geodataframe(fs: FootprintSet):
    import geopandas as gpd

    if fs.crs is None:
        raise ValueError("GeoPackage export requires georeferenced footprints.")
    return gpd.GeoDataFrame([{k: _clean(v) for k, v in a.items()} for a in fs.attrs] or None,
                            geometry=list(fs.geoms), crs=fs.crs)


def write_gpkg(layers: dict[str, FootprintSet], path: str | Path) -> Path:
    """Write several footprint sets as layers of one GeoPackage (each in its own CRS)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for name, fs in layers.items():
        if fs.crs is None or not len(fs):
            continue
        to_geodataframe(fs).to_file(path, layer=name, driver="GPKG")
    return path


def write_mask_geotiff(
    array: np.ndarray,
    path: str | Path,
    transform: Affine | None,
    crs: str | None,
    kind: str = "mask",
    block: int = 2048,
    threshold: float = 0.5,
) -> Path:
    """Write a binary mask (uint8 0/1, ``array >= threshold``) or probability (uint8 0-255) GeoTIFF,
    block by block (works with memory-mapped arrays)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = array.shape
    profile = dict(driver="GTiff", width=w, height=h, count=1, dtype="uint8", compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    if transform is not None and crs is not None:
        profile.update(transform=transform, crs=crs)
    if kind == "mask":
        profile["nodata"] = None
    with rasterio.open(path, "w", **profile) as dst:
        for r in range(0, h, block):
            sub = np.asarray(array[r : r + block])
            data = (sub >= threshold).astype(np.uint8) if kind == "mask" else np.clip(sub * 255, 0, 255).astype(np.uint8)
            dst.write(data, 1, window=rasterio.windows.Window(0, r, w, sub.shape[0]))
        dst.update_tags(content="building candidate mask (1=candidate)" if kind == "mask"
                        else "building probability scaled 0-255")
    return path


def write_report(report: dict, out_dir: str | Path, stem: str = "report") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / f"{stem}.json"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_clean), encoding="utf-8")
    mp = out_dir / f"{stem}.md"
    mp.write_text(report_markdown(report), encoding="utf-8")
    return jp, mp


def report_markdown(r: dict) -> str:
    prov = r.get("provenance", {})
    lines = ["# Quito Building Intelligence Lab — processing report", ""]
    lines += [f"- **Input:** `{prov.get('input', {}).get('path', 'n/a')}`",
              f"- **Input CRS:** {prov.get('input', {}).get('crs')}",
              f"- **Model:** {prov.get('model', {}).get('name')} (`{prov.get('model', {}).get('model_id')}`)",
              f"- **Backend/device:** {prov.get('runtime', {}).get('backend')} / {prov.get('runtime', {}).get('device')}",
              f"- **Processed:** {prov.get('processed_at')}", ""]
    seg = r.get("segmentation", {})
    if seg:
        lines += ["## Segmentation", "", "| item | value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in seg.items()]
        lines.append("")
    for title, key in (("Geometry quality (logical consistency)", "geometry_quality"),
                       ("Detection performance vs reviewed roof labels", "detection_metrics"),
                       ("Cadastral correspondence summary", "correspondence")):
        block = r.get(key)
        if not block:
            continue
        lines += [f"## {title}", "", "| measure | value |", "|---|---|"]
        for k, v in block.items():
            if k == "groups":
                continue
            lines.append(f"| {k} | {json.dumps(v, ensure_ascii=False, default=_clean) if isinstance(v, dict) else v} |")
        lines.append("")
    lines += ["## Limitations", ""] + [f"- {x}" for x in r.get("limitations", [])]
    return "\n".join(lines) + "\n"
