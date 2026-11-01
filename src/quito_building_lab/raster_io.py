"""Raster input, metadata validation and windowed (optionally resampled) reading.

Large mosaics are never read completely: statistics come from a decimated read and
inference tiles are read window by window.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.windows import Window


class RasterValidationError(ValueError):
    """Raised when an input raster cannot be processed."""


@dataclass
class RasterInfo:
    path: str
    width: int
    height: int
    count: int
    dtypes: list[str]
    crs: str | None
    crs_is_geographic: bool | None
    transform: tuple[float, ...]
    res: tuple[float, float]
    gsd_m: float | None  # approximate ground sampling distance in metres
    nodata: float | None
    rgb_bands: list[int]
    has_alpha: bool
    georeferenced: bool
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_gsd_m(crs, transform: Affine, width: int, height: int) -> float | None:
    """Approximate pixel size in metres (mean of x/y). Returns None without a CRS."""
    if crs is None:
        return None
    rx, ry = abs(transform.a), abs(transform.e)
    if crs.is_geographic:
        lat = transform.f + transform.e * height / 2.0
        mx = rx * 111_320.0 * math.cos(math.radians(lat))
        my = ry * 110_574.0
        return float((mx + my) / 2.0)
    try:
        factor = crs.linear_units_factor[1]
    except Exception:  # pragma: no cover - unusual CRS definitions
        factor = 1.0
    return float((rx + ry) / 2.0 * factor)


def _pick_rgb_bands(src) -> tuple[list[int], bool]:
    interp = list(src.colorinterp)
    has_alpha = ColorInterp.alpha in interp
    idx = {ci: i + 1 for i, ci in enumerate(interp)}
    if all(c in idx for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
        return [idx[ColorInterp.red], idx[ColorInterp.green], idx[ColorInterp.blue]], has_alpha
    return [1, 2, 3], has_alpha


def inspect_raster(path: str) -> RasterInfo:
    """Read and validate metadata without loading pixel data."""
    with rasterio.open(path) as src:
        warnings: list[str] = []
        if src.count < 3:
            raise RasterValidationError(
                f"Input has {src.count} band(s); an RGB image (>= 3 bands) is required."
            )
        if src.width < 32 or src.height < 32:
            raise RasterValidationError(f"Image too small ({src.width}x{src.height}); minimum 32x32 px.")
        rgb, has_alpha = _pick_rgb_bands(src)
        georef = src.crs is not None and not src.transform.is_identity
        if src.crs is None:
            warnings.append("No CRS: outputs will be in pixel (image) coordinates.")
        if src.transform.b != 0 or src.transform.d != 0:
            warnings.append("Rotated geotransform detected; areas are computed in the source CRS.")
        if src.count > 3 and not has_alpha:
            warnings.append(f"{src.count} bands found; using bands {rgb} as RGB.")
        gsd = estimate_gsd_m(src.crs, src.transform, src.width, src.height) if georef else None
        return RasterInfo(
            path=str(path),
            width=src.width,
            height=src.height,
            count=src.count,
            dtypes=list(src.dtypes),
            crs=src.crs.to_string() if src.crs else None,
            crs_is_geographic=bool(src.crs.is_geographic) if src.crs else None,
            transform=tuple(src.transform)[:6],
            res=src.res,
            gsd_m=gsd,
            nodata=src.nodata,
            rgb_bands=rgb,
            has_alpha=has_alpha,
            georeferenced=georef,
            warnings=warnings,
        )


def compute_scale(info: RasterInfo, target_gsd: float | None) -> float:
    """Downsampling factor source->processed grid (>= 1; never upsamples)."""
    if target_gsd is None or info.gsd_m is None:
        return 1.0
    return max(1.0, target_gsd / info.gsd_m)


@dataclass
class TileSpec:
    row: int  # processed-grid offsets
    col: int
    size: int
    src_window: Window  # clipped source window
    out_h: int  # valid output pixels in the processed grid (rest is padding)
    out_w: int


def generate_tiles(grid_h: int, grid_w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left offsets of overlapping tiles covering a grid (last tile snapped to the edge)."""
    if tile <= 0:
        raise ValueError("tile size must be positive")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must satisfy 0 <= overlap < tile")
    stride = tile - overlap

    def offsets(n: int) -> list[int]:
        if n <= tile:
            return [0]
        out = list(range(0, n - tile, stride))
        out.append(n - tile)
        return sorted(set(out))

    return [(r, c) for r in offsets(grid_h) for c in offsets(grid_w)]


def plan_tiles(info: RasterInfo, scale: float, tile: int, overlap: int) -> tuple[int, int, list[TileSpec]]:
    grid_h = max(1, int(math.ceil(info.height / scale)))
    grid_w = max(1, int(math.ceil(info.width / scale)))
    specs = []
    for r, c in generate_tiles(grid_h, grid_w, tile, overlap):
        out_h = min(tile, grid_h - r)
        out_w = min(tile, grid_w - c)
        x0, y0 = c * scale, r * scale
        w = min(out_w * scale, info.width - x0)
        h = min(out_h * scale, info.height - y0)
        specs.append(TileSpec(r, c, tile, Window(x0, y0, w, h), out_h, out_w))
    return grid_h, grid_w, specs


def processed_transform(src_transform: Affine, scale: float) -> Affine:
    return src_transform @ Affine.scale(scale, scale)


def read_tile(src, spec: TileSpec, bands: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (image HxWx3 float32 padded to tile size, valid mask HxW bool)."""
    out_shape = (len(bands), spec.out_h, spec.out_w)
    data = src.read(bands, window=spec.src_window, out_shape=out_shape, resampling=Resampling.bilinear)
    masks = src.read_masks(bands, window=spec.src_window, out_shape=out_shape, resampling=Resampling.nearest)
    valid = np.all(masks > 0, axis=0)
    img = np.zeros((spec.size, spec.size, len(bands)), dtype=np.float32)
    vm = np.zeros((spec.size, spec.size), dtype=bool)
    img[: spec.out_h, : spec.out_w] = np.moveaxis(data, 0, -1)
    vm[: spec.out_h, : spec.out_w] = valid
    return img, vm


def band_statistics(src, bands: list[int], max_dim: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std from a decimated read, ignoring NoData."""
    f = max(1.0, max(src.width, src.height) / max_dim)
    shape = (len(bands), max(1, int(src.height / f)), max(1, int(src.width / f)))
    # nearest = pixel sampling; averaging would shrink the standard deviation
    data = src.read(bands, out_shape=shape, resampling=Resampling.nearest).astype(np.float64)
    masks = src.read_masks(bands, out_shape=shape, resampling=Resampling.nearest)
    valid = np.all(masks > 0, axis=0)
    if valid.sum() < 16:
        raise RasterValidationError("Raster contains (almost) only NoData pixels.")
    px = data[:, valid]
    mean = px.mean(axis=1)
    std = px.std(axis=1)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def iter_blocks(h: int, w: int, block: int) -> Iterator[tuple[int, int, int, int]]:
    for r in range(0, h, block):
        for c in range(0, w, block):
            yield r, c, min(block, h - r), min(block, w - c)
