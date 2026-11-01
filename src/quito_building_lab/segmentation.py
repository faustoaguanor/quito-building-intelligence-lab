"""Tiled segmentation of rasters or in-memory arrays with weighted overlap stitching."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.transform import Affine

from .config import PipelineConfig
from .inference import Predictor
from .preprocessing import normalize, to_nchw
from .raster_io import (
    RasterInfo,
    RasterValidationError,
    TileSpec,
    band_statistics,
    compute_scale,
    generate_tiles,
    inspect_raster,
    plan_tiles,
    processed_transform,
    read_tile,
)

log = logging.getLogger(__name__)
ProgressFn = Callable[[int, int], None]
MEMMAP_THRESHOLD_PX = 64_000_000  # above this, accumulators live on disk


@dataclass
class SegmentationResult:
    probability: np.ndarray  # float32 grid (may be a np.memmap)
    valid: np.ndarray  # bool grid
    transform: Affine | None  # processed-grid transform (None = pixel coordinates)
    crs: str | None
    scale: float
    info: RasterInfo | None
    timings: dict = field(default_factory=dict)
    n_tiles: int = 0


def blend_weights(tile: int, overlap: int) -> np.ndarray:
    """2-D weights that decay linearly towards the tile border inside the overlap band."""
    if overlap <= 0:
        return np.ones((tile, tile), np.float32)
    ramp = np.ones(tile, np.float32)
    edge = np.linspace(0.05, 1.0, overlap, dtype=np.float32)
    ramp[:overlap] = edge
    ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


def _alloc(shape: tuple[int, int], dtype, tmpdir: str | None):
    if shape[0] * shape[1] > MEMMAP_THRESHOLD_PX:
        f = tempfile.NamedTemporaryFile(dir=tmpdir, suffix=".dat", delete=False)
        return np.memmap(f.name, dtype=dtype, mode="w+", shape=shape)
    return np.zeros(shape, dtype=dtype)


def _run_tiles(
    grid_h: int,
    grid_w: int,
    specs: list[TileSpec],
    read_fn: Callable[[TileSpec], tuple[np.ndarray, np.ndarray]],
    norm_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    predictor: Predictor,
    cfg: PipelineConfig,
    progress: ProgressFn | None,
    tmpdir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    t = cfg.tiling
    acc = _alloc((grid_h, grid_w), np.float32, tmpdir)
    wsum = _alloc((grid_h, grid_w), np.float32, tmpdir)
    valid_all = _alloc((grid_h, grid_w), np.bool_, tmpdir)
    weights = blend_weights(t.tile_size, min(t.overlap, t.tile_size // 2))
    bs = max(1, int(cfg.inference.batch_size))
    timings = {"read_s": 0.0, "inference_s": 0.0}

    def flush(batch_specs, batch_imgs, batch_valid):
        t0 = time.perf_counter()
        probs = predictor.predict(to_nchw(batch_imgs))
        timings["inference_s"] += time.perf_counter() - t0
        for spec, p, v in zip(batch_specs, probs, batch_valid, strict=True):
            h, w = spec.out_h, spec.out_w
            wt = weights[:h, :w] * v[:h, :w]
            sl = (slice(spec.row, spec.row + h), slice(spec.col, spec.col + w))
            acc[sl] += p[:h, :w] * wt
            wsum[sl] += wt
            valid_all[sl] |= v[:h, :w]

    b_specs, b_imgs, b_valid = [], [], []
    for i, spec in enumerate(specs, 1):
        t0 = time.perf_counter()
        img, valid = read_fn(spec)
        timings["read_s"] += time.perf_counter() - t0
        if not valid.any():
            if progress:
                progress(i, len(specs))
            continue
        b_specs.append(spec)
        b_imgs.append(norm_fn(img, valid))
        b_valid.append(valid)
        if len(b_specs) == bs:
            flush(b_specs, b_imgs, b_valid)
            b_specs, b_imgs, b_valid = [], [], []
        if progress:
            progress(i, len(specs))
    if b_specs:
        flush(b_specs, b_imgs, b_valid)

    # in-place normalisation keeps memory flat for memmaps
    np.divide(acc, wsum, out=acc, where=wsum > 0)
    acc[wsum <= 0] = 0.0
    return acc, valid_all, timings


def segment_raster(
    path: str,
    predictor: Predictor,
    cfg: PipelineConfig | None = None,
    progress: ProgressFn | None = None,
    tmpdir: str | None = None,
) -> SegmentationResult:
    """Segment a (possibly very large) GeoTIFF window by window."""
    cfg = cfg or PipelineConfig()
    t_start = time.perf_counter()
    info = inspect_raster(path)
    scale = compute_scale(info, cfg.tiling.target_gsd)
    grid_h, grid_w, specs = plan_tiles(info, scale, cfg.tiling.tile_size, cfg.tiling.overlap)
    if grid_h * grid_w > cfg.tiling.max_pixels:
        raise RasterValidationError(
            f"Processed grid {grid_w}x{grid_h} exceeds max_pixels={cfg.tiling.max_pixels}; "
            "increase target_gsd or max_pixels."
        )
    log.info("Raster %sx%s, gsd=%s m, scale=%.3f -> grid %sx%s, %d tiles",
             info.width, info.height, info.gsd_m, scale, grid_w, grid_h, len(specs))
    with rasterio.open(path) as src:
        mean = std = None
        if cfg.inference.normalization == "image_stats":
            mean, std = band_statistics(src, info.rgb_bands)

        def read_fn(spec):
            return read_tile(src, spec, info.rgb_bands)

        def norm_fn(img, valid):
            return normalize(img, cfg.inference.normalization, mean, std, valid)

        prob, valid, timings = _run_tiles(grid_h, grid_w, specs, read_fn, norm_fn, predictor, cfg, progress, tmpdir)
    transform = processed_transform(Affine(*info.transform), scale) if info.georeferenced else None
    timings["total_s"] = time.perf_counter() - t_start
    return SegmentationResult(prob, valid, transform, info.crs if info.georeferenced else None,
                              scale, info, timings, len(specs))


def segment_array(
    img: np.ndarray,
    predictor: Predictor,
    cfg: PipelineConfig | None = None,
    valid: np.ndarray | None = None,
    progress: ProgressFn | None = None,
) -> SegmentationResult:
    """Segment an in-memory HxWx3 image (no georeferencing; pixel coordinates)."""
    cfg = cfg or PipelineConfig()
    if img.ndim != 3 or img.shape[2] < 3:
        raise RasterValidationError(f"Expected an RGB image (HxWx3), got shape {img.shape}")
    img = img[..., :3]
    h, w = img.shape[:2]
    valid = np.ones((h, w), bool) if valid is None else valid.astype(bool)
    t0 = time.perf_counter()
    tile = cfg.tiling.tile_size
    specs = []
    for r, c in generate_tiles(h, w, tile, cfg.tiling.overlap):
        specs.append(TileSpec(r, c, tile, None, min(tile, h - r), min(tile, w - c)))
    px = img[valid].astype(np.float64)
    mean = px.mean(axis=0).astype(np.float32)
    std = np.where(px.std(axis=0) < 1e-6, 1.0, px.std(axis=0)).astype(np.float32)

    def read_fn(spec):
        out = np.zeros((tile, tile, 3), np.float32)
        vm = np.zeros((tile, tile), bool)
        out[: spec.out_h, : spec.out_w] = img[spec.row : spec.row + spec.out_h, spec.col : spec.col + spec.out_w]
        vm[: spec.out_h, : spec.out_w] = valid[spec.row : spec.row + spec.out_h, spec.col : spec.col + spec.out_w]
        return out, vm

    def norm_fn(x, v):
        return normalize(x, cfg.inference.normalization, mean, std, v)

    prob, valid_all, timings = _run_tiles(h, w, specs, read_fn, norm_fn, predictor, cfg, progress)
    timings["total_s"] = time.perf_counter() - t0
    return SegmentationResult(prob, valid_all, None, None, 1.0, None, timings, len(specs))
