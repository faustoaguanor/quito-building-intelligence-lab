"""Radiometric preprocessing for the pretrained segmentation model."""

from __future__ import annotations

import numpy as np

# SpaceNet-4 (XD_XD / solaris) training normalisation: x / 65535, then z-score with these constants.
SPACENET4_MEAN = np.array([0.006479, 0.009328, 0.01123], dtype=np.float32)
SPACENET4_STD = np.array([0.004986, 0.004964, 0.004950], dtype=np.float32)

NORMALIZATION_MODES = ("image_stats", "spacenet_dn")


def normalize(
    img: np.ndarray,
    mode: str = "image_stats",
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Normalise an HxWx3 image to the model input domain (float32, HxWx3).

    image_stats: per-band z-score with ``mean``/``std`` of the whole input (8-bit or 16-bit);
                 if not provided they are computed from ``img`` (valid pixels only).
    spacenet_dn: raw WorldView-like DN values normalised with the SpaceNet-4 constants.
    Invalid (NoData) pixels are set to 0 (the band mean in z-score space).
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")
    x = img.astype(np.float32, copy=True)
    if mode == "image_stats":
        if mean is None or std is None:
            px = x[valid] if valid is not None and valid.any() else x.reshape(-1, 3)
            mean = px.mean(axis=0)
            std = px.std(axis=0)
            std = np.where(std < 1e-6, 1.0, std)
        x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    elif mode == "spacenet_dn":
        x = (x / 65535.0 - SPACENET4_MEAN) / SPACENET4_STD
    else:
        raise ValueError(f"Unknown normalization '{mode}', expected one of {NORMALIZATION_MODES}")
    if valid is not None:
        x[~valid] = 0.0
    return x


def to_nchw(batch: list[np.ndarray]) -> np.ndarray:
    return np.ascontiguousarray(np.stack([np.moveaxis(b, -1, 0) for b in batch]).astype(np.float32))
