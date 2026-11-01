"""Pipeline configuration (dataclasses + YAML loading)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InferenceConfig:
    model_id: str = "xdxd_spacenet4_unet_vgg16"
    backend: str = "auto"  # auto | torch | onnx
    device: str = "cpu"  # cpu | cuda | cuda:0 | auto
    batch_size: int = 1
    weights_dir: str | None = None  # defaults to ~/.cache/quito_building_lab or $QBL_WEIGHTS_DIR
    # Radiometric normalization. "image_stats": standardise each band with statistics of the
    # whole input (computed once on a decimated read, so all tiles share them).
    # "spacenet_dn": raw 11/16-bit DN normalised with the SpaceNet-4 training constants.
    normalization: str = "image_stats"


@dataclass
class TilingConfig:
    tile_size: int = 512
    overlap: int = 64
    # Resample input to this ground sampling distance (metres) before inference.
    # The pretrained model was trained on ~0.5 m WorldView imagery; very-high-resolution drone
    # orthophotos (e.g. 5-10 cm) should normally be resampled. None = native resolution.
    target_gsd: float | None = 0.3
    max_pixels: int = 400_000_000  # safety limit for the processed (resampled) grid


@dataclass
class PostprocessConfig:
    threshold: float = 0.5
    sieve_pixels: int = 20  # remove connected components smaller than this (processed grid pixels)
    min_area_m2: float = 8.0  # metric area threshold (only applied with a valid CRS)
    min_area_px: float = 30.0  # used when no CRS is available (image coordinates)
    simplify_tolerance: float = 0.5  # metres (or pixels without CRS)
    fill_holes_smaller_than_m2: float = 4.0
    keep_holes: bool = True


@dataclass
class MatchingConfig:
    strong_iou: float = 0.5
    min_coverage: float = 0.1  # min share of the smaller geometry covered to count as candidate pair
    reference_kind: str = "construction"  # construction | block | uc | reviewed_roofs | other


@dataclass
class PipelineConfig:
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PipelineConfig:
        data = data or {}
        cfg = cls()
        for section in fields(cls):
            values = data.get(section.name) or {}
            target = getattr(cfg, section.name)
            known = {f.name for f in fields(target)}
            unknown = set(values) - known
            if unknown:
                raise ValueError(f"Unknown keys in config section '{section.name}': {sorted(unknown)}")
            for k, v in values.items():
                setattr(target, k, v)
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))
