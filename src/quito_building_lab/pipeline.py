"""End-to-end orchestration shared by the CLI, the local UI and the Streamlit demo."""

from __future__ import annotations

import datetime as dt
import logging
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union

from . import __version__
from .config import PipelineConfig
from .export import write_geojson, write_gpkg, write_mask_geotiff, write_report
from .inference import Predictor, cached_predictor
from .matching import MatchResult, match
from .models import get_spec
from .quality import detection_metrics, geometry_quality, iso19157_report
from .reference import load_reference
from .segmentation import ProgressFn, SegmentationResult, segment_array, segment_raster
from .vectorize import FootprintSet, extract_footprints

log = logging.getLogger(__name__)

LIMITATIONS = [
    "Candidates are visible roof outlines predicted by a model trained on SpaceNet-4 (Atlanta, WorldView-2 "
    "~0.5 m); performance on Quito imagery has not been validated with reviewed Quito labels.",
    "Visible roof geometry is not equivalent to legal cadastral building / construction-unit boundaries.",
    "Discrepancies with cadastral data are review prompts, not proven cadastral errors, unregistered "
    "or illegal constructions.",
    "Matching thresholds are demonstration parameters, not official cadastral tolerances.",
    "Positional accuracy is not assessed without an independent higher-accuracy reference.",
]


@dataclass
class PipelineOutput:
    segmentation: SegmentationResult
    raw: FootprintSet
    candidates: FootprintSet
    matching: MatchResult | None
    report: dict
    files: dict


def _evaluated_area(seg: SegmentationResult):
    """Footprint of valid processed pixels (coarse: bounds of the processed grid)."""
    if seg.transform is None:
        return None
    from rasterio import features

    v = np.asarray(seg.valid)
    step = max(1, max(v.shape) // 1024)
    sub = np.ascontiguousarray(v[::step, ::step]).astype(np.uint8)
    from rasterio.transform import Affine

    tr = seg.transform @ Affine.scale(step, step)
    polys = [shape(g) for g, val in features.shapes(sub, mask=sub > 0, transform=tr) if val]
    return unary_union(polys) if polys else None


def provenance(cfg: PipelineConfig, predictor: Predictor | None, input_meta: dict, extra: dict | None = None) -> dict:
    spec = get_spec(cfg.inference.model_id)
    import rasterio
    import shapely

    return {
        "software": {"name": "quito-building-lab", "version": __version__, "python": platform.python_version(),
                     "rasterio": rasterio.__version__, "shapely": shapely.__version__},
        "model": {"model_id": spec.model_id, "name": spec.name, "architecture": spec.architecture,
                  "source_url": spec.source_url, "source_sha256": spec.source_sha256,
                  "training_data": spec.training_data, "license_note": spec.license_note},
        "runtime": {"backend": getattr(predictor, "backend", None), "device": getattr(predictor, "device", "cpu")},
        "parameters": cfg.to_dict(),
        "input": input_meta,
        "processed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }


def run_raster(
    input_path: str,
    out_dir: str | None,
    cfg: PipelineConfig | None = None,
    reference: str | None = None,
    reference_layer: str | None = None,
    reference_crs: str | None = None,
    reference_id_field: str | None = None,
    reviewed_labels: str | None = None,
    predictor: Predictor | None = None,
    progress: ProgressFn | None = None,
) -> PipelineOutput:
    cfg = cfg or PipelineConfig()
    predictor = predictor or cached_predictor(cfg.inference.model_id, cfg.inference.backend,
                                              cfg.inference.device, cfg.inference.weights_dir)
    seg = segment_raster(input_path, predictor, cfg, progress)
    raw, cand = extract_footprints(seg.probability, seg.valid, seg.transform, seg.crs, cfg.postprocess)
    report: dict = {
        "segmentation": {"processed_grid": f"{seg.probability.shape[1]}x{seg.probability.shape[0]}",
                         "resample_factor": round(seg.scale, 4),
                         "processed_gsd_m": round(seg.info.gsd_m * seg.scale, 4) if seg.info.gsd_m else None,
                         "tiles": seg.n_tiles, "raw_polygons": len(raw), "candidates": len(cand),
                         **{k: round(v, 2) for k, v in seg.timings.items()}},
        "geometry_quality": geometry_quality(cand),
    }
    m = None
    det = None
    if reference:
        ref_fs, ref_rep = load_reference(reference, reference_layer, cfg.matching.reference_kind,
                                         reference_crs, reference_id_field)
        report["reference_import"] = ref_rep.to_dict()
        if cand.crs is None:
            report["correspondence"] = {"status": "not_evaluated", "reason": "input raster has no CRS"}
        else:
            m = match(cand, ref_fs, cfg.matching, _evaluated_area(seg), seg.crs)
            report["correspondence"] = m.summary
    if reviewed_labels:
        truth, _ = load_reference(reviewed_labels, kind="reviewed_roofs", assume_crs=reference_crs)
        det = detection_metrics(cand, truth)
        report["detection_metrics"] = det
    report["iso19157"] = iso19157_report(report["geometry_quality"], det, m)
    report["limitations"] = LIMITATIONS
    report["provenance"] = provenance(cfg, predictor, seg.info.to_dict() if seg.info else {})

    files: dict = {}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files["mask"] = write_mask_geotiff(seg.probability, out / "mask.tif", seg.transform, seg.crs, "mask",
                                           threshold=cfg.postprocess.threshold)
        files["probability"] = write_mask_geotiff(seg.probability, out / "probability.tif", seg.transform, seg.crs,
                                                  "probability")
        files["raw_geojson"] = write_geojson(raw, out / "candidates_raw.geojson")
        files["candidates_geojson"] = write_geojson(m.candidates if m else cand, out / "candidates.geojson")
        if m:
            files["reference_geojson"] = write_geojson(m.references, out / "reference_matches.geojson")
        if cand.crs is not None:
            layers = {"raw_prediction": raw, "candidates": m.candidates if m else cand}
            if m:
                layers["reference_matches"] = m.references
            files["gpkg"] = write_gpkg(layers, out / "results.gpkg")
        files["report_json"], files["report_md"] = write_report(report, out)
    return PipelineOutput(seg, raw, cand, m, report, {k: str(v) for k, v in files.items()})


def run_image(img: np.ndarray, cfg: PipelineConfig | None = None, predictor: Predictor | None = None) -> PipelineOutput:
    """In-memory RGB image without georeferencing (Streamlit demo path). Outputs are in pixel coordinates."""
    cfg = cfg or PipelineConfig()
    predictor = predictor or cached_predictor(cfg.inference.model_id, cfg.inference.backend,
                                              cfg.inference.device, cfg.inference.weights_dir)
    seg = segment_array(img, predictor, cfg)
    raw, cand = extract_footprints(seg.probability, seg.valid, None, None, cfg.postprocess)
    report = {
        "segmentation": {"image_size": f"{img.shape[1]}x{img.shape[0]}", "tiles": seg.n_tiles,
                         "candidates": len(cand), **{k: round(v, 2) for k, v in seg.timings.items()}},
        "geometry_quality": geometry_quality(cand),
        "limitations": LIMITATIONS + ["No georeferencing: polygons are in image pixel coordinates."],
        "provenance": provenance(cfg, predictor, {"path": "in-memory image", "crs": None}),
    }
    return PipelineOutput(seg, raw, cand, None, report, {})
