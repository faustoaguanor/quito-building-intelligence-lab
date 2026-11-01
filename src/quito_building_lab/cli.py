"""Command-line interface: ``qbl <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import PipelineConfig


def _progress(i: int, n: int) -> None:
    if i == n or i % max(1, n // 50) == 0:
        sys.stderr.write(f"\r  tiles {i}/{n} ({100 * i // n}%)")
        if i == n:
            sys.stderr.write("\n")
        sys.stderr.flush()


def _load_cfg(args) -> PipelineConfig:
    cfg = PipelineConfig.from_yaml(args.config) if getattr(args, "config", None) else PipelineConfig()
    ov = {
        ("inference", "backend"): args.backend, ("inference", "device"): args.device,
        ("inference", "batch_size"): args.batch_size, ("inference", "weights_dir"): args.weights_dir,
        ("inference", "normalization"): args.normalization,
        ("tiling", "tile_size"): args.tile_size, ("tiling", "overlap"): args.overlap,
        ("postprocess", "threshold"): args.threshold, ("postprocess", "min_area_m2"): args.min_area,
        ("matching", "reference_kind"): getattr(args, "reference_kind", None),
    }
    for (sec, key), val in ov.items():
        if val is not None:
            setattr(getattr(cfg, sec), key, val)
    if args.target_gsd is not None:
        cfg.tiling.target_gsd = None if args.target_gsd <= 0 else args.target_gsd
    return cfg


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="YAML configuration file (CLI flags override it)")
    p.add_argument("--backend", choices=["auto", "torch", "onnx"])
    p.add_argument("--device", help="cpu | cuda | cuda:0 | auto")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--weights-dir")
    p.add_argument("--normalization", choices=["image_stats", "spacenet_dn"])
    p.add_argument("--tile-size", type=int)
    p.add_argument("--overlap", type=int)
    p.add_argument("--target-gsd", type=float, help="resample to this GSD in metres (<=0: native)")
    p.add_argument("--threshold", type=float)
    p.add_argument("--min-area", type=float, help="minimum polygon area in m2")


def cmd_run(args) -> int:
    from .pipeline import run_raster

    cfg = _load_cfg(args)
    out = run_raster(args.input, args.out, cfg, args.reference, args.reference_layer, args.reference_crs,
                     args.reference_id_field, args.reviewed_labels, progress=_progress)
    print(json.dumps({"segmentation": out.report["segmentation"],
                      "correspondence": {k: v for k, v in (out.report.get("correspondence") or {}).items()
                                         if k != "groups"},
                      "detection_metrics": out.report.get("detection_metrics"),
                      "files": out.files}, indent=2, ensure_ascii=False))
    return 0


def cmd_compare(args) -> int:
    """Compare an existing candidate vector file with a reference dataset (no inference)."""
    from .config import MatchingConfig
    from .export import write_geojson, write_gpkg, write_report
    from .matching import match
    from .quality import geometry_quality, iso19157_report
    from .reference import load_reference

    cand, _ = load_reference(args.candidates, args.candidates_layer, "other", args.candidates_crs)
    for i, a in enumerate(cand.attrs, 1):
        a["cand_id"] = i
    ref, rep = load_reference(args.reference, args.reference_layer, args.reference_kind, args.reference_crs,
                              args.reference_id_field)
    mcfg = MatchingConfig(strong_iou=args.strong_iou, min_coverage=args.min_coverage, reference_kind=args.reference_kind)
    m = match(cand, ref, mcfg)
    gq = geometry_quality(cand)
    report = {"reference_import": rep.to_dict(), "geometry_quality": gq, "correspondence": m.summary,
              "iso19157": iso19157_report(gq, None, m)}
    write_geojson(m.candidates, f"{args.out}/candidates_matched.geojson")
    write_geojson(m.references, f"{args.out}/reference_matches.geojson")
    write_gpkg({"candidates": m.candidates, "reference_matches": m.references}, f"{args.out}/comparison.gpkg")
    write_report(report, args.out, "comparison_report")
    print(json.dumps({k: v for k, v in m.summary.items() if k != "groups"}, indent=2, ensure_ascii=False))
    return 0


def cmd_info(args) -> int:
    from .raster_io import compute_scale, inspect_raster

    info = inspect_raster(args.input)
    d = info.to_dict()
    d["resample_factor_at_0.3m"] = round(compute_scale(info, 0.3), 3)
    print(json.dumps(d, indent=2))
    return 0


def cmd_download(args) -> int:
    from .models import download_weights

    print(download_weights(args.model, args.weights_dir))
    return 0


def cmd_export(args) -> int:
    from .onnx_export import export_onnx, export_safetensors

    print(export_safetensors(args.model, args.weights_dir))
    print(json.dumps(export_onnx(args.model, args.weights_dir, opset=args.opset), indent=2))
    return 0


def cmd_wms_layers(args) -> int:
    from . import ogc

    if args.wfs:
        caps = ogc.wfs_capabilities(args.url)
        rows = [{"name": f.name, "title": f.title, "default_crs": f.default_crs} for f in caps.feature_types]
    else:
        caps = ogc.wms_capabilities(args.url)
        rows = [{"name": layer.name, "title": layer.title, "crs": layer.crs[:6]} for layer in caps.layers]
    print(json.dumps({"url": args.url, "version": caps.version, "count": len(rows), "items": rows[: args.limit]},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_fetch_area(args) -> int:
    """One small WMS GetMap (+ optional WFS GetFeature) for a user-chosen area. Never bulk-downloads."""
    from pathlib import Path

    from . import ogc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bbox = tuple(args.bbox)
    if not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise ValueError("--bbox must be W S E N in lon/lat degrees (west < east, south < north)")
    caps = ogc.wms_capabilities(args.wms_url)
    layer = next((layer for layer in caps.layers if layer.name == args.layer), None)
    if layer is None:
        raise ValueError(f"Layer '{args.layer}' not found; list layers with `qbl wms-layers {args.wms_url}`")
    img = ogc.wms_get_map(args.wms_url, args.layer, bbox, layer.crs, caps.version, args.target_gsd,
                          max_side=args.max_side)
    tif = ogc.write_geotiff(img, str(out / "wms_area.tif"))
    result = {"image": tif, "wms": img.meta}
    if args.wfs_url:
        from .reference import reference_from_geojson

        data, info = ogc.wfs_get_features(args.wfs_url, args.wfs_type, bbox, args.wfs_srs or img.crs,
                                          max_features=args.max_features)
        data["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:{info['srs'].replace(':', '::')}"}}
        ref_path = out / "reference.geojson"
        ref_path.write_text(json.dumps(data), encoding="utf-8")
        _, rep = reference_from_geojson(data, info["srs"], args.reference_kind, source=args.wfs_url)
        result.update({"reference": str(ref_path), "wfs": info, "reference_import": rep.to_dict()})
    (out / "fetch_provenance.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qbl", description="Quito Building Intelligence Lab")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="segment a GeoTIFF, vectorise and (optionally) compare with references")
    p.add_argument("input")
    p.add_argument("--out", required=True)
    p.add_argument("--reference", help="cadastral/reference polygons (GeoJSON / GeoPackage)")
    p.add_argument("--reference-layer")
    p.add_argument("--reference-crs", help="CRS to assign ONLY if the reference file lacks one")
    p.add_argument("--reference-id-field")
    p.add_argument("--reference-kind", choices=["construction", "block", "uc", "reviewed_roofs", "other"])
    p.add_argument("--reviewed-labels", help="independently reviewed roof polygons for detection metrics")
    _add_common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="compare existing candidate polygons with reference polygons")
    p.add_argument("candidates")
    p.add_argument("reference")
    p.add_argument("--out", required=True)
    p.add_argument("--candidates-layer")
    p.add_argument("--candidates-crs")
    p.add_argument("--reference-layer")
    p.add_argument("--reference-crs")
    p.add_argument("--reference-id-field")
    p.add_argument("--reference-kind", default="construction",
                   choices=["construction", "block", "uc", "reviewed_roofs", "other"])
    p.add_argument("--strong-iou", type=float, default=0.5)
    p.add_argument("--min-coverage", type=float, default=0.1)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("wms-layers", help="list the layers of a WMS (or WFS with --wfs) with one GetCapabilities request")
    p.add_argument("url")
    p.add_argument("--wfs", action="store_true")
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_wms_layers)

    p = sub.add_parser("fetch-area", help="fetch ONE small WMS image (and optional WFS polygons) for a bbox")
    p.add_argument("--wms-url", required=True)
    p.add_argument("--layer", required=True)
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"),
                   help="lon/lat degrees (WGS 84)")
    p.add_argument("--out", required=True)
    p.add_argument("--target-gsd", type=float, default=0.3)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--wfs-url")
    p.add_argument("--wfs-type", help="WFS feature type (e.g. workspace:layer)")
    p.add_argument("--wfs-srs", help="projected CRS for the WFS request (default: same as the image)")
    p.add_argument("--max-features", type=int, default=2000)
    p.add_argument("--reference-kind", default="uc", choices=["construction", "block", "uc", "other"])
    p.set_defaults(func=cmd_fetch_area)

    p = sub.add_parser("info", help="validate raster metadata")
    p.add_argument("input")
    p.set_defaults(func=cmd_info)

    for name, fn, hlp in (("download-model", cmd_download, "download trusted weights (SHA-256 verified)"),
                          ("export-onnx", cmd_export, "convert weights to safetensors + ONNX and validate")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--model", default="xdxd_spacenet4_unet_vgg16")
        p.add_argument("--weights-dir")
        if name == "export-onnx":
            p.add_argument("--opset", type=int, default=17)
        p.set_defaults(func=fn)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:  # includes OGCError, CRSError
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
