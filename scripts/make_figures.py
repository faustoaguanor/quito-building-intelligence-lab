"""Regenerate the real demo results and figures used by the portfolio site (needs model weights)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

from quito_building_lab.config import PipelineConfig
from quito_building_lab.export import to_geojson_dict
from quito_building_lab.pipeline import run_raster

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo"
SITE = ROOT / "site" / "assets"


def draw(img, polys, color, width=2):
    d = ImageDraw.Draw(img)
    for g in polys:
        d.line(list(g.exterior.coords), fill=color, width=width)


def main() -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig()
    cfg.inference.backend = "auto"
    ref = str(DEMO / "vegas_sn2_img10_reference_buildings.geojson")
    out = run_raster(str(DEMO / "vegas_sn2_img10_rgb8.tif"), str(ROOT / "examples" / "demo_results"), cfg,
                     reference=ref, reviewed_labels=ref)
    base = Image.open(DEMO / "vegas_sn2_img10_rgb8.png").convert("RGB")
    with rasterio.open(DEMO / "vegas_sn2_img10_rgb8.tif") as s:
        inv = ~s.transform
    from shapely.affinity import affine_transform

    from quito_building_lab.crs_utils import reproject_geom

    def to_px(fs):
        out_g = []
        for g in fs.geoms:
            g4326 = reproject_geom(g, fs.crs, "EPSG:4326")
            out_g.append(affine_transform(g4326, [inv.a, inv.b, inv.d, inv.e, inv.xoff, inv.yoff]))
        return out_g

    prob = np.asarray(out.segmentation.probability)
    heat = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).convert("L").resize(base.size)
    heat_rgb = Image.merge("RGB", (heat, heat.point(lambda v: v // 2), Image.new("L", heat.size, 40)))
    Image.blend(base, heat_rgb, 0.55).save(SITE / "demo_probability.png", optimize=True)
    ov = base.copy()
    draw(ov, to_px(out.matching.references), (60, 200, 255), 2)
    draw(ov, to_px(out.candidates), (255, 235, 0), 2)
    ov.save(SITE / "demo_overlay.png", optimize=True)
    base.save(SITE / "demo_input.png", optimize=True)
    summary = {
        "segmentation": out.report["segmentation"],
        "correspondence": {k: v for k, v in out.report["correspondence"].items() if k != "groups"},
        "detection_metrics": out.report["detection_metrics"],
        "provenance": {k: out.report["provenance"][k] for k in ("model", "runtime", "processed_at")},
    }
    (SITE / "demo_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (SITE / "demo_candidates.geojson").write_text(json.dumps(to_geojson_dict(out.matching.candidates)))
    print(json.dumps(summary["detection_metrics"], indent=2))


if __name__ == "__main__":
    main()
