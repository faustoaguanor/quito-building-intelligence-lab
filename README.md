# Quito Building Intelligence Lab

**AI-assisted building footprint extraction and cadastral quality assessment** — Distrito Metropolitano de Quito, Ecuador.

One reusable Python engine, three delivery targets:

| Target | What it is | Entry point |
|---|---|---|
| Local app / CLI | Tiled processing of large GeoTIFF orthophotos, CPU or CUDA, cadastral comparison, reports | `qbl run …` |
| Streamlit demo | Real CPU inference on one small RGB image (≤ 5 MB, ≤ 512×512) — [live demo](https://quito-building-intelligence-lab-iwqerwtcb7lbmjx7yfd3rx.streamlit.app/) | `app/streamlit_app.py` |
| GitHub Pages | Static portfolio with real example results | `site/` |

> Independent open-source project. It does not imply endorsement by the Municipality of Quito, Streamlit, Hugging Face,
> SpaceNet or any model provider. Visible roof geometry is **not** a legal cadastral boundary, and discrepancies are
> review prompts — **not** proven cadastral errors, unregistered or illegal buildings.

**Links:** [portfolio](https://faustoaguanor.github.io/quito-building-intelligence-lab/) · [Streamlit demo](https://quito-building-intelligence-lab-iwqerwtcb7lbmjx7yfd3rx.streamlit.app/) (may need a moment to wake up) · [ONNX model release](https://github.com/faustoaguanor/quito-building-intelligence-lab/releases/tag/model-onnx-v1). Not published on Hugging Face.

## What works (verified in this repository)

- Pretrained model: **XD_XD SpaceNet-4 U-Net (VGG16 encoder)**, 29.3 M parameters, trusted source + SHA-256 pinned (`docs/MODEL.md`).
- Real inference on PyTorch and ONNX Runtime (max probability difference 3.2e-8, mask agreement 100 %).
- Windowed raster reading, resampling to a target GSD, overlap blending, NoData handling, bounded-memory vectorisation.
- Cadastral import (GeoJSON/GeoPackage), many-to-many matching, IoU/coverage/area differences, ISO 19157-organised report.
- WMS/WFS area picker (Streamlit map + CLI) — tested only against a simulated service; the real Quito services are unverified.
- Exports: GeoJSON (RFC 7946, WGS 84), GeoPackage (metric CRS), GeoTIFF masks, JSON/Markdown reports.
- 50 deterministic tests (`pytest`), ruff lint, CI and Pages workflows.

See [`docs/RESULTS.md`](docs/RESULTS.md) for the measured numbers and their caveats.

## Quick start (Linux)

```bash
git clone https://github.com/faustoaguanor/quito-building-intelligence-lab.git
cd quito-building-intelligence-lab
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu      # or a CUDA build, see docs/INSTALL.md
pip install -e ".[gis,torch,onnx,export,app,dev]"
qbl download-model            # downloads + verifies the original trusted weights (117 MB)
qbl export-onnx               # optional: safetensors + ONNX, validated against PyTorch
pytest
```

Process a large orthophoto (no source edits needed):

```bash
qbl info /data/ortho.tif                         # validate CRS, bands, GSD
qbl run /data/ortho.tif --out outputs/run1 \
    --config configs/quito_drone_7cm.yaml --device auto \
    --reference /data/construcciones.gpkg --reference-layer construcciones --reference-kind construction
```

Fetch a small area from a WMS (and optionally WFS polygons) and process it (URLs and layer names are yours to verify):

```bash
qbl wms-layers https://<server>/geoserver/<workspace>/wms --limit 50
qbl fetch-area --wms-url https://<server>/geoserver/<workspace>/wms --layer <layer> \
    --bbox -78.5002 -0.1802 -78.4992 -0.1792 --out outputs/area1 \
    --wfs-url https://<server>/geoserver/<workspace>/wfs --wfs-type <workspace:layer>
qbl run outputs/area1/wms_area.tif --out outputs/area1/run --reference outputs/area1/reference.geojson --reference-kind uc
```

Run the demo UI locally: `streamlit run app/streamlit_app.py`.

## Documentation

[Architecture](docs/ARCHITECTURE.md) · [Install](docs/INSTALL.md) · [Model](docs/MODEL.md) · [Data sources](docs/DATA_SOURCES.md) ·
[Quality methodology](docs/QUALITY_METHODOLOGY.md) · [Training](docs/TRAINING.md) · [Deployment](docs/DEPLOYMENT.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) · [Results](docs/RESULTS.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

Original code: MIT © 2026 faustoaguanor. Third-party model weights and demo imagery keep their own licenses (see notices).
