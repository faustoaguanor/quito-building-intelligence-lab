# Reproducibility

```bash
pip install -e ".[gis,torch,onnx,export,app,dev]"
qbl download-model && qbl export-onnx
python scripts/make_figures.py      # regenerates examples/demo_results and site/assets
pytest -q                           # 50 tests; the real-inference smoke test is skipped without weights
```
Tested with Python 3.13.18, numpy 2.5.3, rasterio 1.5.2, shapely 2.2.0, pyproj 3.8.0, geopandas 1.2.0, torch 2.15.0 (run on CPU), onnxruntime 1.29.0, streamlit 1.66.1. Versions are minimums in `pyproject.toml`; exact versions are recorded in every `report.json`. Inference is deterministic on CPU for fixed inputs; ONNX hashes may differ across torch versions (set `QBL_ONNX_SHA256` accordingly).
