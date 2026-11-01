# Contributing

1. Fork and branch from `main`.
2. `pip install -e ".[gis,onnx,dev]"`; run `ruff check .` and `pytest -q` before opening a PR.
3. Tests must be deterministic and must not need network access or model weights (mark weight-dependent tests `@pytest.mark.model`).
4. Do not commit rasters, weights, restricted cadastral data, personal data or credentials.
5. Keep code and technical docs in English; user-facing UI text in Spanish.
