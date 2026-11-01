# Measured results (all produced by code in this repo, 2026-11-01, 4-core CPU sandbox)

**Demo inference** (`examples/demo`, SpaceNet-2 Vegas crop, ONNX, default config) vs the 20 SpaceNet reference polygons:

| measure | value |
|---|---|
| candidates / references | 15 / 20 |
| precision / recall / F1 (IoU ≥ 0.5, one-to-one) | 0.667 / 0.500 / 0.571 |
| mean IoU of matched objects | 0.717 |
| area IoU / area precision / area recall | 0.671 / 0.809 / 0.798 |

Caveats: a single 512×512 crop, n = 20, a non-Quito scene, labels from the dataset (not re-reviewed). It shows the pipeline works end-to-end; it is **not** a validation of accuracy in Quito.

**ONNX vs PyTorch** (random input, 512²): max |Δlogit| 5.7e-6, max |Δprob| 3.2e-8, mask agreement 1.0; 1.43 s (ORT) vs 2.21 s (torch) per tile.
**Streamlit path** (ONNX, no torch/geopandas imported), local: ~2 s CPU per image, peak RSS ≈ 790 MB (ORT arena off; with arena on, 1.19 GB).
**Streamlit Community Cloud** (live demo, bundled example image, one run, reported by the app itself): 20 candidates, 68 970 px², 2.2 s CPU, peak memory 829 MB. Identical candidate count and area to the local run. Only this single run was observed: custom uploads, concurrent use and the plan's memory ceiling were not tested.
**Large raster:** synthetic 3072×3072 mosaic made by repeating the demo crop, 49 tiles, 90 s total, peak RSS 779 MB, 390 candidates (a scalability/behaviour test, not an accuracy test).

**Real IGM Ecuador WMS (reported by the user on 2026-11-01, not observed by the development environment, which cannot reach the service):** the layer draws in the app's map; the automatic server-side download fails with `java.lang.NullPointerException` for every request shape tried (256/512 px, PNG/JPEG, with/without transparency, EPSG:3857/32717/4326); the browser-download path works end to end (link opened in a new tab, image saved and uploaded, inference ran). Building-detection quality on that imagery has not been assessed.
