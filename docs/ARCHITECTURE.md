# Architecture

```
GeoTIFF / RGB image
   │ raster_io        validate CRS/bands/GSD, windows, NoData, resampling
   ▼ preprocessing    per-image z-score (or SpaceNet DN constants)
   ▼ inference        TorchPredictor | OnnxPredictor  (cached, CPU/CUDA)
   ▼ segmentation     tiled prediction, linear overlap blending, memmap for huge grids
   ▼ vectorize        threshold → sieve → blockwise polygonize → validity fix → simplify → min area → dedupe
   ▼ reference/matching/quality   import, many-to-many matching, ISO 19157-organised report
   ▼ export           GeoJSON (WGS 84) · GeoPackage (metric CRS) · GeoTIFF masks · JSON/MD reports
```

`pipeline.run_raster` (CLI, local) and `pipeline.run_image` (Streamlit) are thin orchestrators over the same modules.

## Key design decisions

- **Tile-edge reconciliation:** overlapping tiles are blended with linear weights; no per-tile polygons exist, so tile borders cannot split objects. Vectorisation runs on blocks and merges polygons touching block borders.
- **Memory:** the processed grid accumulators switch to disk-backed `np.memmap` above 64 M pixels; reads are windowed. Peak RSS measured at ~780 MB for a 3072² mosaic (CPU, ONNX) — independent of mosaic size apart from the full-grid accumulators (≈ 9 bytes/pixel; on disk above the threshold).
- **GSD handling:** the model saw ~0.5 m imagery. `target_gsd` downsamples finer imagery (never upsamples). The real GSD must be read from the raster (`qbl info`), not assumed.
- **Georeferencing honesty:** no CRS ⇒ outputs stay in pixel coordinates and GeoJSON is flagged `coordinate_space: image_pixels`. Missing CRS on a reference file is an error unless `--reference-crs` is given.
- **Stages are separated:** `raw_prediction`, `processed_prediction`, `reference`, (`human_reviewed` for external labels).
- **Matching model:** many-to-many via connected components (see `docs/QUALITY_METHODOLOGY.md`).
