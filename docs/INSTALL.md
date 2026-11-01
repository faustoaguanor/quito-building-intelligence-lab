# Local installation (Linux)

Requirements: Python ≥ 3.10, ~2 GB disk, 8 GB RAM recommended. Rasterio, Shapely and PyProj ship manylinux wheels (no system GDAL needed).

```bash
python3 -m venv .venv && source .venv/bin/activate
```

## PyTorch (choose ONE; do not assume a CUDA version)
- CPU only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- GPU: check `nvidia-smi` for the driver's supported CUDA, then use the matching index from <https://pytorch.org/get-started/locally/> (e.g. `…/whl/cu121`). Verify: `python -c "import torch;print(torch.cuda.is_available())"`.
- ONNX-only inference (no torch) is possible after exporting once on a machine that has torch: copy `*.onnx` into `~/.cache/quito_building_lab` (or `$QBL_WEIGHTS_DIR`).

```bash
pip install -e ".[gis,torch,onnx,export,app,dev]"
qbl download-model && qbl export-onnx && pytest
```

## Practical limits
- CPU: ≈ 1.4–2.2 s per 512×512 tile on 4 cores (measured); ~100 tiles ≈ 2–4 min. A 10 000 × 10 000 px ortho at 0.3 m is ~1 000 tiles ⇒ ~30–40 min on 4 cores.
- GPU VRAM: tile 512 with batch size 4–8 fits ~4–6 GB (VGG16 U-Net); lower `--batch-size` on OOM. Not benchmarked here (no GPU in the build sandbox).
- RAM: roughly 9 B × processed pixels for accumulators (disk-backed above 64 M px) plus ~0.8 GB for the model/runtime. `max_pixels` guards runaway grids.
- Not unlimited: extremely large mosaics need time, free disk for temp memmaps, and a sane `target_gsd`.
