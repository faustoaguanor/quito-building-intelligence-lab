# Model selection

| Candidate | Weights | Task | Arch / size | CPU 512² | License of weights | Verdict |
|---|---|---|---|---|---|---|
| **XD_XD SpaceNet-4 U-Net VGG16** (chosen) | AWS Open Data (`spacenet-model-weights/spacenet-4`), 117 MB | building footprints, RGB | U-Net / VGG16, 29.3 M params | 1.4 s (ONNX), 2.2 s (torch), 4 cores | SpaceNet data CC BY-SA 4.0; code via Apache-2.0 `solaris` (as declared by that repo) | Selected: RGB input, dedicated building task, ONNX export validated |
| SelimSef SpaceNet-4 ResNet34/DenseNet U-Net | same bucket (62–165 MB) | building footprints | U-Net / ResNet34 or DenseNet; 4-channel config | not measured | same | Not benchmarked; 4-band config and multi-channel mask head make RGB reuse less direct |
| SAM-style promptable model | external | generic assisted segmentation | ViT, hundreds of MB | slow on CPU | varies | **Not** an automatic building detector; only an assisted fallback, not implemented |

Only the first row was installed and run; the rest are compared on documented properties, not on measured results.

## Chosen model details
- Source URL and SHA-256 are pinned in `models.py` (`c870877e…78b31`). `qbl download-model` verifies the hash; checkpoints are loaded with `torch.load(weights_only=True)`; user-supplied checkpoints are never loaded.
- Input: 3×H×W float32 (H, W multiples of 32; trained on 512² chips), output: 1-channel logit (sigmoid → probability).
- Preprocessing: the original training used SpaceNet-4 constants on 16-bit DN. For arbitrary 8-bit orthophotos the default is **per-image z-score** (`image_stats`); `spacenet_dn` is available for raw WorldView-like DN.
- Training domain: Atlanta, WorldView-2 pan-sharpened ~0.5 m. **Generalisation to Quito imagery is unverified.**
- Redistribution: weights stay on their original host; this repo ships only an official download script. A Hugging Face packaging helper (`scripts/package_model.py`) prepares the model card/checksums; publishing requires the maintainer's token and a license review. No Hub upload was performed.
- ONNX (opset 17): max |Δprob| = 3.2e-8, mask agreement 100 % at 0.5 on random input; see `docs/RESULTS.md`.
