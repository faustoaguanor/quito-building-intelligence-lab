---
license: other
library_name: onnx
tags: [building-segmentation, remote-sensing, aerial, onnx]
---
# XD_XD SpaceNet-4 U-Net (VGG16 encoder) (repackaged, not original work)

Third-party weights. Original authors: XD_XD (SpaceNet-4 winner); published by CosmiQ Works / SpaceNet via the
`solaris` model zoo. This repository only repackages them (safetensors / ONNX) for the Quito Building Intelligence Lab.
**Verify the SpaceNet weight/data license (training data CC BY-SA 4.0) before redistributing.**

- Architecture: U-Net, VGG16 encoder, bilinear-upsampling decoder (num_filters=32), ~29.3 M params
- Input: 3xHxW float32, per-image z-score (see project docs); H, W multiples of 32. Output: 1-channel logit.
- Training data: SpaceNet 4 Off-Nadir Building Footprints (Atlanta, WorldView-2 pan-sharpened RGB), ~0.5 m GSD. Not validated on Quito imagery.
- Intended use: candidate roof extraction for human review. Not for legal cadastral decisions.
- Source checkpoint SHA-256: `c870877e4f0140ed8684b0ef368b334a825809992a39bc1c8cb7aca002b78b31`
- Pin downloads by revision and verify `checksums.json`.
- Citation: Weir, N. et al. (2019). SpaceNet MVOI: a Multi-View Overhead Imagery Dataset. ICCV. Model: 1st-place SpaceNet-4 solution by XD_XD, packaged in CosmiQ/solaris.
