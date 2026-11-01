"""Prepare a Hugging Face model-card folder (no upload, no credentials)."""

from __future__ import annotations

import json
from pathlib import Path

from quito_building_lab.models import get_spec, sha256_file, weight_paths

MODEL_ID = "xdxd_spacenet4_unet_vgg16"
OUT = Path(__file__).resolve().parents[1] / "models" / "hf"


def main() -> None:
    spec = get_spec(MODEL_ID)
    OUT.mkdir(parents=True, exist_ok=True)
    sums = {"source_pth": {"sha256": spec.source_sha256, "url": spec.source_url}}
    for kind, p in weight_paths(MODEL_ID).items():
        if kind != "pth" and p.exists():
            sums[kind] = {"file": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size}
    (OUT / "checksums.json").write_text(json.dumps(sums, indent=2))
    card = f"""---
license: other
library_name: onnx
tags: [building-segmentation, remote-sensing, aerial, onnx]
---
# {spec.name} (repackaged, not original work)

Third-party weights. Original authors: XD_XD (SpaceNet-4 winner); published by CosmiQ Works / SpaceNet via the
`solaris` model zoo. This repository only repackages them (safetensors / ONNX) for the Quito Building Intelligence Lab.
**Verify the SpaceNet weight/data license (training data CC BY-SA 4.0) before redistributing.**

- Architecture: {spec.architecture}
- Input: 3xHxW float32, per-image z-score (see project docs); H, W multiples of 32. Output: 1-channel logit.
- Training data: {spec.training_data}, ~{spec.training_gsd_m} m GSD. Not validated on Quito imagery.
- Intended use: candidate roof extraction for human review. Not for legal cadastral decisions.
- Source checkpoint SHA-256: `{spec.source_sha256}`
- Pin downloads by revision and verify `checksums.json`.
- Citation: {spec.citation}
"""
    (OUT / "README.md").write_text(card)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
