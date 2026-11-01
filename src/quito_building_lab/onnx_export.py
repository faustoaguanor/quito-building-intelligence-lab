"""Convert trusted PyTorch weights to safetensors and ONNX, and validate numerical consistency."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .models import ALIAS_RE, get_spec, load_torch_model, sha256_file, weight_paths


def export_safetensors(model_id: str, wdir: str | None = None) -> Path:
    from safetensors.torch import save_file

    model = load_torch_model(model_id, wdir)
    out = weight_paths(model_id, wdir)["safetensors"]
    # convN.* are aliases of encoder.* (shared modules); store each tensor once
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items() if not ALIAS_RE.match(k)}
    save_file(state, str(out), metadata={"model_id": model_id, "source_sha256": get_spec(model_id).source_sha256})
    return out


def export_onnx(model_id: str, wdir: str | None = None, opset: int = 17, tile: int | None = None,
                rtol_check: bool = True) -> dict:
    """Export ONNX (dynamic batch/H/W, multiples of 32) and compare it with PyTorch."""
    import torch

    spec = get_spec(model_id)
    tile = tile or spec.tile_size
    model = load_torch_model(model_id, wdir)
    out = weight_paths(model_id, wdir)["onnx"]
    dummy = torch.randn(1, 3, tile, tile)
    torch.onnx.export(
        model, dummy, str(out), input_names=["image"], output_names=["logits"], opset_version=opset,
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch", 2: "height", 3: "width"}},
        dynamo=False,
    )
    report = {"model_id": model_id, "onnx_path": str(out), "onnx_sha256": sha256_file(out),
              "onnx_size_bytes": out.stat().st_size, "opset": opset, "tile": tile}
    if rtol_check:
        report.update(compare_backends(model, out, tile))
    meta = out.with_suffix(".onnx.json")
    meta.write_text(json.dumps(report, indent=2))
    return report


def compare_backends(model, onnx_path: Path, tile: int, seed: int = 0, repeats: int = 3) -> dict:
    import onnxruntime as ort
    import torch

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 3, tile, tile)).astype(np.float32)
    with torch.inference_mode():
        t0 = time.perf_counter()
        for _ in range(repeats):
            ref = model(torch.from_numpy(x)).numpy()
        t_torch = (time.perf_counter() - t0) / repeats
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sess.run(None, {"image": x})  # warm-up
    t0 = time.perf_counter()
    for _ in range(repeats):
        got = sess.run(None, {"image": x})[0]
    t_ort = (time.perf_counter() - t0) / repeats
    if got.shape != ref.shape:
        raise AssertionError(f"Shape mismatch ONNX {got.shape} vs torch {ref.shape}")
    pa, pb = 1 / (1 + np.exp(-ref)), 1 / (1 + np.exp(-got))
    return {
        "output_shape": list(got.shape),
        "max_abs_logit_diff": float(np.abs(ref - got).max()),
        "max_abs_prob_diff": float(np.abs(pa - pb).max()),
        "mask_agreement_at_0.5": float(((pa >= 0.5) == (pb >= 0.5)).mean()),
        "cpu_seconds_per_tile_torch": round(t_torch, 3),
        "cpu_seconds_per_tile_onnx": round(t_ort, 3),
        "torch_threads": torch.get_num_threads(),
    }
