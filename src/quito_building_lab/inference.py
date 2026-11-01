"""Model inference backends (PyTorch or ONNX Runtime) with a shared interface."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from .models import ModelWeightsError, get_spec, load_torch_model, weight_paths


def resolve_device(device: str) -> str:
    """Normalise the device string; 'auto' picks CUDA if available."""
    device = (device or "cpu").lower()
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if device == "auto":
            return "cpu"
        raise RuntimeError("CUDA requested but PyTorch is not installed.") from None
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class Predictor:
    """Run the segmentation model on NCHW float32 batches and return N x H x W probabilities."""

    backend: str

    def predict(self, batch: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class TorchPredictor(Predictor):
    backend = "torch"

    def __init__(self, model_id: str, device: str = "cpu", wdir: str | None = None):
        import torch

        self.torch = torch
        self.device = resolve_device(device)
        self.model = load_torch_model(model_id, wdir, self.device)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            x = self.torch.from_numpy(batch).to(self.device)
            logits = self.model(x)[:, 0]
            return self.torch.sigmoid(logits).float().cpu().numpy()


class OnnxPredictor(Predictor):
    backend = "onnx"

    def __init__(self, model_id: str, device: str = "cpu", wdir: str | None = None, threads: int | None = None):
        import onnxruntime as ort

        path = weight_paths(model_id, wdir)["onnx"]
        if not path.exists():
            raise ModelWeightsError(f"ONNX model not found: {path}. Run `qbl export-onnx` first.")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.device = "cuda" if providers[0].startswith("CUDA") else "cpu"

    def predict(self, batch: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: batch})[0][:, 0]
        return sigmoid(logits).astype(np.float32)


def create_predictor(model_id: str, backend: str = "auto", device: str = "cpu", wdir: str | None = None) -> Predictor:
    """Create a predictor. 'auto' uses ONNX Runtime if an exported model exists, otherwise PyTorch."""
    get_spec(model_id)
    threads = int(os.environ.get("QBL_ORT_THREADS", "0")) or None
    if backend == "onnx":
        return OnnxPredictor(model_id, device, wdir, threads)
    if backend == "torch":
        return TorchPredictor(model_id, device, wdir)
    if backend != "auto":
        raise ValueError(f"Unknown backend '{backend}'")
    if weight_paths(model_id, wdir)["onnx"].exists() and not device.startswith("cuda"):
        try:
            return OnnxPredictor(model_id, device, wdir, threads)
        except ImportError:
            pass
    try:
        return TorchPredictor(model_id, device, wdir)
    except ImportError as exc:
        raise ModelWeightsError(
            "No usable backend: install onnxruntime (with an exported .onnx) or PyTorch."
        ) from exc


@lru_cache(maxsize=2)
def cached_predictor(model_id: str, backend: str = "auto", device: str = "cpu", wdir: str | None = None) -> Predictor:
    """Process-wide model cache (one loaded model per configuration)."""
    return create_predictor(model_id, backend, device, wdir)
