"""Model registry, trusted weight download and (optional) PyTorch architecture.

Only weights listed in ``MODEL_REGISTRY`` (pinned URL + SHA-256) are ever downloaded or loaded.
User-supplied checkpoints are not accepted by the public demo.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# state-dict keys of the U-Net that alias encoder.* layers (same nn.Module objects)
ALIAS_RE = re.compile(r"^conv[1-5]\.")


class ModelWeightsError(RuntimeError):
    """Weights are missing, corrupt or not registered."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    architecture: str
    source_url: str
    source_sha256: str
    source_size: int
    tile_size: int
    in_channels: int
    output: str
    training_data: str
    training_gsd_m: float
    onnx_sha256: str | None
    license_note: str
    citation: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xdxd_spacenet4_unet_vgg16": ModelSpec(
        model_id="xdxd_spacenet4_unet_vgg16",
        name="XD_XD SpaceNet-4 U-Net (VGG16 encoder)",
        architecture="U-Net, VGG16 encoder, bilinear-upsampling decoder (num_filters=32), ~29.3 M params",
        source_url=(
            "https://s3.amazonaws.com/spacenet-dataset/spacenet-model-weights/spacenet-4/"
            "xdxd_spacenet4_solaris_weights.pth"
        ),
        source_sha256="c870877e4f0140ed8684b0ef368b334a825809992a39bc1c8cb7aca002b78b31",
        source_size=117240148,
        tile_size=512,
        in_channels=3,
        output="1-channel building-footprint logit (apply sigmoid)",
        training_data="SpaceNet 4 Off-Nadir Building Footprints (Atlanta, WorldView-2 pan-sharpened RGB)",
        training_gsd_m=0.5,
        # ONNX produced by `qbl export-onnx` (torch 2.x, opset 17); other torch versions may yield a
        # different hash, in which case verification is skipped only if QBL_ONNX_SHA256 is set to "any".
        onnx_sha256=None,
        license_note=(
            "Weights published by CosmiQ Works / SpaceNet on the AWS Open Data Registry and distributed "
            "through the Apache-2.0 'solaris' library model zoo. The SpaceNet training data are licensed "
            "CC BY-SA 4.0. This project downloads the weights from the original source; it does not "
            "claim ownership or authorship of them."
        ),
        citation=(
            "Weir, N. et al. (2019). SpaceNet MVOI: a Multi-View Overhead Imagery Dataset. ICCV. "
            "Model: 1st-place SpaceNet-4 solution by XD_XD, packaged in CosmiQ/solaris."
        ),
    )
}


def get_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_id]
    except KeyError as exc:
        raise ModelWeightsError(f"Unknown model '{model_id}'. Available: {sorted(MODEL_REGISTRY)}") from exc


def weights_dir(override: str | None = None) -> Path:
    d = override or os.environ.get("QBL_WEIGHTS_DIR") or Path.home() / ".cache" / "quito_building_lab"
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def weight_paths(model_id: str, wdir: str | None = None) -> dict[str, Path]:
    d = weights_dir(wdir)
    return {
        "pth": d / f"{model_id}.pth",
        "safetensors": d / f"{model_id}.safetensors",
        "onnx": d / f"{model_id}.onnx",
    }


def download_weights(model_id: str, wdir: str | None = None, timeout: int = 120) -> Path:
    """Download the original trusted checkpoint and verify its SHA-256."""
    spec = get_spec(model_id)
    dest = weight_paths(model_id, wdir)["pth"]
    if dest.exists() and sha256_file(dest) == spec.source_sha256:
        return dest
    tmp = Path(tempfile.mkstemp(dir=dest.parent, suffix=".part")[1])
    try:
        with urllib.request.urlopen(spec.source_url, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        digest = sha256_file(tmp)
        if digest != spec.source_sha256:
            raise ModelWeightsError(f"Checksum mismatch for {model_id}: {digest} != {spec.source_sha256}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def ensure_onnx(model_id: str, wdir: str | None = None, timeout: int = 180) -> Path:
    """Return the path of the ONNX model, downloading it from ``QBL_ONNX_URL`` if missing.

    The ONNX file is derived from the trusted checkpoint (see ``qbl export-onnx``). Hosting it
    (Hugging Face Hub / GitHub release) is the maintainer's responsibility; the URL and the
    expected SHA-256 are taken from the environment (``QBL_ONNX_URL``, ``QBL_ONNX_SHA256``).
    """
    spec = get_spec(model_id)
    dest = weight_paths(model_id, wdir)["onnx"]
    expected = os.environ.get("QBL_ONNX_SHA256") or spec.onnx_sha256
    if dest.exists():
        return dest
    url = os.environ.get("QBL_ONNX_URL")
    if not url:
        raise ModelWeightsError(
            f"ONNX model not found at {dest}. Run `qbl export-onnx` locally or set QBL_ONNX_URL "
            "(and QBL_ONNX_SHA256) to a hosted copy."
        )
    if not url.startswith("https://"):
        raise ModelWeightsError("QBL_ONNX_URL must be an https:// URL.")
    if not expected:
        raise ModelWeightsError("Set QBL_ONNX_SHA256 so the downloaded model can be verified.")
    tmp = Path(tempfile.mkstemp(dir=dest.parent, suffix=".part")[1])
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        if sha256_file(tmp) != expected:
            raise ModelWeightsError("Checksum mismatch for downloaded ONNX model.")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


# --------------------------------------------------------------------------------------------
# PyTorch architecture (imported lazily so the ONNX path does not need torch)
# --------------------------------------------------------------------------------------------

_VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


def build_xdxd_unet_vgg16(num_filters: int = 32):
    """Re-implementation of solaris ``XDXD_SpaceNet4_UNetVGG16`` (identical state-dict keys),
    without depending on torchvision."""
    import torch
    from torch import nn

    def vgg16_features() -> nn.Sequential:
        layers: list[nn.Module] = []
        c = 3
        for v in _VGG16_CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]
                c = v
        return nn.Sequential(*layers)

    class ConvRelu(nn.Module):
        def __init__(self, in_, out):
            super().__init__()
            self.conv = nn.Conv2d(in_, out, 3, padding=1)
            self.activation = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.activation(self.conv(x))

    class DecoderBlock(nn.Module):
        def __init__(self, in_c, mid_c, out_c):
            super().__init__()
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ConvRelu(in_c, mid_c),
                ConvRelu(mid_c, out_c),
            )

        def forward(self, x):
            return self.block(x)

    class UNetVGG16(nn.Module):
        def __init__(self, nf: int):
            super().__init__()
            self.encoder = vgg16_features()
            e = self.encoder
            self.pool = nn.MaxPool2d(2, 2)
            self.relu = nn.ReLU(inplace=True)
            r = self.relu
            self.conv1 = nn.Sequential(e[0], r, e[2], r)
            self.conv2 = nn.Sequential(e[5], r, e[7], r)
            self.conv3 = nn.Sequential(e[10], r, e[12], r, e[14], r)
            self.conv4 = nn.Sequential(e[17], r, e[19], r, e[21], r)
            self.conv5 = nn.Sequential(e[24], r, e[26], r, e[28], r)
            self.center = DecoderBlock(512, nf * 16, nf * 8)
            self.dec5 = DecoderBlock(512 + nf * 8, nf * 16, nf * 8)
            self.dec4 = DecoderBlock(512 + nf * 8, nf * 16, nf * 8)
            self.dec3 = DecoderBlock(256 + nf * 8, nf * 8, nf * 2)
            self.dec2 = DecoderBlock(128 + nf * 2, nf * 4, nf)
            self.dec1 = ConvRelu(64 + nf, nf)
            self.final = nn.Conv2d(nf, 1, kernel_size=1)

        def forward(self, x):
            c1 = self.conv1(x)
            c2 = self.conv2(self.pool(c1))
            c3 = self.conv3(self.pool(c2))
            c4 = self.conv4(self.pool(c3))
            c5 = self.conv5(self.pool(c4))
            center = self.center(self.pool(c5))
            d5 = self.dec5(torch.cat([center, c5], 1))
            d4 = self.dec4(torch.cat([d5, c4], 1))
            d3 = self.dec3(torch.cat([d4, c3], 1))
            d2 = self.dec2(torch.cat([d3, c2], 1))
            d1 = self.dec1(torch.cat([d2, c1], 1))
            return self.final(d1)

    return UNetVGG16(num_filters)


def load_torch_model(model_id: str, wdir: str | None = None, device: str = "cpu", download: bool = True):
    """Build the architecture and load trusted weights (safetensors preferred, else verified .pth)."""
    import torch

    get_spec(model_id)
    paths = weight_paths(model_id, wdir)
    model = build_xdxd_unet_vgg16()
    if paths["safetensors"].exists():
        from safetensors.torch import load_file

        state = load_file(str(paths["safetensors"]))
    else:
        if not paths["pth"].exists():
            if not download:
                raise ModelWeightsError(
                    f"Weights for '{model_id}' not found in {paths['pth'].parent}. "
                    "Run `qbl download-model` first."
                )
            download_weights(model_id, wdir)
        ckpt = torch.load(paths["pth"], map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k.removeprefix("module."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # convN.* share their modules with encoder.*, so they may be absent from compact files.
    missing = [k for k in missing if not ALIAS_RE.match(k)]
    if missing or unexpected:
        raise ModelWeightsError(f"State dict mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.eval().to(device)
