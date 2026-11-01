"""Optional fine-tuning workflow for manually labelled roof data (never run automatically).

Steps: ``prepare`` (pair image/mask GeoTIFF chips + spatial split) -> ``train`` -> ``evaluate``.
No model is claimed to be fine-tuned unless this workflow was actually executed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import numpy as np


def pair_files(images_dir: str, masks_dir: str, pattern: str = "*.tif") -> list[tuple[Path, Path]]:
    """Pair image and mask chips by identical file stem; raise on any unpaired file."""
    imgs = {p.stem: p for p in Path(images_dir).glob(pattern)}
    msks = {p.stem: p for p in Path(masks_dir).glob(pattern)}
    if set(imgs) != set(msks):
        raise ValueError(f"Unpaired chips: images-only={sorted(set(imgs) - set(msks))[:5]} "
                         f"masks-only={sorted(set(msks) - set(imgs))[:5]}")
    return [(imgs[k], msks[k]) for k in sorted(imgs)]


def spatial_split(centroids: list[tuple[float, float]], block_size: float, fractions=(0.7, 0.15, 0.15),
                  seed: int = 42) -> list[str]:
    """Assign train/val/test by whole spatial blocks (block_size in CRS units) to limit leakage
    between neighbouring, overlapping or autocorrelated chips."""
    if abs(sum(fractions) - 1) > 1e-6:
        raise ValueError("fractions must sum to 1")
    blocks = [(int(x // block_size), int(y // block_size)) for x, y in centroids]
    uniq = sorted(set(blocks))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr, n_va = max(1, round(fractions[0] * n)), round(fractions[1] * n)
    if n >= 3:
        n_va = max(1, n_va)
    assign = {}
    for i, b in enumerate(uniq):
        assign[b] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return [assign[b] for b in blocks]


def segmentation_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    p, t = pred.astype(bool), truth.astype(bool)
    tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"iou": iou, "dice": dice, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}


def prepare(args) -> None:
    import rasterio

    pairs = pair_files(args.images, args.masks)
    cents = []
    for ip, _ in pairs:
        with rasterio.open(ip) as s:
            b = s.bounds
            cents.append(((b.left + b.right) / 2, (b.bottom + b.top) / 2))
    split = spatial_split(cents, args.block_size, seed=args.seed)
    manifest = [{"image": str(i), "mask": str(m), "split": s} for (i, m), s in zip(pairs, split, strict=True)]
    Path(args.out).write_text(json.dumps({"seed": args.seed, "block_size": args.block_size, "items": manifest}, indent=1))
    print({k: split.count(k) for k in ("train", "val", "test")})


def _load(item, size: int, rng):
    import rasterio

    with rasterio.open(item["image"]) as s:
        img = s.read([1, 2, 3]).astype(np.float32)
    with rasterio.open(item["mask"]) as s:
        msk = (s.read(1) > 0).astype(np.float32)
    _, h, w = img.shape
    r, c = rng.randint(0, max(0, h - size)), rng.randint(0, max(0, w - size))
    img, msk = img[:, r : r + size, c : c + size], msk[r : r + size, c : c + size]
    if rng.random() < 0.5:
        img, msk = img[:, :, ::-1], msk[:, ::-1]
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[::-1]
    mean, std = img.mean((1, 2), keepdims=True), img.std((1, 2), keepdims=True) + 1e-6
    return np.ascontiguousarray((img - mean) / std), np.ascontiguousarray(msk)


def train(args) -> None:
    import torch

    from .models import load_torch_model

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    man = json.loads(Path(args.manifest).read_text())["items"]
    tr = [m for m in man if m["split"] == "train"]
    va = [m for m in man if m["split"] == "val"]
    if not tr or not va:
        raise ValueError("Need non-empty train and val splits. Test items are never used here.")
    dev = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model = load_torch_model("xdxd_spacenet4_unet_vgg16", args.weights_dir, dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0

    def batch(items):
        xs, ys = zip(*[_load(i, args.size, rng) for i in items], strict=True)
        return torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ys))[:, None].to(dev)

    for ep in range(1, args.epochs + 1):
        rng.shuffle(tr)
        model.train()
        losses = []
        for k in range(0, len(tr), args.batch_size):
            x, y = batch(tr[k : k + args.batch_size])
            loss = bce(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        ious = []
        with torch.inference_mode():
            for k in range(0, len(va), args.batch_size):
                x, y = batch(va[k : k + args.batch_size])
                ious.append(segmentation_metrics(torch.sigmoid(model(x)).cpu().numpy() > 0.5, y.cpu().numpy())["iou"])
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), "val_iou": float(np.mean(ious))}
        history.append(row)
        print(row)
        if row["val_iou"] > best:
            best = row["val_iou"]
            from safetensors.torch import save_file

            from .models import ALIAS_RE

            save_file({k: v.contiguous() for k, v in model.state_dict().items() if not ALIAS_RE.match(k)},
                      str(out / "finetuned_best.safetensors"))
    (out / "training_metadata.json").write_text(json.dumps({
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base_model": "xdxd_spacenet4_unet_vgg16", "args": vars(args), "history": history,
        "note": "Evaluate on the held-out test split with `evaluate` before any claim."}, indent=1, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qbl-train", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--images", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--out", default="manifest.json")
    p.add_argument("--block-size", type=float, default=500.0, help="spatial block size in CRS units")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=prepare)
    p = sub.add_parser("train")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="runs/finetune")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--weights-dir")
    p.set_defaults(func=train)
    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
