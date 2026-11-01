# Optional fine-tuning (not executed in this repository)

No fine-tuned Quito model exists. This workflow is for future manually labelled data.

```bash
qbl-train prepare --images chips/img --masks chips/mask --out manifest.json --block-size 500   # spatial block split
qbl-train train --manifest manifest.json --epochs 10 --device auto --out runs/ft1               # GPU recommended
```
- Pairing by identical file stem (hard error on unpaired files); split by whole spatial blocks with a fixed seed (train/val/test).
- `train` uses only train/val; the test split must be evaluated once, separately, and never reused for tuning. Do not validate on the same small sample used for training.
- Saves `finetuned_best.safetensors` and `training_metadata.json` (seed, args, per-epoch loss/val IoU).
- Augmentation: random crop and flips (extend in `training._load`). Tests cover pairing, splitting and metrics only; the training loop itself has not been run here.
