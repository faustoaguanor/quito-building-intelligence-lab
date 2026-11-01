# Deployment

## GitHub Pages
Workflow `.github/workflows/pages.yml` publishes `site/` with the official Pages actions. Enable once: *Settings → Pages → Source: GitHub Actions*. The URL is `https://faustoaguanor.github.io/quito-building-intelligence-lab/` only after the first successful run — check it before announcing.

## Streamlit Community Cloud
1. **Build the model once (GitHub does it for you):** Actions → "Build ONNX model" → *Run workflow* (branch `main`). It downloads the original weights (SHA-256 verified), exports and validates the ONNX, and publishes `xdxd_spacenet4_unet_vgg16.onnx` plus its `.sha256` in the release `model-onnx-v1`.
2. **Create the app:** <https://share.streamlit.io> → New app → repo `faustoaguanor/quito-building-intelligence-lab`, branch `main`, file `app/streamlit_app.py`, Python 3.12. Dependencies come from `requirements.txt` (CPU only, no torch).
3. No secrets are needed: the app downloads the ONNX and its checksum from that release on first start. `QBL_ONNX_URL` / `QBL_ONNX_SHA256` are optional overrides (e.g. a Hugging Face copy).
4. Watch memory (about 0.8 GB peak measured locally). The live demo ran once with the bundled example (829 MB peak reported by the app, see `docs/RESULTS.md`); other uses were not tested. If the app is restarted for lack of memory, use the static results on the portfolio site.

If step 1 was not done the app says the model is not available; it never fakes results.

## Hugging Face Hub (optional)
`python scripts/package_model.py` writes `models/hf/` (model card, checksums). Review license obligations of the SpaceNet weights, then `huggingface-cli login` and upload yourself. Never put tokens in the repo.
