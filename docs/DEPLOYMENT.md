# Deployment

## GitHub Pages
Workflow `.github/workflows/pages.yml` publishes `site/` with the official Pages actions. Enable once: *Settings → Pages → Source: GitHub Actions*. The URL is `https://faustoaguanor.github.io/quito-building-intelligence-lab/` only after the first successful run — check it before announcing.

## Streamlit Community Cloud
1. <https://share.streamlit.io> → New app → repo `faustoaguanor/quito-building-intelligence-lab`, branch `main`, file `app/streamlit_app.py`, Python 3.11/3.12. Dependencies come from `requirements.txt` (CPU only, no torch).
2. The 117 MB ONNX must be reachable at runtime but must not be committed. Host it (Hugging Face Hub file or a GitHub Release asset), then in *Advanced settings → Secrets/Env* set `QBL_ONNX_URL` (https) and `QBL_ONNX_SHA256` (hash of your exported file, `sha256sum`). Without them the app shows "model not available" — it never fakes results.
3. Watch memory (≈ 0.8 GB peak measured locally). If the app is OOM-killed, use the static results on the portfolio site as fallback.

## Hugging Face Hub (optional)
`python scripts/package_model.py` writes `models/hf/` (model card, checksums). Review license obligations of the SpaceNet weights, then `huggingface-cli login` and upload yourself. Never put tokens in the repo.
