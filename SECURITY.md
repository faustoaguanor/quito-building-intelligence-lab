# Security

- Report vulnerabilities privately via GitHub *Security → Report a vulnerability* on this repository.
- Only the pinned, checksum-verified checkpoint is loaded (`torch.load(weights_only=True)`); arbitrary user checkpoints are not accepted by the app. Never load untrusted pickle files.
- Never commit tokens or restricted data; the Streamlit demo reads `QBL_ONNX_URL`/`QBL_ONNX_SHA256` from environment/secrets only.
- TLS verification is never disabled.
