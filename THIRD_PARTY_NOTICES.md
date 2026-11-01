# Third-party notices

- **Model weights** `xdxd_spacenet4_solaris_weights.pth` — XD_XD's SpaceNet-4 solution as published by CosmiQ Works / SpaceNet (AWS Open Data Registry) and referenced by the `solaris` model zoo (<https://github.com/CosmiQ/solaris>, Apache-2.0 per that repository). Not redistributed here; downloaded from the original URL and checksum-verified. Training data: SpaceNet-4, CC BY-SA 4.0. Review these terms before redistributing weights or derivatives.
- **Demo imagery and labels** `examples/demo/*` — derived from SpaceNet-2 AOI_2_Vegas (CC BY-SA 4.0, © SpaceNet / Maxar). Derivatives (crop, 8-bit stretch, clipping) are shared under the same license.
- **Libraries** — rasterio, shapely, pyproj, numpy, geopandas, onnxruntime, PyTorch, Streamlit, Pillow, PyYAML under their respective open-source licenses.

The MIT license in `LICENSE` covers the original code only. No endorsement by any third party is implied.
