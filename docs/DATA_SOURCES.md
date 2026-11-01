# Data sources and provenance

## Quito municipal services (candidates, NOT verified)
- `https://geocatastro.quito.gob.ec:8443/geoserver/usigc/wms`, `…/dmot/wms`, `https://geomdmq.quito.gob.ec:8443/geoserver/sgpgeom/wms`.
- Discovery attempt (2026-11-01, from the build sandbox, one `GetCapabilities` request per endpoint, short timeout, TLS verification on): all three closed the connection (`Connection reset by peer`). No retries were made. **Layer names, CRS, GSD (the ~7 cm figure is unverified), WFS availability and redistribution terms remain unknown.** No Quito data is bundled.
- WMS images are rendered pictures, never cadastral vector data, and a small WMS crop does not retain the drone GSD.
- To use them: verify permissions with the owner, read `GetCapabilities` yourself, and pass downloaded GeoTIFF/GeoPackage files to the CLI.

## Bundled demo (`examples/demo/`)
- `vegas_sn2_img10_rgb8.{tif,png}`: 512×512 crop of SpaceNet-2 AOI_2_Vegas image `img10` (pan-sharpened RGB, EPSG:4326, stored 8-bit after 0.5–99.5 percentile stretch per band). **Not Quito.** CC BY-SA 4.0, © SpaceNet / Maxar (source: `s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/`). Acquisition date not recorded here.
- `vegas_sn2_img10_reference_buildings.geojson`: the 20 SpaceNet-2 building polygons intersecting the crop, clipped. They are building-footprint annotations of that dataset, used as *illustrative reference*; they are neither Quito cadastre nor an independent re-review.
- Demo choice reason: a Quito sample could not be obtained/redistributed (see above) and an openly licensed aerial dataset was available.

## Synthetic fixtures
`tests/conftest.py` generates deterministic synthetic scenes in EPSG:32717 (UTM 17S); coordinates are arbitrary test values, not real places.
