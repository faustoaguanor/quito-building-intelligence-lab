"""Streamlit Community Cloud demo: real CPU inference on ONE small RGB image (<= 512x512, <= 5 MB).

Uses the same core as the CLI (`quito_building_lab.pipeline.run_image`) with the ONNX backend.
"""

from __future__ import annotations

import io
import json
import math
import resource
import tempfile
import threading
import time
from pathlib import Path

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw
from PIL import Image, ImageDraw
from shapely.affinity import affine_transform
from streamlit_folium import st_folium

from quito_building_lab import ogc
from quito_building_lab.config import PipelineConfig
from quito_building_lab.crs_utils import reproject_geom
from quito_building_lab.export import to_geojson_dict
from quito_building_lab.inference import cached_predictor
from quito_building_lab.matching import LABELS_ES
from quito_building_lab.models import ModelWeightsError, ensure_onnx, get_spec
from quito_building_lab.pipeline import run_image, run_raster
from quito_building_lab.reference import reference_from_geojson

MAX_BYTES = 5 * 1024 * 1024
MAX_SIDE = 512
MODEL_ID = "xdxd_spacenet4_unet_vgg16"
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "demo" / "vegas_sn2_img10_rgb8.png"
_JOB_LOCK = threading.Lock()  # one inference job at a time per server process

st.set_page_config(page_title="Quito Building Intelligence Lab", page_icon="🏘️", layout="wide")


@st.cache_resource(show_spinner="Cargando modelo ONNX…")
def load_predictor():
    ensure_onnx(MODEL_ID)
    return cached_predictor(MODEL_ID, "onnx", "cpu")


def read_image(data: bytes) -> np.ndarray:
    if len(data) > MAX_BYTES:
        raise ValueError(f"El archivo supera {MAX_BYTES // 2**20} MB.")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as exc:  # noqa: BLE001 - user-facing message
        raise ValueError("No se pudo leer la imagen (use PNG, JPG o TIFF RGB de 8 bits).") from exc
    if im.width > MAX_SIDE or im.height > MAX_SIDE:
        raise ValueError(f"La imagen mide {im.width}×{im.height} px; el máximo es {MAX_SIDE}×{MAX_SIDE}.")
    if im.width < 64 or im.height < 64:
        raise ValueError("La imagen es demasiado pequeña (mínimo 64×64 px).")
    return np.asarray(im.convert("RGB"))


def render(img: np.ndarray, prob: np.ndarray, polys, threshold: float):
    mask = ((prob >= threshold) * 255).astype(np.uint8)
    over = Image.fromarray(img).convert("RGBA")
    tint = np.zeros((*mask.shape, 4), np.uint8)
    tint[mask > 0] = (255, 80, 40, 90)
    over = Image.alpha_composite(over, Image.fromarray(tint))
    d = ImageDraw.Draw(over)
    for g in polys:
        d.line(list(g.exterior.coords), fill=(255, 235, 0, 255), width=2)
    return Image.fromarray(mask), over.convert("RGB")


def png_bytes(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


spec = get_spec(MODEL_ID)
st.title("🏘️ Quito Building Intelligence Lab")
st.caption("Extracción asistida por IA de huellas de edificios y evaluación de calidad catastral — demostración de inferencia en CPU")

with st.expander("¿Qué hace esta demo?", expanded=True):
    st.markdown(
        "Segmenta **cubiertas candidatas** en **una** imagen aérea RGB pequeña con un modelo preentrenado "
        "real y las vectoriza. Es la misma lógica del motor local (CLI, GeoTIFF grandes, comparación catastral).\n\n"
        "- **No** procesa ortofotos grandes ni acepta modelos subidos por el usuario.\n"
        "- Las imágenes PNG/JPG no tienen georreferenciación: los polígonos están en **coordenadas de imagen (píxeles)**, "
        "no geográficas.\n"
        "- La geometría visible de un techo **no equivale** al límite catastral legal."
    )

def tab_upload():
    left, right = st.columns([1, 1])
    with left:
        st.subheader("1. Imagen")
        up = st.file_uploader("Suba una imagen (PNG/JPG/TIFF, ≤ 5 MB, ≤ 512×512 px)", type=["png", "jpg", "jpeg", "tif", "tiff"])
        use_example = st.checkbox("Usar la imagen de ejemplo", value=up is None, disabled=up is not None)
        thr = st.slider("Umbral de probabilidad", 0.30, 0.90, 0.50, 0.05)
        min_px = st.slider("Área mínima del polígono (px²)", 10, 400, 60, 10)
    with right:
        st.subheader("Modelo")
        st.markdown(
            f"**{spec.name}**  \n{spec.architecture}  \nEntrenado con: {spec.training_data} (~{spec.training_gsd_m} m/píxel).  \n"
            f"Backend: ONNX Runtime (CPU). Pesos de terceros; ver licencias y atribución abajo."
        )

    img = None
    try:
        if up is not None:
            img = read_image(up.getvalue())
        elif use_example and EXAMPLE.exists():
            img = read_image(EXAMPLE.read_bytes())
    except ValueError as exc:
        st.error(str(exc))

    if img is not None:
        st.image(img, caption=f"Entrada: {img.shape[1]}×{img.shape[0]} px", width=420)
        if st.button("▶ Ejecutar inferencia", type="primary"):
            if not _JOB_LOCK.acquire(blocking=False):
                st.warning("Hay otra inferencia en curso. Intente de nuevo en unos segundos.")
            else:
                try:
                    try:
                        predictor = load_predictor()
                    except (ModelWeightsError, OSError) as exc:
                        st.error(f"El modelo no está disponible en este despliegue: {exc}")
                        return
                    cfg = PipelineConfig()
                    cfg.tiling.tile_size, cfg.tiling.overlap = 512, 0
                    cfg.postprocess.threshold, cfg.postprocess.min_area_px = thr, float(min_px)
                    with st.spinner("Ejecutando el modelo en CPU…"):
                        t0 = time.perf_counter()
                        out = run_image(img, cfg, predictor)
                        dt = time.perf_counter() - t0
                    mask_im, overlay = render(img, out.segmentation.probability, out.candidates.geoms, thr)
                    c1, c2 = st.columns(2)
                    c1.image(mask_im, caption="Máscara predicha")
                    c2.image(overlay, caption="Superposición y polígonos candidatos")
                    areas = [g.area for g in out.candidates.geoms]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Candidatos", len(areas))
                    m2.metric("Área total (px²)", f"{sum(areas):,.0f}")
                    m3.metric("Tiempo CPU (s)", f"{dt:.1f}")
                    m4.metric("Memoria pico (MB)", f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}")
                    st.info("Coordenadas en píxeles de imagen (x = columna, y = fila). Las áreas están en px², no en m².")
                    gj = json.dumps(to_geojson_dict(out.candidates)).encode()
                    d1, d2, d3, d4 = st.columns(4)
                    d1.download_button("Máscara (PNG)", png_bytes(mask_im), "mascara.png", "image/png")
                    d2.download_button("Superposición (PNG)", png_bytes(overlay), "superposicion.png", "image/png")
                    d3.download_button("Polígonos (GeoJSON, píxeles)", gj, "candidatos_pixeles.geojson", "application/geo+json")
                    d4.download_button("Informe (JSON)", json.dumps(out.report, indent=2, default=str).encode(), "informe.json")
                finally:
                    _JOB_LOCK.release()



@st.cache_data(ttl=600, show_spinner=False)
def cached_wms_caps(url: str):
    return ogc.wms_capabilities(url)


@st.cache_data(ttl=600, show_spinner=False)
def cached_wfs_caps(url: str):
    return ogc.wfs_capabilities(url)


MAX_SIDE_M = 260.0
CLASS_COLORS = {
    "strong_correspondence": (46, 204, 113), "partial_correspondence": (241, 196, 15),
    "ambiguous_match": (230, 126, 34), "candidate_without_reference": (231, 76, 60),
    "not_evaluated": (160, 160, 160),
}


def _bbox_from_drawing(draw) -> tuple[float, float, float, float] | None:
    try:
        ring = draw["geometry"]["coordinates"][0]
        xs, ys = [c[0] for c in ring], [c[1] for c in ring]
        return min(xs), min(ys), max(xs), max(ys)
    except (KeyError, TypeError, IndexError):
        return None


def _bbox_from_center(lon: float, lat: float, side_m: float):
    dlat = side_m / 2 / 110_574.0
    dlon = side_m / 2 / (111_320.0 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _side_m(b) -> float:
    lat = (b[1] + b[3]) / 2
    return max((b[2] - b[0]) * 111_320.0 * math.cos(math.radians(lat)), (b[3] - b[1]) * 110_574.0)


def _to_pixels(geoms, src_crs, img: ogc.GeoImage):
    inv = ~img.transform
    out = []
    for g in geoms:
        g2 = reproject_geom(g, src_crs, img.crs)
        out.append(affine_transform(g2, [inv.a, inv.b, inv.d, inv.e, inv.xoff, inv.yoff]))
    return out


def tab_map():
    st.markdown(
        "Elija un servicio **WMS** (imagen), dibuje un cuadrado en el mapa, descargue **solo esa área** y ejecute la "
        "inferencia sobre ella. Opcionalmente compare con polígonos de un **WFS** (UC / bloques) del mismo cuadrado.\n\n"
        "⚠️ Los servicios de Quito listados son **candidatos sin verificar**: confirme con su titular los permisos y "
        "condiciones de uso. Una imagen WMS es una vista renderizada, **no** los píxeles originales de la ortofoto ni "
        "datos vectoriales catastrales; su resolución real puede ser menor que la del dron."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Servicio de imagen (WMS)")
        preset = st.selectbox("Servicio", list(ogc.CANDIDATE_WMS) + ["Otra URL…"], key="wms_preset")
        wms_url = st.text_input("URL del WMS (https)", ogc.CANDIDATE_WMS.get(preset, ""), key=f"wms_url_{preset}")
        if st.button("Cargar capas del WMS"):
            try:
                st.session_state["wms_caps"] = cached_wms_caps(wms_url)
                st.session_state["wms_caps_url"] = wms_url
            except ogc.OGCError as exc:
                st.session_state.pop("wms_caps", None)
                st.error(f"No se pudo leer el servicio: {exc}")
        caps = st.session_state.get("wms_caps") if st.session_state.get("wms_caps_url") == wms_url else None
        layer = None
        if caps:
            q = st.text_input("Filtrar capas por texto", "", key="wms_filter").lower()
            opts = [la for la in caps.layers if q in la.name.lower() or q in la.title.lower()][:300]
            layer = st.selectbox("Capa", opts, format_func=lambda la: f"{la.title} — {la.name}") if opts else None
            st.caption(f"{len(caps.layers)} capas; WMS {caps.version}.")
    with c2:
        st.subheader("2. Vectores para comparar (opcional)")
        use_wfs = st.checkbox("Comparar con un WFS (UC / bloques / construcciones)")
        wfs_url = wfs_type = None
        kind = "uc"
        if use_wfs:
            wp = st.selectbox("Servicio WFS", list(ogc.CANDIDATE_WFS) + ["Otra URL…"], key="wfs_preset")
            wfs_url = st.text_input("URL del WFS (https)", ogc.CANDIDATE_WFS.get(wp, ""), key=f"wfs_url_{wp}")
            if st.button("Cargar tipos del WFS"):
                try:
                    st.session_state["wfs_caps"] = cached_wfs_caps(wfs_url)
                    st.session_state["wfs_caps_url"] = wfs_url
                except ogc.OGCError as exc:
                    st.session_state.pop("wfs_caps", None)
                    st.error(f"No se pudo leer el servicio: {exc}")
            wcaps = st.session_state.get("wfs_caps") if st.session_state.get("wfs_caps_url") == wfs_url else None
            if wcaps:
                wq = st.text_input("Filtrar tipos", "", key="wfs_filter").lower()
                wopts = [f for f in wcaps.feature_types if wq in f.name.lower() or wq in f.title.lower()][:300]
                ft = st.selectbox("Tipo de entidad", wopts, format_func=lambda f: f"{f.title} — {f.name}") if wopts else None
                wfs_type = ft.name if ft else None
            kind = {"UC": "uc", "Bloque": "block", "Construcción": "construction", "Otro": "other"}[
                st.radio("Qué representa la capa", ["UC", "Bloque", "Construcción", "Otro"], horizontal=True)]

    st.subheader("3. Área (máximo ≈ 260 m de lado)")
    m = folium.Map(location=[-0.1807, -78.4678], zoom_start=13, control_scale=True)
    if layer is not None and caps is not None:
        try:
            folium.raster_layers.WmsTileLayer(url=wms_url, layers=layer.name, fmt="image/png", transparent=True,
                                              version=caps.version, name="WMS seleccionado", overlay=True).add_to(m)
        except Exception:  # noqa: BLE001 - overlay is a convenience only
            pass
    Draw(export=False, draw_options={"rectangle": True, "polyline": False, "polygon": False, "circle": False,
                                     "marker": False, "circlemarker": False}, edit_options={"edit": False}).add_to(m)
    folium.LayerControl().add_to(m)
    mp = st_folium(m, height=420, use_container_width=True, returned_objects=["last_active_drawing"], key="qmap")
    bbox = _bbox_from_drawing((mp or {}).get("last_active_drawing"))
    with st.expander("…o escriba coordenadas (centro + lado)"):
        lon = st.number_input("Longitud", value=-78.4678, format="%.6f")
        lat = st.number_input("Latitud", value=-0.1807, format="%.6f")
        side = st.number_input("Lado del cuadrado (m)", 50.0, MAX_SIDE_M, 120.0, 10.0)
        if st.checkbox("Usar estas coordenadas en lugar del dibujo"):
            bbox = _bbox_from_center(lon, lat, side)
    if bbox:
        st.caption(f"Área: lado ≈ {_side_m(bbox):.0f} m · W {bbox[0]:.5f}, S {bbox[1]:.5f}, E {bbox[2]:.5f}, N {bbox[3]:.5f}")
    if bbox and _side_m(bbox) > MAX_SIDE_M:
        st.error(f"El área es demasiado grande ({_side_m(bbox):.0f} m). Máximo {MAX_SIDE_M:.0f} m de lado.")
        bbox = None

    ready = bool(bbox and layer is not None)
    if st.button("▶ Descargar área y ejecutar inferencia", type="primary", disabled=not ready):
        if not _JOB_LOCK.acquire(blocking=False):
            st.warning("Hay otra inferencia en curso. Intente de nuevo en unos segundos.")
            return
        try:
            try:
                predictor = load_predictor()
            except (ModelWeightsError, OSError) as exc:
                st.error(f"El modelo no está disponible en este despliegue: {exc}")
                return
            try:
                with st.spinner("Pidiendo la imagen al WMS (una sola petición)…"):
                    img = ogc.wms_get_map(wms_url, layer.name, bbox, layer.crs, caps.version, 0.3)
                ref = ref_rep = wfs_info = None
                if use_wfs and wfs_url and wfs_type:
                    with st.spinner("Pidiendo los polígonos al WFS (una sola petición)…"):
                        data, wfs_info = ogc.wfs_get_features(wfs_url, wfs_type, bbox, img.crs)
                    ref, ref_rep = reference_from_geojson(data, wfs_info["srs"], kind, source=wfs_url)
                    if wfs_info["truncated"]:
                        st.warning("El WFS devolvió el máximo de entidades permitido; la comparación puede estar incompleta.")
                    if not len(ref):
                        st.warning("El WFS no devolvió polígonos utilizables en este cuadrado; se omite la comparación.")
                        ref = None
            except (ogc.OGCError, ValueError) as exc:
                st.error(f"No se pudo obtener los datos: {exc}")
                return
            if not img.valid.any():
                st.error("La imagen devuelta está vacía (sin datos en esa zona o capa).")
                return
            cfg = PipelineConfig()
            cfg.tiling.tile_size, cfg.tiling.overlap = 512, 0
            with tempfile.TemporaryDirectory() as tmp:
                tif = ogc.write_geotiff(img, str(Path(tmp) / "wms_area.tif"))
                with st.spinner("Ejecutando el modelo en CPU…"):
                    out = run_raster(tif, None, cfg, reference_set=ref,
                                     reference_report=ref_rep.to_dict() if ref_rep else None, predictor=predictor)
                tif_bytes = Path(tif).read_bytes()
            cands = out.matching.candidates if out.matching else out.candidates
            base = Image.fromarray(img.array).convert("RGB")
            draw = ImageDraw.Draw(base)
            if out.matching:
                for g in _to_pixels(out.matching.references.geoms, out.matching.crs, img):
                    draw.line(list(g.exterior.coords), fill=(80, 200, 255), width=2)
            for g, a in zip(_to_pixels(cands.geoms, cands.crs, img), cands.attrs, strict=True):
                draw.line(list(g.exterior.coords), fill=CLASS_COLORS.get(a.get("match_class"), (255, 235, 0)), width=2)
            a1, a2 = st.columns(2)
            a1.image(img.array, caption=f"Imagen WMS descargada ({img.array.shape[1]}×{img.array.shape[0]} px, "
                                        f"GSD de la petición ≈ {img.meta['approx_gsd_m']} m)")
            a2.image(base, caption="Candidatos (colores por clase) y referencia WFS (cian)")
            k1, k2, k3 = st.columns(3)
            k1.metric("Candidatos", len(out.candidates))
            k2.metric("Tiempo CPU (s)", f"{out.report['segmentation']['total_s']:.1f}")
            k3.metric("Memoria pico (MB)", f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}")
            if out.matching:
                cc = out.matching.summary["candidates_by_class"]
                rc = out.matching.summary["references_by_class"]
                st.markdown("**Comparación con los polígonos del WFS** (umbrales de demostración, no tolerancias oficiales)")
                st.table({"Clase": [LABELS_ES[k] for k in cc], "Candidatos": list(cc.values()),
                          "Referencias": [rc[k] for k in cc]})
                st.info("Las discrepancias son avisos para **revisión humana**; no prueban errores catastrales, "
                        "construcciones no registradas ni ilegales. Un techo visible no es el límite catastral.")
            st.caption("Imagen y vectores obtenidos de servicios externos; compruebe sus condiciones de uso antes de reutilizarlos.")
            report = dict(out.report)
            report["fetch"] = {"wms": img.meta, "wfs": wfs_info}
            d1, d2, d3, d4 = st.columns(4)
            d1.download_button("Imagen (GeoTIFF)", tif_bytes, "wms_area.tif", "image/tiff")
            d2.download_button("Candidatos (GeoJSON)", json.dumps(to_geojson_dict(cands)).encode(),
                               "candidatos.geojson", "application/geo+json")
            if out.matching:
                d3.download_button("Referencia (GeoJSON)", json.dumps(to_geojson_dict(out.matching.references)).encode(),
                                   "referencia.geojson", "application/geo+json")
            d4.download_button("Informe (JSON)", json.dumps(report, indent=2, default=str).encode(), "informe.json")
        finally:
            _JOB_LOCK.release()

tabs = st.tabs(["Imagen propia", "Mapa WMS de Quito (área pequeña)"])
with tabs[0]:
    tab_upload()
with tabs[1]:
    tab_map()

st.divider()
with st.expander("Limitaciones y atribución"):
    st.markdown(
        "- Modelo entrenado con SpaceNet-4 (Atlanta, WorldView-2). **No se ha validado con etiquetas revisadas de Quito.** "
        "Imágenes con resolución muy distinta a ~0,5 m/píxel pueden rendir peor.\n"
        "- Los resultados son **candidatos** para revisión humana; no son límites catastrales ni prueba de errores o construcciones no registradas.\n"
        f"- {spec.license_note}\n- Imagen de ejemplo: recorte de SpaceNet-2 (Las Vegas), CC BY-SA 4.0.\n"
        "- Proyecto independiente: no implica respaldo del Municipio de Quito, Streamlit u otros proveedores.\n"
        "- Código MIT © 2026 faustoaguanor."
    )
