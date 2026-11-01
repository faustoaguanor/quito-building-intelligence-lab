"""Streamlit Community Cloud demo: real CPU inference on ONE small RGB image (<= 512x512, <= 5 MB).

Uses the same core as the CLI (`quito_building_lab.pipeline.run_image`) with the ONNX backend.
"""

# ruff: noqa: E402
from __future__ import annotations

import copy
import io
import json
import math
import resource
import sys
import tempfile
import threading
import time
from pathlib import Path

# Always run the repository's own source: a stale copy of the package installed in the hosting
# environment (Streamlit only reinstalls dependencies when requirements.txt changes) must never win.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir():
    sys.path[:] = [str(_SRC)] + [p for p in sys.path if p != str(_SRC)]  # repository source first
    _loaded = sys.modules.get("quito_building_lab")
    if _loaded is not None and not str(getattr(_loaded, "__file__", "")).startswith(str(_SRC)):
        for _m in [k for k in sys.modules if k == "quito_building_lab" or k.startswith("quito_building_lab.")]:
            del sys.modules[_m]  # a stale copy was already imported in this long-lived process

import folium
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from shapely.affinity import affine_transform
from streamlit_folium import st_folium

from quito_building_lab import ogc
from quito_building_lab.config import PipelineConfig
from quito_building_lab.crs_utils import reproject_geom
from quito_building_lab.export import to_geojson_dict
from quito_building_lab.inference import cached_predictor
from quito_building_lab.models import ModelWeightsError, ensure_onnx, get_spec
from quito_building_lab.pipeline import run_image, run_raster

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


MIN_SIDE_M, MAX_SIDE_M, DEFAULT_SIDE_M = 80.0, 260.0, 150.0
DEFAULT_VIEW = (-1.4, -78.4, 6)  # Ecuador; the map zooms to the layer extent once a layer is chosen


def _bbox_from_center(lon: float, lat: float, side_m: float):
    dlat = side_m / 2 / 110_574.0
    dlon = side_m / 2 / (111_320.0 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def _to_pixels(geoms, src_crs, img: ogc.GeoImage):
    inv = ~img.transform
    out = []
    for g in geoms:
        g2 = reproject_geom(g, src_crs, img.crs)
        out.append(affine_transform(g2, [inv.a, inv.b, inv.d, inv.e, inv.xoff, inv.yoff]))
    return out


def _default_center(extent):
    """Centre of a small coverage (<= 3 km); ``None`` for large ones, where the user must zoom in and click."""
    if not extent:
        return None
    w, s_, e, n = extent
    lat = (s_ + n) / 2
    size_m = max((e - w) * 111_320.0 * math.cos(math.radians(lat)), (n - s_) * 110_574.0)
    return (lat, (w + e) / 2) if size_m <= 3000 else None


def _set_center(lat: float, lon: float) -> None:
    st.session_state["center"] = (float(lat), float(lon))


def tab_map():
    st.markdown(
        "Use **cualquier servicio WMS** con ortofotos: cargue sus capas, haga **clic en el mapa** para elegir el centro, "
        "ajuste el tamaño del cuadrado, y la app descarga **solo esa área** y ejecuta la inferencia.\n\n"
        "⚠️ Compruebe con el titular del servicio sus condiciones de uso. Una imagen WMS es una vista renderizada, **no** "
        "los píxeles originales de la ortofoto; su resolución real puede ser menor. Las capas de vuelos de dron suelen "
        "cubrir solo **zonas concretas**: la línea azul del mapa muestra la cobertura declarada por la capa."
    )
    st.subheader("1. Servicio WMS y capa")
    preset = st.selectbox("Servicio", list(ogc.EXAMPLE_WMS) + ["Otra URL…"], key="wms_preset")
    wms_url = st.text_input("URL del WMS (https)", ogc.EXAMPLE_WMS.get(preset, ""), key=f"wms_url_{preset}")
    wms_url = wms_url.strip().rstrip("?&")
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
    else:
        with st.expander("¿No carga el listado? Escriba la capa a mano"):
            st.caption("Útil si el servidor no responde a la lista de capas pero sí a las imágenes. Si el servidor "
                       "bloquea a Streamlit Cloud, ejecute la app en su computadora: `streamlit run app/streamlit_app.py`.")
            mname = st.text_input("Nombre exacto de la capa", key="manual_layer")
            mcrs = st.selectbox("CRS de la petición", list(ogc.PREFERRED_CRS), key="manual_crs")
            mver = st.selectbox("Versión WMS", ["1.3.0", "1.1.1"], key="manual_ver")
            if mname.strip():
                layer = ogc.WmsLayer(mname.strip(), mname.strip(), [mcrs])
                caps = ogc.WmsCapabilities(wms_url, mver, "(manual)", ["image/png"], [layer])

    st.subheader("2. Elija dónde descargar")
    st.markdown("**① Elija la capa** (el mapa se centra en su cobertura)  ·  **② Haga clic en el mapa** donde quiere el "
                "centro del cuadrado rojo  ·  **③ Ajuste el tamaño** con el deslizador. Puede volver a hacer clic para moverlo.")
    side = st.slider("Tamaño del cuadrado (metros de lado)", MIN_SIDE_M, MAX_SIDE_M, DEFAULT_SIDE_M, 10.0,
                     help=f"Entre {MIN_SIDE_M:.0f} y {MAX_SIDE_M:.0f} m. Más pequeño pierde contexto; más grande "
                          "obliga a bajar la resolución de la imagen que recibe el modelo.")
    px = min(512, math.ceil(side / 0.3))
    st.caption(f"Cuadrado de {side:.0f} × {side:.0f} m ({side * side / 10_000:.1f} ha) → imagen ≈ {px}×{px} px, "
               f"≈ {side / px:.2f} m/píxel (el modelo se entrenó con ~0,5 m).")
    extent = layer.bbox_lonlat if layer is not None else None

    # The base map is built ONCE per (service, layer) and kept in session state. The selected square is
    # added client-side as a feature group and only clicks are reported back, so panning/zooming never
    # reruns the app or rebuilds the map (that caused the flicker and the vanishing layers).
    map_key = (wms_url, layer.name if layer is not None else None, caps.version if caps else None)
    if st.session_state.get("map_key") != map_key:
        base = folium.Map(location=DEFAULT_VIEW[:2], zoom_start=DEFAULT_VIEW[2], control_scale=True)
        if extent:
            w, s_, e, n = extent
            folium.Rectangle([[s_, w], [n, e]], color="#1f77ff", weight=2, fill=False, dash_array="6",
                             interactive=False).add_to(base)  # not interactive: clicks must reach the map
            base.fit_bounds([[s_, w], [n, e]])
        if layer is not None and caps is not None:
            try:
                folium.raster_layers.WmsTileLayer(url=wms_url, layers=layer.name, fmt="image/png", transparent=True,
                                                  version=caps.version, name="Capa WMS", overlay=True).add_to(base)
            except Exception:  # noqa: BLE001 - overlay is a convenience only
                pass
        st.session_state.update(map_key=map_key, base_map=base, map_ver=st.session_state.get("map_ver", 0) + 1,
                                click_seen=None)
        # small coverages (e.g. one drone flight): preselect its centre so a square is visible straight away
        st.session_state["center"] = _default_center(extent)
    center = st.session_state.get("center")
    bbox = None
    group = folium.FeatureGroup(name="Área seleccionada")
    if center:
        bbox = _bbox_from_center(center[1], center[0], side)
        folium.Rectangle([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], color="#e63946", weight=3, fill=True,
                         fill_opacity=0.2, interactive=False).add_to(group)
        folium.CircleMarker(center, radius=4, color="#e63946", fill=True, fill_opacity=1,
                            interactive=False).add_to(group)
    # st_folium mutates the map it receives, so pass a copy (same ids => identical HTML => no re-render)
    mp = st_folium(copy.deepcopy(st.session_state["base_map"]), feature_group_to_add=group,
                   layer_control=folium.LayerControl(), height=460, use_container_width=True,
                   returned_objects=["last_clicked"], key=f"qmap_{st.session_state['map_ver']}") or {}
    click = mp.get("last_clicked")
    if click and click != st.session_state.get("click_seen"):
        st.session_state["click_seen"] = click
        _set_center(click["lat"], click["lng"])
        st.rerun()
    with st.expander("…o escriba las coordenadas del centro"):
        c = center or (-0.1807, -78.4678)
        lat_in = st.number_input("Latitud", value=float(c[0]), format="%.6f", key="lat_in")
        lon_in = st.number_input("Longitud", value=float(c[1]), format="%.6f", key="lon_in")
        if st.button("Usar este centro"):
            _set_center(lat_in, lon_in)
            st.rerun()

    problem = None
    if not center:
        st.info("👆 Acerque el mapa a la zona que le interesa y haga clic para colocar el cuadrado rojo.")
        problem = "sin centro"
    elif layer is None:
        st.info("Elija una capa para continuar.")
        problem = "sin capa"
    elif extent and not ogc.bbox_intersects(bbox, extent):
        st.error("El cuadrado queda **fuera de la cobertura declarada de la capa** (línea azul). Haga clic dentro de ella.")
        problem = "fuera de cobertura"
    elif not extent:
        st.warning("La capa no declara su cobertura: no se puede comprobar si el cuadrado tiene datos.")
    else:
        st.success(f"Área válida: {side:.0f} m de lado dentro de la cobertura de la capa.")

    if st.button("▶ Descargar área y ejecutar inferencia", type="primary", disabled=problem is not None):
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
            except (ogc.OGCError, ValueError) as exc:
                st.error(f"No se pudo obtener la imagen: {exc}")
                return
            reason = ogc.blank_reason(img)
            if reason:
                st.error(f"La capa devolvió una imagen **{reason}** para este cuadrado. Suele pasar cuando la capa no "
                         "tiene datos en esa zona (p. ej. vuelos de dron de áreas concretas). Mueva el centro dentro "
                         "de la zona con imagen o pruebe otra capa.")
                st.image(img.array, caption="Imagen devuelta por el servicio", width=240)
                return
            cfg = PipelineConfig()
            cfg.tiling.tile_size, cfg.tiling.overlap = 512, 0
            with tempfile.TemporaryDirectory() as tmp:
                tif = ogc.write_geotiff(img, str(Path(tmp) / "wms_area.tif"))
                with st.spinner("Ejecutando el modelo en CPU…"):
                    out = run_raster(tif, None, cfg, predictor=predictor)
                tif_bytes = Path(tif).read_bytes()
            base = Image.fromarray(img.array).convert("RGB")
            draw = ImageDraw.Draw(base)
            for g in _to_pixels(out.candidates.geoms, out.candidates.crs, img):
                draw.line(list(g.exterior.coords), fill=(255, 235, 0), width=2)
            a1, a2 = st.columns(2)
            a1.image(img.array, caption=f"Imagen WMS descargada ({img.array.shape[1]}×{img.array.shape[0]} px, "
                                        f"≈ {img.meta['approx_gsd_m']} m/píxel)")
            a2.image(base, caption="Huellas candidatas")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Candidatos", len(out.candidates))
            k2.metric("Área total (m²)", f"{out.candidates.total_area:,.0f}")
            k3.metric("Tiempo CPU (s)", f"{out.report['segmentation']['total_s']:.1f}")
            k4.metric("Memoria pico (MB)", f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}")
            st.info("Son **candidatos** para revisión humana, no límites catastrales. Descargue las huellas y compárelas "
                    "con sus vectores en su SIG o con `qbl compare`.")
            st.caption("Imagen obtenida de un servicio externo; compruebe sus condiciones de uso antes de reutilizarla.")
            report = dict(out.report)
            report["fetch"] = {"wms": img.meta}
            d1, d2, d3 = st.columns(3)
            d1.download_button("Imagen (GeoTIFF)", tif_bytes, "wms_area.tif", "image/tiff")
            d2.download_button("Huellas candidatas (GeoJSON)", json.dumps(to_geojson_dict(out.candidates)).encode(),
                               "candidatos.geojson", "application/geo+json")
            d3.download_button("Informe (JSON)", json.dumps(report, indent=2, default=str).encode(), "informe.json")
        finally:
            _JOB_LOCK.release()


tabs = st.tabs(["Imagen propia", "Mapa WMS (cualquier servicio)"])
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
