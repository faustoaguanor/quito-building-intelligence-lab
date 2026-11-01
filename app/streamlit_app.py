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
    # A hosted process outlives deploys: Python keeps imported modules in memory, so new code on disk would not
    # be used until a restart. Drop the package's modules whenever its source files change (and on first run, in
    # case a stale installed copy was imported earlier in this process) so they are imported again.
    _sig = tuple(sorted((f.name, f.stat().st_mtime_ns) for f in (_SRC / "quito_building_lab").glob("*.py")))
    if getattr(sys, "_qbl_src_signature", None) != _sig:
        for _m in [k for k in sys.modules if k == "quito_building_lab" or k.startswith("quito_building_lab.")]:
            del sys.modules[_m]
        sys._qbl_src_signature = _sig

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

st.set_page_config(page_title="Building Footprint Lab", page_icon="🏘️", layout="wide")

# Streamlit's default text boxes draw their border in the same colour as their fill, so in light mode (light grey
# on white) their edges almost disappear. A mid-grey, semi-transparent border reads well on light and dark pages.
st.markdown(
    """<style>
    [data-testid="stTextInputRootElement"], [data-testid="stNumberInputContainer"],
    [data-testid="stSelectbox"] > div > div[role="group"], [data-testid="stFileUploaderDropzone"],
    [data-testid="stTextAreaRootElement"] { border: 1px solid rgba(128, 128, 128, 0.55) !important; }
    </style>""",
    unsafe_allow_html=True,
)


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
st.title("🏘️ Building Footprint Lab")
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
MAX_ZOOM = 22  # Leaflet hides a layer above its maxZoom (default 18): without this the WMS vanishes when zooming in
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


def _prob_overlay(rgb: np.ndarray, prob: np.ndarray) -> Image.Image:
    """Heat-map of the model probability blended over the image (nearest resize if the grids differ)."""
    h, w = rgb.shape[:2]
    p = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8))
    if p.size != (w, h):
        p = p.resize((w, h), Image.NEAREST)
    a = np.asarray(p).astype(np.float32) / 255.0
    heat = np.stack([a * 255, a * 120, (1 - a) * 80], axis=-1)
    return Image.fromarray((rgb * (1 - 0.55 * a[..., None]) + heat * 0.55 * a[..., None]).clip(0, 255).astype(np.uint8))


def _set_center(lat: float, lon: float) -> None:
    st.session_state["center"] = (float(lat), float(lon))


def _process_and_show(img: ogc.GeoImage, predictor, adv_thr: float, adv_min_area: float, adv_prob: bool,
                      source: str) -> None:
    """Shared by the automatic download and the browser-download path: checks, inference, results, downloads."""
    reason = ogc.blank_reason(img)
    if reason:
        st.error(f"La capa devolvió una imagen **{reason}** para este cuadrado. Suele pasar cuando la capa no "
                 "tiene datos en esa zona (p. ej. vuelos de dron de áreas concretas). Mueva el centro dentro "
                 "de la zona con imagen o pruebe otra capa.")
        st.image(img.array, caption="Imagen devuelta por el servicio", width=240)
        return
    cfg = PipelineConfig()
    cfg.tiling.tile_size, cfg.tiling.overlap = 512, 0
    cfg.postprocess.threshold, cfg.postprocess.min_area_m2 = adv_thr, adv_min_area
    with tempfile.TemporaryDirectory() as tmp:
        tif = ogc.write_geotiff(img, str(Path(tmp) / "wms_area.tif"))
        with st.spinner("Ejecutando el modelo en CPU…"):
            out = run_raster(tif, None, cfg, predictor=predictor)
        tif_bytes = Path(tif).read_bytes()
    base = Image.fromarray(img.array).convert("RGB")
    draw = ImageDraw.Draw(base)
    for g in _to_pixels(out.candidates.geoms, out.candidates.crs, img):
        draw.line(list(g.exterior.coords), fill=(255, 235, 0), width=2)
    prob = np.asarray(out.segmentation.probability)
    cols = st.columns(3 if adv_prob else 2)
    cols[0].image(img.array, caption=f"Imagen {source} ({img.array.shape[1]}×{img.array.shape[0]} px, "
                                     f"≈ {img.meta['approx_gsd_m']} m/píxel)")
    cols[1].image(base, caption="Huellas candidatas")
    if adv_prob:
        cols[2].image(_prob_overlay(img.array, prob), caption="Probabilidad del modelo (más claro = más probable)")
    pmax, share = float(prob.max()), float((prob >= adv_thr).mean()) * 100
    st.caption(f"Probabilidad máxima del modelo: {pmax:.2f} · píxeles por encima del umbral ({adv_thr:.2f}): {share:.1f} %")
    if len(out.candidates) == 0:
        if pmax < adv_thr:
            st.warning("El modelo no ve edificios con esta configuración (su probabilidad máxima no llega al "
                       "umbral). Pruebe bajar el umbral o cambiar la resolución; si sigue igual, la zona es "
                       "demasiado distinta de los datos de entrenamiento.")
        else:
            st.warning("Hay probabilidad por encima del umbral pero las huellas son más pequeñas que el área "
                       "mínima. Pruebe bajar el área mínima.")
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
        if layer is not None and (layer.min_scale or layer.max_scale):
            z0, z1 = layer.zoom_range()
            st.caption("ℹ️ El servicio declara que esta capa solo se dibuja "
                       + (f"hasta una escala 1:{layer.min_scale:,.0f} (zoom ≈ {z1:.0f}) " if layer.min_scale else "")
                       + (f"y desde 1:{layer.max_scale:,.0f} (zoom ≈ {z0:.0f}). " if layer.max_scale else ". ")
                       + "Fuera de ese rango el servidor no dibuja la capa y solo verá el mapa base: no es un fallo de la app.")
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
    with st.expander("Ajustes avanzados (para probar en otras zonas, p. ej. la costa)"):
        st.caption("Si el modelo no detecta bien en una zona, pruebe aquí: bajar el umbral, cambiar la resolución de la "
                   "petición o ver el mapa de probabilidad para saber si el modelo ve algo.")
        adv_thr = st.slider("Umbral de probabilidad", 0.20, 0.90, 0.50, 0.05, key="adv_thr",
                            help="Más bajo detecta más (y más falsos positivos); más alto es más estricto.")
        adv_min_area = st.slider("Área mínima de cada huella (m²)", 4.0, 100.0, 8.0, 2.0, key="adv_min_area")
        adv_gsd = st.slider("Resolución pedida al servicio (m/píxel)", 0.20, 0.60, 0.30, 0.05, key="adv_gsd",
                            help="El modelo se entrenó con ~0,5 m/píxel. Con un cuadrado grande la imagen se limita a "
                                 "512 px, así que la resolución real puede ser más gruesa que la pedida.")
        adv_prob = st.checkbox("Mostrar el mapa de probabilidad del modelo", value=True, key="adv_prob")
    px = min(512, math.ceil(side / adv_gsd))
    st.caption(f"Cuadrado de {side:.0f} × {side:.0f} m ({side * side / 10_000:.1f} ha) → imagen ≈ {px}×{px} px, "
               f"≈ {side / px:.2f} m/píxel (el modelo se entrenó con ~0,5 m).")
    extent = layer.bbox_lonlat if layer is not None else None

    # The base map is built ONCE per (service, layer) and kept in session state. The selected square is
    # added client-side as a feature group and only clicks are reported back, so panning/zooming never
    # reruns the app or rebuilds the map (that caused the flicker and the vanishing layers).
    native = None
    if layer is not None:
        native = st.slider(
            "Detalle máximo del mapa (nivel de zoom que se pide al servidor)", 12, 21, layer.native_zoom(),
            key=f"native_{wms_url}_{layer.name}",
            help="Si la capa desaparece al acercar el mapa, baje este valor un nivel cada vez hasta que se mantenga: "
                 "a partir de ese nivel el mapa amplía las imágenes en lugar de pedir más detalle al servidor (que "
                 "en algunos servicios deja de dibujar). Solo afecta a la vista del mapa, no a la imagen descargada.")
    layer_key = (wms_url, layer.name if layer is not None else None, caps.version if caps else None)
    map_key = layer_key + (native,)
    if st.session_state.get("map_key") != map_key:
        new_layer = st.session_state.get("layer_key") != layer_key
        if new_layer:
            st.session_state.pop("probe", None)
            st.session_state.pop("auto_fail", None)
            # small coverages (e.g. one drone flight): preselect its centre so a square is visible straight away
            st.session_state["center"] = _default_center(extent)
        keep = st.session_state.get("center")
        # same layer, only the map detail changed: stay on the user's square instead of re-framing the coverage
        start = (keep[0], keep[1], 17) if (not new_layer and keep) else DEFAULT_VIEW
        base = folium.Map(location=start[:2], zoom_start=start[2], control_scale=True, tiles=None, max_zoom=MAX_ZOOM)
        folium.TileLayer("OpenStreetMap", name="Mapa base", max_zoom=MAX_ZOOM, max_native_zoom=19).add_to(base)
        if extent:
            w, s_, e, n = extent
            folium.Rectangle([[s_, w], [n, e]], color="#1f77ff", weight=2, fill=False, dash_array="6",
                             interactive=False).add_to(base)  # not interactive: clicks must reach the map
            if new_layer or not keep:
                base.fit_bounds([[s_, w], [n, e]])
        if layer is not None and caps is not None:
            try:
                folium.raster_layers.WmsTileLayer(url=wms_url, layers=layer.name, fmt="image/png", transparent=True,
                                                  version=caps.version, name="Capa WMS", overlay=True,
                                                  max_zoom=MAX_ZOOM,
                                                  # beyond this level, enlarge the last tiles instead of asking the
                                                  # server for detail it does not draw (the layer vanished there)
                                                  max_native_zoom=native).add_to(base)
            except Exception:  # noqa: BLE001 - overlay is a convenience only
                pass
        st.session_state.update(map_key=map_key, layer_key=layer_key, base_map=base,
                                map_ver=st.session_state.get("map_ver", 0) + 1, click_seen=None)
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
    st.caption("🔍 Si la capa desaparece al **acercar**, baje **Detalle máximo del mapa** (encima del mapa) un nivel cada "
               "vez. Si desaparece al **alejar**, acérquese: muchos servicios solo dibujan la ortofoto a escalas grandes. "
               "La línea azul marca dónde dice el servicio que hay datos.")
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

    fail = st.session_state.get("auto_fail")
    if fail and fail.get("key") == st.session_state.get("layer_key"):
        st.error(f"La descarga automática falló: {fail['msg']}")
        st.warning("**Este servicio no responde a la descarga automática de la app**, aunque sí a su navegador (por "
                   "eso ve la capa en el mapa). Use el panel **🌐 Descargar la imagen con su navegador**, abierto aquí "
                   "abajo: abra la imagen, guárdela y súbala.")
        if fail.get("attempts"):
            with st.expander("Detalles técnicos de la descarga automática"):
                for a in fail["attempts"]:
                    st.markdown(f"**{a['crs']}** — {'✅ ok' if a['ok'] else '❌ ' + a.get('error', '')}")
                    st.code(a["url"], language=None)
    else:
        fail = None
    with st.expander("🌐 Descargar la imagen con su navegador (si la descarga automática falla)", expanded=bool(fail)):
        st.caption("Algunos servicios responden a su navegador (por eso ve la capa en el mapa) pero no al servidor de "
                   "esta app. En ese caso descargue usted la imagen del cuadrado y súbala: la app la georreferencia con "
                   "el mismo recuadro y sistema de coordenadas que pidió. Compruebe las condiciones de uso del servicio.")
        if problem is None:
            breq = ogc.browser_getmap_request(wms_url, layer.name, bbox, layer.crs, caps.version)
            st.markdown(f"**1.** Abra la imagen del área ({breq['width']}×{breq['height']} px, "
                        f"≈ {breq['approx_gsd_m']:.2f} m/píxel):")
            st.link_button("Abrir la imagen del cuadrado en una pestaña nueva", breq["url"])
            if breq["approx_gsd_m"] > 0.6:
                st.caption("⚠️ Con este tamaño de cuadrado la imagen queda gruesa para el modelo (entrenado con "
                           "~0,5 m/píxel); un cuadrado de unos 130 m o menos da mejor resolución.")
            st.markdown("**2.** En esa pestaña, clic derecho sobre la imagen → *Guardar imagen como…*  \n"
                        "**3.** Súbala aquí **sin mover el cuadrado** (si lo mueve, vuelva a descargarla):")
            up = st.file_uploader("Imagen guardada (PNG o JPEG)", type=["png", "jpg", "jpeg"], key="browser_upload")
            if up is not None and st.button("▶ Ejecutar inferencia con la imagen subida", type="primary",
                                            key="browser_run"):
                if up.size > MAX_BYTES:
                    st.error("El archivo supera 5 MB.")
                elif not _JOB_LOCK.acquire(blocking=False):
                    st.warning("Hay otra inferencia en curso. Intente de nuevo en unos segundos.")
                else:
                    try:
                        try:
                            predictor = load_predictor()
                        except (ModelWeightsError, OSError) as exc:
                            st.error(f"El modelo no está disponible en este despliegue: {exc}")
                        else:
                            try:
                                bimg = ogc.geoimage_from_bytes(up.getvalue(), breq)
                            except ogc.OGCError as exc:
                                st.error(f"No se pudo usar la imagen subida: {exc}")
                            else:
                                _process_and_show(bimg, predictor, adv_thr, adv_min_area, adv_prob,
                                                  "descargada con su navegador")
                    finally:
                        _JOB_LOCK.release()
        else:
            st.info("Primero elija una capa y coloque el cuadrado dentro de su cobertura.")

    with st.expander("🔧 Diagnóstico del servicio (si la descarga falla)"):
        st.caption("Hace hasta 5 peticiones de prueba, una a una, en el centro del cuadrado, cambiando el tamaño, la "
                   "transparencia, el formato y el sistema de coordenadas, y le dice cuáles acepta el servidor. "
                   "Úselo solo cuando la descarga falle.")
        if st.button("Probar el servicio", disabled=problem is not None, key="probe_btn"):
            try:
                with st.spinner("Probando variantes de la petición…"):
                    st.session_state["probe"] = ogc.wms_probe(wms_url, layer.name, bbox, layer.crs, caps.version)
            except ogc.OGCError as exc:
                st.session_state["probe"] = [{"variant": "No se pudo probar", "ok": False, "detail": str(exc), "url": ""}]
        for r in st.session_state.get("probe", []):
            st.markdown(f"{'✅' if r['ok'] else '❌'} **{r['variant']}** — {r['detail']}")
            if r["url"]:
                st.code(r["url"], language=None)

    rerun_needed = False
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
                    img = ogc.wms_get_map(wms_url, layer.name, bbox, layer.crs, caps.version, adv_gsd)
            except ogc.OGCServiceError as exc:
                # the service answered with an error: remember it for this service/layer and reopen the page with
                # the browser-download panel expanded (the lock is released by ``finally`` before the rerun)
                st.session_state["auto_fail"] = {"key": st.session_state.get("layer_key"), "msg": str(exc),
                                                 "attempts": getattr(exc, "attempts", None)}
                rerun_needed = True
                return
            except (ogc.OGCError, ValueError) as exc:
                st.error(f"No se pudo obtener la imagen: {exc}")
                return
            st.session_state.pop("auto_fail", None)
            _process_and_show(img, predictor, adv_thr, adv_min_area, adv_prob, "WMS descargada")
        finally:
            _JOB_LOCK.release()
            if rerun_needed:
                st.rerun()


tabs = st.tabs(["Imagen propia", "Mapa WMS (cualquier servicio)"])
with tabs[0]:
    tab_upload()
with tabs[1]:
    tab_map()

st.divider()
with st.expander("Limitaciones y atribución"):
    st.markdown(
        "- Modelo entrenado con SpaceNet-4 (Atlanta, WorldView-2). **No se ha validado con etiquetas revisadas de su zona de estudio.** "
        "Imágenes con resolución muy distinta a ~0,5 m/píxel pueden rendir peor, y también las zonas con arquitectura, techos o "
        "vegetación distintos a los de entrenamiento (p. ej. la costa).\n"
        "- Los resultados son **candidatos** para revisión humana; no son límites catastrales ni prueba de errores o construcciones no registradas.\n"
        f"- {spec.license_note}\n- Imagen de ejemplo: recorte de SpaceNet-2 (Las Vegas), CC BY-SA 4.0.\n"
        "- Proyecto independiente: no implica respaldo de Streamlit, de los titulares de los servicios WMS ni de los proveedores del modelo.\n"
        "- Código MIT © 2026 faustoaguanor."
    )
