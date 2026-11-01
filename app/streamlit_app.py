"""Streamlit Community Cloud demo: real CPU inference on ONE small RGB image (<= 512x512, <= 5 MB).

Uses the same core as the CLI (`quito_building_lab.pipeline.run_image`) with the ONNX backend.
"""

from __future__ import annotations

import io
import json
import resource
import threading
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from quito_building_lab.config import PipelineConfig
from quito_building_lab.export import to_geojson_dict
from quito_building_lab.inference import cached_predictor
from quito_building_lab.models import ModelWeightsError, ensure_onnx, get_spec
from quito_building_lab.pipeline import run_image

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
                    st.stop()
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
