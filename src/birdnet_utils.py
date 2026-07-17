"""
Utilidades para comparar el modelo del proyecto (ResNet152V2) con BirdNET
sobre el conjunto de test.

Este módulo NO importa TensorFlow ni birdnet, de forma que puede usarse tanto
desde el entorno aislado de BirdNET (notebook 20) como desde el ``.venv`` del
proyecto con TensorFlow 2.19 (notebook 21).

Contexto: el proyecto ya no conserva los audios originales, solo los
espectrogramas JPEG de test en
``src/data/images_test/images_spectograms/<Especie>/<xc_id>_<offset>.jpeg``.
El ``<xc_id>`` es el identificador de la grabación en Xeno-canto, por lo que el
audio original se puede volver a descargar por ID (API v3).
"""
from __future__ import annotations

import pathlib
import re
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Índice del test set a partir de los espectrogramas
# ---------------------------------------------------------------------------

# Nombre de archivo esperado: <xc_id>_<offset>.jpeg  (ver src/spectograms.py:143)
_FNAME_RE = re.compile(r"^(?P<rec_id>\d+)_(?P<offset>\d+)\.jpe?g$", re.IGNORECASE)


def build_test_index(images_root: str) -> pd.DataFrame:
    """Construye el índice del test a nivel de *grabación*.

    Recorre ``<images_root>/<Especie>/<xc_id>_<offset>.jpeg`` y agrupa los
    chunks por ``(especie, xc_id)``.

    Devuelve un DataFrame con columnas:
        - ``recording_id`` (str): ID de Xeno-canto.
        - ``species`` (str): nombre científico (nombre de la carpeta).
        - ``n_chunks`` (int): número de espectrogramas de esa grabación.
        - ``chunk_paths`` (list[str]): rutas de los chunks, ordenadas por offset.
    """
    root = pathlib.Path(images_root)
    if not root.exists():
        raise FileNotFoundError(f"No existe images_root: {root}")

    rec_map: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir():
            continue
        species = species_dir.name
        for img in species_dir.iterdir():
            m = _FNAME_RE.match(img.name)
            if not m:
                continue
            rec_id = m.group("rec_id")
            offset = int(m.group("offset"))
            rec_map.setdefault((species, rec_id), []).append((offset, str(img)))

    rows = []
    for (species, rec_id), chunks in rec_map.items():
        chunks.sort(key=lambda t: t[0])
        rows.append(
            {
                "recording_id": rec_id,
                "species": species,
                "n_chunks": len(chunks),
                "chunk_paths": [p for _, p in chunks],
            }
        )
    if not rows:
        raise RuntimeError(f"No se encontraron espectrogramas en {root}")
    return (
        pd.DataFrame(rows)
        .sort_values(["species", "recording_id"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 2. Etiquetas de BirdNET y cobertura de especies
# ---------------------------------------------------------------------------

def birdnet_labels_to_df(labels: Sequence[str]) -> pd.DataFrame:
    """Convierte una lista de etiquetas de BirdNET (``'Genus species_Common'``)
    en un DataFrame con columnas ``label``, ``scientific`` y ``common``."""
    rows = []
    for lab in labels:
        lab = str(lab).strip()
        if not lab:
            continue
        if "_" in lab:
            sci, common = lab.split("_", 1)
        else:
            sci, common = lab, ""
        rows.append({"label": lab, "scientific": sci.strip(), "common": common.strip()})
    return pd.DataFrame(rows)


def load_birdnet_labels(labels_path: str) -> pd.DataFrame:
    """Carga las etiquetas de BirdNET desde un archivo de texto (una por línea)."""
    text = pathlib.Path(labels_path).read_text(encoding="utf-8")
    return birdnet_labels_to_df(text.splitlines())


def _norm_sci(name: str) -> str:
    """Normaliza un nombre científico para el cruce (minúsculas, sin dobles espacios)."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def match_species(
    dataset_species: Sequence[str], birdnet_labels: pd.DataFrame
) -> pd.DataFrame:
    """Cruza las especies del dataset con la taxonomía de BirdNET por nombre
    científico (match exacto normalizado).

    Devuelve un DataFrame con columnas: ``species`` (del dataset),
    ``covered`` (bool), ``birdnet_label`` y ``birdnet_scientific``.

    Nota: el match es por binomio exacto. Las especies con cambios
    nomenclaturales recientes (p. ej. *Tangara vitriolina* -> *Stilpnia
    vitriolina*, ver ``src/taxonomia.py``) pueden aparecer como no cubiertas
    aunque BirdNET las incluya bajo otro nombre; conviene revisar manualmente
    la lista de no cubiertas.
    """
    lookup = {
        _norm_sci(sci): (lab, sci)
        for lab, sci in zip(birdnet_labels["label"], birdnet_labels["scientific"])
    }
    rows = []
    for sp in dataset_species:
        hit = lookup.get(_norm_sci(sp))
        rows.append(
            {
                "species": sp,
                "covered": hit is not None,
                "birdnet_label": hit[0] if hit else None,
                "birdnet_scientific": hit[1] if hit else None,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Muestra estratificada
# ---------------------------------------------------------------------------

def stratified_sample(
    index: pd.DataFrame,
    n_per_species: Optional[int],
    covered_species: Optional[Set[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Toma hasta ``n_per_species`` grabaciones por especie.

    Si ``n_per_species`` es ``None`` incluye **todas** las grabaciones de cada
    especie (todo el test). Si ``covered_species`` se especifica, solo incluye
    esas especies (las que BirdNET puede predecir), para una comparación justa.
    """
    df = index
    if covered_species is not None:
        df = df[df["species"].isin(set(covered_species))]
    parts = []
    for _, g in df.groupby("species", sort=True):
        if n_per_species is None:
            parts.append(g)
        else:
            parts.append(g.sample(n=min(n_per_species, len(g)), random_state=seed))
    if not parts:
        raise RuntimeError("La muestra estratificada quedó vacía (revisa covered_species).")
    return pd.concat(parts).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Descarga desde Xeno-canto (URL de descarga directa, SIN API key)
# ---------------------------------------------------------------------------

# La URL de descarga directa entrega el audio original y NO requiere API key
# (a diferencia de la API v3, que sí la exige). Es el mismo acceso público que
# usaba históricamente xenopy.
XC_DOWNLOAD_URL = "https://xeno-canto.org/{xc_id}/download"


def download_xc_recording(
    xc_id: str,
    out_dir: str,
    species: Optional[str] = None,
    session=None,
    api_key: Optional[str] = None,  # aceptado por compatibilidad; se ignora
    max_retries: int = 3,
    pause: float = 0.5,
    timeout: int = 90,
) -> Optional[str]:
    """Descarga la grabación de Xeno-canto ``xc_id`` por su URL de descarga
    directa (``https://xeno-canto.org/<id>/download``), que **no requiere API
    key**.

    - Guarda en ``<out_dir>/[<species>/]<xc_id>.<ext>`` (extensión detectada del
      encabezado ``Content-Disposition``/``Content-Type``; por defecto ``.mp3``).
    - Si ya existe un archivo para ese ID, lo devuelve sin volver a descargar
      (checkpointing).
    - Devuelve la ruta local, o ``None`` si la grabación fue eliminada (404/410)
      o no se pudo descargar.

    ``api_key`` se acepta por compatibilidad pero se ignora. Requiere el paquete
    ``requests`` (import perezoso).
    """
    import re as _re
    import requests

    if session is None:
        session = requests.Session()

    base = pathlib.Path(out_dir)
    if species:
        base = base / species
    base.mkdir(parents=True, exist_ok=True)

    existing = [p for p in base.glob(f"{xc_id}.*") if not p.name.endswith(".part")]
    if existing:
        return str(existing[0])

    url = XC_DOWNLOAD_URL.format(xc_id=xc_id)
    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout,
                             allow_redirects=True) as r:
                if r.status_code in (404, 410):
                    return None  # grabación eliminada / inexistente
                if r.status_code == 429:  # rate limit
                    time.sleep(pause * attempt * 3)
                    continue
                r.raise_for_status()

                # Detectar la extensión del archivo original
                ext = ".mp3"
                cd = r.headers.get("content-disposition", "")
                m = _re.search(r'filename="?[^"]*?(\.[A-Za-z0-9]+)"?\s*$', cd)
                if m:
                    ext = m.group(1).lower()
                else:
                    ct = r.headers.get("content-type", "").lower()
                    for key, e in (("wav", ".wav"), ("flac", ".flac"),
                                   ("ogg", ".ogg"), ("mpeg", ".mp3")):
                        if key in ct:
                            ext = e
                            break

                out_path = base / f"{xc_id}{ext}"
                tmp = out_path.with_name(out_path.name + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
                tmp.rename(out_path)
            time.sleep(pause)  # cortesía / rate-limit
            return str(out_path)
        except Exception:
            if attempt == max_retries:
                return None
            time.sleep(pause * attempt)
    return None


def is_valid_audio(path: str, min_duration_s: float = 0.5) -> bool:
    """Comprueba que ``path`` es un audio legible y con duración suficiente.

    BirdNET procesa los archivos en lote y **cancela todo el análisis** si uno
    está vacío o corrupto (la FFT falla con "invalid number of data points").
    Esta comprobación **decodifica** el audio (no solo el encabezado, que puede
    mentir en MP3 con Xing header roto) y descarta los patológicos.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        return False
    return len(data) > 0 and sr > 0 and (len(data) / sr) >= min_duration_s


def load_mono_48k(path: str, target_sr: int = 48000, min_duration_s: float = 0.5):
    """Carga un audio como array mono ``float32`` resampleado a ``target_sr``.

    Devuelve ``(array, target_sr)`` o ``None`` si es ilegible o dura menos de
    ``min_duration_s``. Pre-decodificar así —en vez de dejar que BirdNET lea el
    archivo— **esquiva el bug de segmento vacío al resamplear** que provocan los
    MP3 corruptos de Xeno-canto (y evita el multiprocessing frágil de la lectura).
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    import numpy as np
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    if len(data) == 0 or sr <= 0 or (len(data) / sr) < min_duration_s:
        return None
    if int(sr) != int(target_sr):
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), int(target_sr))
            data = resample_poly(data, int(target_sr) // g, int(sr) // g)
        except Exception:
            return None
    return (np.ascontiguousarray(data, dtype=np.float32), int(target_sr))


def predict_birdnet_robust(
    model,
    paths,
    custom_species_list,
    top_k: int = 5,
    batch_size: int = 64,
    default_confidence_threshold: float = 0.0,
    on_progress=None,
):
    """Ejecuta ``model.predict`` de forma robusta ante archivos problemáticos.

    BirdNET procesa en lote y **cancela todo el análisis** si un solo archivo
    produce un segmento vacío al resamplear (algunos MP3 de Xeno-canto). Esta
    función procesa por lotes y, cuando un lote falla, lo **reintenta archivo por
    archivo** para aislar y omitir únicamente al culpable, sin perder el resto.

    Devuelve ``(dataframe_concatenado, lista_de_paths_fallidos)``. El DataFrame
    tiene las columnas de ``AcousticFilePredictionResult.to_dataframe()``.
    """
    import pandas as pd

    def _run(subset):
        res = model.predict(
            subset,
            top_k=top_k,
            n_producers=8,
            default_confidence_threshold=default_confidence_threshold,
            custom_species_list=custom_species_list,
            show_stats=None,
        )
        return res.to_dataframe()

    dfs, failed = [], []

    def _process(subset):
        # Divide y vencerás: si un subconjunto falla, se parte a la mitad hasta
        # aislar el/los archivo(s) culpable(s) (mucho más rápido que uno por uno).
        if not subset:
            return
        try:
            dfs.append(_run(subset))
        except Exception:
            if len(subset) == 1:
                failed.append(subset[0])
            else:
                mid = len(subset) // 2
                _process(subset[:mid])
                _process(subset[mid:])

    paths = list(paths)
    for i in range(0, len(paths), batch_size):
        _process(paths[i:i + batch_size])
        if on_progress is not None:
            on_progress(min(i + batch_size, len(paths)), len(paths))

    cols = ["input", "start_time", "end_time", "species_name", "confidence"]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=cols)
    return df, failed


# ---------------------------------------------------------------------------
# 5. Agregación de predicciones a nivel de grabación
# ---------------------------------------------------------------------------

def aggregate_recording_level(
    chunk_scores: Dict[str, List[float]], how: str = "max"
) -> Tuple[Optional[str], float]:
    """Colapsa las puntuaciones por especie de varios chunks a una única
    predicción top-1 por grabación.

    ``chunk_scores``: ``{especie: [confianzas en los chunks donde aparece]}``.
    ``how``: ``'max'`` (por defecto) o ``'mean'``.
    Devuelve ``(especie_top1, confianza)`` o ``(None, 0.0)`` si no hay nada.
    """
    agg: Dict[str, float] = {}
    for sp, scores in chunk_scores.items():
        if not scores:
            continue
        agg[sp] = max(scores) if how == "max" else sum(scores) / len(scores)
    if not agg:
        return None, 0.0
    top_sp, top_conf = max(agg.items(), key=lambda kv: kv[1])
    return top_sp, float(top_conf)


def birdnet_top1_from_dataframe(df, how: str = "max") -> Tuple[Optional[str], float]:
    """Predicción top-1 a nivel de grabación a partir de la salida de BirdNET.

    ``df`` es ``AcousticFilePredictionResult.to_dataframe()`` (paquete
    ``birdnet``), con una fila por segmento de 3 s y columnas ``species_name`` y
    ``confidence``. Se agrega por especie a lo largo de todos los segmentos con
    ``how`` (``'max'`` por defecto, o ``'mean'``) y se devuelve
    ``(especie_top1, confianza)``. Si ``df`` está vacío, ``(None, 0.0)``.

    La ``species_name`` conserva el formato de BirdNET ``'Genus species_Common'``;
    el binomio científico se obtiene con :func:`birdnet_label_to_scientific`.
    """
    if df is None or len(df) == 0:
        return None, 0.0
    grp = df.groupby("species_name")["confidence"]
    agg = grp.max() if how == "max" else grp.mean()
    top = agg.idxmax()
    return str(top), float(agg.loc[top])


def birdnet_label_to_scientific(label: Optional[str]) -> Optional[str]:
    """Extrae el binomio científico de una etiqueta de BirdNET
    (``'Genus species_Common Name'`` -> ``'Genus species'``)."""
    if label is None:
        return None
    return str(label).split("_", 1)[0].strip()


# ---------------------------------------------------------------------------
# 6. Taxonomía (familia / género) reutilizando src/taxonomia.py
# ---------------------------------------------------------------------------

def get_genus_family_map() -> Dict[str, str]:
    """Devuelve ``{Genus: Family}`` a partir de ``src/taxonomia.py``.

    Requiere que el directorio ``src`` esté en ``sys.path`` (los notebooks
    hacen ``sys.path.append('../src')``).
    """
    from taxonomia import genus_family, capitalize_family_genus

    return capitalize_family_genus(genus_family)


def family_of(species: str, gf_map: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Devuelve la familia de una especie (nombre científico ``'Genus species'``)."""
    if gf_map is None:
        gf_map = get_genus_family_map()
    genus = str(species).strip().split()[0].capitalize()
    return gf_map.get(genus)
