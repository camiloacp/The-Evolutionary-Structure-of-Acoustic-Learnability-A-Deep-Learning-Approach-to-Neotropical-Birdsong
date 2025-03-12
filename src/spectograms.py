# Librerías del sistema
import os, pathlib
import gc
from typing import Tuple

# Librerías de procesamiento de datos
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

# Librerías de procesamiento de audio
import librosa
import soundfile as sf

# Librerías de visualización
import matplotlib.pyplot as plt
from IPython.display import Audio
import cv2

# Librerías de machine learning
import tensorflow as tf
from pydantic import BaseModel as ConfigBaseModel

# Utilidades
from tqdm.notebook import tqdm

class SpectogramConfig():
    def __init__(
        self,
        # Datos de configuración principales
        audio_dir: str = "data/audio",
        output_dir: str = "data/spectograms",
        sample_rate: int = 32_000,
        # Configuración del espectrograma
        img_size: Tuple[int, int] = (224, 224),
        seconds: int = 5,
        num_offset_max: int = 24,
        min_duration: float = 0.5,
        n_fft: int = 2048,
        n_mels: int = 224,
        # Cálculo predefinido de hop_length
        hop_length: int = (5 * 32_000 - 2048) // (224 - 1),
        center: bool = True,
        fmin: int = 500,
        fmax: int = 12_500,
        power: float = 1.0,
        top_db: int = 80,
        # Configuración de salida
        out_dir: str = "/content/drive/My Drive/images_spectograms/",
        jpeg_quality: int = 100
    ):

        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.img_size = img_size
        self.seconds = seconds
        self.num_offset_max = num_offset_max
        self.min_duration = min_duration
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.center = center
        self.fmin = fmin
        self.fmax = fmax
        self.power = power
        self.top_db = top_db
        self.out_dir = out_dir
        self.jpeg_quality = jpeg_quality

    def load_data(self):
        self.data = pd.read_csv(self.audio_dir)
        return self.data

    def get_duration(self, audio_path: str) -> float:
        try:
            return sf.info(audio_path).duration
        except:
            return 0.0

    def process_chunk(self, paths_chunk):
        return [self.get_duration(path) for path in paths_chunk]

    def duration(self, audio_path: str) -> float:
        if not hasattr(self, 'data') or self.data is None:
            self.load_data()

        paths = self.data[audio_path].tolist()
        chunks = np.array_split(paths, os.cpu_count())

        durations = Parallel(n_jobs=-1, verbose=1)(
            delayed(self.process_chunk)(chunk) for chunk in chunks
        )

        self.data["duration"] = [dur for sublist in durations for dur in sublist]
        self.data["num_offset"] = (1 + (self.data["duration"] - self.min_duration) // self.seconds).astype('int')
        self.data["num_offset"] = self.data["num_offset"].clip(upper=self.num_offset_max)

        return self.data

    def get_mel_spec_db(self, path_ogg: str, offset: int = 0):
        required_len = self.seconds * self.sample_rate
        sig, dr = librosa.load(path=path_ogg, sr=self.sample_rate, offset=(offset * self.seconds), duration=self.seconds)

        if len(sig) < required_len:
            sig = np.concatenate([sig, np.zeros((required_len - len(sig)), dtype=sig.dtype)])

        mel_spec = librosa.feature.melspectrogram(
            y=sig,
            hop_length=self.hop_length,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            center=self.center,
            fmin=self.fmin,
            fmax=self.fmax,
            power=self.power
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.top_db)
        return mel_spec_db

    def normalize_img(self, img: np.ndarray):
        assert img.ndim == 2, "Unexpected image dimension"
        v_min, v_max = np.min(img), np.max(img)
        return ((img - v_min) / (v_max - v_min) * 255).astype(np.uint8)

    def process_record(self, rec):
        """Procesa un único registro de audio para generar espectrogramas"""
        rec_dir = os.path.join(self.out_dir, rec.label)
        os.makedirs(rec_dir, exist_ok=True)
        stats = []
        base_stat = {"label": rec.label, "orig_filename": rec.audio_path}

        if not os.path.exists(rec.audio_path):
            raise FileNotFoundError(f"Archivo no encontrado: {rec.audio_path}")

        for offset in range(rec.num_offset):
            try:
                mel_spec_db = self.get_mel_spec_db(rec.audio_path, offset=offset)
                img = self.normalize_img(mel_spec_db)
                fname = f"{pathlib.Path(rec.audio_path).stem}_{offset}.jpeg"
                path_img = os.path.join(rec_dir, fname)
                ret = cv2.imwrite(path_img, img, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                stat = base_stat.copy()
                stat.update({
                    "offset": offset,
                    "ret": ret,
                    "filename": "/".join(pathlib.Path(path_img).parts[-2:]),
                })
                stats.append(stat)
            except Exception as e:
                print(f"Error procesando offset {offset} para {rec.audio_path}: {e}")
                continue
        return pd.DataFrame(stats)

    def process_data(self, data=None):
        """Procesa todos los datos del DataFrame"""
        if data is None:
            data = self.data

        errors = []
        l_stats = []
        for rec in data.itertuples():
            try:
                stats = self.process_record(rec)
                l_stats.append(stats)
            except FileNotFoundError as e:
                print(e)
                errors.append((rec.audio_path, str(e)))
            except Exception as e:
                print(f"Error leyendo {rec.audio_path}: {e}")
                errors.append((rec.audio_path, str(e)))
            gc.collect()

        if l_stats:
            combined_stats = pd.concat(l_stats, ignore_index=True)
            return combined_stats, errors
        else:
            return pd.DataFrame(), errors

    def process_parallel(self):
        """Procesa los datos en paralelo utilizando múltiples núcleos"""
        from joblib import parallel_backend

        with parallel_backend("threading"):
            results = Parallel(n_jobs=os.cpu_count(), verbose=1)(
                delayed(self.process_data)(sub) for sub in np.array_split(self.data, os.cpu_count())
            )

        # Combinar resultados
        all_stats = []
        all_errors = []
        for stats, errors in results:
            if not stats.empty:
                all_stats.append(stats)
            all_errors.extend(errors)

        if all_stats:
            combined_stats = pd.concat(all_stats, ignore_index=True)
            return combined_stats, all_errors
        else:
            return pd.DataFrame(), all_errors

    def spectograms_paths(self):
        data = []
        # Se listan las carpetas dentro de ruta_base
        image_db = os.listdir(self.out_dir)
        image_db.sort()
        for i in image_db:
            ruta_db = os.path.join(self.out_dir, i)
            print("Procesando:", ruta_db)
            # Solo procesar si es un directorio
            if os.path.isdir(ruta_db):
                for img in os.listdir(ruta_db):
                    ruta_img = os.path.join(ruta_db, img)
                    # Verificar que sea un archivo (opcional)
                    if os.path.isfile(ruta_img):
                        data.append({
                            "label": i,  # Se usa el nombre de la carpeta como etiqueta
                            "image_path": ruta_img
                        })
        return pd.DataFrame(data)
