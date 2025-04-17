import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
from typing import Tuple, List, Dict, Optional, Union
from pydantic import BaseModel
import keras_cv
from keras_cv.layers import BaseImageAugmentationLayer
from sklearn.preprocessing import LabelEncoder

class ImagePreprocessorConfig(BaseModel):
    """
    Configuración para el preprocesamiento de imágenes de espectogramas.
    """
    # Configuración general
    img_size: Tuple[int, int] = (128, 256)
    channels: int = 1
    img_shape: Tuple[int, int, int] = (*img_size, channels)

    # Configuración de datasets
    batch_size: int = 128
    valid_batch_size: int = 128
    shuffle_size: int = 1028

    # Configuración de aumentación
    aug_proba: float = 0.8
    brightness_factor: float = 0.2
    grid_mask_ratio: Tuple[float, float] = (0.05, 0.10)
    blur_kernel_size: int = 2
    blur_factor: float = 0.1
    time_mask_param: int = 10
    time_mask_max: int = 3
    freq_mask_param: int = 40
    freq_mask_max: int = 2

    # Configuración de etiquetas
    label_column: str = "label"
    n_classes: int = 667

class RandomRowMask(BaseImageAugmentationLayer):
    """
    Capa de aumentación que aplica máscaras en filas (time masking) a la imagen.

    Esta capa aplica un número aleatorio (entre 1 y max_num_mask) de máscaras en la dimensión
    temporal (filas) de la imagen usando la función tfio.audio.time_mask.

    Args:
        param (int): Longitud máxima de la máscara a aplicar. Valor típico: 10.
        max_num_mask (int): Número máximo de máscaras a aplicar por imagen. Valor típico: 3.
    """

    def __init__(self, param=10, max_num_mask=3, **kwargs):
        super().__init__(**kwargs)
        self.param = param
        self.max_num_mask = max_num_mask

    @tf.function
    def augment_image(self, image, transformation=None, **kwargs):
        """
        Aplica máscaras en filas a la imagen.

        Args:
            image (tf.Tensor): Tensor de imagen de forma [altura, anchura, canales] o
                                [batch, altura, anchura, canales].
            transformation: Parámetro no usado, reservado para compatibilidad con la API.

        Returns:
            tf.Tensor: Imagen con las máscaras aplicadas, manteniendo la misma forma que la entrada.
        """
        # Función interna que procesa una sola imagen.
        def _apply_mask(img):
            # Asegurarse de que img tenga forma [H, W, C]
            img = tf.convert_to_tensor(img)
            original_shape = tf.shape(img)

            # Si la imagen tiene un canal único, podemos quitar la dimensión del canal para aplicar la máscara
            if original_shape[-1] == 1:
                img_2d = tf.squeeze(img, axis=-1)  # forma [H, W]
            else:
                img_2d = img  # Se asume que la función soporta la entrada directamente

            # Determinar el número de máscaras a aplicar (entre 1 y max_num_mask, inclusive).
            num_masks = tf.random.uniform([], minval=1, maxval=self.max_num_mask + 1, dtype=tf.int32)

            # Aplicar la función de enmascarado de manera iterativa
            for _ in tf.range(num_masks):
                img_2d = tfio.audio.time_mask(img_2d, param=self.param)

            # Si se quitó la dimensión del canal, se la reincorpora para mantener la forma original.
            if original_shape[-1] == 1:
                img_2d = tf.expand_dims(img_2d, axis=-1)

            # Restaurar la forma original de la imagen
            img_out = tf.reshape(img_2d, original_shape)
            return img_out

        # Si 'image' tiene batch dimension (4D) se aplica map_fn para procesar cada imagen individualmente.
        if tf.rank(image) == 4:
            image = tf.map_fn(_apply_mask, image)
        else:
            image = _apply_mask(image)
        return image

class RandomColumnMask(BaseImageAugmentationLayer):
    """
    Capa de aumentación que aplica máscaras en columnas (frequency masking) a la imagen.

    Args:
        param (int): Longitud máxima de la máscara a aplicar. Valor típico: 40.
        max_num_mask (int): Número máximo de máscaras a aplicar por imagen. Valor típico: 2.
    """

    def __init__(self, param=40, max_num_mask=2, img_shape=(128, 256, 1), **kwargs):
        super().__init__(**kwargs)
        self.param = param
        self.max_num_mask = max_num_mask
        self.img_shape = img_shape

    def augment_image(self, image, transformation=None, **kwargs):
        """
        Aplica máscaras en columnas a la imagen.

        Args:
            image (tf.Tensor): Tensor de imagen.
            transformation: Parámetro no usado, reservado para compatibilidad con la API.

        Returns:
            tf.Tensor: Imagen con las máscaras aplicadas.
        """
        num = tf.random.uniform([], minval=1, maxval=self.max_num_mask + 1, dtype=tf.int32)

        def condition(i, img):
            return tf.less(i, num)

        def body(i, img):
            img = tfio.audio.freq_mask(tf.squeeze(img), param=self.param)
            img = tf.reshape(img, self.img_shape)
            return tf.add(i, 1), img

        _, image = tf.while_loop(
            condition,
            body,
            loop_vars=[0, image],
            shape_invariants=[tf.TensorShape([]), tf.TensorShape(self.img_shape)]
        )

        return image

class ImagePreprocessor:
    """
    Clase para el preprocesamiento de imágenes de espectogramas, incluyendo
    lectura, redimensionamiento, aumentación y creación de datasets.
    """

    def __init__(
        self,
        config: Optional[ImagePreprocessorConfig] = None,
        label_encoder: Optional[LabelEncoder] = None
        ):
        """
        Inicializa el preprocesador de imágenes.

        Args:
            config: Configuración para el preprocesamiento. Si es None, se usa la configuración por defecto.
        """
        self.config = config or ImagePreprocessorConfig()
        self.label_encoder = label_encoder
        self._create_augmenter()

    def _create_augmenter(self):
        """Crea la capa de aumentación usando keras_cv."""
        self.augmenter = keras_cv.layers.Augmenter(
            layers=[
                keras_cv.layers.RandomBrightness(factor=self.config.brightness_factor),
                keras_cv.layers.GridMask(ratio_factor=self.config.grid_mask_ratio),
                keras_cv.layers.RandomGaussianBlur(
                    kernel_size=self.config.blur_kernel_size,
                    factor=self.config.blur_factor
                ),
                RandomRowMask(
                    param=self.config.time_mask_param,
                    max_num_mask=self.config.time_mask_max
                ),
                RandomColumnMask(
                    param=self.config.freq_mask_param,
                    max_num_mask=self.config.freq_mask_max,
                    img_shape=self.config.img_shape
                )
            ]
        )

    def fit_label_encoder(self, labels: List[str]):
        """
        Entrena el codificador de etiquetas.

        Args:
            labels: Lista de etiquetas de clase.
        """
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)

    def encode_labels(self, labels: List[str]) -> tf.Tensor:
        """
        Codifica las etiquetas en one-hot.

        Args:
            labels: Lista de etiquetas de clase.

        Returns:
            Tensor de etiquetas codificadas en one-hot.
        """
        if self.label_encoder is None:
            self.fit_label_encoder(labels)

        # Transformamos a etiquetas numéricas
        integer_encoded = self.label_encoder.transform(labels)
        # Convertimos a one-hot
        return tf.one_hot(integer_encoded, depth=self.config.n_classes)

    def read_image(self, path_img: str) -> tf.Tensor:
        """
        Lee una imagen de un archivo y la preprocesa.

        Args:
            path_img: Ruta al archivo de imagen.

        Returns:
            Tensor con la imagen preprocesada.
        """
        img_data = tf.io.read_file(path_img)
        img = tf.io.decode_jpeg(img_data, channels=self.config.channels)
        img = tf.image.resize(img, self.config.img_size)
        img = tf.cast(img, tf.float32)
        return img

    def augment_image(self, img: tf.Tensor) -> tf.Tensor:
        """
        Aplica aumentación a una imagen con una probabilidad definida.

        Args:
            img: Tensor de imagen a aumentar.

        Returns:
            Tensor con la imagen aumentada.
        """
        if tf.random.uniform([]) <= self.config.aug_proba:
            img = self.augmenter(img)
        return img

    def show_image_stats(self, img: tf.Tensor) -> Dict:
        """
        Muestra estadísticas de una imagen.

        Args:
            img: Tensor de imagen.

        Returns:
            Diccionario con estadísticas de la imagen.
        """
        stats = {}
        if isinstance(img, tf.Tensor):
            stats = {
                "shape": img.shape,
                "dtype": img.dtype,
                "min": tf.reduce_min(img).numpy(),
                "max": tf.reduce_max(img).numpy()
            }
        elif isinstance(img, np.ndarray):
            stats = {
                "shape": img.shape,
                "dtype": img.dtype,
                "min": img.min(),
                "max": img.max()
            }
        else:
            stats = {"error": f"Tipo inesperado: {type(img)}"}

        return stats

    def create_dataset(
        self,
        data: pd.DataFrame,
        include_label: bool = True,
        repeat: bool = False,
        shuffle: bool = False,
        augment: bool = False,
        prefetch: bool = False,
        batch_size: Optional[int] = None
    ) -> tf.data.Dataset:
        """
        Crea un dataset de TensorFlow a partir de un DataFrame.

        Args:
            data: DataFrame con las rutas de las imágenes y etiquetas.
            include_label: Si se deben incluir las etiquetas.
            repeat: Si el dataset debe repetirse.
            shuffle: Si el dataset debe mezclarse.
            augment: Si se debe aplicar aumentación.
            prefetch: Si se debe usar prefetch.
            batch_size: Tamaño del batch.

        Returns:
            Dataset de TensorFlow.
        """
        AUTOTUNE = tf.data.AUTOTUNE

        slices = data["path_img"].values if "path_img" in data.columns else data["image_path"].values

        read_func = self.read_image
        aug_func = self.augment_image

        if include_label:
            labels = data[self.config.label_column].values
            if self.label_encoder is None:
                self.fit_label_encoder(labels)

            slices = (slices, self.encode_labels(labels))
            read_func = lambda path_img, label: (self.read_image(path_img), label)
            aug_func = lambda img, label: (self.augment_image(img), label)

        ds = tf.data.Dataset.from_tensor_slices(slices)
        ds = ds.map(read_func, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()

        if repeat:
            ds = ds.repeat()

        if shuffle:
            ds = ds.shuffle(buffer_size=self.config.shuffle_size)

        if augment:
            ds = ds.map(aug_func, num_parallel_calls=AUTOTUNE)

        if batch_size:
            ds = ds.batch(batch_size)

        if prefetch:
            ds = ds.prefetch(AUTOTUNE)

        return ds

    def create_training_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Crea un dataset de entrenamiento.

        Args:
            data: DataFrame con las rutas de las imágenes y etiquetas.

        Returns:
            Dataset de entrenamiento.
        """
        return self.create_dataset(
            data,
            include_label=True,
            repeat=True,
            shuffle=True,
            augment=True,
            prefetch=True,
            batch_size=self.config.batch_size,
        )

    def create_validation_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Crea un dataset de validación.

        Args:
            data: DataFrame con las rutas de las imágenes y etiquetas.

        Returns:
            Dataset de validación.
        """
        return self.create_dataset(
            data,
            include_label=True,
            repeat=False,
            shuffle=False,
            augment=False,
            prefetch=True,
            batch_size=self.config.valid_batch_size,
        )

    def load_data_from_directory(self, base_path: str) -> pd.DataFrame:
        """
        Carga datos de un directorio donde cada subdirectorio es una clase.

        Args:
            base_path: Ruta base donde se encuentran los subdirectorios de clases.

        Returns:
            DataFrame con rutas y etiquetas.
        """
        data = []

        # Listar las carpetas dentro de ruta_base
        image_directories = os.listdir(base_path)
        image_directories.sort()

        for directory in image_directories:
            dir_path = os.path.join(base_path, directory)

            # Solo procesar si es un directorio
            if os.path.isdir(dir_path):
                for img in os.listdir(dir_path):
                    img_path = os.path.join(dir_path, img)

                    # Verificar que sea un archivo
                    if os.path.isfile(img_path):
                        data.append({
                            "label": directory,  # Usar el nombre de la carpeta como etiqueta
                            "image_path": img_path
                        })

        return pd.DataFrame(data)
