import os, pathlib
from typing import Dict, Optional, Tuple, List, Union, Any

os.environ["KERAS_BACKEND"] = "tensorflow"

from datetime import datetime
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel as PydanticBaseModel
from IPython.display import Image, display
import tensorflow as tf
import keras_cv
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split
# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from typing import Optional, Tuple, List, Union


# Diccionarios de configuración para los diferentes modelos
BASE_MODEL_NAMES: Dict[str, str] = {
    "ResNet152V2": "resnet152v2",
    "MobileNetV3Large": "MobileNetV3Large",
    "EfficientNetV2L": "efficientnetv2-l",
    "EfficientNetB7": "efficientnetb7"
}

CONV_LAYERS: Dict[str, str] = {
    "ResNet152V2": "conv5_block3_3_conv",
    "MobileNetV3Large": "conv_1",
    "EfficientNetV2L": "top_conv",
    "EfficientNetB7": "top_conv"
}

class GradCam:
    """
    Implementación de Gradient-weighted Class Activation Mapping (Grad-CAM).

    Grad-CAM utiliza gradientes de la clase objetivo que fluyen hacia la capa convolucional
    final para producir un mapa de calor que resalta las regiones importantes de una imagen
    para la predicción de un modelo.

    Attributes
    ----------
    model : tf.keras.Model
        Modelo de TensorFlow/Keras pre-entrenado
    model_name : str, optional
        Nombre del modelo para identificar la arquitectura correspondiente
    """

    def __init__(
        self,
        model: tf.keras.Model,
        model_name: Optional[str] = None
    ) -> None:
        """
        Inicializa la clase GradCam.

        Parameters
        ----------
        model : tf.keras.Model
            Modelo pre-entrenado de TensorFlow/Keras
        model_name : str, optional
            Nombre del modelo para determinar las capas específicas y el preprocesamiento
        """
        self.model = model
        self.model_name = model_name

    def prepare_image(self, img_path: str, img_size: Tuple[int, int]) -> np.ndarray:
        """
        Prepara una imagen para la inferencia.

        Carga la imagen desde el disco, la redimensiona al tamaño deseado y
        aplica el preprocesamiento adecuado según el modelo seleccionado.

        Parameters
        ----------
        img_path : str
            Ruta a la imagen que se va a procesar
        img_size : Tuple[int, int]
            Dimensiones a las que se redimensionará la imagen (altura, anchura)

        Returns
        -------
        np.ndarray
            Imagen procesada lista para la inferencia con forma [1, altura, anchura, canales]
        """
        img = load_img(img_path, target_size=img_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Aplicar el preprocesamiento específico según el modelo
        if self.model_name == "ResNet152V2":
            x = tf.keras.applications.resnet_v2.preprocess_input(x)
        elif self.model_name == "MobileNetV3Large":
            x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
        elif self.model_name == "EfficientNetV2L":
            x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
        elif self.model_name == "EfficientNetB7":
            x = tf.keras.applications.efficientnet.preprocess_input(x)

        return x

    def generate_gradcam(self, img_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Genera un mapa de calor de Grad-CAM para una imagen.

        Parameters
        ----------
        img_path : str
            Ruta a la imagen para la que se generará el mapa de calor

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Una tupla que contiene:
            - heatmap: Mapa de calor normalizado
            - original_img: Imagen original redimensionada
            - predictions: Predicciones del modelo para la imagen

        Notes
        -----
        Este método asume que las imágenes se redimensionan a (128, 256).
        """
        # Tamaño fijo para las imágenes de entrada
        img_size = (128, 256)

        # Preparar la imagen y obtener la original para visualización
        img_array = self.prepare_image(img_path, img_size=img_size)
        original_img = np.array(load_img(img_path, target_size=img_size))

        # Obtener el modelo base y la última capa convolucional
        if self.model_name not in BASE_MODEL_NAMES:
            raise ValueError(f"Modelo no soportado: {self.model_name}")

        base_model = self.model.get_layer(BASE_MODEL_NAMES[self.model_name])
        last_conv_layer = base_model.get_layer(CONV_LAYERS[self.model_name])

        # Crear un modelo que mapea desde la entrada hasta la última capa convolucional
        # y hasta la salida del modelo base
        grad_model = tf.keras.Model(
            inputs=[base_model.inputs],
            outputs=[last_conv_layer.output, base_model.output]
        )

        # Calcular los gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            top_pred_index = tf.argmax(predictions[0])
            class_output = predictions[:, top_pred_index]

        # Obtener los gradientes de la clase con respecto a la salida de la capa convolucional
        grads = tape.gradient(class_output, conv_outputs)

        # Calcular los pesos mediante promediado global
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Obtener la salida de la capa convolucional para la primera imagen (lote de 1)
        conv_outputs = conv_outputs[0]

        # Multiplicar cada canal por su importancia y sumarlos
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

        # Aplicar ReLU y normalizar
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()

        # Redimensionar el mapa de calor al tamaño de la imagen
        heatmap = tf.image.resize(
            heatmap[..., tf.newaxis],
            img_size,
            method='bilinear'
        ).numpy()[:,:,0]

        return heatmap, original_img, predictions
