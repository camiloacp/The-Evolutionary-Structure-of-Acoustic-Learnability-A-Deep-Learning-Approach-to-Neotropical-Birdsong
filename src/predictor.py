import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .model_trainer import ModelTrainer
from .image_preprocessor import ImagePreprocessor

class Predictor:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        img_shape: Tuple[int, int, int] = (128, 256, 1),
        n_classes: int = 667
    ):
        """
        Inicializa el predictor con un modelo entrenado.

        Args:
            model_name: Nombre del modelo a utilizar
            model_path: Ruta al archivo de pesos del modelo
            img_shape: Forma de las imágenes de entrada
            n_classes: Número de clases
        """
        self.model_name = model_name
        self.model_path = model_path
        self.img_shape = img_shape
        self.n_classes = n_classes

        # Inicializar el entrenador y cargar el modelo
        self.model_trainer = ModelTrainer(
            model_name=model_name,
            img_shape=img_shape,
            n_classes=n_classes
        )
        self.model = self.model_trainer.create_model()
        self.model.load_weights(model_path)

        # Inicializar el preprocesador
        self.preprocessor = ImagePreprocessor()

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza predicciones para un lote de imágenes.

        Args:
            image_paths: Lista de rutas a las imágenes
            batch_size: Tamaño del lote para procesar

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicciones y probabilidades
        """
        # Crear dataset para predicción
        test_data = pd.DataFrame({"image_path": image_paths})
        test_dataset = self.preprocessor.create_dataset(
            test_data,
            include_label=False,
            repeat=False,
            shuffle=False,
            augment=False,
            prefetch=True,
            batch_size=batch_size
        )

        # Realizar predicciones
        predictions = self.model.predict(test_dataset)

        # Obtener clases predichas y probabilidades
        predicted_classes = np.argmax(predictions, axis=1)
        probabilities = np.max(predictions, axis=1)

        return predicted_classes, probabilities

    def predict_single(self, image_path: str) -> Tuple[int, float]:
        """
        Realiza predicción para una única imagen.

        Args:
            image_path: Ruta a la imagen

        Returns:
            Tuple[int, float]: Clase predicha y probabilidad
        """
        predicted_classes, probabilities = self.predict_batch([image_path], batch_size=1)
        return predicted_classes[0], probabilities[0]

    def predict_with_top_k(
        self,
        image_path: str,
        k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Realiza predicción y devuelve las top-k clases más probables.

        Args:
            image_path: Ruta a la imagen
            k: Número de clases más probables a devolver

        Returns:
            List[Tuple[int, float]]: Lista de tuplas (clase, probabilidad) ordenadas por probabilidad
        """
        predictions = self.model.predict(
            self.preprocessor.create_dataset(
                pd.DataFrame({"image_path": [image_path]}),
                include_label=False,
                repeat=False,
                shuffle=False,
                augment=False,
                prefetch=True,
                batch_size=1
            )
        )

        # Obtener top-k predicciones
        top_k_indices = np.argsort(predictions[0])[-k:][::-1]
        top_k_probabilities = predictions[0][top_k_indices]

        return list(zip(top_k_indices, top_k_probabilities))

"""
from src.predictor import Predictor
import pandas as pd

# Inicializar el predictor
predictor = Predictor(
    model_name="MobileNetV3Large",  # o el modelo que hayas usado
    model_path="ruta/a/tu/modelo/weights.h5",
    img_shape=(128, 256, 1),
    n_classes=667
)

# Asumiendo que tu DataFrame se llama df_test y tiene una columna 'image_path'
# Realizar predicciones en lotes
predicted_classes, probabilities = predictor.predict_batch(
    image_paths=df_test['image_path'].tolist(),
    batch_size=32
)

# Agregar las predicciones al DataFrame
df_test['predicted_class'] = predicted_classes
df_test['prediction_probability'] = probabilities

# Si quieres las top-5 predicciones para cada imagen
df_test['top_5_predictions'] = df_test['image_path'].apply(
    lambda x: predictor.predict_with_top_k(x, k=5)
)

# Guardar resultados
df_test.to_csv('resultados_predicciones.csv', index=False)
"""
