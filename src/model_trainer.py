import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
import os
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple, List, Callable


class ModelTrainer:
    # Diccionario que mapea nombres de modelos a sus respectivas clases e importaciones
    AVAILABLE_MODELS = {
        "ResNet152V2": {
            "model": "tensorflow.keras.applications.ResNet152V2",
            "preprocess": "tensorflow.keras.applications.resnet_v2.preprocess_input"
        },
        "MobileNetV3Large": {
            "model": "tensorflow.keras.applications.MobileNetV3Large",
            "preprocess": "tensorflow.keras.applications.mobilenet_v3.preprocess_input"
        },
        "EfficientNetV2L": {
            "model": "tensorflow.keras.applications.efficientnet_v2.EfficientNetV2L",
            "preprocess": "tensorflow.keras.applications.efficientnet_v2.preprocess_input"
        },
        "EfficientNetB7": {
            "model": "tensorflow.keras.applications.EfficientNetB7",
            "preprocess": "tensorflow.keras.applications.efficientnet.preprocess_input"
        }
    }

    def __init__(
        self,
        model_name: str,
        img_shape: Tuple[int, int, int] = (128, 256, 1),
        n_classes: int = 667,
        dropout_rate: float = 0.2,
        label_smoothing: float = 0.1,
        weights: str = "imagenet",
        model_dir: str = "./models",
        fine_tune_layers: int = 200
    ):
        """
        Inicializa el entrenador de modelos.

        Args:
            model_name: Nombre del modelo a utilizar (debe estar en AVAILABLE_MODELS)
            img_shape: Forma de las imágenes de entrada (altura, ancho, canales)
            n_classes: Número de clases para la clasificación
            dropout_rate: Tasa de dropout para regularización
            label_smoothing: Valor de suavizado de etiquetas para la pérdida
            weights: Pesos pre-entrenados a utilizar ('imagenet' o None)
            model_dir: Directorio donde guardar los modelos entrenados
            fine_tune_layers: Número de capas finales a ajustar (fine-tuning)
        """
        self.model_name = model_name
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing
        self.weights = weights
        self.model_dir = model_dir
        self.fine_tune_layers = fine_tune_layers

        # Validar que el modelo solicitado está disponible
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modelo '{model_name}' no disponible. Opciones: {list(self.AVAILABLE_MODELS.keys())}")

        # Importar dinámicamente el modelo base y la función de preprocesamiento
        self._import_model_and_preprocess()

        # Crear callback para guardar modelos
        os.makedirs(model_dir, exist_ok=True)

    def _import_model_and_preprocess(self):
        """Importa dinámicamente la clase del modelo base y la función de preprocesamiento."""
        model_info = self.AVAILABLE_MODELS[self.model_name]

        # Importar la clase del modelo base
        module_path, class_name = model_info["model"].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        self.BaseModel = getattr(module, class_name)

        # Importar la función de preprocesamiento
        preprocess_path = model_info["preprocess"].rsplit(".", 1)[0]
        preprocess_func = model_info["preprocess"].rsplit(".", 1)[1]
        preprocess_module = __import__(preprocess_path, fromlist=[preprocess_func])
        self.preprocess_input = getattr(preprocess_module, preprocess_func)

    def create_model(self, learning_rate: float = 1e-4) -> Model:
        """
        Crea un modelo de clasificación utilizando el modelo base especificado.

        Args:
            learning_rate: Tasa de aprendizaje para el optimizador Adam

        Returns:
            Modelo de Keras compilado
        """
        # Definir la entrada del modelo
        inputs = layers.Input(shape=self.img_shape, dtype=tf.float32, name="input_layer")

        # Convertir imágenes en escala de grises a RGB si es necesario
        if self.img_shape[-1] == 1:
            x = layers.Lambda(lambda img: tf.image.grayscale_to_rgb(img))(inputs)
        else:
            x = inputs

        # Aplicar preprocesamiento específico del modelo
        x = layers.Lambda(self.preprocess_input, name="preprocess_input")(x)

        # Inicializar el modelo base
        base_model = self.BaseModel(include_top=False, weights=self.weights, pooling="avg")

        # Configurar capas a entrenar (fine-tuning)
        base_model.trainable = True
        if self.fine_tune_layers > 0:
            for layer in base_model.layers[:-self.fine_tune_layers]:
                layer.trainable = False

        # Construir el modelo completo
        x = base_model(x)
        x = layers.Dropout(self.dropout_rate, name="top_dropout")(x, training=True)
        outputs = layers.Dense(self.n_classes, name="logits")(x)

        # Crear y compilar el modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{self.model_name}_classifier")

        # Compilar el modelo
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, label_smoothing=self.label_smoothing
            ),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy")
            ]
        )

        return model

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        learning_rate: float = 1e-4,
        epochs: int = 20,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ) -> tf.keras.Model:
        """
        Entrena el modelo con los datasets proporcionados.

        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            learning_rate: Tasa de aprendizaje para el optimizador
            epochs: Número de épocas a entrenar
            callbacks: Lista de callbacks adicionales para el entrenamiento

        Returns:
            Modelo entrenado
        """
        # Crear el modelo
        model = self.create_model(learning_rate=learning_rate)

        # Definir callbacks por defecto
        default_callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f"{self.model_name}_best.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                monitor="val_accuracy",
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.model_dir, 'logs', self.model_name),
                write_graph=True,
                update_freq='epoch'
            )
        ]

        # Combinar callbacks
        if callbacks:
            all_callbacks = default_callbacks + callbacks
        else:
            all_callbacks = default_callbacks

        # Entrenar el modelo
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=all_callbacks
        )

        # Guardar el modelo final
        model.save(os.path.join(self.model_dir, f"{self.model_name}_final.h5"))

        return model, history
