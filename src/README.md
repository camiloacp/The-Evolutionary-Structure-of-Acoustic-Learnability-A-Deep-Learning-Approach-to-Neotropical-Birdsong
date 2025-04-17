# ModelTrainer

Esta clase permite entrenar modelos de clasificación utilizando diferentes arquitecturas de `tensorflow.keras.applications` de manera flexible y reutilizable.

## Características principales

- Soporte para múltiples modelos pre-entrenados de Keras Applications
- Importación dinámica de modelos y funciones de preprocesamiento
- Fácil configuración del proceso de fine-tuning
- Callbacks por defecto para optimizar el entrenamiento
- Almacenamiento automático de modelos entrenados

## Modelos soportados

Actualmente, la clase soporta los siguientes modelos:

- `ResNet152V2`: Ideal para conjuntos de datos grandes y tareas complejas
- `MobileNetV3Large`: Optimizado para dispositivos móviles y aplicaciones de baja latencia
- `EfficientNetV2L`: Excelente equilibrio entre precisión y eficiencia computacional
- `EfficientNetB7`: Versión anterior del EfficientNet con alto rendimiento

## Uso básico

```python
from model_trainer import ModelTrainer
import tensorflow as tf

# Crear datasets
train_dataset = ...  # Tu dataset de entrenamiento
val_dataset = ...    # Tu dataset de validación

# Inicializar el entrenador con el modelo deseado
trainer = ModelTrainer(
    model_name="ResNet152V2",
    img_shape=(224, 224, 1),  # Dimensiones de la imagen (altura, ancho, canales)
    n_classes=450,            # Número de clases
    dropout_rate=0.2,         # Regularización
    label_smoothing=0.1,      # Suavizado de etiquetas
    weights="imagenet",       # Pesos pre-entrenados
    model_dir="./models",     # Directorio para guardar modelos
    fine_tune_layers=200      # Número de capas para fine-tuning
)

# Entrenar el modelo
model, history = trainer.train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    learning_rate=1e-4,
    epochs=20
)
```

## Añadir nuevos modelos

Para añadir soporte a un nuevo modelo, simplemente actualiza el diccionario `AVAILABLE_MODELS` en la clase:

```python
# Ejemplo de cómo añadir un nuevo modelo
AVAILABLE_MODELS = {
    # Modelos existentes...
    "NuevoModelo": {
        "model": "tensorflow.keras.applications.NuevoModelo",
        "preprocess": "tensorflow.keras.applications.nuevo_modulo.preprocess_input"
    }
}
```

## Ejemplo de notebook

Para un ejemplo completo de uso, consulta el notebook `notebooks/ejemplo_model_trainer.ipynb`.
