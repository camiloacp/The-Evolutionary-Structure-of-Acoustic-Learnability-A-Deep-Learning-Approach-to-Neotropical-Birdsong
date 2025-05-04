# Proyecto de Clasificación de Aves mediante Sonidos

Este proyecto implementa un sistema completo para el análisis, procesamiento y clasificación de sonidos de aves utilizando técnicas avanzadas de aprendizaje profundo.

## Estructura del Proyecto

El proyecto está organizado en dos carpetas principales:

### 1. Carpeta `src`

Contiene los módulos de código Python con la implementación de las clases y funciones principales del sistema:

- **image_preprocessor.py**: Preprocesamiento y aumento de imágenes de espectrogramas.
- **model_trainer.py**: Entrenamiento de modelos utilizando diferentes arquitecturas de redes neuronales.
- **predictor.py**: Realización de predicciones con modelos entrenados.
- **spectograms.py**: Transformación de archivos de audio en espectrogramas.
- **taxonomia.py**: Manejo de la taxonomía de aves.
- **data_engineering.py**: Funciones para manipulación y transformación de datos.
- **gradcam.py**: Implementación de GradCAM para visualizar áreas de decisión del modelo.
- **incertidumbres.py**: Análisis de incertidumbres en las predicciones.

### 2. Carpeta `notebooks`

Contiene los cuadernos Jupyter (notebooks) que demuestran el flujo de trabajo completo:

- **00_Download data.ipynb**: Descarga de datos de audio de aves.
- **01_Paths.ipynb**: Configuración de rutas y directorios de trabajo.
- **02_Overview.ipynb**: Visión general del dataset y análisis exploratorio.
- **05_fine-tune-efficientnet-b7.ipynb**: Ajuste fino del modelo EfficientNetB7.
- **06_fine_tune_efficientnetv2_L.ipynb**: Ajuste fino del modelo EfficientNetV2L.
- **07_mobilenet_v3.ipynb**: Entrenamiento utilizando MobileNetV3.
- **08_resnet_v2_152.ipynb**: Entrenamiento utilizando ResNetV2-152.
- **09_GradCam.ipynb**: Visualización de mapas de activación con GradCAM.
- **10_Shap.ipynb**: Análisis de atribución de características con SHAP.
- **11_Incertidumbres.ipynb**: Evaluación de incertidumbres en las predicciones.

## Componentes Principales

### Preprocesamiento de Imágenes

El componente `ImagePreprocessor` (en `image_preprocessor.py`) permite:

- Cargar y transformar imágenes de espectrogramas.
- Aplicar técnicas de aumento de datos específicas para audio:
  - Time Masking: Enmascaramiento temporal de filas.
  - Frequency Masking: Enmascaramiento de frecuencias (columnas).
  - Ajustes de brillo, desenfoque, y enmascaramiento de cuadrícula.
- Generar datasets de TensorFlow optimizados para entrenamiento y validación.

```python
# Ejemplo de uso
from src.image_preprocessor import ImagePreprocessor, ImagePreprocessorConfig

# Configuración personalizada
config = ImagePreprocessorConfig(
    img_size=(128, 256),
    channels=1,
    batch_size=64,
    aug_proba=0.8
)

# Inicializar el preprocesador
preprocessor = ImagePreprocessor(config)

# Crear datasets para entrenamiento
train_ds = preprocessor.create_training_dataset(df_train)
val_ds = preprocessor.create_validation_dataset(df_val)
```

### Generación de Espectrogramas

El componente `SpectogramConfig` (en `spectograms.py`) permite:

- Convertir archivos de audio de aves en espectrogramas mel.
- Configurar parámetros de procesamiento de audio (tasa de muestreo, FFT, etc.).
- Procesar lotes de archivos en paralelo.
- Normalizar y guardar espectrogramas como imágenes JPEG.

```python
# Ejemplo de uso
from src.spectograms import SpectogramConfig

# Inicializar configuración
spectogram_generator = SpectogramConfig(
    audio_dir="data/audio",
    output_dir="data/spectograms",
    img_size=(224, 224),
    sample_rate=32000,
    seconds=5
)

# Procesar datos y generar espectrogramas
data = spectogram_generator.load_data()
data_with_duration = spectogram_generator.duration("audio_path")
spectogram_data, errors = spectogram_generator.process_data()
```

### Entrenamiento de Modelos

El componente `ModelTrainer` (en `model_trainer.py`) permite:

- Entrenar modelos de clasificación utilizando diferentes arquitecturas:
  - ResNet152V2
  - MobileNetV3Large
  - EfficientNetV2L
  - EfficientNetB7
- Configurar ajuste fino (fine-tuning) de capas específicas.
- Implementar técnicas de regularización (dropout, suavizado de etiquetas).
- Monitoreo del entrenamiento con callbacks optimizados.

```python
# Ejemplo de uso
from src.model_trainer import ModelTrainer

# Inicializar el entrenador con la arquitectura deseada
trainer = ModelTrainer(
    model_name="EfficientNetV2L",
    img_shape=(224, 224, 1),
    n_classes=667,
    dropout_rate=0.2,
    label_smoothing=0.1,
    weights="imagenet",
    fine_tune_layers=200
)

# Entrenar el modelo
model = trainer.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    learning_rate=1e-4,
    epochs=20
)
```

### Predicción

El componente `Predictor` (en `predictor.py`) permite:

- Cargar modelos entrenados para realizar predicciones.
- Procesar imágenes individuales o lotes para clasificación.
- Obtener las N clases más probables para cada predicción.
- Integrar el preprocesamiento con el proceso de predicción.

```python
# Ejemplo de uso
from src.predictor import Predictor

# Inicializar el predictor
predictor = Predictor(
    model_name="EfficientNetV2L",
    model_path="models/EfficientNetV2L_best.h5",
    img_shape=(224, 224, 1),
    n_classes=667
)

# Predecir clase para una imagen
class_id, probability = predictor.predict_single("path/to/spectogram.jpg")

# Obtener top-5 predicciones
top_predictions = predictor.predict_with_top_k("path/to/spectogram.jpg", k=5)
```

### Visualización y Explicabilidad

El proyecto incluye componentes avanzados para interpretación de modelos:

- **GradCAM** (en `gradcam.py`): Visualiza regiones importantes en el espectrograma que contribuyen a la predicción.
- **SHAP** (en el notebook `10_Shap.ipynb`): Análisis de atribución de características para entender el comportamiento del modelo.
- **Incertidumbres** (en `incertidumbres.py`): Métodos para cuantificar la incertidumbre en las predicciones, importante para aplicaciones críticas.

## Flujo de Trabajo

El proceso completo de trabajo con este proyecto sigue estos pasos:

1. **Preparación de Datos**:

   - Descargar archivos de audio de aves (`00_Download data.ipynb`)
   - Convertir audio en espectrogramas (`spectograms.py`)
   - Organizar imágenes en estructura de directorios

2. **Preprocesamiento**:

   - Configurar el preprocesador de imágenes (`image_preprocessor.py`)
   - Aplicar técnicas de aumento de datos
   - Crear datasets para entrenamiento y validación

3. **Entrenamiento**:

   - Seleccionar arquitectura y configurar hiperparámetros
   - Entrenar modelo con ajuste fino (notebooks 05-08)
   - Evaluar rendimiento y guardar mejor modelo

4. **Análisis y Explicabilidad**:

   - Visualizar regiones de decisión con GradCAM (notebook 09)
   - Analizar importancia de características con SHAP (notebook 10)
   - Evaluar incertidumbres en las predicciones (notebook 11)

5. **Predicción**:
   - Configurar el predictor con el modelo entrenado
   - Realizar predicciones en nuevos datos
   - Analizar resultados y probablidades

## Requisitos y Dependencias

El proyecto utiliza las siguientes bibliotecas principales:

- TensorFlow 2.x: Framework principal para modelos de aprendizaje profundo
- Keras: API de alto nivel para redes neuronales
- Librosa: Análisis de audio y extracción de características
- Numpy/Pandas: Manejo y procesamiento de datos
- Matplotlib: Visualización de datos
- OpenCV (cv2): Procesamiento de imágenes
- Scikit-learn: Herramientas de aprendizaje automático

## Uso del proyecto

Para utilizar este proyecto, se recomienda seguir el flujo de trabajo demostrado en los notebooks, comenzando con la descarga de datos y siguiendo el proceso hasta la predicción y análisis.

Los módulos en la carpeta `src` pueden importarse directamente para aplicaciones personalizadas o integrarse en otros proyectos de clasificación basados en sonido.

## Consideraciones y Limitaciones

- El rendimiento depende de la calidad de los espectrogramas generados.
- Las grabaciones de audio deben tener una duración mínima para generar espectrogramas adecuados.
- El entrenamiento de modelos como EfficientNetV2L requiere significativos recursos computacionales (GPU recomendada).
- Para aplicaciones en tiempo real, considerar modelos más ligeros como MobileNetV3Large.
