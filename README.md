[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18464327-blue)](https://doi.org/10.5281/zenodo.18464327)

# Bird Sound Classification and Evolution

Automatic classification system for **667 bird species** using convolutional neural networks on audio spectrograms, with comparative phylogenetic analysis.

## Project Overview

This project implements a complete pipeline for:

1. **Audio to mel spectrogram conversion** - Transforming bird recordings into visual representations
2. **Deep Learning classification** - Training CNNs (ResNet152V2, EfficientNet, MobileNet) with transfer learning
3. **Uncertainty analysis** - Confidence quantification using Monte Carlo Dropout
4. **Explainability** - Visualization of decision regions with GradCAM and SHAP
5. **Phylogenetic analysis** - Evaluation of factors influencing performance using PGLS/PGLMM

## Methodology

### Audio Processing

- **Input**: Audio files (.ogg, .mp3) of bird recordings
- **Conversion**: Mel spectrograms (128x224 pixels)
- **Parameters**: Sample rate 32kHz, frequency range 500-12500 Hz, 5-second segments

### Model Architecture

Four architectures were evaluated with transfer learning from ImageNet:

| Model           | Features                     |
| --------------- | ---------------------------- |
| ResNet152V2     | Best overall performance     |
| EfficientNetV2L | Accuracy/efficiency balance  |
| EfficientNetB7  | High capacity                |
| MobileNetV3     | Optimized for fast inference |

### Data Augmentation

Audio-specific techniques:

- **Time Masking**: Temporal band masking
- **Frequency Masking**: Frequency band masking
- GridMask, brightness adjustments, Gaussian blur

### Phylogenetic Analysis

Integration with AVONET morphological data to analyze which factors influence species classifiability:

- **PGLS** (Phylogenetic Generalized Least Squares)
- **PGLMM** (Phylogenetic Generalized Linear Mixed Models)

## Key Findings

1. **Performance**: The ResNet152V2 model classifies 667 bird species with detailed precision, recall, and F1-score metrics per species

2. **Uncertainty**: Monte Carlo Dropout (2000 samples) enables prediction confidence quantification with 95% confidence intervals

3. **Taxonomic confusion**: Species confusion is correlated with phylogenetic proximity - species from the same genus or family tend to be confused more often

4. **Morphological traits**: PGLS/PGLMM analysis reveals which morphological features (AVONET) are associated with better or worse classifiability

5. **Explainability**: GradCAM shows that the model focuses on species-specific temporal and frequency patterns

## Project Structure

```
birds-sounds/
├── src/                          # Main modules
│   ├── spectograms.py            # Spectrogram generation
│   ├── image_preprocessor.py     # Preprocessing and data augmentation
│   ├── model_trainer.py          # Model training
│   ├── predictor.py              # Predictions
│   ├── gradcam.py                # GradCAM visualization
│   ├── incertidumbres.py         # Monte Carlo analysis
│   ├── data_engineering.py       # Audio feature extraction
│   ├── data_preprocessor.py      # DataFrame preprocessing
│   ├── taxonomia.py              # Taxonomic mappings
│   ├── analisis_confusion.py     # Confusion analysis
│   └── data/                     # Data and results
│       ├── AVONET.xlsx           # Morphological traits
│       ├── reporte_resnet.csv    # Per-species metrics
│       └── pgls_*.csv            # Phylogenetic analysis results
├── notebooks/                    # Complete workflow
│   ├── 00-04                     # Data preparation
│   ├── 05-08                     # Model training
│   ├── 09-11                     # Explainability and uncertainty
│   └── 12-18                     # Phylogenetic analysis
└── fig/                          # Generated visualizations
```

## Usage

### Installation

```bash
# Clone repository
git clone <repo-url>
cd birds-sounds

# Install dependencies (requires Python >= 3.11.11)
pip install -e .
# or with uv
uv sync
```

### Generate Spectrograms

```python
from src.spectograms import SpectogramConfig

config = SpectogramConfig(
    audio_dir="data/audio",
    output_dir="data/spectograms",
    img_size=(128, 224),
    sample_rate=32000,
    seconds=5
)
data, errors = config.process_data()
```

### Train Model

```python
from src.model_trainer import ModelTrainer
from src.image_preprocessor import ImagePreprocessor, ImagePreprocessorConfig

# Prepare data
preprocessor = ImagePreprocessor(ImagePreprocessorConfig(img_size=(128, 224)))
train_ds = preprocessor.create_training_dataset(df_train)
val_ds = preprocessor.create_validation_dataset(df_val)

# Train
trainer = ModelTrainer(
    model_name="ResNet152V2",
    img_shape=(128, 224, 1),
    n_classes=667,
    fine_tune_layers=100
)
model = trainer.train(train_ds, val_ds, epochs=20)
```

### Make Predictions

```python
from src.predictor import Predictor

predictor = Predictor(
    model_name="ResNet152V2",
    model_path="models/resnet_best.h5",
    n_classes=667
)
class_id, probability = predictor.predict_single("spectrogram.jpg")
```

### Uncertainty Analysis

```python
from src.incertidumbres import calcular_incertidumbre, plot_combinado

# Get prediction distribution (2000 MC samples)
results = calcular_incertidumbre(model, image, n_samples=2000)

# Visualize
plot_combinado(image, results, true_class, labels)
```

### GradCAM Visualization

```python
from src.gradcam import GradCAM

gradcam = GradCAM(model)
heatmap = gradcam.generate_heatmap(image)
```

## Notebook Workflow

| Notebook | Description                                      |
| -------- | ------------------------------------------------ |
| 00-02    | Data download and exploratory analysis           |
| 03-04    | Spectrogram generation and organization          |
| 05-08    | Model training (EfficientNet, MobileNet, ResNet) |
| 09       | GradCAM visualization                            |
| 10       | SHAP analysis                                    |
| 11       | Uncertainty quantification                       |
| 12-13    | Feature engineering and AVONET data              |
| 14-15    | Phylogenetic tree and QuaSSE analysis            |
| 16       | Multimodal model (audio + morphology)            |
| 17-18    | Visualizations and PGLS/PGLMM analysis           |

## Main Dependencies

- TensorFlow >= 2.19
- Librosa >= 0.11 (audio processing)
- Keras-CV >= 0.9 (data augmentation)
- Scikit-learn >= 1.6
- Pandas, NumPy, Matplotlib, Seaborn

## Requirements

- Python >= 3.11.11
- GPU recommended for training (CUDA compatible)
- ~25GB disk space for complete data
