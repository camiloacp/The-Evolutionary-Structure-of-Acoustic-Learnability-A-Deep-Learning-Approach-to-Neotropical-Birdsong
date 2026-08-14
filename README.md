[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21911159.svg)](https://doi.org/10.5281/zenodo.21911159)

# The Evolutionary Structure of Acoustic Learnability

**A Deep Learning Approach to Neotropical Birdsong**

This repository contains the complete code, notebooks, and data products behind the paper
*"The Evolutionary Structure of Acoustic Learnability: A Deep Learning Approach to Neotropical Birdsong"*
(Cortés-Parra, Hortúa & Ríos-Orjuela). It implements an end-to-end pipeline that:

1. **Builds a large-scale bioacoustic dataset** — 215,188 mel spectrograms from Xeno-Canto recordings of **667 Neotropical bird species**.
2. **Trains and benchmarks four CNN architectures** with transfer learning (EfficientNetV2L, EfficientNetB7, MobileNetV3Large, ResNet152V2).
3. **Quantifies predictive uncertainty** with Monte Carlo Dropout and explains model decisions with Grad-CAM and SHAP.
4. **Benchmarks against BirdNET v2.4** on a recording-level sample drawn from the held-out test partition.
5. **Repurposes classification performance as a biological signal**: species-level F1-scores are analyzed with phylogenetically aware comparative methods (Pagel's λ, PGLS, Beta-PGLMM) to test macroevolutionary hypotheses about acoustic distinctiveness.

## Key Results

| Result | Value |
| --- | --- |
| Best architecture | **EfficientNetV2L** |
| Test accuracy (32,279 held-out segments, 667 species) | **94.48%** |
| Macro F1-score | **94.30%** |
| Macro ROC-AUC / PR-AUC (test set) | 0.998 / 0.972 |
| vs. BirdNET v2.4 (11,695 recordings, 664 shared species) | accuracy **0.957 vs. 0.903**, macro F1 **0.952 vs. 0.900** |
| Phylogenetic signal of learnability (Pagel's λ) | 0.159 (*p* < 0.001) |
| Variance explained by best trait model (Geographic PGLS) | ~2.2% — broad biological predictors explain little interspecific variation |

## Repository Structure

```
birds-sounds/
├── src/                    # Reusable Python modules (the pipeline core)
│   ├── spectograms.py          # Audio → mel-spectrogram generation
│   ├── image_preprocessor.py   # Image loading, tf.data pipelines, data augmentation
│   ├── model_trainer.py        # Architecture builder + fine-tuning loop
│   ├── predictor.py            # Single/batch inference utilities
│   ├── incertidumbres.py       # Monte Carlo Dropout uncertainty estimation
│   ├── gradcam.py              # Grad-CAM heatmaps
│   ├── analisis_confusion.py   # Confusion-matrix analysis helpers
│   ├── birdnet_utils.py        # BirdNET interoperability helpers
│   ├── run_birdnet_inference.py# BirdNET batch inference
│   ├── data_engineering.py     # Acoustic feature extraction (librosa)
│   ├── data_preprocessor.py    # Tabular data preprocessing
│   ├── taxonomia.py            # Taxonomic mappings and label alignment
│   ├── viz_style.py            # Shared plotting style
│   └── data/                   # Data products (metrics, predictions, results)
├── notebooks/              # Numbered, end-to-end analysis workflow (see guide below)
├── fig/                    # Generated figures
├── data/                   # External/sample data
├── pyproject.toml          # Dependencies (uv / pip)
└── main.py
```

## Notebook Guide

The notebooks are numbered in workflow order. Everything reported in the paper can be
traced back to one of them.

### Stage 1 — Dataset construction (00–04)

| Notebook | What it does |
| --- | --- |
| `00_Download data.ipynb` | Downloads quality-"A" recordings from Xeno-Canto, organized by family. |
| `01_Paths.ipynb` | Builds the catalog of audio file paths used by later stages. |
| `02_Overview.ipynb` | Exploratory overview of the corpus: recordings per family/genus/species. |
| `03_image_spectogram_creation.ipynb` | Converts audio to mel spectrograms: 5-s non-overlapping segments (max 24 per file), 32 kHz, FFT 2048, 128 Mel bands, hop 619, 500–12,500 Hz, dB scale relative to segment peak, min–max normalized to 128×256 single-channel JPEGs. |
| `04_images_paths.ipynb` | Catalogs the spectrogram images and organizes the train/validation/test file lists. The split is stratified by species (70/15/15) and performed **at the recording level**, so segments of one audio file never appear in more than one partition. |

### Stage 2 — Model training (05–08)

All four architectures share the same protocol: ImageNet-pretrained backbone, 1→3 channel
Lambda projection, global average pooling, classification head `Dropout(0.2) → Dense(667)`,
top 200 layers unfrozen, Adam (lr = 1e-4), categorical cross-entropy with label smoothing
(α = 0.1), ModelCheckpoint + EarlyStopping + ReduceLROnPlateau, max 100 epochs.
Trained on Google Colab with NVIDIA A100 GPUs.

| Notebook | Model |
| --- | --- |
| `05_fine-tune-efficientnet-b7.ipynb` | EfficientNetB7 |
| `06_fine_tune_efficientnetv2_L.ipynb` | **EfficientNetV2L** (best-performing; reference model for all downstream analyses) |
| `07_mobilenet_v3.ipynb` | MobileNetV3Large |
| `08_resnet_v2_152.ipynb` | ResNet152V2 |

### Stage 3 — Explainability and uncertainty (09–11)

| Notebook | What it does |
| --- | --- |
| `09_GradCam.ipynb` | Grad-CAM activation maps over the last convolutional layer (e.g., *Acropternis orthonyx*, *Anthus rubescens* case studies in the paper). |
| `10_Shap.ipynb` | SHAP-based feature attribution (exploratory). |
| `11_Incertidumbres_base.ipynb` | Base Monte Carlo Dropout run: rebuilds the architecture, loads the trained weights, and performs 2,000 stochastic forward passes per sample with the dropout layer active. **Note:** MC Dropout inference uses dropout rate 0.3, while training used 0.2 (the rate is a layer configuration, not a trained weight). |
| `11_Incertidumbres.ipynb` | Species-level uncertainty profiles and case studies (*Volatinia jacarina*, *Mimus gilvus*, *Riparia riparia*). |

### Stage 4 — Traits, phylogeny, and comparative data (12–17)

| Notebook | What it does |
| --- | --- |
| `12_Data_engineering.ipynb` | Acoustic feature extraction with librosa (exploratory). |
| `13_Data_avonet.ipynb` | Merges model metrics with the AVONET trait database; outlier treatment and documentation of acoustic features. |
| `14_tree.ipynb` | **(R)** Phylogenetic tree handling with `phytools`: pruning the BirdTree phylogeny (Jetz et al.) to the study species, MCC tree preparation. |
| `15_QuaSSE.ipynb` | **(R)** QuaSSE models with `diversitree` (exploratory; not reported in the paper). |
| `16_Modelo_avonet.ipynb` | Joins per-species model metrics with AVONET traits; EDA and correlation analysis. |
| `17_plots_performance_model.ipynb` | Performance visualizations (per-species F1 distributions, training curves). |

### Stage 5 — Phylogenetic comparative analysis (18–19)

| Notebook | What it does |
| --- | --- |
| `18_PGLS_analysis.ipynb` | Core comparative analysis. Pagel's λ; PGLS on logit-transformed F1; custom `PhyloBetaRegression`; archived AIC calculations include a Jacobian correction for the transformed PGLS response. In the manuscript, model selection is interpreted within PGLS and PGLMM separately rather than using absolute cross-framework AIC differences to rank the two modelling approaches. |
| `19_confusion-matrix.ipynb` | Confusion-matrix construction and family-level aggregation; engineering characterization of the four architectures (parameters, FLOPs at 128×256, latency/throughput). |

### Stage 6 — BirdNET benchmark (20–23)

The comparison uses a **random recording-level sample of 11,695 recordings drawn from the
held-out test partition**, restricted to the 664 study species covered by BirdNET.
Our model: argmax of softmax probabilities averaged over all 5-s segments of a recording.
BirdNET: maximum top-1 score over its 3-s windows.

| Notebook | What it does |
| --- | --- |
| `20_birdnet_comparison.ipynb` | Runs BirdNET v2.4 inference over the evaluation sample (uses the separate `.venv-birdnet` environment). |
| `21_resnet_recording_eval_and_compare.ipynb` | Paired recording-level evaluation of both models on the same recordings; global and per-species comparison tables. |
| `22_birdnet_vs_resnet_insights.ipynb` | Visual comparison and confidence analysis. **Important:** the ROC-AUC/PR-AUC values computed here (0.967/0.881, 0.998/0.984) use top-1 confidence as a discriminator of *correct vs. incorrect* predictions — they are confidence-separation metrics, **not** macro one-vs-rest classification AUCs. |
| `23_reporte_efficientnetv2l.ipynb` | Final per-species classification report for EfficientNetV2L (precision, recall, F1, support for all 667 species). |

## Data Products (`src/data/`)

Selected outputs referenced by the paper:

| File | Contents |
| --- | --- |
| `reporte_efficientnetv2l.csv` | Per-species precision/recall/F1/support for EfficientNetV2L from the archived single-pass stochastic evaluation used in the comparative analyses (Table S7). |
| `reporte_mc_efficientnetv2l.csv` | Same, computed from the MC Dropout predictive mean. |
| `reporte_resnet.csv`, `reporte_mc_resnet.csv` | Equivalent reports for ResNet152V2. |
| `engineering_characterization.csv`, `inference_time_benchmark.csv` | Parameters, FLOPs, latency and throughput per architecture (Table S2, Fig. S1). |
| `pgls_*.csv`, `feature_importance_pglmm_pgls.csv` | PGLS/PGLMM model comparison, species-level results, variable importance. |
| `bayes_confidence_matrix_*.npz`, `bayes_uncertainty_per_species_*.csv` | Corpus-level MC Dropout predictive-confidence matrices and per-species uncertainty (Figs. S4–S5). |
| `birdnet_*.csv`, `birdnet_vs_*` | BirdNET predictions and the paired comparison tables (Table S1, Fig. S6). |
| `*_incertidumbres.csv` / `.pkl` | MC Dropout predictive distributions for the case-study species. |
| `AVONET.xlsx` | Trait database (Tobias et al. 2022). |

## Installation and Usage

Requires **Python ≥ 3.11.11**. A CUDA-capable GPU is recommended for training
(the paper's experiments ran on Google Colab with NVIDIA A100 GPUs).

```bash
git clone https://github.com/camiloacp/birds-sounds.git
cd birds-sounds

# with uv (recommended)
uv sync

# or with pip
pip install -e .
```

Main dependencies: TensorFlow ≥ 2.19, keras-cv, librosa, scikit-learn, pandas,
opencv-python, seaborn. Notebooks 14–15 additionally require **R** with `phytools`,
`tidyverse`, and `diversitree`. BirdNET inference (notebook 20) runs in the separate
`.venv-birdnet` environment.

### Minimal example

```python
from src.model_trainer import ModelTrainer

trainer = ModelTrainer(
    model_name="EfficientNetV2L",
    img_shape=(128, 256, 1),
    n_classes=667,
    dropout_rate=0.2,        # training configuration
    label_smoothing=0.1,
    fine_tune_layers=200,
)
model = trainer.create_model()
```

```python
from src.incertidumbres import calcular_incertidumbre

# Monte Carlo Dropout: 2,000 stochastic forward passes
results = calcular_incertidumbre(model, image, n_samples=2000)
```

## Methodological Notes

- **Split leakage control:** the train/validation/test split is performed at the
  recording-file level, never at the segment level. Crowdsourced repositories such as
  Xeno-Canto lack persistent individual-bird identifiers, so recordings of the same
  individual may still appear across splits; this is discussed as a limitation in the paper.
- **Dropout and inference:** all four architectures were trained with dropout rate 0.2. The
  archived final EfficientNetV2L species-level report (notebook 23) was generated after
  reconstructing the classification head with dropout rate 0.3. Because
  `ModelTrainer.create_model()` fixes the classification-head Dropout layer with
  `training=True`, these primary EfficientNetV2L predictions represent a single stochastic
  realization rather than deterministic inference. The separate Monte Carlo Dropout analyses
  use dropout rate 0.3 and aggregate 2,000 stochastic forward passes. The manuscript reports
  this distinction explicitly.
- **Boundary handling in comparative models:** F1 = 1 values (26 species) are clipped to
  [1e-6, 1 − 1e-6] for the Beta-PGLMM and to [0.001, 0.999] before the logit transform
  for PGLS.
- **Comparative subset:** phylogenetic analyses use the 583 species with complete
  phylogenetic + trait data; the deep learning benchmark uses all 667; the BirdNET
  comparison uses the 664 species covered by BirdNET.

## Citation

If you use this code or data, please cite the paper and the archived release:

> Cortés-Parra, C. A., Hortúa, H. J., & Ríos-Orjuela, J. C. (2026).
> *The Evolutionary Structure of Acoustic Learnability: A Deep Learning Approach to
> Neotropical Birdsong.* Code and data: https://doi.org/10.5281/zenodo.21911159
