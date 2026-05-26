# ECG Classification by AI — 1D CNN on MIT-BIH

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Dataset-MIT--BIH-005A9C?style=for-the-badge&logo=databricks&logoColor=white" alt="MIT-BIH"/>
  <img src="https://img.shields.io/badge/Accuracy-91.3%25-2ECC71?style=for-the-badge&logo=checkmarx&logoColor=white" alt="Accuracy 91.3%"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License MIT"/>
</p>

> **Automated ECG arrhythmia classification using a 1D Convolutional Neural Network trained on the MIT-BIH Arrhythmia Database, with inference pipeline for real-time signals acquired by an embedded STM32L476RG cardiograph.**

---

## Table of Contents

- [Overview](#overview)
- [Hardware Context](#hardware-context)
- [Installation](#installation)
- [Workflow](#workflow)
  - [Step 1 — Download MIT-BIH](#step-1--download-mit-bih)
  - [Step 2 — Prepare the dataset](#step-2--prepare-the-dataset-prepare_datapy)
  - [Step 3 — Train the model](#step-3--train-the-model-trainpy)
  - [Step 4 — Classify a STM32 recording](#step-4--classify-a-stm32-recording-classify_stm32py)
- [Model Architecture](#model-architecture)
- [AAMI Beat Classes](#aami-beat-classes)
- [Repository Structure](#repository-structure)
- [Authors](#authors)

---

## Overview

This project implements an end-to-end pipeline for ECG arrhythmia detection:

1. **Training data**: 48 recordings from the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) (~109 000 labelled beats at 360 Hz).
2. **Model**: A compact 1D CNN trained to classify beats into 5 AAMI categories (N / S / V / F / Q).
3. **Inference**: A dedicated script resamples CSV files exported by the embedded STM32 acquisition system (1000 Hz → 360 Hz) and runs the classifier in a sliding-window fashion.

**Achieved accuracy: ~91.3 % on the MIT-BIH test set (80/20 stratified split).**

---

## Hardware Context

The classifier is designed to operate on signals recorded by a custom embedded system:

| Component | Description |
|-----------|-------------|
| MCU | STM32L476RG |
| Acquisition | Analog front-end → 12-bit ADC @ 1 000 Hz |
| Export | CSV file on SD card (with GPS timestamp) |

The embedded firmware (acquisition, display, GPS, SD) is maintained in a separate repository:  
**[Projet-ECG-STM32L476RG — acquisition, affichage, GPS, SD](https://github.com/YanisMeziane817/Projet-ECG-STM32L476RG---acquisition-affichage-GPS-SD)**

---

## Installation

> **Python 3.11 is required.**  
> Python 3.13 is **not compatible** with TensorFlow 2.x.

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv ecg_venv
# Windows
ecg_venv\Scripts\activate
# Linux / macOS
source ecg_venv/bin/activate

# 2. Install dependencies
pip install tensorflow wfdb numpy pandas scikit-learn matplotlib scipy
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

---

## Workflow

### Step 1 — Download MIT-BIH

1. Create a free account at <https://physionet.org>
2. Go to <https://physionet.org/content/mitdb/1.0.0/>
3. Click **"Download the ZIP file"** (≈ 108 MB)
4. Extract the archive so that `.hea`, `.dat` and `.atr` files are directly inside a `mitdb/` folder at the project root:

```
mitdb/
├── 100.hea
├── 100.dat
├── 100.atr
├── 101.hea
└── ...
```

---

### Step 2 — Prepare the dataset (`prepare_data.py`)

```bash
cd src
python prepare_data.py
```

The script reads all 48 MIT-BIH records, extracts individual beats using R-peak annotations, maps them to AAMI classes, normalises each beat, and saves train/test splits as NumPy arrays.

**Expected output:**

```
Extraction des battements MIT-BIH...

Total battements extraits : 109494
Distribution :
  F :    803  (0.7%)
  N :  90588 (82.7%)
  Q :   8088  (7.4%)
  S :   2779  (2.5%)
  V :   7236  (6.6%)

Train : (87595, 36), Test : (21899, 36)
Classes : ['F', 'N', 'Q', 'S', 'V']

Dataset sauvegardé dans dataset/
Prêt pour l'entraînement → lancer train.py
```

**Generated files:**

```
dataset/
├── X_train.npy   # (87595, 36) — training beats
├── X_test.npy    # (21899, 36) — test beats
├── y_train.npy   # integer labels
├── y_test.npy
└── classes.npy   # ['F', 'N', 'Q', 'S', 'V']
```

---

### Step 3 — Train the model (`train.py`)

```bash
python train.py
```

Training runs for up to 50 epochs with early stopping (patience = 5). Estimated runtime: **10–20 min on CPU**, 2–5 min on GPU.

**Expected output:**

```
Epoch  1/50: loss=1.4200 acc=0.3800 val_acc=0.4100
Epoch 10/50: loss=0.5500 acc=0.8100 val_acc=0.7800
Epoch 20/50: loss=0.2100 acc=0.9300 val_acc=0.9100
Early stopping at epoch 24

========================================
  Test accuracy : 91.30%
  Test loss     : 0.2841
========================================

Classification report:
              precision    recall  f1-score
           F       0.72      0.68      0.70
           N       0.96      0.97      0.96
           Q       0.94      0.91      0.92
           S       0.73      0.69      0.71
           V       0.93      0.90      0.91
```

**Generated files:**

```
model/
├── ecg_cnn.h5           # final model
├── ecg_cnn_best.h5      # best checkpoint (val_accuracy)
├── training_curves.png  # loss & accuracy curves
├── confusion_matrix.png # per-class confusion matrix
└── results.json         # accuracy, loss, epochs
```

---

### Step 4 — Classify a STM32 recording (`classify_stm32.py`)

Place your CSV file (exported from the STM32 SD card) anywhere accessible, then run:

```bash
python classify_stm32.py --csv ECG_14_22_05_2026.CSV
```

**Expected CSV format:**

```
heure_utc;latitude;longitude;altitude_m
143022 UTC 22/05/2026;47.729168;7.311351;254.3
index;raw;filtered
0;2007;2007
1;2011;2010
...
```

**Expected output:**

```
=======================================================
  Classification ECG — ECG_14_22_05_2026.CSV
=======================================================

  Données GPS : Heure=143022 UTC 22/05/2026 | LAT=47.729168 | LON=7.311351 | ALT=254.3m
  Signal      : 10000 samples à 1000Hz (10.0s)
  Fenêtres    : 55 fenêtres de 36 samples

─────────────────────────────────────────────────────
  RÉSULTATS DE CLASSIFICATION
─────────────────────────────────────────────────────
  N (Normal                        ) :  87.3% ████████████████████████████
  S (Supraventriculaire (SVEB)     ) :   8.2% ██
  V (Ventriculaire (VEB)           ) :   4.5% █
  F (Fusion                        ) :   0.0%
  Q (Inconnu / Paced               ) :   0.0%

─────────────────────────────────────────────────────
  DIAGNOSTIC DOMINANT : N — Normal
  Présence             : 87.3% des fenêtres
  Confiance moyenne    : 92.1%
=======================================================

  Graphique sauvegardé : ECG_14_22_05_2026_classification.png
```

The script also saves a two-panel figure showing the raw ECG signal and the class distribution bar chart.

---

## Model Architecture

The 1D CNN processes each beat as a time series of 36 samples (resampled from the original 360-sample window):

```
Input: (36, 1)
│
├─ Conv1D(32, k=5, ReLU, same) ─ BatchNorm ─ MaxPool(2) ─ Dropout(0.25)
│       ↳ detects fine-grained patterns (R peaks, P waves)
│
├─ Conv1D(64, k=5, ReLU, same) ─ BatchNorm ─ MaxPool(2) ─ Dropout(0.25)
│       ↳ captures complex morphologies (QRS complex)
│
├─ Conv1D(128, k=3, ReLU, same) ─ BatchNorm ─ GlobalAveragePooling1D
│       ↳ high-level abstraction
│
├─ Dense(64, ReLU) ─ Dropout(0.4)
│
└─ Dense(5, Softmax)   →   [F, N, Q, S, V]
```

| Layer | Output Shape | Parameters |
|-------|-------------|-----------|
| Conv1D(32, 5) | (36, 32) | 192 |
| BatchNorm | (36, 32) | 128 |
| MaxPool(2) | (18, 32) | — |
| Conv1D(64, 5) | (18, 64) | 10 304 |
| BatchNorm | (18, 64) | 256 |
| MaxPool(2) | (9, 64) | — |
| Conv1D(128, 3) | (9, 128) | 24 704 |
| BatchNorm | (9, 128) | 512 |
| GlobalAvgPool | (128,) | — |
| Dense(64) | (64,) | 8 256 |
| Dense(5) | (5,) | 325 |
| **Total** | | **≈ 44 677** |

**Training settings:**

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | Adam (lr = 0.001) |
| Loss | Sparse categorical cross-entropy |
| Batch size | 64 |
| Max epochs | 50 |
| Early stopping | patience = 5 on val_loss |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Validation split | 20 % of training set |

---

## AAMI Beat Classes

The model follows the **ANSI/AAMI EC57** standard for arrhythmia classification:

| Class | Full name | MIT-BIH symbols included | Clinical severity |
|-------|-----------|--------------------------|------------------|
| **N** | Normal | N, L, R, e, j | None |
| **S** | Supraventricular ectopic beat (SVEB) | A, a, J, S | Low |
| **V** | Ventricular ectopic beat (VEB) | V, E | Moderate to high |
| **F** | Fusion beat (N + V) | F | Moderate |
| **Q** | Unknown / Paced | /, f, Q | Variable |

> Classes **V** and **S** are clinically most significant; their under-representation in MIT-BIH (6.6 % and 2.5 % respectively) motivates careful evaluation of recall scores in addition to overall accuracy.

---

## Repository Structure

```
ecg-ia-classification/
│
├── src/
│   ├── prepare_data.py       # MIT-BIH → NumPy dataset (beats + labels)
│   ├── train.py              # CNN 1D training + evaluation + plots
│   └── classify_stm32.py     # Inference on STM32 CSV recordings
│
├── mitdb/                    # MIT-BIH raw files (.hea .dat .atr) — NOT in repo
│   └── ...                   # Download from physionet.org
│
├── dataset/                  # Generated by prepare_data.py — NOT in repo
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
│   └── classes.npy
│
├── model/                    # Generated by train.py — NOT in repo
│   ├── ecg_cnn.h5
│   ├── ecg_cnn_best.h5
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── results.json
│
├── Cours & explications/     # Educational HTML documents
│   ├── cours_cnn_complet_ecg.html
│   ├── classes_ecg_explication.html
│   └── etat_art_simple.html
│
├── requirements.txt
└── README.md
```

---

## Related Project

This AI module is part of a larger embedded system project.  
The STM32 firmware (ECG acquisition, ILI9341 display, GPS parsing, SD card logging) is available here:

**[Projet-ECG-STM32L476RG — acquisition, affichage, GPS, SD](https://github.com/YanisMeziane817/Projet-ECG-STM32L476RG---acquisition-affichage-GPS-SD)**

---

## Authors

This project was developed as part of a 2nd-year engineering school project (ASE — Architecture des Systèmes Embarqués) at ENSISA.

| Name | GitHub |
|------|--------|
| Yanis Meziane | [@YanisMeziane817](https://github.com/YanisMeziane817) |
| Evan Legland | [@Evan-Legd](https://github.com/Evan-Legd) |

---

## License

This project is licensed under the MIT License.  
MIT-BIH Arrhythmia Database is provided by PhysioNet under the [ODC Attribution License](https://physionet.org/content/mitdb/1.0.0/).
