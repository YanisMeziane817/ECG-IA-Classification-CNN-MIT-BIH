"""
classify_stm32.py
=================
Charge un fichier CSV produit par le STM32 et classifie le signal ECG
avec le modèle CNN 1D entraîné sur MIT-BIH.

UTILISATION :
    python classify_stm32.py --csv ECG_14_22_05_2026.CSV

FORMAT CSV ATTENDU :
    heure_utc;latitude;longitude;altitude_m
    143022 UTC 22/05/2026;47.729168;7.311351;254.3
    index;raw;filtered
    0;2007;2007
    1;2011;2010
    ...
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.signal import resample
import matplotlib.pyplot as plt
import argparse, os
from collections import Counter

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_PATH   = 'model/ecg_cnn.h5'
CLASSES_PATH = 'dataset/classes.npy'
STM32_FS     = 1000   # fréquence STM32 (Hz)
MITBIH_FS    = 360    # fréquence MIT-BIH (Hz)
WINDOW_STM32 = 1000   # fenêtre STM32 : 1 seconde
WINDOW_MITBIH = 36   # fenêtre MIT-BIH après rééchantillonnage
ADC_BASELINE  = 2000  # baseline ADC STM32

CLASSES_FULL = {
    'N': 'Normal',
    'S': 'Supraventriculaire (SVEB)',
    'V': 'Ventriculaire (VEB)',
    'F': 'Fusion',
    'Q': 'Inconnu / Paced'
}

# ─── CHARGEMENT CSV STM32 ─────────────────────────────────────────────────────
def load_stm32_csv(csv_path):
    """Charge le CSV STM32 et retourne le signal filtré + métadonnées GPS."""

    # Lire les 2 premières lignes pour les métadonnées GPS
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    gps_info = "N/A"
    if len(lines) >= 2 and ';' in lines[1]:
        parts = lines[1].strip().split(';')
        if len(parts) >= 4:
            gps_info = (f"Heure={parts[0]} | "
                        f"LAT={parts[1]} | LON={parts[2]} | ALT={parts[3]}m")

    # Charger les données ECG (à partir de la ligne 3, après le header GPS)
    df = pd.read_csv(csv_path, sep=';', skiprows=2, header=0)
    signal = df['filtered'].values.astype(np.float32)

    return signal, gps_info

# ─── PRÉTRAITEMENT ────────────────────────────────────────────────────────────
def preprocess_signal(signal):
    """Centre, rééchantillonne, découpe en fenêtres."""

    # 1. Centrer le signal (retirer la baseline ADC)
    signal = signal - ADC_BASELINE

    # 2. Rééchantillonner 1000Hz → 360Hz
    n_samples_360 = int(len(signal) * MITBIH_FS / STM32_FS)
    signal_360 = resample(signal, n_samples_360)

    # 3. Découper en fenêtres de 360 samples (overlap 50%)
    step = WINDOW_MITBIH // 2
    windows = []
    for i in range(0, len(signal_360) - WINDOW_MITBIH, step):
        w = signal_360[i:i + WINDOW_MITBIH].astype(np.float32)
        # Normalisation par fenêtre
        w = (w - w.mean()) / (w.std() + 1e-8)
        windows.append(w)

    return np.array(windows)[..., np.newaxis]  # (N, 360, 1)

# ─── CLASSIFICATION ───────────────────────────────────────────────────────────
def classify(csv_path):
    print(f"\n{'='*55}")
    print(f"  Classification ECG — {os.path.basename(csv_path)}")
    print(f"{'='*55}")

    # Charger
    signal, gps_info = load_stm32_csv(csv_path)
    print(f"\n  Données GPS : {gps_info}")
    print(f"  Signal      : {len(signal)} samples à {STM32_FS}Hz "
          f"({len(signal)/STM32_FS:.1f}s)")

    # Prétraiter
    X = preprocess_signal(signal)
    print(f"  Fenêtres    : {len(X)} fenêtres de {WINDOW_MITBIH} samples")

    # Charger modèle
    model   = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)

    # Prédire
    print(f"\n  Classification en cours...")
    preds  = model.predict(X, verbose=0)
    labels = np.argmax(preds, axis=1)
    confs  = np.max(preds, axis=1)

    # Résultats par classe
    counts = Counter(labels)
    total  = len(labels)

    print(f"\n{'─'*45}")
    print(f"  RÉSULTATS DE CLASSIFICATION")
    print(f"{'─'*45}")
    for i, cls in enumerate(classes):
        count = counts.get(i, 0)
        pct   = count / total * 100
        bar   = '█' * int(pct / 3)
        print(f"  {cls} ({CLASSES_FULL.get(cls, cls):30s}) : "
              f"{pct:5.1f}% {bar}")

    # Diagnostic dominant
    dominant_idx = max(counts, key=counts.get)
    dominant_cls = classes[dominant_idx]
    dominant_pct = counts[dominant_idx] / total * 100
    mean_conf    = confs.mean() * 100

    print(f"\n{'─'*45}")
    print(f"  DIAGNOSTIC DOMINANT : {dominant_cls} — "
          f"{CLASSES_FULL.get(dominant_cls, dominant_cls)}")
    print(f"  Présence             : {dominant_pct:.1f}% des fenêtres")
    print(f"  Confiance moyenne    : {mean_conf:.1f}%")
    print(f"{'='*55}\n")

    # ─── GRAPHIQUE ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Classification ECG STM32 — {os.path.basename(csv_path)}',
                 fontsize=13, fontweight='bold')

    # Signal brut
    t = np.arange(len(signal)) / STM32_FS
    axes[0].plot(t, signal - ADC_BASELINE, color='#378ADD', linewidth=0.8)
    axes[0].set_title('Signal ECG (filtré, centré)')
    axes[0].set_xlabel('Temps (s)')
    axes[0].set_ylabel('Amplitude (ADC - 2000)')
    axes[0].grid(True, alpha=0.3)

    # Résultats classification
    cls_labels = [f"{c}\n{CLASSES_FULL.get(c,'')}" for c in classes]
    cls_counts = [counts.get(i, 0) / total * 100 for i in range(len(classes))]
    colors = ['#1D9E75' if classes[i] == dominant_cls else '#E6F1FB'
              for i in range(len(classes))]
    bars = axes[1].bar(cls_labels, cls_counts, color=colors,
                       edgecolor='#0C447C', linewidth=0.5)
    for bar, val in zip(bars, cls_counts):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    axes[1].set_title('Distribution des classes détectées')
    axes[1].set_ylabel('Pourcentage des fenêtres (%)')
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_fig = csv_path.replace('.CSV', '_classification.png').replace('.csv', '_classification.png')
    plt.savefig(out_fig, dpi=150)
    print(f"  Graphique sauvegardé : {out_fig}")
    plt.show()

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classification ECG STM32')
    parser.add_argument('--csv', required=True, help='Fichier CSV STM32')
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"ERREUR : modèle '{MODEL_PATH}' introuvable.")
        print("Lance d'abord : python train.py")
        exit(1)

    classify(args.csv)
