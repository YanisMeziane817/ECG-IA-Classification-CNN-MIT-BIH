"""
prepare_data.py
===============
Charge MIT-BIH depuis le dossier local et prépare X_train, X_test, y_train, y_test

UTILISATION :
    1. Télécharger MIT-BIH : https://physionet.org/content/mitdb/1.0.0/
       (créer un compte gratuit PhysioNet, puis télécharger le ZIP)
    2. Extraire dans un dossier 'mitdb/'
    3. Lancer : python prepare_data.py
"""

import wfdb
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MITDB_PATH   = './mitdb'      # dossier contenant les fichiers .hea .dat .atr
OUTPUT_PATH  = './dataset'
WINDOW_SIZE  = 360            # samples par battement (= 1 seconde à 360Hz MIT-BIH)
BEFORE_PEAK  = 9             # samples avant le pic R
AFTER_PEAK   = 27            # samples après le pic R

# Classes AAMI (standard de classification ECG)
# N=Normal, S=Supraventriculaire, V=Ventriculaire, F=Fusion, Q=Inconnu
AAMI_CLASSES = {
    'N': 'N',  # Normal
    'L': 'N',  # Left bundle branch block → Normal
    'R': 'N',  # Right bundle branch block → Normal
    'e': 'N',  # Atrial escape → Normal
    'j': 'N',  # Nodal escape → Normal
    'A': 'S',  # Atrial premature
    'a': 'S',  # Aberrated atrial premature
    'J': 'S',  # Nodal premature
    'S': 'S',  # Supraventricular premature
    'V': 'V',  # Ventricular premature
    'E': 'V',  # Ventricular escape
    'F': 'F',  # Fusion
    '/': 'Q',  # Paced
    'f': 'Q',  # Fusion of paced
    'Q': 'Q',  # Unclassifiable
}

# Enregistrements MIT-BIH (48 au total)
RECORDS = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119','121',
    '122','123','124','200','201','202','203','205','207','208',
    '209','210','212','213','214','215','217','219','220','221',
    '222','223','228','230','231','232','233','234'
]

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─── EXTRACTION DES BATTEMENTS ────────────────────────────────────────────────
print("Extraction des battements MIT-BIH...")
signals_list = []
labels_list  = []

for rec_name in RECORDS:
    rec_path = os.path.join(MITDB_PATH, rec_name)
    if not os.path.exists(rec_path + '.hea'):
        print(f"  Manquant : {rec_name}")
        continue

    try:
        record = wfdb.rdrecord(rec_path)
        annot  = wfdb.rdann(rec_path, 'atr')

        signal = record.p_signal[:, 0]  # Lead II

        for i, (sample, symbol) in enumerate(zip(annot.sample, annot.symbol)):
            if symbol not in AAMI_CLASSES:
                continue

            start = sample - BEFORE_PEAK
            end   = sample + AFTER_PEAK

            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end].astype(np.float32)

            # Normalisation par battement
            beat = (beat - beat.mean()) / (beat.std() + 1e-8)

            signals_list.append(beat)
            labels_list.append(AAMI_CLASSES[symbol])

    except Exception as e:
        print(f"  Erreur {rec_name}: {e}")
        continue

print(f"\nTotal battements extraits : {len(signals_list)}")
print("Distribution :")
for cls, count in sorted(Counter(labels_list).items()):
    print(f"  {cls} : {count:6d} ({count/len(labels_list)*100:.1f}%)")

# ─── ENCODAGE + SPLIT ─────────────────────────────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(labels_list)
X  = np.array(signals_list)

# Split stratifié 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain : {X_train.shape}, Test : {X_test.shape}")
print(f"Classes : {list(le.classes_)}")

# ─── SAUVEGARDE ───────────────────────────────────────────────────────────────
np.save(f'{OUTPUT_PATH}/X_train.npy', X_train)
np.save(f'{OUTPUT_PATH}/X_test.npy',  X_test)
np.save(f'{OUTPUT_PATH}/y_train.npy', y_train)
np.save(f'{OUTPUT_PATH}/y_test.npy',  y_test)
np.save(f'{OUTPUT_PATH}/classes.npy', le.classes_)

print(f"\nDataset sauvegardé dans {OUTPUT_PATH}/")
print("Prêt pour l'entraînement → lancer train.py")
