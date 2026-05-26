"""
train.py
========
Entraîne un CNN 1D sur MIT-BIH pour classifier les battements ECG.

UTILISATION :
    python train.py

SORTIE :
    - model/ecg_cnn.h5         → modèle entraîné
    - model/training_curves.png → courbes loss/accuracy
    - model/confusion_matrix.png → matrice de confusion
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os, json

os.makedirs('model', exist_ok=True)

# ─── 1. Charger les données ────────────────────────────────────────────────
print("Chargement des données...")
X_train  = np.load('dataset/X_train.npy')[..., np.newaxis]  # (N, 360, 1)
X_test   = np.load('dataset/X_test.npy')[...,  np.newaxis]
y_train  = np.load('dataset/y_train.npy')
y_test   = np.load('dataset/y_test.npy')
classes  = np.load('dataset/classes.npy', allow_pickle=True)
n_classes = len(classes)

print(f"  Train : {X_train.shape}  Test : {X_test.shape}")
print(f"  Classes : {list(classes)}")

# ─── 2. Architecture CNN 1D ────────────────────────────────────────────────
def build_model(input_shape, n_classes):
    model = models.Sequential([
        # Bloc 1 — détecte les motifs fins (pics R, ondes P)
        layers.Conv1D(32, kernel_size=5, activation='relu',
                      padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.25),

        # Bloc 2 — détecte les formes complexes (complexe QRS)
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.25),

        # Bloc 3 — abstraction haute niveau
        layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # Décision finale
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ], name='ECG_CNN_1D')
    return model

model = build_model((360, 1), n_classes)
model.summary()

# ─── 3. Compilation ────────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ─── 4. Callbacks ──────────────────────────────────────────────────────────
cb = [
    callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, min_lr=1e-6, verbose=1),
    callbacks.ModelCheckpoint(
        'model/ecg_cnn_best.h5',
        monitor='val_accuracy', save_best_only=True, verbose=0)
]

# ─── 5. Entraînement ───────────────────────────────────────────────────────
print("\nEntraînement...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=cb,
    verbose=1
)

# ─── 6. Évaluation ─────────────────────────────────────────────────────────
print("\nÉvaluation sur le set de test...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n{'='*40}")
print(f"  Test accuracy : {test_acc:.2%}")
print(f"  Test loss     : {test_loss:.4f}")
print(f"{'='*40}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=list(classes)))

# ─── 7. Courbes d'apprentissage ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Courbes d\'apprentissage — CNN 1D ECG MIT-BIH', fontsize=13)

epochs_range = range(1, len(history.history['loss']) + 1)

ax1.plot(epochs_range, history.history['loss'],     label='Train loss',      linewidth=2)
ax1.plot(epochs_range, history.history['val_loss'], label='Validation loss', linewidth=2, linestyle='--')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history.history['accuracy'],     label='Train accuracy',      linewidth=2, color='green')
ax2.plot(epochs_range, history.history['val_accuracy'], label='Validation accuracy', linewidth=2, color='green', linestyle='--')
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model/training_curves.png', dpi=150)
print("\nCourbes sauvegardées : model/training_curves.png")

# ─── 8. Matrice de confusion ───────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig2, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=range(n_classes), yticks=range(n_classes),
       xticklabels=classes, yticklabels=classes,
       title='Matrice de confusion', ylabel='Vraie classe', xlabel='Classe prédite')
for i in range(n_classes):
    for j in range(n_classes):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png', dpi=150)
print("Matrice de confusion : model/confusion_matrix.png")

# ─── 9. Sauvegarde modèle ──────────────────────────────────────────────────
model.save('model/ecg_cnn.h5')
print("Modèle sauvegardé : model/ecg_cnn.h5")

# Résultats dans un JSON
results = {
    'test_accuracy': float(test_acc),
    'test_loss':     float(test_loss),
    'epochs_trained': len(history.history['loss']),
    'classes': list(classes)
}
with open('model/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Entraînement terminé — Accuracy : {test_acc:.2%}")
