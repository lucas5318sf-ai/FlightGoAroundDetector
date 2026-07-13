"""
Simple 1D CNN — 1 Conv Layer
DASHlink Anomaly Detection — Binary Classification
Each layer: Conv1D → MaxPooling1D. Ends with Flatten → Dense → output.
"""

import numpy as np, gc, time, warnings
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np

VARIANT_NAME = "1_conv_layer"

print("Loading data...")
data  = np.load('DASHlink_full_fourclass_raw_comp.npz')
X_raw = data['data']
y_raw = data['label']
y_all = np.where(y_raw == 0, 0, 1).astype(np.float32)
X_seq = X_raw.astype(np.float32)
del X_raw, y_raw, data; gc.collect()

N, T, F = X_seq.shape
print(f"{N} samples | {T} timesteps | {F} features | anomaly rate: {np.mean(y_all):.2%}")

X_trainval, X_test, y_trainval, y_test = train_test_split(X_seq, y_all, test_size=0.15, random_state=42, stratify=y_all)
del X_seq, y_all; gc.collect()
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval)
del X_trainval, y_trainval; gc.collect()
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")


# Number of samples to display
num_samples = 6

# Random training examples
idx = np.random.choice(len(X_train), num_samples, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for i, sample_idx in enumerate(idx):
    sample = X_train[sample_idx]  # shape (T, F)

    im = axes[i].imshow(
        sample,
        aspect='auto',
        cmap='viridis',
        origin='lower'
    )

    axes[i].set_title(
        f"Sample {sample_idx}\nLabel={'Anomaly' if y_train[sample_idx] else 'Nominal'}"
    )
    axes[i].set_xlabel("Features")
    axes[i].set_ylabel("Timesteps")

fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
plt.suptitle("Random Training Samples", fontsize=14)
plt.tight_layout()
plt.show()

# scaler = StandardScaler()
# X_train[:] = scaler.fit_transform(X_train.reshape(-1, F)).reshape(X_train.shape)
# X_val[:]   = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
# X_test[:]  = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)
# for arr in [X_train, X_val, X_test]:
#     np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
# gc.collect()

# # ── MODEL ──────────────────────────────────────
# model = models.Sequential([
#     layers.Conv1D(32, 3, activation='relu', input_shape=(T, F)),
#     layers.MaxPooling1D(2),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy',
#     metrics=['accuracy',
#              tf.keras.metrics.AUC(name='auc', curve='ROC'),
#              tf.keras.metrics.AUC(name='pr_auc', curve='PR')])
# model.summary()

# callbacks = [
#     EarlyStopping(monitor='val_pr_auc', patience=5, restore_best_weights=True, mode='max', verbose=1),
#     ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5, patience=3, min_lr=1e-6, mode='max', verbose=1),
#     tf.keras.callbacks.ModelCheckpoint(f'best_{VARIANT_NAME}.keras', monitor='val_pr_auc', save_best_only=True, mode='max', verbose=1)
# ]

# print("\nTraining...")
# start = time.time()
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
#     epochs=20, batch_size=512, callbacks=callbacks, verbose=1)
# print(f"Training time: {time.time()-start:.1f}s")

# y_val_proba = model.predict(X_val, batch_size=1024).ravel()
# best_f1, best_thresh = 0.0, 0.5
# for t in np.arange(0.05, 0.95, 0.01):
#     f1 = f1_score(y_val, (y_val_proba >= t).astype(int), zero_division=0)
#     if f1 > best_f1: best_f1, best_thresh = f1, t
# print(f"Optimal threshold: {best_thresh:.2f} (val F1={best_f1:.4f})")

# y_test_proba = model.predict(X_test, batch_size=1024).ravel()
# y_test_pred  = (y_test_proba >= best_thresh).astype(int)
# acc       = accuracy_score(y_test, y_test_pred)
# precision = precision_score(y_test, y_test_pred, zero_division=0)
# recall    = recall_score(y_test, y_test_pred, zero_division=0)
# f1        = f1_score(y_test, y_test_pred, zero_division=0)
# roc_auc   = roc_auc_score(y_test, y_test_proba)
# pr_auc    = average_precision_score(y_test, y_test_proba)
# cm        = confusion_matrix(y_test, y_test_pred)

# print(f"\n{'='*50}\nTEST RESULTS — {VARIANT_NAME}\n{'='*50}")
# print(f"  Accuracy:  {acc:.4f} | Precision: {precision:.4f}")
# print(f"  Recall:    {recall:.4f} | F1: {f1:.4f}")
# print(f"  ROC-AUC:   {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
# print(f"\nConfusion Matrix:\n{cm}")
# print(classification_report(y_test, y_test_pred, target_names=['Nominal','Anomaly']))

# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle(f'Simple 1D CNN — {VARIANT_NAME}\nDASHlink Anomaly Detection', fontsize=14, fontweight='bold')
# axes[0,0].plot(history.history['loss'], label='Train', lw=2)
# axes[0,0].plot(history.history['val_loss'], label='Val', lw=2)
# axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
# axes[0,1].plot(history.history['accuracy'], label='Train', lw=2)
# axes[0,1].plot(history.history['val_accuracy'], label='Val', lw=2)
# axes[0,1].set_title('Accuracy'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
# fpr, tpr, _ = roc_curve(y_test, y_test_proba)
# axes[0,2].plot(fpr, tpr, lw=2, label=f'AUC={roc_auc:.4f}')
# axes[0,2].plot([0,1],[0,1],'k--'); axes[0,2].set_title('ROC Curve'); axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)
# prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_test_proba)
# axes[1,0].plot(rec_arr, prec_arr, lw=2, label=f'PR-AUC={pr_auc:.4f}')
# axes[1,0].axvline(recall, color='red', ls='--', alpha=0.6, label=f'recall={recall:.2f}')
# axes[1,0].set_title('Precision-Recall'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)
# im = axes[1,1].imshow(cm, cmap='Blues')
# axes[1,1].set_xticks([0,1]); axes[1,1].set_xticklabels(['Nominal','Anomaly'])
# axes[1,1].set_yticks([0,1]); axes[1,1].set_yticklabels(['Nominal','Anomaly'])
# axes[1,1].set_title('Confusion Matrix')
# for i in range(2):
#     for j in range(2):
#         axes[1,1].text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=13, fontweight='bold',
#             color='white' if cm[i,j] > cm.max()/2 else 'black')
# plt.colorbar(im, ax=axes[1,1])
# thr_arr = np.arange(0.05, 0.95, 0.01)
# f1_arr = [f1_score(y_test,(y_test_proba>=t).astype(int),zero_division=0) for t in thr_arr]
# axes[1,2].plot(thr_arr, f1_arr, lw=2, color='steelblue')
# axes[1,2].axvline(best_thresh, color='red', ls='--', label=f'Optimal={best_thresh:.2f}')
# axes[1,2].set_title('F1 vs Threshold'); axes[1,2].legend(); axes[1,2].grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(f'cnn_{VARIANT_NAME}_results.png', dpi=150, bbox_inches='tight')
# plt.show()
# model.save(f'cnn_{VARIANT_NAME}.keras')
# print(f"\nDone — {VARIANT_NAME}")