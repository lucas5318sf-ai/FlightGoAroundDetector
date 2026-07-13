"""
1D CNN — All 20 Features, Full 160 Seconds
DASHlink Curated 4-Class Anomaly Detection Dataset
Binary classification: Nominal (0) vs Any Anomaly (1)

Features (all 20):
  0  AILERON POSITION LH       - Left aileron position (degrees)
  1  AILERON POSITION RH       - Right aileron position (degrees)
  2  CORRECTED ANGLE OF ATTACK - Angle of attack (degrees)
  3  BARO CORRECT ALTITUDE     - Altitude (feet)
  4  COMPUTED AIRSPEED LSP     - Airspeed (knots)
  5  SELECTED COURSE           - Autopilot selected course (degrees)
  6  DRIFT ANGLE               - Heading vs track angle (degrees)
  7  ELEVATOR POSITION LEFT    - Left elevator position (degrees)
  8  T.E. FLAP POSITION        - Flap deployment (discrete)
  9  GLIDESLOPE DEVIATION      - Deviation from glideslope (%)
 10  SELECTED HEADING          - Autopilot desired heading (degrees)
 11  LOCALIZER DEVIATION       - Runway axis deviation (%)
 12  CORE SPEED AVG            - Avg engine compressor speed (%)
 13  TOTAL PRESSURE            - Total pressure (millibar)
 14  PITCH ANGLE LSP           - Pitch angle (degrees)
 15  ROLL ANGLE LSP            - Roll angle (degrees)
 16  RUDDER POSITION           - Rudder position (degrees)
 17  TRUE HEADING LSP          - True heading (degrees)
 18  VERTICAL ACCELERATION     - Vertical acceleration (G)
 19  WIND SPEED                - Wind speed (knots)

Anomaly classes:
  0 = Nominal
  1 = Speed High  ─┐
  2 = Path High    ├─ all mapped to label 1 (anomaly)
  3 = Flaps Late  ─┘

Architecture:
  - Residual 1D CNN blocks with dilated causal convolutions
  - Multi-head self-attention
  - Focal loss to handle class imbalance
  - Memory-efficient: deletes intermediate arrays immediately
"""

import numpy as np
import gc
import time

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout, Add,
    GlobalAveragePooling1D, Dense, Activation,
    MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

FEATURE_NAMES = [
    "Aileron LH", "Aileron RH", "Angle of Attack", "Altitude",
    "Airspeed", "Selected Course", "Drift Angle", "Elevator Left",
    "Flap Position", "Glideslope Dev", "Selected Heading", "Localizer Dev",
    "Core Speed Avg", "Total Pressure", "Pitch Angle", "Roll Angle",
    "Rudder Position", "True Heading", "Vertical Accel", "Wind Speed"
]

# ──────────────────────────────────────────────
# 1. LOAD & PREP — free raw array ASAP
# ──────────────────────────────────────────────
print("Loading data...")
FILE_PATH = 'DASHlink_full_fourclass_raw_comp.npz'
data  = np.load(FILE_PATH)
X_raw = data['data']    # (N, 160, 20)
y_raw = data['label']   # (N,)

print(f"Full dataset shape: {X_raw.shape}")
print(f"Class distribution: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

# Binary labels: anomaly = any non-zero class
y_all  = np.where(y_raw == 0, 0, 1).astype(np.float32)
X_seq  = X_raw.astype(np.float32)   # keep ALL samples (no filtering needed)

del X_raw, y_raw, data
gc.collect()

N, T, F = X_seq.shape
print(f"\nUsing all {N} samples, {T} timesteps, {F} features")
print(f"Anomaly rate: {np.mean(y_all):.2%}")

# ──────────────────────────────────────────────
# 2. SPLIT
# ──────────────────────────────────────────────
print("\nSplitting data...")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_seq, y_all, test_size=0.15, random_state=42, stratify=y_all
)
del X_seq, y_all
gc.collect()

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.15, random_state=42, stratify=y_trainval
)
del X_trainval, y_trainval
gc.collect()

print(f"Train: {X_train.shape[0]}  |  Val: {X_val.shape[0]}  |  Test: {X_test.shape[0]}")

# ──────────────────────────────────────────────
# 3. SCALE IN-PLACE
# ──────────────────────────────────────────────
print("\nScaling features...")
scaler = StandardScaler()

Ntr, Ttr, Ftr = X_train.shape
X_train_2d = X_train.reshape(-1, Ftr)
scaler.fit(X_train_2d)

X_train[:] = scaler.transform(X_train_2d).reshape(Ntr, Ttr, Ftr)
del X_train_2d; gc.collect()

Nv = X_val.shape[0]
X_val[:] = scaler.transform(X_val.reshape(-1, Ftr)).reshape(Nv, T, F)

Nte = X_test.shape[0]
X_test[:] = scaler.transform(X_test.reshape(-1, Ftr)).reshape(Nte, T, F)

# Fix any NaN/Inf from zero-variance features
for arr in [X_train, X_val, X_test]:
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
gc.collect()
print("Scaling done.")
#Ask for what is printed
# ──────────────────────────────────────────────
# 4. FOCAL LOSS (handles class imbalance better
#    than simple binary cross-entropy)
# ──────────────────────────────────────────────
def focal_loss(gamma=2.0, alpha=0.75):
    """
    gamma: how much to down-weight easy examples (2.0 is standard)
    alpha: weight for the positive (anomaly) class (>0.5 = emphasize anomalies)
    """
    def loss_fn(y_true, y_pred):
        y_pred  = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce     = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return K.mean(alpha_t * K.pow(1.0 - p_t, gamma) * bce)
    return loss_fn

# ──────────────────────────────────────────────
# 5. MODEL ARCHITECTURE
# ──────────────────────────────────────────────
class GaussianNoise(tf.keras.layers.Layer):
    """Adds Gaussian noise during training for regularization."""
    def __init__(self, stddev=0.05, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    def call(self, inputs, training=None):
        if training:
            return inputs + tf.random.normal(tf.shape(inputs), stddev=self.stddev)
        return inputs


def residual_block(x, filters, kernel_size, dropout_rate=0.2, dilation=1):
    """
    Dilated causal residual block. Dilation lets the network
    see further back in time without increasing parameters.
    """
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='causal',
               dilation_rate=dilation, activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv1D(filters, kernel_size, padding='causal',
               dilation_rate=dilation, activation=None)(x)
    x = BatchNormalization()(x)
    # Project shortcut if number of filters changed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    x = Add()([x, shortcut])
    return Activation('relu')(x)


def build_model(T, F, lr=5e-4):
    inp = Input(shape=(T, F))

    # Light augmentation noise
    x = GaussianNoise(0.05)(inp)

    # Initial conv to project to 64 channels
    x = Conv1D(64, 7, padding='causal', activation='relu')(x)
    x = BatchNormalization()(x)

    # Stacked residual blocks with increasing dilation
    # Dilation 1,2,4,8 means the receptive field covers 1+2+4+8=15 s at each layer
    x = residual_block(x, 64,  5, dropout_rate=0.20, dilation=1)
    x = residual_block(x, 128, 5, dropout_rate=0.25, dilation=2)
    x = residual_block(x, 128, 3, dropout_rate=0.25, dilation=4)
    x = residual_block(x, 128, 3, dropout_rate=0.30, dilation=8)

    # Self-attention: lets the model focus on the most anomalous timesteps
    attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x    = LayerNormalization()(attn + x)

    # Global pooling: aggregate across the time dimension
    x = GlobalAveragePooling1D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64,  activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    m = Model(inp, out)
    m.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.75),
        metrics=[
            tf.keras.metrics.AUC(name='auc',    curve='ROC'),
            tf.keras.metrics.AUC(name='pr_auc', curve='PR'),
        ]
    )
    return m


print("\nBuilding model...")
model = build_model(T, F)
model.summary()

# ──────────────────────────────────────────────
# 6. TRAIN
# ──────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_pr_auc', patience=5,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_pr_auc', factor=0.5,
                     patience=3, min_lr=1e-6, mode='max', verbose=1),
    tf.keras.callbacks.ModelCheckpoint(
        'best_cnn_model.keras', monitor='val_pr_auc',
        save_best_only=True, mode='max', verbose=1
    )
]

print("\nTraining...")
start = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epoch=60 ,
    batch_size=512,
    callbacks=callbacks,
    verbose=1
)
elapsed = time.time() - start
print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ──────────────────────────────────────────────
# 7. FIND OPTIMAL THRESHOLD (on validation set)
#    We optimize for F1 — balances recall & precision
# ──────────────────────────────────────────────
print("\nFinding optimal decision threshold on validation set...")
y_val_proba = model.predict(X_val, batch_size=1024).ravel()
best_f1, best_thresh = 0.0, 0.5
for t in np.arange(0.05, 0.95, 0.01):
    f1 = f1_score(y_val, (y_val_proba >= t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t
print(f"Optimal threshold: {best_thresh:.2f}  (val F1 = {best_f1:.4f})")

# ──────────────────────────────────────────────
# 8. EVALUATE ON TEST SET
# ──────────────────────────────────────────────
print("\n" + "="*55)
print("TEST SET RESULTS")
print("="*55)

y_test_proba = model.predict(X_test, batch_size=1024).ravel()
y_test_pred  = (y_test_proba >= best_thresh).astype(int)

acc       = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall    = recall_score(y_test, y_test_pred, zero_division=0)
f1        = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_test_proba)
pr_auc    = average_precision_score(y_test, y_test_proba)

print(f"  Threshold:        {best_thresh:.2f}")
print(f"  Accuracy:         {acc:.4f}")
print(f"  Precision:        {precision:.4f}")
print(f"  Recall (anomaly): {recall:.4f}   ← most important for safety")
print(f"  F1:               {f1:.4f}")
print(f"  ROC-AUC:          {roc_auc:.4f}")
print(f"  PR-AUC:           {pr_auc:.4f}")

cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)
print()
print(classification_report(y_test, y_test_pred,
      target_names=['Nominal', 'Anomaly']))

# ──────────────────────────────────────────────
# 9. THRESHOLD SWEEP
# ──────────────────────────────────────────────
print("\nThreshold sweep:")
for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    y_tmp = (y_test_proba >= t).astype(int)
    r = recall_score(y_test, y_tmp, zero_division=0)
    p = precision_score(y_test, y_tmp, zero_division=0)
    f = f1_score(y_test, y_tmp, zero_division=0)
    print(f"  {t:.2f} → Recall: {r:.3f}  Precision: {p:.3f}  F1: {f:.3f}")

# ──────────────────────────────────────────────
# 10. PLOTS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('1D CNN — All 20 Features, 160s\nDASHlink Anomaly Detection (Binary)',
             fontsize=14, fontweight='bold')

# Training curves — Loss
ax = axes[0, 0]
ax.plot(history.history['loss'],     label='Train', lw=2)
ax.plot(history.history['val_loss'], label='Val',   lw=2)
ax.set_title('Focal Loss'); ax.set_xlabel('Epoch')
ax.legend(); ax.grid(True, alpha=0.3)

# Training curves — PR-AUC
ax = axes[0, 1]
ax.plot(history.history['pr_auc'],     label='Train', lw=2)
ax.plot(history.history['val_pr_auc'], label='Val',   lw=2)
ax.set_title('PR-AUC'); ax.set_xlabel('Epoch')
ax.legend(); ax.grid(True, alpha=0.3)

# ROC curve
ax = axes[0, 2]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve'); ax.legend(); ax.grid(True, alpha=0.3)

# Precision-Recall curve
ax = axes[1, 0]
prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_test_proba)
ax.plot(rec_arr, prec_arr, lw=2, label=f'PR-AUC = {pr_auc:.4f}')
ax.axvline(recall, color='red', ls='--', alpha=0.6,
           label=f'Operating point (recall={recall:.2f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve'); ax.legend(); ax.grid(True, alpha=0.3)

# Confusion matrix
ax = axes[1, 1]
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Nominal', 'Anomaly'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Nominal', 'Anomaly'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=13, fontweight='bold',
                color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar(im, ax=ax)

# F1 vs Threshold
ax = axes[1, 2]
thr_arr = np.arange(0.05, 0.95, 0.01)
f1_arr  = [f1_score(y_test, (y_test_proba >= t).astype(int), zero_division=0)
           for t in thr_arr]
ax.plot(thr_arr, f1_arr, lw=2, color='steelblue')
ax.axvline(best_thresh, color='red', ls='--', label=f'Optimal = {best_thresh:.2f}')
ax.set_xlabel('Threshold'); ax.set_ylabel('F1 Score')
ax.set_title('F1 vs Decision Threshold'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_all_features_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to cnn_all_features_results.png")

# Save the model
model.save('cnn_anomaly_detector.keras')
print("Model saved to cnn_anomaly_detector.keras")
print("\n✅ 1D CNN (all features) complete!")

