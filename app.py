import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. LOAD DATA & INITIALIZE SHAPES
# ──────────────────────────────────────────────
print("Loading .npz file...")
data = np.load('DASHlink_full_fourclass_raw_comp.npz')

X = data['data'].astype(np.float32)
y = np.where(data['label'] == 0, 0, 1).astype(np.float32)

num_samples, timesteps, num_features = X.shape
print(f"Loaded {num_samples} samples ({timesteps} timesteps, {num_features} features).")

# ──────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT (80% Train, 20% Test)
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ──────────────────────────────────────────────
# 3. BUILD THE SEQUENTIAL RNN MODEL
# ──────────────────────────────────────────────
model = Sequential([
    Input(shape=(timesteps, num_features)),
    
    GRU(64, return_sequences=True),
    Dropout(0.2),
    
    GRU(64, return_sequences=False),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve='ROC')]
)

model.summary()

# ──────────────────────────────────────────────
# 4. TRAIN MODEL (Saving to history variable)
# ──────────────────────────────────────────────
print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

# ──────────────────────────────────────────────
# 5. GENERATE PREDICTIONS
# ──────────────────────────────────────────────
print("\nEvaluating model performance...")
y_probs = model.predict(X_test, batch_size=512).ravel()
y_pred = (y_probs >= 0.5).astype(int)  # Standard 0.5 classification threshold

# ──────────────────────────────────────────────
# 6. GRAPH THE PLOTS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Sequential RNN (GRU) Performance Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Training & Validation Loss
ax = axes[0, 0]
ax.plot(history.history['loss'], label='Train Loss', lw=2)
ax.plot(history.history['val_loss'], label='Val Loss', lw=2)
ax.set_title('Model Loss Progression')
ax.set_xlabel('Epoch')
ax.set_ylabel('Binary Cross-Entropy Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: ROC Curve (Receiver Operating Characteristic)
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-AUC = {roc_auc:.4f}')
ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
ax.set_title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

# Plot 3: Precision-Recall Curve
ax = axes[1, 0]
precision, recall, _ = precision_recall_curve(y_test, y_probs)
ax.plot(recall, precision, color='forestgreen', lw=2, label='PR Curve')
ax.set_title('Precision-Recall Curve')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix
ax = axes[1, 1]
cm = confusion_matrix(y_test, y_pred)
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Nominal', 'Anomaly'])
ax.set_yticks([0, 1])
ax.set_yticklabels(['Nominal', 'Anomaly'])
ax.set_title('Confusion Matrix (Threshold = 0.5)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

# Write the numbers inside the confusion matrix squares
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontweight='bold', fontsize=12)

# Adjust layout and render the plots
plt.tight_layout()
plt.show()
