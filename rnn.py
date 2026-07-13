import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

# 1. Load Data & Create Binary Labels
data = np.load('DASHlink_full_fourclass_raw_comp.npz')
X = data['data'].astype(np.float32)
y = np.where(data['label'] == 0, 0, 1).astype(np.float32)

# 2. Simple Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Model Architecture
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
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
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# 4. Train (saving to history variable for plotting)
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=256,
    validation_data=(X_test, y_test)
)

# 5. Generate Predictions
y_probs = model.predict(X_test).ravel()
y_pred = (y_probs >= 0.5).astype(int)  # Using default 0.5 threshold

# ──────────────────────────────────────────────
# 6. PLOTS
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('RNN Model Performance Baseline', fontsize=14, fontweight='bold')

# Plot 1: Loss Curves
ax = axes[0, 0]
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Val Loss')
ax.set_title('Model Loss')
ax.set_xlabel('Epoch')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: ROC Curve
ax = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.4f}')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('Receiver Operating Characteristic (ROC)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Precision-Recall Curve
ax = axes[1, 0]
precision, recall, _ = precision_recall_curve(y_test, y_probs)
ax.plot(recall, precision, color='blue', label='PR Curve')
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
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# Annotate numbers inside the confusion matrix
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontweight='bold')

plt.tight_layout()
plt.show()