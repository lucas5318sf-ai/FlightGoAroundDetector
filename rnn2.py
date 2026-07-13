import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# 1. LOAD DATA & INITIALIZE SHAPES
# ──────────────────────────────────────────────
print("Loading .npz file...")
data = np.load('DASHlink_full_fourclass_raw_comp.npz')

# Extract features and map labels to binary (0 = Nominal, 1 = Anomaly)
X = data['data'].astype(np.float32)
y = np.where(data['label'] == 0, 0, 1).astype(np.float32)

# Get sequence length (160) and number of features (20) dynamically
num_samples, timesteps, num_features = X.shape
print(f"Loaded {num_samples} samples with {timesteps} timesteps and {num_features} features.")

# ──────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
# 80% Training, 20% Testing (stratified to preserve anomaly ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ──────────────────────────────────────────────
# 3. BUILD THE SEQUENTIAL RNN MODEL
# ──────────────────────────────────────────────
model = Sequential([
    # Explicitly define the input shape matching your npz dimensions (160, 20)
    Input(shape=(timesteps, num_features)),
    
    # First recurrent layer (must return sequences to feed into the next RNN layer)
    GRU(64, return_sequences=True),
    Dropout(0.2),
    
    # Second recurrent layer (returns only the final timestep vector to flatten the output)
    GRU(64, return_sequences=False),
    Dropout(0.2),
    
    # Dense classification layers
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid output for binary anomaly detection
])

# ──────────────────────────────────────────────
# 4. COMPILE AND TRAIN
# ──────────────────────────────────────────────
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc', curve='ROC')]
)

model.summary()

print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

# ──────────────────────────────────────────────
# 5. FINAL EVALUATION
# ──────────────────────────────────────────────
print("\n" + "="*40)
print("FINAL TEST SET PERFORMANCE")
print("="*40)
loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test ROC-AUC:  {auc:.4f}")