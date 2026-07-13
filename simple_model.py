import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#  Paramaters
numSeconds = 160
# Select features: Airspeed, Localizer, Glideslope, Pitch
selected_features = [4, 11, 12, 14]

# ===== 1. LOAD DATA =====
print("Loading data...")
file_path = 'DASHlink_full_fourclass_raw_comp.npz'
data = np.load(file_path)
X = data['data']
y = data['label']

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# ===== 2. USE FIRST 80 SECONDS & SELECT FEATURES =====
print("\nProcessing features...")

X_seconds = X[:, :numSeconds, :]  
X_sel = X_seconds[:, :, selected_features]

# Flatten for scikit-learn
num_samples = X_sel.shape[0]
X_model = X_sel.reshape(num_samples, -1)

# Binary labels: 0 = Normal, 1 = Any Anomaly
y_binary = np.where(y == 0, 0, 1)

print(f"Feature matrix shape: {X_model.shape}")
print(f"Normal flights: {np.sum(y_binary == 0)}")
print(f"Anomaly flights: {np.sum(y_binary == 1)}")

# ===== 3. SPLIT DATA =====
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_model, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# ===== 4. SCALE FEATURES =====
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== 5. TRAIN RANDOM FOREST =====
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight='balanced',
    n_jobs=-1  # Use all CPU cores
)

model.fit(X_train_scaled, y_train)

# ===== 6. EVALUATE =====
print("\nEvaluating model...")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
print("Confusion Matrix")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


