import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
import joblib

# ─────────────────────────────────────────────
# Custom objects needed to load the model
# ─────────────────────────────────────────────
class GaussianNoise(tf.keras.layers.Layer):
    def __init__(self, stddev=0.05, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev
    def call(self, inputs, training=None):
        if training:
            return inputs + tf.random.normal(tf.shape(inputs), stddev=self.stddev)
        return inputs

def focal_loss(gamma=2.0, alpha=0.75):
    def loss_fn(y_true, y_pred):
        y_pred  = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        bce     = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return K.mean(alpha_t * K.pow(1.0 - p_t, gamma) * bce)
    return loss_fn

# ─────────────────────────────────────────────
# Feature definitions
# ─────────────────────────────────────────────
FEATURE_NAMES = [
    "Aileron LH", "Aileron RH", "Angle of Attack", "Altitude",
    "Airspeed", "Selected Course", "Drift Angle", "Elevator Left",
    "Flap Position", "Glideslope Dev", "Selected Heading", "Localizer Dev",
    "Core Speed Avg", "Total Pressure", "Pitch Angle", "Roll Angle",
    "Rudder Position", "True Heading", "Vertical Accel", "Wind Speed"
]

FEATURE_UNITS = [
    "degrees", "degrees", "degrees", "feet",
    "knots", "degrees", "degrees", "degrees",
    "discrete", "%", "degrees", "%",
    "%", "millibar", "degrees", "degrees",
    "degrees", "degrees", "G", "knots"
]

THRESHOLD = 0.50  # update this to your best_thresh from training output

# ─────────────────────────────────────────────
# Load model (cached so it only loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return keras.models.load_model(
        'cnn_anomaly_detector.keras',
        custom_objects={'GaussianNoise': GaussianNoise, 'loss_fn': focal_loss()}
    )

@st.cache_resource
def load_scaler():
    # If you saved your scaler with joblib.dump(scaler, 'scaler.joblib')
    # uncomment the next line and remove the return None
    # return joblib.load('scaler.joblib')
    return None  # remove this if you have a saved scaler

model  = load_model()
scaler = load_scaler()

# ─────────────────────────────────────────────
# App UI
# ─────────────────────────────────────────────
st.title("✈️ Flight Anomaly Detector")
st.markdown("Detects **Speed High**, **Path High**, and **Flaps Late** anomalies using a 1D CNN trained on DASHlink data.")

tab1, tab2 = st.tabs(["📁 Upload CSV", "🎛️ Manual Input"])

# ─────────────────────────────────────────────
# Helper: preprocess & predict
# ─────────────────────────────────────────────
def preprocess(arr):
    """
    arr: numpy array of shape (160, 20)
    Returns scaled array of shape (1, 160, 20) ready for the model.
    """
    arr = arr.astype(np.float32)
    if scaler is not None:
        arr = scaler.transform(arr).astype(np.float32)
    # If no scaler saved, data goes in as-is (not recommended — see note below)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.reshape(1, 160, 20)

def run_prediction(X):
    score = float(model.predict(X, verbose=0)[0][0])
    label = "⚠️ Anomaly Detected" if score >= THRESHOLD else "✅ Nominal Flight"
    return score, label

def show_result(score, label):
    st.metric("Anomaly Score", f"{score:.4f}", help="Score ≥ threshold = anomaly")
    if score >= THRESHOLD:
        st.error(label)
    else:
        st.success(label)
    st.progress(score)
    st.caption(f"Decision threshold: {THRESHOLD}")

# ─────────────────────────────────────────────
# TAB 1 — CSV Upload
# ─────────────────────────────────────────────
with tab1:
    st.markdown("Upload a CSV with **160 rows × 20 columns** (one row per timestep, one column per feature, in the order below).")
    
    with st.expander("Expected column order"):
        for i, (name, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS)):
            st.markdown(f"`{i}` — **{name}** ({unit})")

    uploaded_file = st.file_uploader("Upload flight CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            st.subheader("Preview")
            st.dataframe(df.head(), use_container_width=True)

            if df.shape != (160, 20):
                st.error(f"Expected shape (160, 20) but got {df.shape}. Please check your file.")
            else:
                st.line_chart(df, use_container_width=True)
                if st.button("Run Detection", key="csv_predict"):
                    with st.spinner("Analysing..."):
                        X   = preprocess(df.values)
                        score, label = run_prediction(X)
                    show_result(score, label)

        except Exception as e:
            st.error(f"Could not read file: {e}")

# ─────────────────────────────────────────────
# TAB 2 — Manual Input
# ─────────────────────────────────────────────
with tab2:
    st.markdown("Enter a **single snapshot** of sensor readings. The same values are repeated across all 160 timesteps.")
    st.info("For a realistic prediction, upload a full 160-row CSV instead. Manual input is best for quick sanity checks.")

    cols = st.columns(2)
    manual_values = []

    DEFAULTS = [0.0, 0.0, 3.0, 10000.0, 150.0, 270.0, 0.0, 0.0,
                0.0, 0.0, 270.0, 0.0, 85.0, 900.0, 2.0, 1.0,
                0.0, 270.0, 1.0, 10.0]

    for i, (name, unit, default) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS, DEFAULTS)):
        col = cols[i % 2]
        val = col.number_input(f"{name} ({unit})", value=float(default), key=f"feat_{i}")
        manual_values.append(val)

    if st.button("Run Detection", key="manual_predict"):
        with st.spinner("Analysing..."):
            arr = np.array(manual_values, dtype=np.float32)
            arr = np.tile(arr, (160, 1))   # repeat snapshot across 160 timesteps
            X   = preprocess(arr)
            score, label = run_prediction(X)
        show_result(score, label)