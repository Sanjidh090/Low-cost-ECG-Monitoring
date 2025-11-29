# app.py
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ---- Load MIT-BIH model ----
MODEL_PATH = 'mitbih_model.h5'  # put your model in the same folder as app.py
model = load_model(MODEL_PATH)
st.sidebar.success("Model loaded!")

WINDOW_SIZE = 187  # same as used in training

# ---- Helper functions ----
def load_ecg_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df.columns = ['timestamp_ms', 'ecg_value']  # normalize column names
    return df

def split_windows(signal, window_size):
    windows = []
    for start in range(0, len(signal) - window_size + 1, window_size):
        win = signal[start:start+window_size]
        windows.append(win)
    return np.array(windows)

def predict_heartbeats(signal):
    signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    windows = split_windows(signal_norm, WINDOW_SIZE)
    X = windows.reshape(-1, WINDOW_SIZE, 1)
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_pred

def calculate_heart_rate(signal, window_size=WINDOW_SIZE, sampling_rate=360):
    # Number of beats in signal * (60 sec / total seconds)
    num_beats = len(split_windows(signal, window_size))
    total_seconds = len(signal) / sampling_rate
    heart_rate = num_beats * 60 / total_seconds
    return round(heart_rate, 1)

# Map numeric classes to descriptive names
CLASS_LABELS = {
    0: "N (0) → Normal beats (non‑ectopic, sinus rhythm)",
    1: "S (1) → Supraventricular ectopic beats (atrial premature, etc.)",
    2: "V (2) → Ventricular ectopic beats (PVCs, ventricular tachycardia)",
    3: "F (3) → Fusion beats (a normal beat fused with a ventricular beat)",
    4: "Q (4) → Unknown/unclassifiable beats (noisy or ambiguous signals)"
}

# ---- Streamlit UI ----
st.title("ECG Heartbeat Analysis")
st.write("Upload your ECG CSV file and explore heartbeat types, signal, and heart rate.")

uploaded_file = st.file_uploader("Upload ECG CSV", type=['csv'])

# Unique button style
def styled_button(label):
    return st.markdown(f"""
        <style>
        div.stButton > button:first-child {{
            background-color: #4CAF50;
            color: white;
            height: 3em;
            width: 100%;
            border-radius: 12px;
            border: none;
            font-size: 16px;
        }}
        div.stButton > button:hover {{
            background-color: #45a049;
            color: white;
        }}
        </style>
        <form action="" target="_self">
        <button>{label}</button>
        </form>
        """, unsafe_allow_html=True)

if uploaded_file is not None:
    df = load_ecg_file(uploaded_file)
    signal = df['ecg_value'].values

    # Show ECG signal button
    if st.button("Show Raw ECG Signal"):
        st.subheader("Raw ECG Signal")
        demo_samples = 500 if len(signal) > 500 else len(signal)
        demo_time = df['timestamp_ms'].values[:demo_samples]
        demo_signal = signal[:demo_samples]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(demo_time, demo_signal, label="ECG")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Raw ECG Signal")
        ax.grid(True)
        st.pyplot(fig)

    # Predict heartbeats button
    if st.button("Predict Heartbeats"):
        predictions = predict_heartbeats(signal)
        labels = [CLASS_LABELS.get(p, "Unknown") for p in predictions]

        results = pd.DataFrame({
            'Heartbeat_index': np.arange(1, len(labels)+1),
            'Heartbeat_type': labels
        })

        st.success(f"Prediction completed! Total heartbeats: {len(labels)}")
        st.dataframe(results)

    # Show heart rate button
    if st.button("Show Heart Rate"):
        heart_rate = calculate_heart_rate(signal)
        st.subheader("Estimated Heart Rate")
        st.metric(label="Heart Rate (BPM)", value=heart_rate)
