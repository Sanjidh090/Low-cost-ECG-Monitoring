import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# CONFIGURATION
n_samples = 5000
np.random.seed(42)

# FEATURES WE WILL SIMULATE:
# 1. BPM (Rate)
# 2. RMSSD (Variability/Stress)
# 3. SDNN (Irregularity)
# 4. QRS_WIDTH (Width of the spike in ms) -> Detects PVCs
# 5. QT_INTERVAL (Time from beat start to end in ms) -> Detects Long QT
# 6. ST_ELEVATION (Voltage difference after spike in mV) -> Detects Heart Attack

# --- HELPER TO GENERATE DATA ---
def generate_class(n, bpm_mu, rmssd_mu, sdnn_mu, qrs_mu, qt_mu, st_mu, label):
    # We add noise (np.random.normal) to make it realistic
    data = np.column_stack((
        np.random.normal(bpm_mu, 10, n),    # BPM
        np.random.normal(rmssd_mu, 10, n),  # RMSSD
        np.random.normal(sdnn_mu, 10, n),   # SDNN
        np.random.normal(qrs_mu, 10, n),    # QRS Width
        np.random.normal(qt_mu, 20, n),     # QT Interval
        np.random.normal(st_mu, 0.1, n)     # ST Elevation
    ))
    labels = np.full(n, label)
    return data, labels

# --- GENERATE PATIENT GROUPS ---

# 0. NORMAL SINUS RHYTHM
# Healthy: 75 BPM, Relaxed (High RMSSD), Regular (Low SDNN), Narrow QRS, Normal QT, No ST
X0, y0 = generate_class(n_samples, 75, 50, 40, 90, 400, 0.0, 0)

# 1. ATRIAL FIBRILLATION (AFib)
# Irregularly Irregular: High SDNN, High RMSSD. Normal shape.
X1, y1 = generate_class(n_samples, 110, 80, 150, 90, 400, 0.0, 1)

# 2. PVC (Premature Ventricular Contraction)
# The "Skipped Beat": Normalish rate, but WIDE QRS (>120ms).
X2, y2 = generate_class(n_samples, 80, 50, 60, 140, 400, 0.0, 2)

# 3. STEMI (Myocardial Infarction / Heart Attack)
# The Danger Zone: ST Elevation > 0.2mV. Often elevated HR.
X3, y3 = generate_class(n_samples, 95, 30, 40, 90, 410, 0.4, 3)

# 4. LONG QT SYNDROME
# Genetic Risk: QT Interval > 460ms.
X4, y4 = generate_class(n_samples, 70, 50, 40, 95, 490, 0.0, 4)

# 5. BRADYCARDIA
# Too Slow: BPM < 60.
X5, y5 = generate_class(n_samples, 45, 60, 40, 90, 420, 0.0, 5)

# 6. TACHYCARDIA
# Too Fast: BPM > 100.
X6, y6 = generate_class(n_samples, 130, 20, 30, 90, 380, 0.0, 6)

# --- MERGE & TRAIN ---
X = np.vstack((X0, X1, X2, X3, X4, X5, X6))
y = np.concatenate((y0, y1, y2, y3, y4, y5, y6))

print("🧠 Training Diagnostic Model on 7 Conditions...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

filename = "ecg_brain_advanced.pkl"
joblib.dump(model, filename)
print(f"✅ Model saved to {filename}")
print("Classes: 0=Normal, 1=AFib, 2=PVC, 3=STEMI, 4=LongQT, 5=Brady, 6=Tachy")