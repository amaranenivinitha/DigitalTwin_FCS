import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# -----------------------------------
# Load fault dataset
# -----------------------------------
df = pd.read_csv(
    'data/flight_logs/imu_faulted.csv'
)

# -----------------------------------
# Load reconstruction errors
# -----------------------------------
fault_recon_err = np.load(
    'data/models/fault_recon_err.npy'
)

# -----------------------------------
# Load threshold
# -----------------------------------
with open(
    'data/models/threshold.txt',
    'r'
) as f:

    threshold = float(f.read())

# -----------------------------------
# Generate predictions
# -----------------------------------
seq_len = 50

pred = np.zeros(len(df))

for i, err in enumerate(fault_recon_err):

    if err > threshold:
        pred[i:i+seq_len] = 1

pred = pred.astype(int)

# -----------------------------------
# Ground truth
# -----------------------------------
gt = df['fault_label'].values

# -----------------------------------
# Confusion Matrix
# -----------------------------------
cm = confusion_matrix(gt, pred)

TN, FP, FN, TP = cm.ravel()

# -----------------------------------
# False Alarm Rate
# -----------------------------------
far = FP / (FP + TN)

# -----------------------------------
# Time-to-Detect
# -----------------------------------
fault_indices = np.where(gt == 1)[0]

fault_start_idx = fault_indices[0]

detection_indices = np.where(pred == 1)[0]

detection_after_fault = detection_indices[
    detection_indices >= fault_start_idx
]

if len(detection_after_fault) > 0:

    detect_idx = detection_after_fault[0]

    t_fault = df.iloc[fault_start_idx]['time_s']

    t_detect = df.iloc[detect_idx]['time_s']

    ttd = t_detect - t_fault

else:

    t_fault = None
    t_detect = None
    ttd = None

# -----------------------------------
# Results
# -----------------------------------
print("\n========== Detection Evaluation ==========\n")

print(f"TN = {TN}")
print(f"FP = {FP}")
print(f"FN = {FN}")
print(f"TP = {TP}")

print("\nFalse Alarm Rate:")
print(f"{far:.4f}")

print("\nFault Injection Time:")
print(t_fault)

print("\nDetection Time:")
print(t_detect)

print("\nTime-to-Detect:")
print(f"{ttd:.4f} seconds")

print("\n==========================================")