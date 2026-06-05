import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# -----------------------------
# LOAD DATA
# -----------------------------
faulted = pd.read_csv("data/flight_logs/imu_faulted.csv")

# Ground truth labels
y_true = faulted["fault_label"].values

# Anomaly scores
scores = np.abs(faulted["gyro_pitch_rad_s_faulty"].values)

# -----------------------------
# PRECISION-RECALL COMPUTATION
# -----------------------------
precision, recall, thresholds = precision_recall_curve(y_true, scores)

ap_score = average_precision_score(y_true, scores)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(7,7))

plt.plot(
    recall,
    precision,
    linewidth=2,
    label=f"PR Curve (AP = {ap_score:.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve — AI Fault Detection")

plt.grid(True)

plt.legend(loc="lower left")

# -----------------------------
# SAVE
# -----------------------------
os.makedirs("results", exist_ok=True)

save_path = "results/precision_recall_curve.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"✅ Precision-Recall curve saved -> {save_path}")

plt.show()