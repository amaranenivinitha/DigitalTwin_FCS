import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# -----------------------------
# LOAD DATA
# -----------------------------
faulted = pd.read_csv("data/flight_logs/imu_faulted.csv")

# Ground truth labels
y_true = faulted["fault_label"].values

# Use absolute gyro signal magnitude as anomaly score
# (simple proxy for visualization)
scores = np.abs(faulted["gyro_pitch_rad_s_faulty"].values)

# -----------------------------
# ROC COMPUTATION
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_true, scores)

roc_auc = auc(fpr, tpr)

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(7,7))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"ROC Curve (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    linewidth=1
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve — AI Fault Detection")

plt.legend(loc="lower right")

plt.grid(True)

# -----------------------------
# SAVE
# -----------------------------
os.makedirs("results", exist_ok=True)

save_path = "results/roc_curve.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"✅ ROC curve saved -> {save_path}")

plt.show()