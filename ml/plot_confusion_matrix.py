import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# -----------------------------
# CONFUSION MATRIX VALUES
# -----------------------------
TN = 2766
FP = 234
FN = 0
TP = 1000

cm = np.array([
    [TN, FP],
    [FN, TP]
])

# -----------------------------
# CREATE FIGURE
# -----------------------------
fig, ax = plt.subplots(figsize=(6,6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fault"]
)

disp.plot(ax=ax, cmap="Blues", colorbar=True)

plt.title("AI Fault Detection Confusion Matrix")

# -----------------------------
# SAVE FIGURE
# -----------------------------
os.makedirs("results", exist_ok=True)

save_path = "results/confusion_matrix.png"

plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"✅ Confusion matrix saved -> {save_path}")

plt.show()