import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# -----------------------------------
# LOAD DATA
# -----------------------------------

nominal = pd.read_csv("data/flight_logs/imu_emulated.csv")
faulted = pd.read_csv("data/flight_logs/imu_faulted.csv")

# Reference signal
reference = nominal["gyro_pitch_rad_s"].values

# Twin signal
twin = faulted["gyro_pitch_rad_s_faulty"].values

# Time
time = faulted["time_s"].values

# -----------------------------------
# COMPUTE METRICS
# -----------------------------------

rmse = np.sqrt(mean_squared_error(reference, twin))

corr = np.corrcoef(reference, twin)[0,1]

error_signal = twin - reference

print("\n========== Digital Twin Fidelity ==========\n")

print(f"RMSE                 : {rmse:.6f}")
print(f"Correlation Coefficient : {corr:.6f}")

print("\n===========================================\n")

# -----------------------------------
# CREATE FIGURE
# -----------------------------------

fig, axs = plt.subplots(2, 1, figsize=(12,8))

# -------------------------------
# Overlay Plot
# -------------------------------

axs[0].plot(
    time,
    reference,
    label="Reference Signal",
    linewidth=2
)

axs[0].plot(
    time,
    twin,
    label="Digital Twin Signal",
    linewidth=1.5
)

axs[0].set_title("Twin vs Reference Signal")

axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Gyro Rate (rad/s)")

axs[0].legend()
axs[0].grid(True)

# -------------------------------
# Error Plot
# -------------------------------

axs[1].plot(
    time,
    error_signal,
    color="red",
    linewidth=1
)

axs[1].set_title("Twin Tracking Error")

axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Error")

axs[1].grid(True)

plt.tight_layout()

# -----------------------------------
# SAVE FIGURE
# -----------------------------------

os.makedirs("results", exist_ok=True)

save_path = "results/fidelity_validation.png"

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"✅ Fidelity plot saved -> {save_path}")

plt.show()