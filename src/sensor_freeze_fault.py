import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import chirp

# =========================================================
# Generate synthetic gyroscope signal
# =========================================================

dt = 0.01
t = np.arange(0, 40, dt)

gyro_signal = (
    0.5 * np.sin(0.5 * t)
    + 0.3 * np.sin(2 * t)
    + 0.05 * np.random.randn(len(t))
)

# Add aggressive maneuver spikes
gyro_signal += 0.8 * chirp(t, f0=0.2, f1=3, t1=40)

# =========================================================
# Inject sensor freeze fault
# =========================================================

freeze_start = 20
freeze_end = 32

freeze_mask = (t >= freeze_start) & (t <= freeze_end)

gyro_frozen = gyro_signal.copy()

# Freeze sensor value
freeze_value = gyro_signal[np.where(t >= freeze_start)[0][0]]

gyro_frozen[freeze_mask] = freeze_value

# =========================================================
# Fault labels
# =========================================================

fault_label = np.zeros(len(t))
fault_label[freeze_mask] = 1

# =========================================================
# Save dataset
# =========================================================

df = pd.DataFrame({
    "time_s": t,
    "gyro_nominal": gyro_signal,
    "gyro_frozen": gyro_frozen,
    "fault_label": fault_label
})

save_path = "data/flight_logs/sensor_freeze.csv"
df.to_csv(save_path, index=False)

print(f"✅ Sensor freeze dataset saved -> {save_path}")

# =========================================================
# Plot fault injection
# =========================================================

plt.figure(figsize=(14, 6))

plt.plot(t, gyro_signal,
         label="Nominal Signal",
         linewidth=2)

plt.plot(t, gyro_frozen,
         label="Frozen Signal",
         linewidth=2)

plt.axvspan(
    freeze_start,
    freeze_end,
    color='red',
    alpha=0.2,
    label="Freeze Region"
)

plt.xlabel("Time (s)")
plt.ylabel("Gyro Rate (rad/s)")
plt.title("Sensor Freeze Fault Injection")
plt.legend()
plt.grid(True)

fig_path = "results/sensor_freeze_fault.png"

plt.savefig(fig_path, dpi=300, bbox_inches='tight')

print(f"✅ Figure saved -> {fig_path}")

plt.show()