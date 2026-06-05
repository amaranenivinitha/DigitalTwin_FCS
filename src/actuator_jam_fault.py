import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# LOAD CONTROLLER DATA
# ---------------------------------------

df = pd.read_csv("data/flight_logs/pitch_control.csv")

# ---------------------------------------
# CREATE JAMMED ELEVATOR SIGNAL
# ---------------------------------------

# Assume elevator command column exists
# If your column name differs, adapt later

elevator = df["elevator_cmd"].values.copy()

time = df["time_s"].values

# ---------------------------------------
# JAM PARAMETERS
# ---------------------------------------

jam_start = 20.0
jam_end   = 32.0

# Find jam start index
jam_idx = np.where(time >= jam_start)[0][0]

# Freeze actuator value
jam_value = elevator[jam_idx]

jammed = elevator.copy()

# Apply jam
for i in range(len(time)):

    if jam_start <= time[i] <= jam_end:
        jammed[i] = jam_value

# ---------------------------------------
# CREATE LABELS
# ---------------------------------------

fault_label = np.zeros(len(time), dtype=int)

fault_label[
    (time >= jam_start) &
    (time <= jam_end)
] = 1

# ---------------------------------------
# SAVE DATASET
# ---------------------------------------

out = pd.DataFrame({
    "time_s": time,
    "elevator_nominal": elevator,
    "elevator_jammed": jammed,
    "fault_label": fault_label
})

os.makedirs("data/flight_logs", exist_ok=True)

save_path = "data/flight_logs/actuator_jam.csv"

out.to_csv(save_path, index=False)

print(f"✅ Actuator jam dataset saved -> {save_path}")

# ---------------------------------------
# VISUALIZATION
# ---------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    time,
    elevator,
    label="Nominal Elevator",
    linewidth=2
)

plt.plot(
    time,
    jammed,
    label="Jammed Elevator",
    linewidth=2
)

plt.axvspan(
    jam_start,
    jam_end,
    color="red",
    alpha=0.2,
    label="Jam Region"
)

plt.title("Actuator Jam Fault Injection")

plt.xlabel("Time (s)")
plt.ylabel("Elevator Command")

plt.legend()

plt.grid(True)

plt.tight_layout()

os.makedirs("results", exist_ok=True)

fig_path = "results/actuator_jam_fault.png"

plt.savefig(
    fig_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"✅ Figure saved -> {fig_path}")

plt.show()