import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

print("\nLoading actuator dataset...\n")

df = pd.read_csv("data/flight_logs/actuator_jam.csv")

time = df["time_s"]

nominal = df["elevator_nominal"]

faulted = df["elevator_jammed"]

fault_start = df[df["fault_label"] == 1]["time_s"].iloc[0]

rmse = np.sqrt(mean_squared_error(nominal, faulted))

peak_error = np.max(np.abs(nominal - faulted))

steady_state_error = np.mean(
    np.abs(nominal.tail(100) - faulted.tail(100))
)

print("========== Controller Validation ==========\n")

print(f"RMSE              : {rmse:.6f}")
print(f"Peak Error        : {peak_error:.6f}")
print(f"Steady-State Error: {steady_state_error:.6f}")

print("\n===========================================\n")

plt.figure(figsize=(12,6))

plt.plot(
    time,
    nominal,
    label="Nominal Elevator"
)

plt.plot(
    time,
    faulted,
    label="Faulted Elevator"
)

plt.axvline(
    fault_start,
    linestyle="--",
    label="Fault Injection"
)

plt.title(
    "Controller Validation Under Actuator Jam Fault"
)

plt.xlabel("Time (s)")
plt.ylabel("Elevator Command")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/controller_validation.png",
    dpi=300
)

plt.show()

print(
    "✅ Plot saved -> results/controller_validation.png"
)