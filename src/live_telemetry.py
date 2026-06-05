import numpy as np
import pandas as pd
import time
import os

# =========================================================
# Output file
# =========================================================

OUTPUT_FILE = "data/live/live_telemetry.csv"

os.makedirs(
    "data/live",
    exist_ok=True
)

# =========================================================
# Time settings
# =========================================================

dt = 0.05

t = 0

print("\nStarting live telemetry stream...\n")

# =========================================================
# Infinite stream loop
# =========================================================

while True:

    # Nominal aircraft gyro signal
    gyro = (
        np.sin(0.5 * t)
        +
        0.3 * np.sin(5 * t)
    )

    # Inject fault after 20 sec
    if 20 < t < 30:

        gyro_faulty = gyro + np.random.normal(
            2.0,
            0.5
        )

        fault = 1

    else:

        gyro_faulty = gyro

        fault = 0

    # Create row
    row = pd.DataFrame({
        "time_s": [t],
        "gyro_nominal": [gyro],
        "gyro_faulty": [gyro_faulty],
        "fault_label": [fault]
    })

    # Append live data
    if not os.path.exists(OUTPUT_FILE):

        row.to_csv(
            OUTPUT_FILE,
            index=False
        )

    else:

        row.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=False,
            index=False
        )

    print(
        f"t={t:.2f} | gyro={gyro_faulty:.3f}"
    )

    time.sleep(dt)

    t += dt