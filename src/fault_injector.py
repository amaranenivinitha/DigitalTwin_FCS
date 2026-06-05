import pandas as pd
import numpy as np

# -----------------------------------
# Load IMU data
# -----------------------------------
df = pd.read_csv('data/flight_logs/imu_emulated.csv')

gyro = df['gyro_pitch_rad_s'].values.copy()

fault_label = np.zeros(len(gyro))

# -----------------------------------
# Inject realistic faults
# -----------------------------------

# Fault region
fault_start = 2500
fault_end = 3500

# 1. Bias drift
gyro[fault_start:fault_end] += np.linspace(
    0,
    1.5,
    fault_end - fault_start
)

# 2. Oscillatory corruption
gyro[fault_start:fault_end] += (
    0.8 * np.sin(
        np.linspace(0, 80, fault_end - fault_start)
    )
)

# 3. Random spikes
np.random.seed(42)

spike_idx = np.random.randint(
    fault_start,
    fault_end,
    50
)

gyro[spike_idx] += np.random.normal(
    3,
    1,
    len(spike_idx)
)

# 4. Sensor dropout
gyro[3200:3250] = 0

# Labels
fault_label[fault_start:fault_end] = 1

# -----------------------------------
# Save
# -----------------------------------
fault_df = pd.DataFrame({
    'time_s': df['time_s'],
    'gyro_pitch_rad_s_faulty': gyro,
    'fault_label': fault_label.astype(int)
})

fault_df.to_csv(
    'data/flight_logs/imu_faulted.csv',
    index=False
)

print("✅ Advanced fault scenarios injected")