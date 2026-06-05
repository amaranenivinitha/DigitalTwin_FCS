import pandas as pd
import numpy as np

df = pd.read_csv('data/flight_logs/pitch_control.csv')

np.random.seed(42)

gyro = df['pitch_rate_rad_s'].values

gyro_noise = np.random.normal(0, 0.01, len(gyro))

gyro_bias = 0.02

gyro_signal = gyro + gyro_noise + gyro_bias

imu_df = pd.DataFrame({
    'time_s': df['time_s'],
    'gyro_pitch_rad_s': gyro_signal
})

imu_df.to_csv(
    'data/flight_logs/imu_emulated.csv',
    index=False
)

print("✅ Improved IMU data generated")