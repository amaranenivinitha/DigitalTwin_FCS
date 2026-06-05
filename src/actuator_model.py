import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Parameters ---
dt = 0.01           # 100 Hz control rate
t_end = 10.0        # seconds
time = np.arange(0, t_end, dt)

# --- Simulated command input (elevator command in range -1..1) ---
# Example: a pulse command sequence
cmd = np.zeros_like(time)
cmd[(time > 2) & (time <= 4)] = 0.8
cmd[(time > 6) & (time <= 8)] = -0.8

# --- Actuator dynamics ---
rate_limit = 0.3        # max rate of change (unit/s)
delay_steps = 10        # 10 samples = 0.1 s delay
saturation_limit = 1.0  # physical travel limit

# Initialize
actuator_pos = np.zeros_like(time)
cmd_buffer = [0.0] * delay_steps

for i in range(1, len(time)):
    # simulate command delay
    cmd_buffer.append(cmd[i])
    delayed_cmd = cmd_buffer.pop(0)

    # apply rate limit
    delta = delayed_cmd - actuator_pos[i-1]
    delta = np.clip(delta, -rate_limit * dt, rate_limit * dt)

    # update actuator position with saturation
    actuator_pos[i] = np.clip(actuator_pos[i-1] + delta, -saturation_limit, saturation_limit)

# --- Save data ---
df = pd.DataFrame({'time_s': time, 'cmd': cmd, 'actuator_pos': actuator_pos})
df.to_csv('data/flight_logs/actuator_model.csv', index=False)
print("✅ Actuator data saved -> data/flight_logs/actuator_model.csv")

# --- Plot results ---
plt.figure(figsize=(8,4))
plt.plot(time, cmd, label='Command Input', linestyle='--')
plt.plot(time, actuator_pos, label='Actuator Output', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Normalized Deflection')
plt.legend()
plt.title('Actuator Dynamics: Lag + Delay + Saturation')
plt.tight_layout()
plt.show()
