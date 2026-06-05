import jsbsim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# Initialize JSBSim
# -----------------------------------
sim = jsbsim.FGFDMExec(None)

sim.load_model('c172p')

sim.set_dt(0.01)

# -----------------------------------
# Initial Flight Conditions
# -----------------------------------
sim['ic/h-sl-ft'] = 5000
sim['ic/u-fps'] = 200
sim['ic/terrain-elevation-ft'] = 0

sim.run_ic()

# -----------------------------------
# PID Parameters
# -----------------------------------
Kp = 0.08
Ki = 0.02
Kd = 0.03

integral = 0.0
prev_error = 0.0

dt = sim.get_delta_t()

# -----------------------------------
# Logging
# -----------------------------------
time_log = []
pitch_log = []
pitch_rate_log = []
elevator_log = []
altitude_log = []

# -----------------------------------
# Simulation Loop
# -----------------------------------
for i in range(4000):

    t = i * dt

    sim.run()

    # Dynamic pitch target (changes over time)
    if t < 10:
        target_pitch = 5
    elif t < 20:
        target_pitch = -5
    elif t < 30:
        target_pitch = 10
    else:
        target_pitch = 0

    current_pitch = sim['attitude/theta-deg']

    error = target_pitch - current_pitch

    # PID
    integral += error * dt

    derivative = (error - prev_error) / dt

    prev_error = error

    elevator_cmd = (
        Kp * error +
        Ki * integral +
        Kd * derivative
    )

    elevator_cmd = np.clip(elevator_cmd, -1, 1)

    # Apply command
    sim['fcs/elevator-cmd-norm'] = elevator_cmd

    # Logs
    time_log.append(t)

    pitch_log.append(current_pitch)

    pitch_rate_log.append(
        sim['velocities/q-rad_sec']
    )

    elevator_log.append(elevator_cmd)

    altitude_log.append(
        sim['position/h-sl-ft']
    )

# -----------------------------------
# Save CSV
# -----------------------------------
df = pd.DataFrame({
    'time_s': time_log,
    'pitch_deg': pitch_log,
    'pitch_rate_rad_s': pitch_rate_log,
    'elevator_cmd': elevator_log,
    'altitude_ft': altitude_log
})

df.to_csv(
    'data/flight_logs/pitch_control.csv',
    index=False
)

print("✅ Dynamic flight data generated")

# -----------------------------------
# Plots
# -----------------------------------
plt.figure(figsize=(10,6))

plt.subplot(311)
plt.plot(time_log, pitch_log)
plt.title("Pitch Angle")

plt.subplot(312)
plt.plot(time_log, pitch_rate_log)
plt.title("Pitch Rate")

plt.subplot(313)
plt.plot(time_log, elevator_log)
plt.title("Elevator Command")

plt.tight_layout()
plt.show()