import jsbsim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Initialize JSBSim flight dynamics model
sim = jsbsim.FGFDMExec(None)
sim.load_model('c172p')  # Cessna 172 model
sim.set_dt(0.01)         # Simulation step: 100 Hz

time = []
altitude = []
pitch = []

# Run simulation for 10 seconds (1000 steps)
for i in range(1000):
    sim.run()
    time.append(i * 0.01)
    altitude.append(sim['position/h-sl-ft'])
    pitch.append(sim['attitude/theta-deg'])

# Save data to CSV for later ML training
df = pd.DataFrame({'time_s': time, 'alt_ft': altitude, 'pitch_deg': pitch})
df.to_csv('data/flight_logs/nominal.csv', index=False)

# Plot quick results
plt.figure(figsize=(8,4))
plt.subplot(211)
plt.plot(time, altitude)
plt.title("Altitude (ft)")

plt.subplot(212)
plt.plot(time, pitch)
plt.title("Pitch Angle (deg)")

plt.tight_layout()
plt.show()

print("✅ Saved nominal flight data -> data/flight_logs/nominal.csv")
