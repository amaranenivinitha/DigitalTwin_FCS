import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler

# =========================================================
# Load dataset
# =========================================================

print("\nLoading sensor freeze dataset...\n")

df = pd.read_csv(
    "data/flight_logs/sensor_freeze.csv"
)

time = df["time_s"].values

nominal = df["gyro_nominal"].values

frozen = df["gyro_frozen"].values

labels = df["fault_label"].values

# =========================================================
# Residual
# =========================================================

residual = frozen - nominal

# =========================================================
# Scale
# =========================================================

scaler = StandardScaler()

residual_scaled = scaler.fit_transform(
    residual.reshape(-1, 1)
)

# =========================================================
# Sequence generation
# =========================================================

SEQ_LEN = 30

X = []

for i in range(len(residual_scaled) - SEQ_LEN):

    X.append(
        residual_scaled[i:i+SEQ_LEN]
    )

X = np.array(X)

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
)

# =========================================================
# LSTM Autoencoder
# =========================================================

class LSTMAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=32,
            batch_first=True
        )

        self.latent = nn.Linear(32, 16)

        self.decoder_input = nn.Linear(16, 32)

        self.decoder = nn.LSTM(
            input_size=32,
            hidden_size=32,
            batch_first=True
        )

        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)

        latent = self.latent(hidden[-1])

        repeated = self.decoder_input(latent)

        repeated = repeated.unsqueeze(1).repeat(
            1,
            x.shape[1],
            1
        )

        decoded, _ = self.decoder(repeated)

        output = self.output_layer(decoded)

        return output

# =========================================================
# Load trained model
# =========================================================

model = LSTMAutoencoder()

model.load_state_dict(
    torch.load(
        "ml/sensor_freeze_detector.pth"
    )
)

model.eval()

# =========================================================
# Reconstruction error
# =========================================================

with torch.no_grad():

    reconstructed = model(X_tensor)

errors = torch.mean(
    (X_tensor - reconstructed) ** 2,
    dim=(1, 2)
).numpy()

# =========================================================
# Threshold
# =========================================================

threshold = np.percentile(errors, 95)

predictions = (
    errors > threshold
).astype(int)

# =========================================================
# Time alignment
# =========================================================

aligned_time = time[SEQ_LEN:]

aligned_labels = labels[SEQ_LEN:]

# =========================================================
# Plot
# =========================================================

plt.figure(figsize=(15, 6))

plt.plot(
    aligned_time,
    errors,
    label="Reconstruction Error",
    linewidth=2
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

# Actual freeze region
fault_region = aligned_labels == 1

plt.fill_between(
    aligned_time,
    0,
    max(errors),
    where=fault_region,
    alpha=0.2,
    label="Actual Freeze Region"
)

# Detected anomalies
anomaly_indices = np.where(
    predictions == 1
)[0]

plt.scatter(
    aligned_time[anomaly_indices],
    errors[anomaly_indices],
    s=40,
    label="Detected Anomalies"
)

plt.xlabel("Time (s)")

plt.ylabel("Reconstruction Error")

plt.title(
    "Sensor Freeze Anomaly Timeline"
)

plt.legend()

plt.grid(True)

# =========================================================
# Save figure
# =========================================================

save_path = (
    "results/sensor_freeze_timeline.png"
)

plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"\n✅ Figure saved -> {save_path}"
)

plt.show()