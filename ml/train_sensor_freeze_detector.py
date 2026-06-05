import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

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
# Residual generation
# =========================================================

residual = (
    frozen - nominal
).reshape(-1, 1)

# =========================================================
# Scaling
# =========================================================

print("Scaling residual sensor data...\n")

scaler = StandardScaler()

scaled = scaler.fit_transform(
    residual
)

# =========================================================
# Sequence creation
# =========================================================

SEQ_LEN = 20

sequences = []

for i in range(len(scaled) - SEQ_LEN):

    sequences.append(
        scaled[i:i + SEQ_LEN]
    )

sequences = np.array(sequences)

X = torch.tensor(
    sequences,
    dtype=torch.float32
)

# =========================================================
# LSTM Autoencoder
# =========================================================

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_dim=32,
        latent_dim=16
    ):

        super().__init__()

        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )

        self.latent = nn.Linear(
            hidden_dim,
            latent_dim
        )

        self.decoder_input = nn.Linear(
            latent_dim,
            hidden_dim
        )

        self.decoder = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            input_dim
        )

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)

        latent = self.latent(
            hidden[-1]
        )

        repeated = self.decoder_input(
            latent
        )

        repeated = repeated.unsqueeze(1).repeat(
            1,
            x.shape[1],
            1
        )

        decoded, _ = self.decoder(
            repeated
        )

        output = self.output_layer(
            decoded
        )

        return output

# =========================================================
# Model
# =========================================================

model = LSTMAutoencoder()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================================================
# Training
# =========================================================

print("Starting sensor freeze training...\n")

EPOCHS = 40

for epoch in range(EPOCHS):

    model.train()

    output = model(X)

    loss = criterion(
        output,
        X
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"| Loss = {loss.item():.6f}"
        )

# =========================================================
# Reconstruction error
# =========================================================

model.eval()

with torch.no_grad():

    reconstructed = model(X)

errors = torch.mean(
    (X - reconstructed) ** 2,
    dim=(1, 2)
).numpy()

# =========================================================
# Threshold
# =========================================================

print("\nComputing reconstruction threshold...\n")

threshold = np.percentile(
    errors,
    95
)

print(
    f"Threshold = {threshold:.8f}"
)

# =========================================================
# Predictions
# =========================================================

predictions = (
    errors > threshold
).astype(int)

aligned_labels = labels[SEQ_LEN:]

# =========================================================
# Metrics
# =========================================================

print("\nEvaluating sensor freeze detection...\n")

cm = confusion_matrix(
    aligned_labels,
    predictions
)

precision = precision_score(
    aligned_labels,
    predictions
)

recall = recall_score(
    aligned_labels,
    predictions
)

f1 = f1_score(
    aligned_labels,
    predictions
)

tn, fp, fn, tp = cm.ravel()

far = fp / (fp + tn)

# =========================================================
# Time-to-detect
# =========================================================

fault_indices = np.where(
    aligned_labels == 1
)[0]

detect_indices = np.where(
    predictions == 1
)[0]

if len(detect_indices) > 0:

    detection_index = detect_indices[0]

    detection_time = time[
        detection_index + SEQ_LEN
    ]

else:

    detection_time = -1

fault_start = time[
    fault_indices[0] + SEQ_LEN
]

ttd = detection_time - fault_start

# =========================================================
# Results
# =========================================================

print("\nConfusion Matrix:")

print(cm)

print(f"\nPrecision = {precision:.3f}")

print(f"Recall    = {recall:.3f}")

print(f"F1 Score  = {f1:.3f}")

print(f"\nFalse Alarm Rate = {far:.4f}")

print(f"\nTime-to-Detect = {ttd:.4f} sec")

# =========================================================
# Save model
# =========================================================

torch.save(
    model.state_dict(),
    "ml/sensor_freeze_detector.pth"
)

print(
    "\n✅ Model saved -> ml/sensor_freeze_detector.pth"
)

print(
    "\n✅ Sensor freeze detector trained successfully"
)