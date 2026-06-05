import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support
)

from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# LSTM AUTOENCODER
# ============================================================

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        latent_size=16,
        seq_len=50
    ):
        super().__init__()

        self.seq_len = seq_len

        # ---------------- ENCODER ----------------

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.latent = nn.Linear(
            hidden_size,
            latent_size
        )

        # ---------------- DECODER ----------------

        self.decoder_input = nn.Linear(
            latent_size,
            hidden_size
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            input_size
        )

    def forward(self, x):

        _, (hidden, _) = self.encoder(x)

        z = self.latent(hidden[-1])

        dec_input = self.decoder_input(z)

        dec_input = dec_input.unsqueeze(1).repeat(
            1,
            self.seq_len,
            1
        )

        dec_out, _ = self.decoder(dec_input)

        out = self.output_layer(dec_out)

        return out

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading actuator jam dataset...\n")

df = pd.read_csv(
    "data/flight_logs/actuator_jam.csv"
)

# ============================================================
# RESIDUAL ERROR SIGNAL
# ============================================================

nominal = df["elevator_nominal"].values

jammed = df["elevator_jammed"].values

# residual tracking error
residual = (jammed - nominal).reshape(-1, 1)

labels = df["fault_label"].values

time = df["time_s"].values

# ------------------------------------------------------------
# TRAIN ONLY ON HEALTHY DATA
# ------------------------------------------------------------

healthy_mask = labels == 0

x_nominal = residual[healthy_mask]

# ------------------------------------------------------------
# TEST ON FULL DATA
# ------------------------------------------------------------

x_faulted = residual

# ============================================================
# NORMALIZATION
# ============================================================

print("Scaling residual actuator data...\n")

scaler = StandardScaler()

x_nominal_scaled = scaler.fit_transform(x_nominal)

x_faulted_scaled = scaler.transform(x_faulted)

# ============================================================
# WINDOW CREATION
# ============================================================

SEQ_LEN = 50

def create_windows(x, seq_len):

    windows = []

    for i in range(len(x) - seq_len):

        windows.append(
            x[i:i+seq_len]
        )

    return np.array(windows)

train_windows = create_windows(
    x_nominal_scaled,
    SEQ_LEN
)

test_windows = create_windows(
    x_faulted_scaled,
    SEQ_LEN
)

# ============================================================
# TORCH DATA
# ============================================================

train_tensor = torch.tensor(
    train_windows,
    dtype=torch.float32
)

train_loader = DataLoader(
    TensorDataset(train_tensor, train_tensor),
    batch_size=64,
    shuffle=True
)

# ============================================================
# MODEL
# ============================================================

device = torch.device("cpu")

model = LSTMAutoencoder(
    input_size=1,
    hidden_size=64,
    latent_size=16,
    seq_len=SEQ_LEN
).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

# ============================================================
# TRAINING
# ============================================================

print("Starting actuator fault training...\n")

EPOCHS = 40

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for xb, _ in train_loader:

        xb = xb.to(device)

        recon = model(xb)

        loss = criterion(recon, xb)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if epoch == 0 or (epoch + 1) % 5 == 0:

        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss = {avg_loss:.6f}"
        )

# ============================================================
# TRAIN RECONSTRUCTION ERROR
# ============================================================

print("\nComputing reconstruction threshold...\n")

model.eval()

with torch.no_grad():

    recon_train = model(
        train_tensor.to(device)
    ).cpu().numpy()

train_error = np.mean(
    (recon_train - train_windows) ** 2,
    axis=(1, 2)
)

threshold = np.percentile(
    train_error,
    95
)

print(f"Threshold = {threshold:.8f}")

# ============================================================
# TEST ON FULL DATA
# ============================================================

print("\nEvaluating actuator jam detection...\n")

test_tensor = torch.tensor(
    test_windows,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    recon_test = model(
        test_tensor
    ).cpu().numpy()

test_error = np.mean(
    (recon_test - test_windows) ** 2,
    axis=(1, 2)
)

# ============================================================
# DETECTION
# ============================================================

predictions = (
    test_error > threshold
).astype(int)

# Align labels
gt = labels[SEQ_LEN:]

# ============================================================
# METRICS
# ============================================================

cm = confusion_matrix(
    gt,
    predictions
)

tn, fp, fn, tp = cm.ravel()

precision, recall, f1, _ = precision_recall_fscore_support(
    gt,
    predictions,
    average="binary",
    zero_division=0
)

print("\nConfusion Matrix:")
print(cm)

print(f"\nPrecision = {precision:.3f}")
print(f"Recall    = {recall:.3f}")
print(f"F1 Score  = {f1:.3f}")

# ============================================================
# FALSE ALARM RATE
# ============================================================

far = fp / (fp + tn)

print(f"\nFalse Alarm Rate = {far:.4f}")

# ============================================================
# TIME TO DETECT
# ============================================================

fault_indices = np.where(gt == 1)[0]

detect_indices = np.where(predictions == 1)[0]

if len(fault_indices) > 0:

    fault_start = fault_indices[0]

    valid_detect = detect_indices[
        detect_indices >= fault_start
    ]

    if len(valid_detect) > 0:

        detect_time = valid_detect[0]

        ttd = (
            detect_time - fault_start
        ) * 0.01

        print(
            f"\nTime-to-Detect = "
            f"{ttd:.4f} sec"
        )

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

np.save(
    "results/actuator_test_error.npy",
    test_error
)

np.save(
    "results/actuator_threshold.npy",
    np.array([threshold])
)

torch.save(
    model.state_dict(),
    "data/models/actuator_lstm_ae.pth"
)

print(
    "\n✅ Actuator fault detector "
    "trained successfully"
)