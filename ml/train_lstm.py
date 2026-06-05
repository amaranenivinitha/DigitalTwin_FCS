import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from ml.lstm_autoencoder import LSTMAutoencoder
import argparse

# -----------------------------
# Argument Parser
# -----------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--seq_len', type=int, default=50)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

# -----------------------------
# Load Data
# -----------------------------
print("Loading datasets...")

nominal = pd.read_csv('data/flight_logs/imu_emulated.csv')
faulted = pd.read_csv('data/flight_logs/imu_faulted.csv')

# -----------------------------
# Extract Features
# -----------------------------
x_nom = nominal['gyro_pitch_rad_s'].values.reshape(-1, 1)

# Clean NaN / Inf
x_nom = np.nan_to_num(
    x_nom,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

# Clip extreme values
x_nom = np.clip(x_nom, -10, 10)

x_fault = faulted[['gyro_pitch_rad_s_faulty', 'fault_label', 'time_s']].copy()

# -----------------------------
# Standardization
# -----------------------------
print("Scaling data...")

scaler = StandardScaler()
x_nom_s = scaler.fit_transform(x_nom)

# -----------------------------
# Window Creation
# -----------------------------
def make_windows(x, seq_len):
    N = len(x)

    windows = []

    for i in range(0, N - seq_len + 1):
        windows.append(x[i:i + seq_len])

    return np.stack(windows)

seq_len = args.seq_len

train_windows = make_windows(x_nom_s, seq_len)

# -----------------------------
# Torch Dataset
# -----------------------------
train_tensor = torch.tensor(train_windows, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(train_tensor, train_tensor),
    batch_size=args.batch,
    shuffle=True
)

# -----------------------------
# Model Setup
# -----------------------------
device = torch.device(args.device)

model = LSTMAutoencoder(
    input_size=1,
    hidden_size=64,
    latent_size=16,
    seq_len=seq_len
).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr
)

print("\nStarting training...\n")

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(1, args.epochs + 1):

    model.train()

    epoch_loss = 0.0

    for xb, _ in train_loader:

        xb = xb.to(device)

        recon = model(xb)

        loss = criterion(recon, xb)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)

    epoch_loss /= len(train_loader.dataset)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch}/{args.epochs} | Loss = {epoch_loss:.6f}")

# -----------------------------
# Reconstruction Error (Train)
# -----------------------------
print("\nComputing reconstruction threshold...")

model.eval()

with torch.no_grad():

    recon = model(train_tensor.to(device)).cpu().numpy()

train_recon_err = np.mean(
    (recon - train_windows) ** 2,
    axis=(1, 2)
)

threshold = np.percentile(train_recon_err, 95.0)

print(f"Threshold = {threshold:.8f}")

# -----------------------------
# Evaluate on Faulted Data
# -----------------------------
print("\nEvaluating on faulted data...")

x_fault_series = x_fault['gyro_pitch_rad_s_faulty'].values.reshape(-1, 1)

# Clean
x_fault_series = np.nan_to_num(
    x_fault_series,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

x_fault_series = np.clip(x_fault_series, -10, 10)

x_fault_s = scaler.transform(x_fault_series)

fault_windows = make_windows(x_fault_s, seq_len)

fault_tensor = torch.tensor(
    fault_windows,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    recon_fault = model(fault_tensor).cpu().numpy()

fault_recon_err = np.mean(
    (recon_fault - fault_windows) ** 2,
    axis=(1, 2)
)

# -----------------------------
# Generate Predictions
# -----------------------------
num_samples = len(x_fault_series)

sample_flags = np.zeros(num_samples, dtype=int)

for i, err in enumerate(fault_recon_err):

    if err > threshold:
        sample_flags[i:i + seq_len] = 1

# -----------------------------
# Ground Truth Alignment
# -----------------------------
gt = x_fault['fault_label'].values[seq_len - 1:]

pred = sample_flags[seq_len - 1:]

# -----------------------------
# Metrics
# -----------------------------
prec, rec, f1, _ = precision_recall_fscore_support(
    gt,
    pred,
    average='binary',
    zero_division=0
)

cm = confusion_matrix(gt, pred)

print("\nConfusion Matrix:")
print(cm)

print(f"\nPrecision = {prec:.3f}")
print(f"Recall    = {rec:.3f}")
print(f"F1 Score  = {f1:.3f}")

# -----------------------------
# Save Model + Results
# -----------------------------
os.makedirs('data/models', exist_ok=True)

torch.save(
    model.state_dict(),
    'data/models/lstm_ae.pth'
)

np.save(
    'data/models/train_recon_err.npy',
    train_recon_err
)

np.save(
    'data/models/fault_recon_err.npy',
    fault_recon_err
)

with open('data/models/threshold.txt', 'w') as f:
    f.write(str(threshold))

print("\n✅ Training and evaluation completed successfully")
print("✅ Model saved -> data/models/lstm_ae.pth")