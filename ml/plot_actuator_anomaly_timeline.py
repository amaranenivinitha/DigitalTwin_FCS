import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch

# ============================================
# LSTM AUTOENCODER
# ============================================

class LSTMAutoencoder(torch.nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        latent_size=16,
        seq_len=50
    ):

        super().__init__()

        self.seq_len = seq_len

        # Encoder
        self.encoder = torch.nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )

        # Latent space
        self.latent = torch.nn.Linear(
            hidden_size,
            latent_size
        )

        # Decoder input
        self.decoder_input = torch.nn.Linear(
            latent_size,
            hidden_size
        )

        # Decoder
        self.decoder = torch.nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True
        )

        # Output layer
        self.output_layer = torch.nn.Linear(
            hidden_size,
            input_size
        )

    def forward(self, x):

        enc_out, _ = self.encoder(x)

        z = self.latent(enc_out[:, -1, :])

        dec_input = torch.relu(
            self.decoder_input(z)
        )

        dec_input = dec_input.unsqueeze(1).repeat(
            1,
            self.seq_len,
            1
        )

        dec_out, _ = self.decoder(dec_input)

        out = self.output_layer(dec_out)

        return out

# ============================================
# LOAD DATA
# ============================================

print("\nLoading actuator dataset...\n")

df = pd.read_csv(
    "data/flight_logs/actuator_jam.csv"
)

time = df["time_s"].values

nominal = df["elevator_nominal"].values

jammed = df["elevator_jammed"].values

fault_label = df["fault_label"].values

# ============================================
# RESIDUAL SIGNAL
# ============================================

residual = jammed - nominal

residual = residual.reshape(-1, 1)

# ============================================
# SCALE
# ============================================

scaler = StandardScaler()

residual_scaled = scaler.fit_transform(
    residual
)

# ============================================
# WINDOW FUNCTION
# ============================================

SEQ_LEN = 50

def make_windows(x, seq_len):

    windows = []

    for i in range(len(x) - seq_len + 1):

        windows.append(
            x[i:i+seq_len]
        )

    return np.array(windows)

windows = make_windows(
    residual_scaled,
    SEQ_LEN
)

# ============================================
# LOAD MODEL
# ============================================

device = torch.device("cpu")

model = LSTMAutoencoder(
    input_size=1,
    hidden_size=64,
    latent_size=16,
    seq_len=SEQ_LEN
)

model.load_state_dict(
    torch.load(
        "data/models/actuator_lstm_ae.pth",
        map_location=device
    )
)

model.eval()

# ============================================
# RECONSTRUCTION ERROR
# ============================================

tensor_data = torch.tensor(
    windows,
    dtype=torch.float32
)

with torch.no_grad():

    recon = model(
        tensor_data
    ).numpy()

recon_error = np.mean(
    (recon - windows) ** 2,
    axis=(1, 2)
)

# ============================================
# THRESHOLD
# ============================================

threshold = np.percentile(
    recon_error,
    95
)

# ============================================
# DETECTIONS
# ============================================

detections = recon_error > threshold

# ============================================
# TIMELINE
# ============================================

timeline = time[SEQ_LEN - 1:]

fault_region = fault_label[SEQ_LEN - 1:]

# ============================================
# PLOT
# ============================================

plt.figure(figsize=(14, 6))

# Reconstruction error
plt.plot(
    timeline,
    recon_error,
    linewidth=2,
    label="Reconstruction Error"
)

# Threshold
plt.axhline(
    threshold,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

# Actual jam region
plt.fill_between(
    timeline,
    0,
    np.max(recon_error),
    where=fault_region == 1,
    alpha=0.2,
    label="Actual Jam Region"
)

# Detected anomalies
plt.scatter(
    timeline[detections],
    recon_error[detections],
    s=30,
    label="Detected Anomalies"
)

# Labels
plt.title(
    "Actuator Fault Anomaly Timeline"
)

plt.xlabel("Time (s)")

plt.ylabel("Reconstruction Error")

plt.legend()

plt.grid(True)

# ============================================
# SAVE FIGURE
# ============================================

plt.savefig(
    "results/actuator_anomaly_timeline.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "\n✅ Figure saved -> results/actuator_anomaly_timeline.png\n"
)

# ============================================
# SHOW
# ============================================

plt.show()