import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, latent_size=16, seq_len=50):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.enc_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.enc_fc = nn.Linear(hidden_size, latent_size)
        self.dec_fc = nn.Linear(latent_size, hidden_size)
        self.dec_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        enc_out, _ = self.enc_lstm(x)                  # (b, seq, hidden)
        z = self.enc_fc(enc_out[:, -1, :])             # (b, latent)
        dec_init = torch.relu(self.dec_fc(z)).unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out, _ = self.dec_lstm(dec_init)
        out = self.out_fc(dec_out)
        return out
