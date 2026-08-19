import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# ─── Configuration ────────────────────────────────────────────────────────────
HISTORY_LEN: int = 30  # days of historical input
FORECAST_LEN: int = 7  # days to predict
N_CHANNELS: int = 3
GRID_H: int = 21  # spatial grid height (not used by the model class itself,
GRID_W: int = 13  # but needed when building input tensors for /predict)


# ─── Building Blocks ──────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Two-conv residual block with BatchNorm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        # Channel attention
        self.ca_fc1 = nn.Linear(channels, mid)
        self.ca_fc2 = nn.Linear(mid, channels)
        # Spatial attention
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg_pool = x.mean(dim=[2, 3])  # (B, C)
        max_pool = x.amax(dim=[2, 3])  # (B, C)
        ca = (
            torch.sigmoid(
                self.ca_fc2(F.relu(self.ca_fc1(avg_pool)))
                + self.ca_fc2(F.relu(self.ca_fc1(max_pool)))
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # (B, C, 1, 1)
        x = x * ca
        # Spatial attention
        sa_in = torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1
        )  # (B, 2, H, W)
        sa = torch.sigmoid(self.sa_conv(sa_in))  # (B, 1, H, W)
        return x * sa


class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell (one timestep)."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3) -> None:
        super().__init__()
        pad = kernel // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch, kernel, padding=pad, bias=True
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
        self, batch: int, h: int, w: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(batch, self.hidden_ch, h, w, device=device)
        return z, z.clone()


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM that processes a sequence of spatial feature maps."""

    def __init__(
        self,
        in_ch: int,
        hidden_ch: int,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        cells = [
            ConvLSTMCell(in_ch if i == 0 else hidden_ch, hidden_ch)
            for i in range(n_layers)
        ]
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x : (B, T, C_in, H, W)
        returns:
            all_hidden : (B, T, hidden_ch, H, W)  — all hidden states
            states     : list of (h, c) for each layer at the last timestep
        """
        B, T, _, H, W = x.shape
        device = x.device

        # Initialise hidden/cell states
        states = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        layer_input = x  # (B, T, C, H, W)
        for layer_idx, cell in enumerate(self.cells):
            h, c = states[layer_idx]
            outputs = []
            for t in range(T):
                h, c = cell(layer_input[:, t], h, c)
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)  # (B, T, hidden, H, W)
            if layer_idx < self.n_layers - 1:
                # Apply spatial dropout between layers
                B2, T2, C2, H2, W2 = layer_input.shape
                layer_input = self.dropout(layer_input.view(B2 * T2, C2, H2, W2)).view(
                    B2, T2, C2, H2, W2
                )
            states[layer_idx] = (h, c)

        return layer_input, states  # (B, T, hidden, H, W)


class TemporalSelfAttention(nn.Module):
    """
    Lightweight temporal self-attention over the T time dimension.
    Uses spatial average pooling to reduce memory footprint.
    """

    def __init__(self, hidden_ch: int, n_heads: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B*T, C) → scalar per channel
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_ch, num_heads=n_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_ch)
        self.proj = nn.Linear(hidden_ch, hidden_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, C, H, W)  →  returns (B, T, C, H, W) attention-weighted."""
        B, T, C, H, W = x.shape
        # Summarise spatial → temporal token sequence
        tokens = self.pool(x.view(B * T, C, H, W)).view(B, T, C)  # (B, T, C)
        attn_out, _ = self.attn(tokens, tokens, tokens)  # (B, T, C)
        weights = torch.sigmoid(self.proj(self.norm(attn_out)))  # (B, T, C)
        # Broadcast weight back to spatial dims
        return x * weights.view(B, T, C, 1, 1)


# ─── Full Model: ClimateDualNet ────────────────────────────────────────────────


class ClimateDualNet(nn.Module):
    """
    Spatiotemporal climate forecasting model.

    Pipeline:
      1. Per-frame residual CNN encoder  (32 → 64 → 128 channels)
      2. CBAM attention on each frame
      3. Two-layer ConvLSTM  (hidden=128)
      4. Temporal self-attention over all 30 input frames
      5. UNet decoder with skip connections (from encoder stages)
      6. Three independent prediction heads
         forecasting FORECAST_LEN frames for Rainfall, Tmax, Tmin
    """

    def __init__(
        self,
        in_channels: int,
        history_len: int,
        forecast_len: int,
        enc_channels: Tuple[int, ...] = (32, 64, 128),
        lstm_hidden: int = 128,
        n_lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.lstm_hidden = lstm_hidden

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = ResidualBlock(in_channels, enc_channels[0])
        self.enc2 = ResidualBlock(enc_channels[0], enc_channels[1])
        self.enc3 = ResidualBlock(enc_channels[1], enc_channels[2])

        # CBAM after final encoder stage
        self.cbam = CBAM(enc_channels[2])

        # ── ConvLSTM ─────────────────────────────────────────────────────────
        self.convlstm = ConvLSTM(
            in_ch=enc_channels[2],
            hidden_ch=lstm_hidden,
            n_layers=n_lstm_layers,
            dropout=lstm_dropout,
        )

        # ── Temporal Attention ───────────────────────────────────────────────
        self.temp_attn = TemporalSelfAttention(lstm_hidden, n_heads=4)

        # ── UNet Decoder ─────────────────────────────────────────────────────
        # Skip connections: enc1→dec3, enc2→dec2, enc3→dec1
        self.dec1 = ResidualBlock(lstm_hidden + enc_channels[2], enc_channels[1])
        self.dec2 = ResidualBlock(enc_channels[1] + enc_channels[1], enc_channels[0])
        self.dec3 = ResidualBlock(enc_channels[0] + enc_channels[0], enc_channels[0])

        # ── Prediction Heads ─────────────────────────────────────────────────
        # Each head outputs (B, forecast_len, H, W)
        head_in = enc_channels[0]
        self.head_rainfall = nn.Sequential(
            nn.Conv2d(head_in, head_in, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, forecast_len, 1),
        )
        self.head_tmax = nn.Sequential(
            nn.Conv2d(head_in, head_in, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, forecast_len, 1),
        )
        self.head_tmin = nn.Sequential(
            nn.Conv2d(head_in, head_in, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, forecast_len, 1),
        )

    def _encode_frame(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a single spatial frame through the residual encoder."""
        s1 = self.enc1(x)  # (B, 32, H, W)
        s2 = self.enc2(s1)  # (B, 64, H, W)
        s3 = self.cbam(self.enc3(s2))  # (B, 128, H, W)
        return s1, s2, s3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T_in, C, H, W)

        Returns
        -------
        out : (B, T_out, C, H, W)
        """
        B, T, C, H, W = x.shape

        # ── Per-frame encoding ────────────────────────────────────────────────
        enc_frames = []  # list of (s1, s2, s3) per timestep
        for t in range(T):
            s1, s2, s3 = self._encode_frame(x[:, t])  # each (B, ch, H, W)
            enc_frames.append((s1, s2, s3))

        # Stack encoder-3 outputs for ConvLSTM input
        enc3_seq = torch.stack([f[2] for f in enc_frames], dim=1)  # (B, T, 128, H, W)

        # Save last-frame encoder states for skip connections in decoder
        skip_s1 = enc_frames[-1][0]  # (B, 32, H, W)
        skip_s2 = enc_frames[-1][1]  # (B, 64, H, W)
        skip_s3 = enc_frames[-1][2]  # (B, 128, H, W)

        # ── ConvLSTM ─────────────────────────────────────────────────────────
        lstm_out, _ = self.convlstm(enc3_seq)  # (B, T, 128, H, W)

        # ── Temporal Self-Attention ───────────────────────────────────────────
        attn_out = self.temp_attn(lstm_out)  # (B, T, 128, H, W)

        # Use temporally-attended last state as decoder input
        context = attn_out[:, -1]  # (B, 128, H, W)

        # ── UNet Decoder ──────────────────────────────────────────────────────
        d1 = self.dec1(torch.cat([context, skip_s3], dim=1))  # (B, 64, H, W)
        d2 = self.dec2(torch.cat([d1, skip_s2], dim=1))  # (B, 32, H, W)
        d3 = self.dec3(torch.cat([d2, skip_s1], dim=1))  # (B, 32, H, W)

        # ── Prediction Heads ─────────────────────────────────────────────────
        rain = self.head_rainfall(d3)  # (B, T_out, H, W)
        tmax = self.head_tmax(d3)
        tmin = self.head_tmin(d3)

        # Stack into (B, T_out, 3, H, W)
        out = torch.stack([rain, tmax, tmin], dim=2)  # (B, T_out, C, H, W)
        return out
