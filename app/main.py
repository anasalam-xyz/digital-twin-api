# app/main.py
from pathlib import Path
import torch
from model.model_def import ClimateDualNet

HISTORY_LEN: int = 30  # days of historical input
FORECAST_LEN: int = 7  # days to predict
N_CHANNELS: int = 3
GRID_H: int = 21  # spatial grid height (not used by the model class itself,
GRID_W: int = 13
MODEL_PATH = Path(__file__).parent / "model" / "weights" / "best_model.pt"

model = ClimateDualNet(
    in_channels=N_CHANNELS,
    history_len=HISTORY_LEN,
    forecast_len=FORECAST_LEN,
)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

print("Loaded successfully.")
print("Epoch:", checkpoint["epoch"])
print("Val RMSE:", checkpoint["best_val_rmse"])
