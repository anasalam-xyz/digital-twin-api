"""
digital-twin-api — FastAPI backend serving ClimateDualNet (Kerala pilot region).

Endpoints:
  GET  /health     liveness + data range check
  POST /forecast   predicted + actual (if known) + 7-day trend for a given date
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# On CPU-constrained hosting (e.g. free-tier Render), torch's default of
# spawning one thread per detected core causes severe contention instead of
# speedup, since the container often only gets a fraction of a real core.
# Pinning to 1 thread avoids that and is typically faster in this scenario.
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.model.model_def import ClimateDualNet, N_CHANNELS, HISTORY_LEN, FORECAST_LEN
from app.model.normalization import (
    load_norm_stats,
    normalize_tensor,
    denormalize_output,
)

torch.set_num_threads(1)
# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "weights" / "best_model.pt"
DATA_DIR = BASE_DIR / "model" / "data"
TRAIN_PATH = DATA_DIR / "train_tensor_kerala.npz"
VAL_PATH = DATA_DIR / "validation_tensor_kerala.npz"

CHANNEL_NAMES = ["rainfall", "tmax", "tmin"]

# ─── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="digital-twin-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # add your deployed frontend URL too
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load everything ONCE at startup ────────────────────────────────────────
print("Loading model...")
model = ClimateDualNet(
    in_channels=N_CHANNELS,
    history_len=HISTORY_LEN,
    forecast_len=FORECAST_LEN,
)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()
print(
    f"Model loaded. Epoch {checkpoint['epoch']}, val RMSE {checkpoint['best_val_rmse']:.4f}"
)

print("Loading normalization stats...")
norm_stats = load_norm_stats()

print("Loading + combining train + validation tensors...")
train_npz = np.load(TRAIN_PATH, allow_pickle=True)
val_npz = np.load(VAL_PATH, allow_pickle=True)

# Combine chronologically: train (2012-2024) + validation (2025) into one
# continuous array so the API can serve the full historical range, not just 2025.
tensor = np.concatenate(
    [train_npz["tensor"], val_npz["tensor"]], axis=0
)  # (Time, 3, 21, 13)
dates_raw = np.concatenate([train_npz["dates"], val_npz["dates"]], axis=0)
latitudes = train_npz["latitudes"]
longitudes = train_npz["longitudes"]

# .npz stores dates as plain "YYYY-MM-DD" strings — parse directly, no datetime64 involved.
dates = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in dates_raw]
date_to_index = {d: i for i, d in enumerate(dates)}

print(f"Combined tensor: {tensor.shape}, date range {dates[0]} to {dates[-1]}")


# ─── Request / response schemas ─────────────────────────────────────────────
class ForecastRequest(BaseModel):
    date: str  # the single date the map should display, "YYYY-MM-DD"
    rainfall_delta: float = 0.0  # mm/day added to every input day's rainfall (what-if)
    temp_delta: float = 0.0  # °C added to every input day's tmax/tmin (what-if)


class ForecastResponse(BaseModel):
    date: str
    latitudes: List[float]
    longitudes: List[float]
    predicted: Dict[str, list]  # {channel: (21, 13)} — model's forecast for `date`
    actual: Optional[
        Dict[str, list]
    ]  # {channel: (21, 13)} if ground truth exists, else null
    trend_dates: List[str]  # the FORECAST_LEN dates starting at `date`
    trend: Dict[str, list]  # {channel: (FORECAST_LEN, 21, 13)} — forecast trend
    scenario: Optional[Dict[str, list]] = (
        None  # {channel: (21, 13)} — what-if forecast, if deltas given
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_epoch": checkpoint["epoch"],
        "val_rmse": checkpoint["best_val_rmse"],
        "data_range": {"start": str(dates[0]), "end": str(dates[-1])},
    }


def _get_input_window(window_end_idx: int) -> Optional[np.ndarray]:
    """Slice HISTORY_LEN days ending at window_end_idx (inclusive) from the raw
    tensor, in physical units. Returns None if there isn't enough history."""
    start_idx = window_end_idx - HISTORY_LEN + 1
    if start_idx < 0:
        return None
    return tensor[start_idx : window_end_idx + 1].copy()  # (HISTORY_LEN, 3, 21, 13)


def _run_model_on_window(window: np.ndarray) -> np.ndarray:
    """Normalize a raw physical-units input window, run the model, and
    denormalize the output. window: (HISTORY_LEN, 3, 21, 13).
    Returns physical-units output of shape (FORECAST_LEN, 3, 21, 13)."""
    x_norm = normalize_tensor(window, norm_stats)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).float()

    with torch.no_grad():
        y_norm = model(x_tensor)

    y_phys = denormalize_output(y_norm.squeeze(0).numpy(), norm_stats)
    y_phys[:, 0] = np.maximum(y_phys[:, 0], 0.0)  # rainfall can't be negative
    return y_phys  # (FORECAST_LEN, 3, 21, 13)


def _apply_scenario_deltas(
    window: np.ndarray, rainfall_delta: float, temp_delta: float
) -> np.ndarray:
    """Return a copy of the input window with what-if deltas applied to every
    input day. Rainfall (channel 0) shifted by mm/day and clamped at 0;
    tmax/tmin (channels 1, 2) shifted by °C."""
    perturbed = window.copy()
    perturbed[:, 0] = np.maximum(perturbed[:, 0] + rainfall_delta, 0.0)
    perturbed[:, 1] += temp_delta
    perturbed[:, 2] += temp_delta
    return perturbed


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    try:
        target_date = datetime.strptime(req.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "date must be in YYYY-MM-DD format")

    # To forecast `target_date`, we need HISTORY_LEN days ending the day BEFORE it.
    day_before = target_date - timedelta(days=1)
    day_before_idx = date_to_index.get(day_before)

    if day_before_idx is None:
        raise HTTPException(
            404,
            f"cannot forecast {target_date}: no data for {day_before} "
            f"(available range: {dates[0]} to {dates[-1]})",
        )

    window = _get_input_window(day_before_idx)
    if window is None:
        raise HTTPException(
            400,
            f"not enough history before {target_date} — need {HISTORY_LEN} days prior data",
        )

    y_phys = _run_model_on_window(window)

    trend_dates = [
        (target_date + timedelta(days=i)).isoformat() for i in range(FORECAST_LEN)
    ]
    trend = {name: y_phys[:, c].tolist() for c, name in enumerate(CHANNEL_NAMES)}
    predicted = {name: y_phys[0, c].tolist() for c, name in enumerate(CHANNEL_NAMES)}

    # Ground truth only exists if target_date is inside our loaded dataset.
    actual = None
    if target_date in date_to_index:
        actual_grid = tensor[
            date_to_index[target_date]
        ]  # (3, 21, 13), already physical units
        actual = {name: actual_grid[c].tolist() for c, name in enumerate(CHANNEL_NAMES)}

    # What-if scenario: perturb the input window and re-run the model, only
    # if the caller actually asked for a deviation from baseline.
    scenario = None
    if req.rainfall_delta != 0.0 or req.temp_delta != 0.0:
        perturbed_window = _apply_scenario_deltas(
            window, req.rainfall_delta, req.temp_delta
        )
        y_phys_scenario = _run_model_on_window(perturbed_window)
        scenario = {
            name: y_phys_scenario[0, c].tolist() for c, name in enumerate(CHANNEL_NAMES)
        }

    return ForecastResponse(
        date=target_date.isoformat(),
        latitudes=latitudes.tolist(),
        longitudes=longitudes.tolist(),
        predicted=predicted,
        actual=actual,
        trend_dates=trend_dates,
        trend=trend,
        scenario=scenario,
    )
