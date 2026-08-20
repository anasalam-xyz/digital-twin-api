"""
digital-twin-api — FastAPI backend serving ClimateDualNet (Kerala pilot region).

Endpoints:
  GET  /health           basic liveness check
  POST /predict           7-day forecast given a start date
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.model.model_def import ClimateDualNet, N_CHANNELS, HISTORY_LEN, FORECAST_LEN
from app.model.normalization import (
    load_norm_stats,
    normalize_tensor,
    denormalize_output,
)

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "weights" / "best_model.pt"
DATA_PATH = BASE_DIR / "model" / "data" / "validation_tensor_kerala.npz"

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

print("Loading climate tensor...")
npz = np.load(DATA_PATH, allow_pickle=True)
tensor = npz["tensor"]  # (Time, 3, 21, 13)
dates_raw = npz["dates"]  # chronological daily timestamps
latitudes = npz["latitudes"]
longitudes = npz["longitudes"]

# The .npz stores dates as plain "YYYY-MM-DD" strings (numpy.str_, dtype <U10).
# Convert to real datetime.date objects for arithmetic (timedelta, comparisons),
# but keep lookups keyed by date objects derived directly from those strings.
dates = [datetime.strptime(str(d), "%Y-%m-%d").date() for d in dates_raw]
date_to_index = {d: i for i, d in enumerate(dates)}

print(f"Tensor loaded: {tensor.shape}, date range {dates[0]} to {dates[-1]}")


# ─── Request / response schemas ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    date: str  # "YYYY-MM-DD" — last day of the 30-day input window


class PredictResponse(BaseModel):
    input_start_date: str
    input_end_date: str
    forecast_dates: List[str]
    forecast: dict  # {"rainfall": [[...]], "tmax": [[...]], "tmin": [[...]]}
    latitudes: List[float]
    longitudes: List[float]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_epoch": checkpoint["epoch"],
        "val_rmse": checkpoint["best_val_rmse"],
        "data_range": {"start": str(dates[0]), "end": str(dates[-1])},
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # ── Parse and validate the requested date ───────────────────────────────
    try:
        target_date = datetime.strptime(req.date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(400, "date must be in YYYY-MM-DD format")

    if target_date not in date_to_index:
        raise HTTPException(
            404,
            f"date {target_date} not found in dataset "
            f"(available range: {dates[0]} to {dates[-1]})",
        )

    end_idx = date_to_index[target_date]
    start_idx = end_idx - HISTORY_LEN + 1  # inclusive window of HISTORY_LEN days

    if start_idx < 0:
        raise HTTPException(
            400,
            f"not enough history before {target_date} — need {HISTORY_LEN} days, "
            f"only {end_idx + 1} available from start of dataset",
        )

    # ── Slice the 30-day input window ────────────────────────────────────────
    window = tensor[start_idx : end_idx + 1]  # (HISTORY_LEN, 3, 21, 13)

    if window.shape[0] != HISTORY_LEN:
        raise HTTPException(500, "internal error: input window has wrong length")

    # ── Normalize, add batch dim, run inference ──────────────────────────────
    x_norm = normalize_tensor(window, norm_stats)  # (30, 3, 21, 13)
    x_tensor = torch.from_numpy(x_norm).unsqueeze(0).float()  # (1, 30, 3, 21, 13)

    with torch.no_grad():
        y_norm = model(x_tensor)  # (1, FORECAST_LEN, 3, 21, 13)

    y_norm = y_norm.squeeze(0).numpy()  # (FORECAST_LEN, 3, 21, 13)

    # ── Denormalize back to physical units (mm, °C) ──────────────────────────
    y_phys = denormalize_output(y_norm, norm_stats)  # (FORECAST_LEN, 3, 21, 13)

    # ── Build forecast dates (day after target_date, for FORECAST_LEN days) ──
    forecast_dates = [
        (target_date + timedelta(days=i + 1)).isoformat() for i in range(FORECAST_LEN)
    ]

    # ── Split into per-channel grids for a friendlier response shape ─────────
    forecast = {}
    for c, name in enumerate(CHANNEL_NAMES):
        values = y_phys[:, c]
        if name == "rainfall":
            values = np.maximum(values, 0.0)  # rainfall can't be negative
        forecast[name] = values.tolist()

    return PredictResponse(
        input_start_date=dates[start_idx].isoformat(),
        input_end_date=dates[end_idx].isoformat(),
        forecast_dates=forecast_dates,
        forecast=forecast,
        latitudes=latitudes.tolist(),
        longitudes=longitudes.tolist(),
    )
