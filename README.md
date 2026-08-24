# digital-twin-api

FastAPI backend serving **ClimateDualNet**, a spatiotemporal deep learning model that forecasts daily rainfall, max temperature, and min temperature for the **Kerala pilot region** (8.0°N–13.0°N, 74.5°E–77.5°E, 21×13 grid), trained on IMD gridded climate data (2012–2025).

Built for ISRO Bharatiya Antariksh Hackathon (BAH) 2026.

## What it does

- Serves a trained PyTorch model (`ClimateDualNet`: residual CNN encoder + CBAM attention + ConvLSTM + temporal self-attention + UNet decoder) over HTTP.
- Given a date, returns:
  - **Predicted** — the model's forecast for that date, using the prior 30 days as input.
  - **Actual** — real observed values for that date, if it falls within the loaded dataset (2012–2025). `null` where no ground truth exists (e.g. future dates, or ocean grid cells with no land data).
  - **7-day trend** — a forecast trend starting at the selected date, for per-point charting.
  - **Scenario (what-if)** — optionally, perturb the 30-day input window by a rainfall/temperature delta and re-run the model to see how the forecast shifts.

## Tech stack

- **FastAPI** — HTTP API
- **PyTorch** (CPU-only build) — model inference
- **NumPy** — tensor/data handling
- **uv** — dependency management

## Project structure

```
digital-twin-api/
├── app/
│   ├── main.py                  # FastAPI app, endpoints, inference logic
│   └── model/
│       ├── model_def.py         # ClimateDualNet architecture + building blocks
│       ├── normalization.py     # per-channel normalization (log1p + z-score)
│       ├── norm_stats.json      # fixed mean/std stats computed at training time
│       ├── weights/
│       │   └── best_model.pt    # trained model checkpoint
│       └── data/
│           ├── train_tensor_kerala.npz       # 2012–2024, (Time, 3, 21, 13)
│           └── validation_tensor_kerala.npz  # 2025, (Time, 3, 21, 13)
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`pyproject.toml` is already configured to install the **CPU-only** build of PyTorch (not the much larger CUDA build), via a custom index:

```toml
[tool.uv.sources]
torch = { index = "pytorch" }

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

## Running locally

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Startup logs will confirm the model, normalization stats, and combined train+validation tensor all loaded correctly, including the model's validation epoch/RMSE and the full usable date range.

## API

### `GET /health`

Liveness check. Returns model metadata and the dataset's available date range.

```json
{
  "status": "ok",
  "model_epoch": 10,
  "val_rmse": 0.591,
  "data_range": { "start": "2012-01-01", "end": "2025-12-31" }
}
```

### `POST /forecast`

**Request:**
```json
{
  "date": "2025-03-15",
  "rainfall_delta": 0,
  "temp_delta": 0
}
```

- `date` (required) — the date to forecast, `YYYY-MM-DD`. Must have at least 30 days of prior data available (so the earliest usable date is ~30 days after the dataset start).
- `rainfall_delta` (optional, default `0`) — mm/day added to every day in the 30-day input window, for what-if scenarios. Rainfall is clamped at 0 after perturbation.
- `temp_delta` (optional, default `0`) — °C added to both tmax and tmin across the input window.

**Response:**
```json
{
  "date": "2025-03-15",
  "latitudes": [8.0, 8.25, ...],
  "longitudes": [74.5, 74.75, ...],
  "predicted": { "rainfall": [[...]], "tmax": [[...]], "tmin": [[...]] },
  "actual": { "rainfall": [[...]], "tmax": [[...]], "tmin": [[...]] },
  "trend_dates": ["2025-03-15", "...", "2025-03-21"],
  "trend": { "rainfall": [[[...]]], "tmax": [[[...]]], "tmin": [[[...]]] },
  "scenario": { "rainfall": [[...]], "tmax": [[...]], "tmin": [[...]] }
}
```

- `predicted` / `actual` / `scenario` — each a 21×13 grid per channel. `actual` and `scenario` are `null` when not applicable (no ground truth for that date, or no deltas supplied).
- `trend` — 7-day forecast trend (`FORECAST_LEN` days) starting at `date`, shape `(7, 21, 13)` per channel.

## Data pipeline summary

1. Raw IMD `.GRD` binary files parsed into aligned NumPy arrays (rainfall, tmax, tmin) on a common grid.
2. Missing land values interpolated; ocean pixels preserved as `NaN` (not interpolated).
3. Daily grids stacked into a `(Time, Variables, Lat, Lon)` tensor.
4. Cropped from full-India (129×135) to the Kerala pilot region (21×13).
5. Split chronologically: train 2012–2024, validation 2025 onward (no shuffling, to avoid temporal leakage).

Normalization (`norm_stats.json`): rainfall uses `log1p` then z-score; tmax/tmin use z-score only. Stats were computed once on the training split and must never be recomputed from live data.

## Performance notes

- Model is small (~3M params) and the grid is tiny (21×13), so inference is fast on a real CPU (well under a second).
- `torch.set_num_threads(1)` is set at startup — on CPU-limited hosting (e.g. free-tier deploys), PyTorch's default of spawning one thread per detected core causes severe contention rather than speedup, since the container often only gets a fraction of a real core. Pinning to 1 thread avoids that.
- On free-tier hosting, cold starts and shared-CPU throttling can still make individual requests slow (~seconds, occasionally more) even with the thread fix — this is a hosting tier limitation, not a model or code issue.

## Known limitations

- Model only covers the **Kerala pilot region** — no coverage elsewhere in India.
- `actual` ground truth is only available for dates within the loaded 2012–2025 dataset; forecasts for dates outside that range have no ground truth to compare against.
- What-if scenarios apply a uniform delta across the entire 30-day input window and full spatial grid — they are a coarse perturbation, not a physically-informed climate simulation.
