"""
Normalization utilities for ClimateDualNet inference.

Mirrors the exact logic used at training time:
  - Rainfall (channel 0): log1p, then z-score
  - Tmax, Tmin (channels 1, 2): z-score only

The stats below (mean/std per channel) were computed once on the TRAINING
split only, and must never be recomputed from live/incoming data — using
different stats at inference than at training silently breaks predictions.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

STATS_PATH = Path(__file__).parent / "norm_stats.json"

CHANNEL_NAMES = ["Rainfall", "Tmax", "Tmin"]


@dataclass
class NormStats:
    """Per-channel normalization statistics."""
    mean: List[float]
    std: List[float]
    use_log1p: List[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "use_log1p": self.use_log1p}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NormStats":
        return cls(mean=d["mean"], std=d["std"], use_log1p=d["use_log1p"])


def load_norm_stats(path: Path = STATS_PATH) -> NormStats:
    """Load the fixed normalization stats saved from training."""
    with open(path) as f:
        return NormStats.from_dict(json.load(f))


def normalize_tensor(tensor: np.ndarray, stats: NormStats) -> np.ndarray:
    """
    Normalize a raw physical-units tensor before feeding it to the model.

    Parameters
    ----------
    tensor : np.ndarray, shape (..., C, H, W) or (days, C, H, W)
             Must have channel axis matching len(stats.mean) == 3.
    stats  : NormStats loaded from training.

    Returns
    -------
    Normalized tensor, same shape, dtype float32. NaNs/Infs zeroed out
    (matching the cleaning step used at training time).
    """
    out = np.empty_like(tensor, dtype=np.float32)
    n_channels = tensor.shape[-3]  # assumes (..., C, H, W)
    for c in range(n_channels):
        ch = tensor[..., c, :, :].copy()
        if stats.use_log1p[c]:
            ch = np.log1p(np.maximum(ch, 0.0))
        out[..., c, :, :] = (ch - stats.mean[c]) / stats.std[c]

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def denormalize_channel(x: np.ndarray, channel: int, stats: NormStats) -> np.ndarray:
    """
    Inverse-normalize model OUTPUT back to physical units (mm, °C).

    Parameters
    ----------
    x       : model output for a single channel, any shape
    channel : 0=Rainfall, 1=Tmax, 2=Tmin
    stats   : same NormStats used for normalize_tensor
    """
    x_phys = x * stats.std[channel] + stats.mean[channel]
    if stats.use_log1p[channel]:
        x_phys = np.expm1(x_phys)
    return x_phys


def denormalize_output(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """
    Inverse-normalize a full model output tensor with shape (..., C, H, W)
    where C matches len(stats.mean) == 3, in one call.
    """
    out = np.empty_like(x, dtype=np.float32)
    n_channels = x.shape[-3]
    for c in range(n_channels):
        out[..., c, :, :] = denormalize_channel(x[..., c, :, :], c, stats)
    return out
