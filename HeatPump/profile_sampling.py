from __future__ import annotations

import numpy as np
import pandas as pd

from .simulate_ata import build_setpoint_profile

__all__ = ["perturb_hp_house", "build_hp_setpoint_profile"]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sample_clamped(
    rng: np.random.Generator,
    mean: float,
    sigma: float,
    low: float,
    high: float,
) -> float:
    return float(_clamp(rng.normal(mean, sigma), low, high))


def perturb_hp_house(base: dict, rng: np.random.Generator) -> dict:
    """Apply small random parameter offsets to a heat-pump building model."""
    scale = lambda sigma: _clamp(rng.normal(1.0, sigma), 0.90, 1.10)

    perturbed = dict(base)
    perturbed["Aref"] = base["Aref"] * scale(0.03)
    perturbed["h"] = base["h"] * scale(0.01)
    perturbed["Htr"] = base["Htr"] * scale(0.04)
    perturbed["Hve"] = base["Hve"] * scale(0.04)
    perturbed["U_window"] = base["U_window"] * scale(0.03)
    perturbed["window_total"] = base["window_total"] * scale(0.03)
    perturbed["Awin_raw"] = {
        direction: max(0.0, value * scale(0.04))
        for direction, value in base["Awin_raw"].items()
    }
    return perturbed


def build_hp_setpoint_profile(
    idx: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Build a seasonal 22/25 °C setpoint with household-level variation."""
    profile = build_setpoint_profile(idx, t_cold=22.0, t_summer=25.0).astype(float)
    offset_c = _sample_clamped(rng, mean=0.0, sigma=0.35, low=-0.8, high=0.8)
    night_delta_c = _sample_clamped(rng, mean=-0.35, sigma=0.20, low=-1.0, high=0.0)

    hours = idx.hour.to_numpy()
    is_night = (hours < 6) | (hours >= 22)
    profile += offset_c
    profile[is_night] += night_delta_c

    return profile, {
        "setpoint_offset_c": float(offset_c),
        "night_delta_c": float(night_delta_c),
    }
