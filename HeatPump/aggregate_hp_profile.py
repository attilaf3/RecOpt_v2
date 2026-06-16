import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulate_ata import HOUSES_RAW, load_or_make_inputs, simulate_5r2c, solar_gain_sepsi, tabula_to_5r2c_iso_sepsi

DEFAULT_GEOPROFIL = (
    "C:\\NextCloud\\Doktori\\8_felev\\Onlab\\EnergiakozossegOptimalizalas\\bemeneti_fajlok\\hoszivattyu"
    "\\geoprofil.csv"
)

# -----------------------------------------------------------------------------
# Editable run configuration (constants, no command line arguments required)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
NUM_HOUSES = 200
GEOPROFIL_PATH = DEFAULT_GEOPROFIL
OUTPUT_CSV_PATH = ""  # e.g. str(ROOT / "HeatPump" / "hp_profile_avg.csv")
ENABLE_SEASONAL_PLOTS = True

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def perturb_house(base: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
    # Smaller perturbations to avoid identical houses while staying realistic.
    scale = lambda s: _clamp(rng.normal(1.0, s), 0.85, 1.15)

    perturbed = dict(base)
    perturbed["Aref"] = base["Aref"] * scale(0.04)
    perturbed["h"] = base["h"] * scale(0.02)
    perturbed["Htr"] = base["Htr"] * scale(0.06)
    perturbed["Hve"] = base["Hve"] * scale(0.06)
    perturbed["U_window"] = base["U_window"] * scale(0.04)
    perturbed["window_total"] = base["window_total"] * scale(0.04)

    awin_raw = {}
    for d, v in base["Awin_raw"].items():
        awin_raw[d] = max(0.0, v * scale(0.06))
    perturbed["Awin_raw"] = awin_raw

    return perturbed


def make_house_samples(num_houses: int, seed: Optional[int]) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    keys = list(HOUSES_RAW.keys())
    samples = {}

    for i in range(num_houses):
        base_key = keys[int(rng.integers(0, len(keys)))]
        samples[f"house_{i:03d}"] = perturb_house(HOUSES_RAW[base_key], rng)

    return samples


def load_geoprofile(path: str, idx: pd.DatetimeIndex) -> pd.Series | None:
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, header=0)
    if df.empty:
        return None

    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return None
    s = numeric.iloc[:, 0]
    if len(s) == len(idx):
        s.index = idx
        return s

    return None


def _sample_clamped(rng: np.random.Generator, mean: float, sigma: float, low: float, high: float) -> float:
    return float(_clamp(rng.normal(mean, sigma), low, high))


def build_house_setpoint_profile(idx: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    # Seasonal baseline: 22C outside summer, 25C in summer.
    months = idx.month.values
    is_summer = (months >= 6) & (months <= 8)
    base = np.where(is_summer, 25.0, 22.0).astype(float)

    # Day-night schedule and per-house offset.
    offset = _sample_clamped(rng, mean=0.0, sigma=0.4, low=-1.0, high=1.0)
    night_delta = _sample_clamped(rng, mean=-1, sigma=0.3, low=-2, high=0.0)
    hours = idx.hour.values

    # Small per-house variation of night window (e.g. 21->5, 23->8).
    night_start = int(_sample_clamped(rng, mean=22.0, sigma=0.8, low=20.0, high=24.0)) % 24
    night_end = int(_sample_clamped(rng, mean=6.0, sigma=0.8, low=4.0, high=9.0)) % 24

    if night_start > night_end:
        is_night = (hours >= night_start) | (hours < night_end)
    elif night_start < night_end:
        is_night = (hours >= night_start) & (hours < night_end)
    else:
        # Degenerate fallback: if both collapse to the same hour, use baseline 22->6.
        is_night = (hours >= 22) | (hours < 6)

    profile = base + offset
    profile[is_night] += night_delta
    return profile


def sample_control_params(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "deadband": _sample_clamped(rng, mean=2.0, sigma=0.4, low=1.0, high=3.5),
        "min_on_min": _sample_clamped(rng, mean=20.0, sigma=5.0, low=5.0, high=40.0),
        "min_off_min": _sample_clamped(rng, mean=20.0, sigma=5.0, low=5.0, high=40.0),
        "p_ramp_kW_per_step": _sample_clamped(rng, mean=1.5, sigma=0.4, low=0.2, high=3.0),
        "p_th_min_kW": _sample_clamped(rng, mean=2.0, sigma=0.6, low=0.5, high=4.0),
    }


def compute_sum_profile(
    idx: pd.DatetimeIndex,
    T_env: np.ndarray,
    I_by_dir: Dict[str, pd.Series],
    num_houses: int,
    seed: Optional[int],
    progress_every: int,
) -> Tuple[pd.Series, Dict[str, Dict[str, float]]]:
    print(f"Generating {num_houses} houses (seed={seed})...")
    rng = np.random.default_rng(seed)
    houses = make_house_samples(num_houses, seed)
    avg_profile = np.zeros(len(idx), dtype=float)

    for i, house in enumerate(houses.values(), start=1):
        pars = tabula_to_5r2c_iso_sepsi(
            Aref_m2=house["Aref"],
            h_m=house["h"],
            Htr_W_per_K=house["Htr"],
            Hve_W_per_K=house["Hve"],
            U_window_W_m2K=house["U_window"],
            window_total_m2=house["window_total"],
            Awin_raw=house["Awin_raw"],
        )

        Q_sol = solar_gain_sepsi(I_by_dir, pars.Awin_by_dir_m2, idx)
        t_set_profile = build_house_setpoint_profile(idx, rng)
        ctrl = sample_control_params(rng)
        res = simulate_5r2c(
            idx,
            T_env,
            Q_sol.values,
            pars,
            deadband=ctrl["deadband"],
            min_on_min=ctrl["min_on_min"],
            min_off_min=ctrl["min_off_min"],
            p_ramp_kW_per_step=ctrl["p_ramp_kW_per_step"],
            power_mode="min_cap",
            p_th_min_kW=ctrl["p_th_min_kW"],
            t_set_profile=t_set_profile,
            enforce_hysteresis=True,
            enforce_ramp_limit=True,
            enforce_min_on_off=True,
            hp_block_windows={"summer": ((7, 9), (15, 17)), "winter": ((8, 10), (16, 18) ), "default": ((8, 10), (16, 18) ), "autumn": ((7, 9), (16, 18) ) },
        )
        avg_profile += res["P_hp_el"]

        if progress_every > 0 and i % progress_every == 0:
            print(f"Simulated {i}/{num_houses} houses...")

    print("Aggregation complete.")
    avg_profile /= max(1, num_houses)
    return pd.Series(avg_profile, index=idx, name="P_hp_el_avg"), houses


def _season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def pick_season_days(idx: pd.DatetimeIndex) -> Dict[str, pd.Timestamp]:
    season_days: Dict[str, pd.Timestamp] = {}
    unique_days = pd.to_datetime(pd.Index(idx.normalize().unique()))
    for day in unique_days:
        season = _season_for_month(day.month)
        if season not in season_days:
            season_days[season] = day
        if len(season_days) == 4:
            break
    return season_days


def plot_profiles(sum_profile: pd.Series, geo_profile: pd.Series | None) -> None:
    plt.figure(figsize=(12, 4.2))
    plt.plot(sum_profile.index, sum_profile.values, label="Average HP P_el (kW)", lw=1.2)
    if geo_profile is not None:
        geo_profile *= sum_profile.sum() / geo_profile.sum()
        plt.plot(sum_profile.index, geo_profile.values, label="Geoprofil", lw=1.0)
    plt.title("Aggregated heat pump electric profile")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_seasonal_days(sum_profile: pd.Series, geo_profile: pd.Series | None) -> None:
    season_days = pick_season_days(sum_profile.index)
    if not season_days:
        print("No seasonal days available in the input index.")
        return

    geo_scaled = None
    if geo_profile is not None and geo_profile.sum() != 0:
        geo_scaled = geo_profile * (sum_profile.sum() / geo_profile.sum())

    seasons = ["Winter", "Spring", "Summer", "Fall"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.0), sharey=True)
    axes = axes.ravel()

    for ax, season in zip(axes, seasons):
        day = season_days.get(season)
        if day is None:
            ax.set_title(f"{season}: no data")
            ax.set_xlabel("Time")
            ax.set_ylabel("kW")
            ax.tick_params(axis="x", labelrotation=60)
            continue

        start = day
        end = day + pd.Timedelta(days=1)
        mask = (sum_profile.index >= start) & (sum_profile.index < end)
        ax.plot(sum_profile.index[mask], sum_profile.values[mask], label="Average P_el", lw=1.2)
        if geo_scaled is not None:
            ax.plot(geo_scaled.index[mask], geo_scaled.values[mask], label="Geoprofil", lw=1.0)
        ax.set_title(f"{season} example day: {day.date()}")
        ax.set_xlabel("Time")
        ax.set_ylabel("kW")
        ax.tick_params(axis="x", labelrotation=60)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()


def generate_synthetic_inputs() -> Tuple[pd.DatetimeIndex, np.ndarray, Dict[str, pd.Series]]:
    idx = pd.date_range("2024-01-01", periods=96, freq="15min")
    T_env = np.zeros(len(idx), dtype=float)
    I_by_dir = {d: pd.Series(0.0, index=idx) for d in ["E", "S", "W", "N"]}
    return idx, T_env, I_by_dir


def main() -> int:
    print("Loading inputs...")
    idx, T_env, I_by_dir = load_or_make_inputs()

    print("Loading geoprofile for comparison...")
    geo_profile = load_geoprofile(GEOPROFIL_PATH, idx)
    if geo_profile is None:
        print(f"Geoprofile not found or could not be parsed: {GEOPROFIL_PATH}")

    sum_profile, _ = compute_sum_profile(
        idx,
        T_env,
        I_by_dir,
        NUM_HOUSES,
        42,
        10,
    )

    if OUTPUT_CSV_PATH:
        print(f"Saving aggregated profile to {OUTPUT_CSV_PATH}...")
        sum_profile.to_csv(OUTPUT_CSV_PATH, index=True)

    print("Plotting profiles...")
    plot_profiles(sum_profile, geo_profile)

    if ENABLE_SEASONAL_PLOTS:
        print("Plotting seasonal example days...")
        plot_seasonal_days(sum_profile, geo_profile)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

