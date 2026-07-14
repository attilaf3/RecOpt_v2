from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from Economics import calculate_component_grid_bill
from HeatPump.profile_sampling import build_hp_setpoint_profile, perturb_hp_house
from HeatPump.simulate_ata import (HOUSES_RAW,
                                   load_or_make_inputs as load_hp_weather, simulate_5r2c, solar_gain_sepsi,
                                   tabula_to_5r2c_iso_sepsi, )
from InputReading import read_simulation_inputs
from Utility.configuration import config
from Visualization import (
    plot_community_energy_balance,
    plot_household_percentiles_by_group,
    plot_household_percentiles_by_group_with_global_scurve,
)

DT = config.getfloat("simulation", "dt_hours")
HP_DT = config.getfloat("simulation", "hp_dt_hours")
LOW_TARIFF_LIMIT_KWH = config.getfloat("tariffs", "grid_a_low_limit_kwh")
LOW_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_a_low_ft_per_kwh")
HIGH_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_a_high_ft_per_kwh")
# Separate tariffs for Boilers and Heat Pumps (two-tier, same annual low-tariff limit)
BOILER_LOW_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_b_low_ft_per_kwh")
BOILER_HIGH_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_b_high_ft_per_kwh")
HP_LOW_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_hp_low_ft_per_kwh")
HP_HIGH_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_hp_high_ft_per_kwh")


def build_inputs(sim_yaml_path: os.PathLike, profiles_csv_path: os.PathLike, dhw_profile_path: os.PathLike,
        max_users: int | None = 10, search_roots: Iterable[os.PathLike] | None = None,
        pv_ratio: float = 1.0, ) -> tuple:
    """Compatibility adapter backed exclusively by :mod:`InputReading`."""
    inputs = read_simulation_inputs(
        sim_yaml_path=sim_yaml_path,
        profiles_csv_path=profiles_csv_path,
        dhw_profile_path=dhw_profile_path,
        max_users=max_users,
        pv_ratio=pv_ratio,
        dt=DT,
        search_roots=search_roots,
    )
    return (
        inputs.e_pv_kwh, inputs.e_ue_kwh, inputs.e_el_heater_kwh,
        inputs.size_bess, inputs.eta_bess_in, inputs.eta_bess_out,
        inputs.eta_bess_stor, inputs.soc_bess_min, inputs.soc_bess_max,
        inputs.t_bess_min, inputs.user_names,
    )


LOW_TARIFF_LIMIT_KWH = config.getfloat("tariffs", "grid_a_low_limit_kwh")
LOW_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_a_low_ft_per_kwh")
HIGH_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_a_high_ft_per_kwh")

B_LOW_TARIFF_LIMIT_KWH = config.getfloat("tariffs", "grid_b_low_limit_kwh")
B_LOW_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_b_low_ft_per_kwh")
B_HIGH_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_b_high_ft_per_kwh")

EXPORT_FT_PER_KWH = config.getfloat("tariffs", "pv_export_ft_per_kwh")
# Default HP forbidden windows: allow season-specific configuration.
# By default use earlier-morning forbidden window in winter (5-8) and
# slightly later in summer (6-9). The evening window remains the same.
# The simulate_5r2c function accepts either a sequence of (start,end) tuples
# (applies all year) or a dict with keys 'winter' and 'summer' (and optional
# 'default') mapping to sequences of tuples.
HP_DEFAULT_BLOCK_WINDOWS = {"winter": ((5, 8), (16, 18)), "summer": ((6, 9), (16, 18)), # fallback for other seasons
    "default": ((5, 8), (16, 18)), }


def _calc_import_cost(e_import_steps: np.ndarray, low_limit_kwh: float, low_price_ft_per_kwh: float,
        high_price_ft_per_kwh: float, ) -> tuple[float, float, float]:
    remaining_low = float(low_limit_kwh)
    low_kwh = 0.0
    high_kwh = 0.0
    cost_ft = 0.0

    for e_imp in np.maximum(np.asarray(e_import_steps, dtype=float), 0.0):
        low_part = min(float(e_imp), max(remaining_low, 0.0))
        high_part = max(float(e_imp) - low_part, 0.0)

        cost_ft += low_part * low_price_ft_per_kwh
        cost_ft += high_part * high_price_ft_per_kwh

        low_kwh += low_part
        high_kwh += high_part
        remaining_low -= low_part

    return low_kwh, high_kwh, cost_ft


def calc_bill_15min_brutto(e_grid_to_load: np.ndarray, e_inj: np.ndarray,
        e_grid_to_boiler: np.ndarray | None = None, ) -> dict:
    """
    15 perces bruttó elszámolás.

    A bemenetek kWh / időlépés egységűek.
    - Normál háztartási fogyasztás: A tarifa, 36 / 71 Ft/kWh.
    - Bojler: B tarifa, 23 / 61 Ft/kWh.

    Ha nincs külön `e_grid_to_boiler`, akkor minden import A-tarifára kerül.
    Ha van, akkor `e_grid_to_load` a teljes import, ebből levonjuk a bojler
    importját, és csak a maradék kerül A-tarifára.
    """

    e_grid_to_load = np.maximum(np.asarray(e_grid_to_load, dtype=float), 0.0)
    e_inj = np.maximum(np.asarray(e_inj, dtype=float), 0.0)

    if e_grid_to_boiler is None:
        e_grid_to_boiler = np.zeros_like(e_grid_to_load)
    else:
        e_grid_to_boiler = np.maximum(np.asarray(e_grid_to_boiler, dtype=float), 0.0)

    e_grid_to_boiler = np.minimum(e_grid_to_boiler, e_grid_to_load)
    e_grid_to_base = np.maximum(e_grid_to_load - e_grid_to_boiler, 0.0)

    a_low_kwh, a_high_kwh, a_import_cost_ft = _calc_import_cost(e_import_steps=e_grid_to_base,
        low_limit_kwh=LOW_TARIFF_LIMIT_KWH, low_price_ft_per_kwh=LOW_TARIFF_FT_PER_KWH,
        high_price_ft_per_kwh=HIGH_TARIFF_FT_PER_KWH, )
    b_low_kwh, b_high_kwh, b_import_cost_ft = _calc_import_cost(e_import_steps=e_grid_to_boiler,
        low_limit_kwh=B_LOW_TARIFF_LIMIT_KWH, low_price_ft_per_kwh=B_LOW_TARIFF_FT_PER_KWH,
        high_price_ft_per_kwh=B_HIGH_TARIFF_FT_PER_KWH, )

    # Bruttó elszámolás:
    #   importköltség = A tarifás import költsége + B tarifás import költsége
    #   exportbevétel = PV exportált energia * átvételi ár
    #   bruttó villanyszámla = importköltség - exportbevétel
    #
    # A tarifa:
    #   normál háztartási fogyasztás
    # B tarifa:
    #   külön bojler import

    export_revenue_ft = e_inj.sum() * EXPORT_FT_PER_KWH
    import_cost_ft = a_import_cost_ft + b_import_cost_ft
    brt_bill_ft = import_cost_ft - export_revenue_ft

    return {"grid_import_low_kwh": a_low_kwh + b_low_kwh, "grid_import_high_kwh": a_high_kwh + b_high_kwh,
        "grid_import_a_low_kwh": a_low_kwh, "grid_import_a_high_kwh": a_high_kwh, "grid_import_b_low_kwh": b_low_kwh,
        "grid_import_b_high_kwh": b_high_kwh, "grid_import_a_kwh": float(e_grid_to_base.sum()),
        "grid_import_b_kwh": float(e_grid_to_boiler.sum()), "import_cost_a_ft": a_import_cost_ft,
        "import_cost_b_ft": b_import_cost_ft, "import_cost_ft": import_cost_ft, "export_revenue_ft": export_revenue_ft,
        "brt_bill_ft": brt_bill_ft, }


def calc_bill_15min_brutto_breakdown(
        e_grid_base: np.ndarray, e_grid_boiler: np.ndarray, e_grid_hp: np.ndarray, e_inj: np.ndarray, ) -> dict:
    """Compatibility wrapper for the centralized component tariff calculation."""
    return calculate_component_grid_bill(e_grid_base, e_grid_boiler, e_grid_hp, e_inj)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sample_clamped(rng: np.random.Generator, mean: float, sigma: float, low: float, high: float) -> float:
    return float(_clamp(rng.normal(mean, sigma), low, high))


def _sample_hp_control_params(rng: np.random.Generator) -> dict:
    return {"deadband": _sample_clamped(rng, mean=2.0, sigma=0.30, low=1.0, high=3.0),
        "min_on_min": _sample_clamped(rng, mean=20.0, sigma=4.0, low=5.0, high=35.0),
        "min_off_min": _sample_clamped(rng, mean=20.0, sigma=4.0, low=5.0, high=35.0),
        "p_ramp_kW_per_step": _sample_clamped(rng, mean=1.5, sigma=0.25, low=0.25, high=3.0),
        "p_th_min_kW": _sample_clamped(rng, mean=2.0, sigma=0.40, low=0.5, high=4.0), }


def _simulate_heat_pump_hourly_load(idx_hp: pd.DatetimeIndex, T_env_hp: np.ndarray, I_by_dir_hp: dict[str, pd.Series],
        rng: np.random.Generator, ) -> tuple[np.ndarray, dict]:
    """Egy háztartás HP villamos profiljának előállítása a 5R2C modellből."""

    base_key = str(rng.choice(list(HOUSES_RAW.keys())))
    house = perturb_hp_house(HOUSES_RAW[base_key], rng)

    pars = tabula_to_5r2c_iso_sepsi(Aref_m2=house["Aref"], h_m=house["h"], Htr_W_per_K=house["Htr"],
        Hve_W_per_K=house["Hve"], U_window_W_m2K=house["U_window"], window_total_m2=house["window_total"],
        Awin_raw=house["Awin_raw"], )

    q_sol = solar_gain_sepsi(I_by_dir_hp, pars.Awin_by_dir_m2, idx_hp)
    t_set_profile, setpoint_meta = build_hp_setpoint_profile(idx_hp, rng)
    ctrl = _sample_hp_control_params(rng)

    res = simulate_5r2c(idx_hp, T_env_hp, q_sol.values, pars, dt_h=HP_DT, deadband=ctrl["deadband"],
        min_on_min=ctrl["min_on_min"], min_off_min=ctrl["min_off_min"], p_ramp_kW_per_step=ctrl["p_ramp_kW_per_step"],
        hp_block_windows=HP_DEFAULT_BLOCK_WINDOWS, t_set_profile=t_set_profile, enforce_hysteresis=True,
        enforce_min_on_off=True, enforce_ramp_limit=True, enforce_participation_factor=True, power_mode="min_cap",
        p_th_min_kW=ctrl["p_th_min_kW"], )

    e_hp_hourly = pd.Series(res["P_hp_el"] * HP_DT, index=idx_hp).resample("h").sum()
    meta = {"hp_house_key": base_key, "hp_Aref_m2": float(house["Aref"]), "hp_h_m": float(house["h"]),
        "hp_deadband_c": float(ctrl["deadband"]), "hp_min_on_min": float(ctrl["min_on_min"]),
        "hp_min_off_min": float(ctrl["min_off_min"]), "hp_ramp_kW_per_step": float(ctrl["p_ramp_kW_per_step"]),
        "hp_p_th_min_kW": float(ctrl["p_th_min_kW"]), "hp_setpoint_offset_c": float(setpoint_meta["setpoint_offset_c"]),
        "hp_night_delta_c": float(setpoint_meta["night_delta_c"]), "hp_energy_kwh": float(e_hp_hourly.sum()),
        "hp_peak_kw": float(np.max(res["P_hp_el"])) if len(res["P_hp_el"]) else 0.0, }
    return e_hp_hourly.to_numpy(dtype=float), meta


def _group_label(has_pv: bool, has_bess: bool, has_heat_pump: bool, has_boiler: bool = False) -> str:
    """Return a compact group label. Optionally include Boiler when present.

    Examples: 'PV+BESS+HP', 'BESS+HP+Boiler', 'HP+Boiler', 'Nincs PV'
    """
    parts = []
    if has_pv:
        parts.append("PV")
    if has_bess:
        parts.append("BESS")
    if has_heat_pump:
        parts.append("HP")
    if has_boiler:
        parts.append("Boiler")
    return "+".join(parts) if parts else "Nincs PV"


def _ordered_group_labels(df: pd.DataFrame) -> list[str]:
    preferred = ["Nincs PV", "PV", "HP", "PV+HP", "PV+BESS", "PV+BESS+HP", "BESS+HP", "BESS+HP+Boiler", "BESS+Boiler",
        "HP+Boiler", "PV+BESS+HP+Boiler", "PV+BESS+Boiler", "PV+HP+Boiler", ]
    groups = list(dict.fromkeys(df["group_label"].tolist()))
    ordered = [g for g in preferred if g in groups]
    ordered.extend([g for g in groups if g not in ordered])
    return ordered


def plot_community_load_breakdown(community_ts: pd.DataFrame, out_case: Path):
    if not {"e_load_total", "e_load_base_total", "e_hp_total"}.issubset(community_ts.columns):
        return

    x = np.arange(len(community_ts))
    plt.figure(figsize=(12, 4.8))
    plt.plot(x, community_ts["e_load_base_total"].to_numpy(dtype=float), label="Alapterhelés (UE + boiler)", lw=1.0)
    plt.plot(x, community_ts["e_hp_total"].to_numpy(dtype=float), label="HP terhelés", lw=1.0)
    plt.plot(x, community_ts["e_load_total"].to_numpy(dtype=float), label="Összes terhelés", lw=1.2)
    plt.xlabel("Órás időlépés")
    plt.ylabel("kWh / óra")
    plt.title("Közösségi villamos terhelés HP opcióval")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_case / "community_load_breakdown_with_heat_pumps.png", dpi=200)
    plt.close()


def _as_datetime_index(index: pd.Index, length: int) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex) and len(index) == length:
        return pd.DatetimeIndex(index)
    return pd.date_range("2021-01-01", periods=length, freq="h")


def _season_windows(index: pd.DatetimeIndex, window_len: int) -> list[tuple[str, slice]]:
    if len(index) == 0:
        return []

    window_len = int(max(1, min(window_len, len(index))))
    if len(index) < 4:
        return [("Általános", slice(0, len(index)))]

    season_centers = [("Tél", pd.Timestamp(index[0].year, 1, 15, 12)),
        ("Tavasz", pd.Timestamp(index[0].year, 4, 15, 12)), ("Nyár", pd.Timestamp(index[0].year, 7, 15, 12)),
        ("Ősz", pd.Timestamp(index[0].year, 10, 15, 12)), ]

    windows: list[tuple[str, slice]] = []
    for label, center in season_centers:
        pos = int(index.get_indexer(pd.DatetimeIndex([center]), method="nearest")[0])
        start = max(0, pos - window_len // 2)
        stop = min(len(index), start + window_len)
        start = max(0, stop - window_len)
        windows.append((label, slice(start, stop)))
    return windows


def _col_or_zero(frame: pd.DataFrame, *names: str) -> np.ndarray:
    for name in names:
        if name in frame.columns:
            return frame[name].to_numpy(dtype=float)
    return np.zeros(len(frame), dtype=float)


def _series_or_zero(series_map: Optional[Dict[str, np.ndarray]], key: str, shape: Tuple[int, ...]) -> np.ndarray:
    if series_map is None or key not in series_map:
        return np.zeros(shape, dtype=float)
    return np.asarray(series_map[key], dtype=float)


def _select_households(per_user_df: Optional[pd.DataFrame], limit: int = 3) -> List[int]:
    if per_user_df is None or len(per_user_df) == 0:
        return []

    preferred_groups = ["PV+BESS+HP+Boiler", "PV+BESS+HP", "BESS+HP+Boiler", "BESS+HP", "PV+HP+Boiler", "PV+HP",
        "HP+Boiler", "HP", "PV+BESS+Boiler", "PV+BESS", "PV+Boiler", "PV", "Nincs PV", ]

    selected: list[int] = []
    used: set[int] = set()
    for group in preferred_groups:
        sub = per_user_df[per_user_df["group_label"] == group]
        if len(sub) == 0:
            continue
        idx = int(sub.index[0])
        if idx not in used:
            selected.append(idx)
            used.add(idx)
        if len(selected) >= limit:
            return selected

    for idx in per_user_df.index.tolist():
        idx = int(idx)
        if idx not in used:
            selected.append(idx)
            used.add(idx)
        if len(selected) >= limit:
            break
    return selected


def _stack_signed_bars(ax: plt.Axes, x: np.ndarray, components: List[Tuple[str, np.ndarray, str]], *,
        width: float = 0.8, alpha: float = 0.88, ) -> Dict[str, object]:
    pos_bottom = np.zeros_like(x, dtype=float)
    neg_bottom = np.zeros_like(x, dtype=float)
    handles: dict[str, object] = {}

    for label, values, color in components:
        arr = np.asarray(values, dtype=float)
        if arr.shape != x.shape:
            continue
        pos = np.clip(arr, 0.0, None)
        neg = np.clip(arr, None, 0.0)

        if np.any(pos > 0.0):
            container = ax.bar(x, pos, width=width, bottom=pos_bottom, color=color, alpha=alpha, edgecolor="none",
                               label=label)
            pos_bottom = pos_bottom + pos
            handles[label] = container[0]
        elif np.any(neg < 0.0):
            handles[label] = Patch(facecolor=color, edgecolor="none", alpha=alpha, label=label)

        if np.any(neg < 0.0):
            ax.bar(x, neg, width=width, bottom=neg_bottom, color=color, alpha=alpha, edgecolor="none")
            neg_bottom = neg_bottom + neg

    ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25, axis="y")
    return handles


def _legacy_plot_community_energy_balance(community_ts: pd.DataFrame, out_case: Path,
        per_user_df: Optional[pd.DataFrame] = None, household_ts: Optional[Dict[str, np.ndarray]] = None,
        user_names: Optional[List[str]] = None, ):
    """Seasonal bar-balance plot for the community and a few sample households."""

    if len(community_ts) == 0:
        return

    required_any = {"e_pv_total", "e_load_total"}
    if not required_any.issubset(community_ts.columns):
        return

    idx = _as_datetime_index(community_ts.index, len(community_ts))
    windows = _season_windows(idx, window_len=min(96, len(community_ts)))
    if not windows:
        return

    available = {"pv": float(_col_or_zero(community_ts, "e_pv_total").sum()) > 1e-9,
        "grid": float(_col_or_zero(community_ts, "e_grid_to_load_total").sum()) > 1e-9,
        "bess": float(_col_or_zero(community_ts, "e_bess_to_load_total").sum()) > 1e-9 or float(
            _col_or_zero(community_ts, "e_bess_total").sum()) > 1e-9,
        "hp": float(_col_or_zero(community_ts, "e_hp_total").sum()) > 1e-9,
        "boiler": float(_col_or_zero(community_ts, "e_boiler_total").sum()) > 1e-9,
        "base": float(_col_or_zero(community_ts, "e_ue_total", "e_load_base_total").sum()) > 1e-9, }

    has_bess_col = available["bess"]
    nrows = len(windows)
    ncols = 3 if has_bess_col else 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.2 * ncols, 4.3 * nrows), squeeze=False)
    fig.suptitle("Közösségi energiamérleg szezononkénti bontásban", fontsize=16, y=0.995)

    legend_handles: Dict[str, object] = {}
    household_indices = _select_households(per_user_df, limit=3)
    household_colors = ["#3b82f6", "#ef4444", "#10b981"]

    for row, (season_label, slc) in enumerate(windows):
        x = np.arange(slc.stop - slc.start, dtype=float)
        season_start = idx[slc.start]
        season_stop = idx[slc.stop - 1]
        row_axes = list(axes[row])

        season_pv = _col_or_zero(community_ts, "e_pv_total")[slc]
        season_grid = _col_or_zero(community_ts, "e_grid_to_load_total")[slc]
        season_bess_to_load = _col_or_zero(community_ts, "e_bess_to_load_total")[slc]
        season_pv_to_bess = _col_or_zero(community_ts, "e_pv_to_bess_total")[slc]
        season_inj = _col_or_zero(community_ts, "e_inj_total")[slc]
        season_load_base = _col_or_zero(community_ts, "e_ue_total", "e_load_base_total")[slc]
        season_boiler = _col_or_zero(community_ts, "e_boiler_total")[slc]
        season_hp = _col_or_zero(community_ts, "e_hp_total")[slc]
        season_bess_soc = _col_or_zero(community_ts, "e_bess_total")[slc]

        # One combined community balance panel:
        # positive (top) = consumption, negative (bottom) = production/supply
        balance_components: list[tuple[str, np.ndarray, str]] = []
        if available["base"]:
            balance_components.append(("UE / alap", season_load_base, "#7f8c8d"))
        if available["boiler"]:
            balance_components.append(("Boiler", season_boiler, "#a65628"))
        if available["hp"]:
            balance_components.append(("HP", season_hp, "#3d5afe"))
        if has_bess_col and np.any(season_pv_to_bess > 1e-9):
            balance_components.append(("BESS töltés", season_pv_to_bess, "#37cdd2"))
        if np.any(season_inj > 1e-9):
            balance_components.append(("Hálózati export", season_inj, "#d65a5a"))

        if available["pv"]:
            balance_components.append(("PV", -season_pv, "#f4c20d"))
        if available["grid"]:
            balance_components.append(("Hálózati import", -season_grid, "#4c78a8"))
        if has_bess_col and np.any(season_bess_to_load > 1e-9):
            balance_components.append(("BESS kisütés", -season_bess_to_load, "#2f8f4e"))

        balance_ax = row_axes[0]
        handles = _stack_signed_bars(balance_ax, x, balance_components)
        for k, v in handles.items():
            legend_handles.setdefault(k, v)
        balance_ax.set_title(
            f"{season_label} – fogyasztás (felül) / termelés (alul)\n{season_start:%m-%d} → {season_stop:%m-%d}")
        balance_ax.set_ylabel("kWh / óra (+ fogyasztás/import, − termelés/export)")

        hh_col = 1
        if has_bess_col:
            stor_components: list[tuple[str, np.ndarray, str]] = []
            if np.any(season_pv_to_bess > 1e-9):
                stor_components.append(("Töltés a PV-ből", season_pv_to_bess, "#ff8c00"))
            if np.any(season_bess_to_load > 1e-9):
                stor_components.append(("Kisütés a terhelésre", -season_bess_to_load, "#5b8cc0"))

            storage_ax = row_axes[1]
            handles = _stack_signed_bars(storage_ax, x, stor_components)
            for k, v in handles.items():
                legend_handles.setdefault(k, v)
            if np.any(season_bess_soc > 1e-9):
                ax2 = storage_ax.twinx()
                soc_line, = ax2.plot(x, season_bess_soc, color="#6b7280", ls="--", lw=1.2)
                legend_handles.setdefault("BESS töltöttség", soc_line)
                ax2.set_ylabel("Energia (kWh)", color="#6b7280")
                ax2.tick_params(axis="y", colors="#6b7280")
            storage_ax.set_title(f"{season_label} – villamos tároló")
            hh_col = 2

        # Household balances
        hh_ax = row_axes[hh_col]
        if household_ts is not None and household_indices:
            if user_names is None:
                local_names = [f"H{u + 1}" for u in household_indices]
            else:
                local_names = [str(user_names[u]) for u in household_indices]

            hh_grid = _series_or_zero(household_ts, "e_grid_to_load", (len(idx), len(user_names or household_indices)))
            hh_inj = _series_or_zero(household_ts, "e_inj", (len(idx), len(user_names or household_indices)))

            hh_x = np.arange(slc.stop - slc.start, dtype=float)
            household_components: list[tuple[str, np.ndarray, str]] = []
            for pos, u in enumerate(household_indices):
                if hh_grid.ndim != 2 or u >= hh_grid.shape[1]:
                    continue
                net_balance = hh_grid[slc, u] - hh_inj[slc, u]
                household_components.append(
                    (local_names[pos], net_balance, household_colors[pos % len(household_colors)]))

            if household_components:
                handles = _stack_signed_bars(hh_ax, hh_x, household_components)
                for k, v in handles.items():
                    legend_handles.setdefault(k, v)
            hh_ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
            hh_ax.grid(True, alpha=0.25, axis="y")
            hh_ax.set_title(f"{season_label} – kiválasztott háztartások\n(+ fogyasztás/import, − termelés/export)")
            hh_ax.set_ylabel("kWh / óra")
        else:
            hh_ax.text(0.5, 0.5, "Nincs háztartási idősor", ha="center", va="center", transform=hh_ax.transAxes)
            hh_ax.set_axis_off()

        # Shared formatting for the row
        for col in range(ncols):
            if not row_axes[col].get_visible():
                continue
            row_axes[col].set_xlim(-0.5, max(0.5, len(x) - 0.5))
            if len(x) > 0:
                tick_positions = np.linspace(0, len(x) - 1, num=min(5, len(x)), dtype=int)
                tick_labels = [f"{int(p):d}" for p in tick_positions]
                row_axes[col].set_xticks(tick_positions)
                row_axes[col].set_xticklabels(tick_labels, rotation=60)
            if row == nrows - 1:
                row_axes[col].set_xlabel("Óra az ablakban")

    # Also add a compact, visible legend along the bottom for quick reference.
    handles_list = list(legend_handles.values())
    labels_list = list(legend_handles.keys())
    if handles_list:
        ncol_bottom = min(6, max(1, len(handles_list)))
        fig.legend(handles_list, labels_list, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=ncol_bottom,
            frameon=True, )

    # leave a small bottom margin so the bottom legend is visible
    fig.tight_layout(rect=(0.0, 0.06, 0.86, 0.985))
    fig.savefig(out_case / "community_energy_balance.png", dpi=200)
    plt.close(fig)


def split_grid_import_base_boiler(e_base_load: np.ndarray, e_boiler_load: np.ndarray, e_pv_to_load: np.ndarray,
        e_bess_to_load: np.ndarray, ) -> dict:
    """
    A teljes lokális ellátást először az alapfogyasztásra, majd a bojlerre osztja.

    Így a bojler csak akkor kap PV/BESS energiát, ha az adott 15 perces lépésben
    az alapfogyasztás már ki van szolgálva. A maradék bojlerigény B-tarifás
    hálózati import lesz.
    """
    e_base_load = np.maximum(np.asarray(e_base_load, dtype=float), 0.0)
    e_boiler_load = np.maximum(np.asarray(e_boiler_load, dtype=float), 0.0)
    local_supply = np.maximum(np.asarray(e_pv_to_load, dtype=float) + np.asarray(e_bess_to_load, dtype=float), 0.0, )

    e_local_to_base = np.minimum(e_base_load, local_supply)
    remaining_local = np.maximum(local_supply - e_local_to_base, 0.0)
    e_local_to_boiler = np.minimum(e_boiler_load, remaining_local)

    e_grid_to_base = np.maximum(e_base_load - e_local_to_base, 0.0)
    e_grid_to_boiler = np.maximum(e_boiler_load - e_local_to_boiler, 0.0)

    return {"e_local_to_base": e_local_to_base, "e_local_to_boiler": e_local_to_boiler,
        "e_grid_to_base": e_grid_to_base, "e_grid_to_boiler": e_grid_to_boiler, }


def simulate_one_user_greedy(e_load_base: np.ndarray, e_boiler: np.ndarray, e_hp: np.ndarray, e_pv: np.ndarray,
        use_bess: bool, bess_size_kwh: float, eta_bess_in: float, eta_bess_out: float, eta_bess_stor: float,
        soc_bess_min: float, soc_bess_max: float, t_bess_min_h: float, e_boiler_load: np.ndarray | None = None,
        boiler_tariff: str = "B", ) -> dict:
    """Greedy simulator that tracks PV/bess/grid interactions and attributes grid imports
    to base, boiler and heat-pump loads separately so separate tariffs can be applied.
    """
    T = len(e_load_base)

    e_load_base = np.asarray(e_load_base, dtype=float)
    e_boiler = np.asarray(e_boiler, dtype=float)
    e_hp = np.asarray(e_hp, dtype=float)
    e_pv = np.asarray(e_pv, dtype=float)
    if e_boiler_load is None:
        e_boiler_load = np.zeros_like(e_load_base)
    else:
        e_boiler_load = np.maximum(np.asarray(e_boiler_load, dtype=float), 0.0)
    e_base_load = np.maximum(e_load_base - e_boiler_load, 0.0)

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    if boiler_tariff == "A":
        # A tarifás bojler: sima fogyasztó, kaphat PV-t és BESS-t.
        e_dispatch_load = e_load_base
    else:
        # B tarifás bojler: külön mérő, külön áramkör.
        # PV/BESS csak az általános fogyasztást látja el.
        e_dispatch_load = e_base_load

    # per-component time series
    e_pv_to_load = np.zeros(T, dtype=float)
    e_pv_to_bess = np.zeros(T, dtype=float)
    e_bess_to_load = np.zeros(T, dtype=float)
    e_grid_to_load_base = np.zeros(T, dtype=float)
    e_grid_to_load_boiler = np.zeros(T, dtype=float)
    e_grid_to_load_hp = np.zeros(T, dtype=float)
    e_grid_to_load = np.zeros(T, dtype=float)
    e_grid_to_bess = np.zeros(T, dtype=float)
    e_inj = np.zeros(T, dtype=float)
    e_bess = np.zeros(T, dtype=float)
    d_bess_ch = np.zeros(T, dtype=float)
    d_bess_dis = np.zeros(T, dtype=float)

    e_load = e_load_base + e_boiler + e_hp

    if not use_bess or bess_size_kwh <= 1e-12:
        # No battery: PV directly to loads proportionally, leftover is grid import or injection
        for t in range(T):
            lb = max(e_load_base[t], 0.0)
            lo = max(e_boiler[t], 0.0)
            lh = max(e_hp[t], 0.0)
            total_load = lb + lo + lh
            pv_t = max(e_pv[t], 0.0)

            if total_load > 1e-12:
                # proportional allocation of PV to components
                pv_base = min(lb, pv_t * (lb / total_load))
                pv_boiler = min(lo, pv_t * (lo / total_load))
                pv_hp = min(lh, pv_t * (lh / total_load))
                assigned = pv_base + pv_boiler + pv_hp
                # distribute any remaining PV to components with remaining demand
                remaining_pv = pv_t - assigned
                if remaining_pv > 1e-12:
                    for comp_need, setter in ((lb - pv_base, 'b'), (lo - pv_boiler, 'o'), (lh - pv_hp, 'h')):
                        if remaining_pv <= 1e-12:
                            break
                        take = min(comp_need, remaining_pv)
                        if setter == 'b':
                            pv_base += take
                        elif setter == 'o':
                            pv_boiler += take
                        else:
                            pv_hp += take
                        remaining_pv -= take
            else:
                pv_base = pv_boiler = pv_hp = 0.0

            assigned_to_load = pv_base + pv_boiler + pv_hp
            e_pv_to_load[t] = assigned_to_load
            e_pv_to_bess[t] = 0.0

            # compute remaining deficits
            def_base = max(lb - pv_base, 0.0)
            def_boiler = max(lo - pv_boiler, 0.0)
            def_hp = max(lh - pv_hp, 0.0)

            e_grid_to_load_base[t] = def_base
            e_grid_to_load_boiler[t] = def_boiler
            e_grid_to_load_hp[t] = def_hp

            e_inj[t] = max(pv_t - assigned_to_load, 0.0)
            e_bess[t] = 0.0

        # Preserve the component-wise allocation calculated above. Recomputing
        # these flows here would discard the boiler/HP split.
        e_grid_to_load = (
            e_grid_to_load_base
            + e_grid_to_load_boiler
            + e_grid_to_load_hp
        )
    else:
        soc_min_kwh = max(0.0, soc_bess_min) * bess_size_kwh
        soc_max_kwh = max(soc_min_kwh, soc_bess_max * bess_size_kwh)

        if t_bess_min_h is None or t_bess_min_h <= 1e-12:
            p_bess_max_kw = 1e18
        else:
            p_bess_max_kw = bess_size_kwh / t_bess_min_h

        e_bess_max_step = p_bess_max_kw * DT

        soc = 0.5 * bess_size_kwh
        soc = min(max(soc, soc_min_kwh), soc_max_kwh)

        min_mode_steps = 4
        mode = "idle"
        lock_steps_left = 0

        # Greedy logika:
        # 1) PV először közvetlenül a fogyasztást látja el.
        # 2) Ha PV többlet van, az BESS-be tölt.
        # 3) Ha a BESS már nem tud több energiát felvenni, a maradék PV export.
        # 4) Ha fogyasztási hiány van, a BESS kisüt.
        # 5) Ha még mindig hiány van, hálózati import történik.
        # 6) A BESS SOC minden időlépésben önkisüléssel csökken.

        # A tarifás bojler esetén:
        #   a bojler a teljes háztartási fogyasztás része,
        #   ezért PV és BESS is kiszolgálhatja.
        #
        # B tarifás bojler esetén:
        #   a bojler külön mérőn van,
        #   ezért nem kap PV-t és nem kap BESS energiát,
        #   a teljes bojlerigény B tarifás hálózati import.

        for t in range(T):
            soc *= eta_bess_stor

            lb = max(e_load_base[t], 0.0)
            lo = max(e_boiler[t], 0.0)
            lh = max(e_hp[t], 0.0)
            total_load = lb + lo + lh
            load_t = max(e_dispatch_load[t], 0.0)
            pv_t = max(e_pv[t], 0.0)

            # allocate PV to loads proportionally
            if total_load > 1e-12:
                pv_base = min(lb, pv_t * (lb / total_load))
                pv_boiler = min(lo, pv_t * (lo / total_load))
                pv_hp = min(lh, pv_t * (lh / total_load))
                assigned = pv_base + pv_boiler + pv_hp
                remaining_pv = pv_t - assigned
                if remaining_pv > 1e-12:
                    for comp_need, setter in ((lb - pv_base, 'b'), (lo - pv_boiler, 'o'), (lh - pv_hp, 'h')):
                        if remaining_pv <= 1e-12:
                            break
                        take = min(comp_need, remaining_pv)
                        if setter == 'b':
                            pv_base += take
                        elif setter == 'o':
                            pv_boiler += take
                        else:
                            pv_hp += take
                        remaining_pv -= take
            else:
                pv_base = pv_boiler = pv_hp = 0.0

            assigned_to_load = pv_base + pv_boiler + pv_hp

            # charge battery with remaining PV
            surplus = max(pv_t - assigned_to_load, 0.0)
            can_charge = (lock_steps_left <= 0) or (mode == "charge")
            can_discharge = (lock_steps_left <= 0) or (mode == "discharge")

            charge = 0.0
            if surplus > 1e-12 and can_charge:
                room_kwh = max(soc_max_kwh - soc, 0.0)
                max_charge_by_soc = room_kwh / max(eta_bess_in, 1e-12)
                charge = min(surplus, e_bess_max_step, max_charge_by_soc)
                soc += charge * eta_bess_in
                surplus -= charge

            # deficits after PV
            def_base = max(lb - pv_base, 0.0)
            def_boiler = max(lo - pv_boiler, 0.0)
            def_hp = max(lh - pv_hp, 0.0)
            deficit = def_base + def_boiler + def_hp

            # discharge to cover deficit
            discharge = 0.0
            if deficit > 1e-12 and can_discharge:
                avail_kwh = max(soc - soc_min_kwh, 0.0)
                max_discharge_by_soc = avail_kwh * eta_bess_out
                discharge = min(deficit, e_bess_max_step, max_discharge_by_soc)
                # distribute discharge proportionally to component deficits
                if deficit > 1e-12:
                    frac = discharge / deficit
                else:
                    frac = 0.0
                d_base = min(def_base, def_base * frac)
                d_boiler = min(def_boiler, def_boiler * frac)
                d_hp = min(def_hp, def_hp * frac)
                # correct any rounding so sum equals discharge
                sum_d = d_base + d_boiler + d_hp
                if sum_d < discharge and deficit > 1e-12:
                    rem = discharge - sum_d
                    # distribute remainder to components with remaining deficit
                    for need, arr in ((def_base - d_base, 'b'), (def_boiler - d_boiler, 'o'), (def_hp - d_hp, 'h')):
                        if rem <= 1e-12:
                            break
                        add = min(need, rem)
                        if arr == 'b':
                            d_base += add
                        elif arr == 'o':
                            d_boiler += add
                        else:
                            d_hp += add
                        rem -= add

                soc -= discharge / max(eta_bess_out, 1e-12)
                def_base = max(def_base - d_base, 0.0)
                def_boiler = max(def_boiler - d_boiler, 0.0)
                def_hp = max(def_hp - d_hp, 0.0)

            e_pv_to_load[t] = assigned_to_load
            e_pv_to_bess[t] = charge
            e_bess_to_load[t] = discharge
            e_grid_to_load_base[t] = def_base
            e_grid_to_load_boiler[t] = def_boiler
            e_grid_to_load_hp[t] = def_hp

            grid_charge = 0.0
            if soc < soc_min_kwh - 1e-12 and can_charge and surplus <= 1e-12:
                need_to_soc = soc_min_kwh - soc
                grid_charge = min(e_bess_max_step, need_to_soc / max(eta_bess_in, 1e-12), )
                soc += grid_charge * eta_bess_in

            if charge > 1e-12 or grid_charge > 1e-12:
                if mode != "charge":
                    mode = "charge"
                    lock_steps_left = min_mode_steps - 1

            elif discharge > 1e-12:
                if mode != "discharge":
                    mode = "discharge"
                    lock_steps_left = min_mode_steps - 1

            else:
                if lock_steps_left <= 0:
                    mode = "idle"

            if lock_steps_left > 0:
                lock_steps_left -= 1

            e_pv_to_bess[t] = charge
            e_bess_to_load[t] = discharge
            e_grid_to_load[t] = max(deficit, 0.0) + grid_charge
            e_inj[t] = max(surplus, 0.0)
            e_bess[t] = soc
            e_grid_to_bess[t] = grid_charge

            d_bess_ch[t] = 1.0 if (charge > 1e-12 or grid_charge > 1e-12) else 0.0
            d_bess_dis[t] = 1.0 if discharge > 1e-12 else 0.0

    load_kwh = e_load.sum()
    pv_kwh = e_pv.sum()
    pv_to_load_kwh = e_pv_to_load.sum()
    pv_to_bess_kwh = e_pv_to_bess.sum()
    bess_to_load_kwh = e_bess_to_load.sum()
    grid_import_kwh = (e_grid_to_load_base + e_grid_to_load_boiler + e_grid_to_load_hp).sum()
    injection_kwh = e_inj.sum()

    if boiler_tariff == "A":
        # A tarifás bojler: nincs külön B tarifás import.
        e_grid_to_base = e_grid_to_load.copy()
        e_grid_to_boiler = np.zeros_like(e_grid_to_load)

        e_local_total = e_pv_to_load + e_bess_to_load
        e_local_to_base = np.minimum(e_base_load, e_local_total)
        e_local_to_boiler = np.minimum(np.maximum(e_local_total - e_local_to_base, 0.0), e_boiler_load, )

    else:
        # B tarifás bojler: teljes bojlerfogyasztás B tarifás hálózati import.
        e_grid_to_base = e_grid_to_load.copy()
        e_grid_to_boiler = e_boiler_load.copy()

        e_local_to_base = e_pv_to_load + e_bess_to_load
        e_local_to_boiler = np.zeros_like(e_boiler_load)

        # Teljes hálózati import = A tarifás alapimport + B tarifás bojlerimport.
        e_grid_to_load = e_grid_to_base + e_grid_to_boiler

    grid_import_kwh = e_grid_to_load.sum()

    bill = calc_bill_15min_brutto(e_grid_to_load=e_grid_to_load, e_inj=e_inj, e_grid_to_boiler=e_grid_to_boiler, )

    # SCI: önfogyasztási index.
    # A megtermelt PV mekkora része marad helyben:
    # - közvetlen PV -> fogyasztás
    # - PV -> BESS töltés
    #
    # Fontos: SCI-ben a PV->BESS töltést számoljuk,
    # nem a későbbi BESS->load kisütést, mert az már veszteségekkel csökkentett energia.
    self_consumed_pv_kwh = pv_to_load_kwh + pv_to_bess_kwh
    self_consumption_ratio = self_consumed_pv_kwh / pv_kwh if pv_kwh > 1e-12 else 0.0
    self_consumption_ratio = min(max(self_consumption_ratio, 0.0), 1.0)

    # SSI: önellátási index.
    # A fogyasztás mekkora részét fedezi helyi energia:
    # - közvetlen PV -> fogyasztás
    # - BESS -> fogyasztás
    #
    # Itt a BESS kisütés számít, mert ez ténylegesen fogyasztást lát el.
    locally_supplied_load_kwh = pv_to_load_kwh + bess_to_load_kwh
    self_sufficiency_ratio = locally_supplied_load_kwh / load_kwh if load_kwh > 1e-12 else 0.0
    self_sufficiency_ratio = min(max(self_sufficiency_ratio, 0.0), 1.0)

    return {"timeseries": {"e_load": e_load, "e_pv": e_pv, "e_pv_to_load": e_pv_to_load, "e_pv_to_bess": e_pv_to_bess,
        "e_bess_to_load": e_bess_to_load,
        "e_grid_to_load": e_grid_to_load, "e_grid_to_bess": e_grid_to_bess, "e_grid_to_base": e_grid_to_base,
        "e_grid_to_boiler": e_grid_to_boiler, "e_local_to_base": e_local_to_base,
        "e_local_to_boiler": e_local_to_boiler, "e_inj": e_inj, "e_bess": e_bess, "d_bess_ch": d_bess_ch,
        "d_bess_dis": d_bess_dis, },

        "annual": {"load_kwh": load_kwh, "pv_kwh": pv_kwh, "pv_to_load_kwh": pv_to_load_kwh,
            "pv_to_bess_kwh": pv_to_bess_kwh, "bess_to_load_kwh": bess_to_load_kwh, "grid_import_kwh": grid_import_kwh,
            "grid_import_a_kwh": bill["grid_import_a_kwh"], "grid_import_b_kwh": bill["grid_import_b_kwh"],
            "grid_to_bess_kwh": float(e_grid_to_bess.sum()), "injection_kwh": injection_kwh,
            "self_consumed_pv_kwh": self_consumed_pv_kwh, "locally_supplied_load_kwh": locally_supplied_load_kwh,
            "self_consumption_ratio": self_consumption_ratio, "self_sufficiency_ratio": self_sufficiency_ratio,
            **bill, }, }


def _legacy_plot_household_percentiles_by_group(per_user_df: pd.DataFrame, out_case: Path):
    df = per_user_df.copy().sort_values("brt_bill_ft").reset_index(drop=True)
    n = len(df)

    if n == 0:
        return

    df["percentile"] = 100.0 * (np.arange(n) + 0.5) / n

    marker_map = {"Nincs PV": "o", "PV": "s", "HP": "D", "PV+HP": "P", "PV+BESS": "^", "PV+BESS+HP": "X",
        # Boiler-aware groups
        "BESS+HP": "H", "BESS+HP+Boiler": "*", "BESS+Boiler": "v", "HP+Boiler": "d", "PV+BESS+HP+Boiler": "*",
        "PV+BESS+Boiler": "v", "PV+HP+Boiler": "d", }

    plt.figure(figsize=(9, 5.5))

    for group in _ordered_group_labels(df):
        sub = df[df["group_label"] == group].copy()
        if len(sub) == 0:
            continue

        plt.scatter(sub["brt_bill_ft"], sub["percentile"], s=28, marker=marker_map.get(group, "o"), alpha=0.8,
            label=group, )
    x_lo = np.percentile(df["brt_bill_ft"], 1)
    x_hi = np.percentile(df["brt_bill_ft"], 99)
    plt.xlim(x_lo, x_hi)
    plt.xlabel("Éves bruttó villanyszámla [Ft/év]")
    plt.ylabel("Háztartások aránya [%]")
    plt.title("Háztartások villanyszámla szerinti eloszlása")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_case / "household_bill_percentiles_by_group.png", dpi=200)
    plt.close()


def _legacy_plot_household_percentiles_by_group_with_global_scurve(per_user_df: pd.DataFrame, out_case: Path):
    from scipy.optimize import curve_fit

    df = per_user_df.copy().sort_values("brt_bill_ft").reset_index(drop=True)
    n = len(df)

    if n == 0:
        return

    df["percentile"] = 100.0 * (np.arange(n) + 0.5) / n

    marker_map = {"Nincs PV": "o", "PV": "s", "HP": "D", "PV+HP": "P", "PV+BESS": "^", "PV+BESS+HP": "X",
        # Boiler-aware groups
        "BESS+HP": "H", "BESS+HP+Boiler": "*", "BESS+Boiler": "v", "HP+Boiler": "d", "PV+BESS+HP+Boiler": "*",
        "PV+BESS+Boiler": "v", "PV+HP+Boiler": "d", }

    plt.figure(figsize=(9, 5.5))

    # pontok csoportonként
    for group in _ordered_group_labels(df):
        sub = df[df["group_label"] == group].copy()
        if len(sub) == 0:
            continue

        plt.scatter(sub["brt_bill_ft"], sub["percentile"], s=28, marker=marker_map.get(group, "o"), alpha=0.8,
            label=group, )

    # egyetlen globális S-görbe az összes háztartásra
    x = df["brt_bill_ft"].to_numpy(dtype=float)
    y = df["percentile"].to_numpy(dtype=float)

    def logistic(x_, x0, k):
        return 100.0 / (1.0 + np.exp(-k * (x_ - x0)))

    if len(df) >= 4:
        x0_init = np.median(x)
        spread = max(np.std(x), 1.0)
        k_init = 1.0 / spread

        try:
            popt = curve_fit(logistic, x, y, p0=[x0_init, k_init], maxfev=20000)[0]
            x_grid = np.linspace(x.min(), x.max(), 400)
            y_fit = logistic(x_grid, *popt)
            plt.plot(x_grid, y_fit, color="darkred", linewidth=2, label="Illesztett S-görbe (összes háztartás)")
        except Exception:
            pass

    x_lo = np.percentile(df["brt_bill_ft"], 1)
    x_hi = np.percentile(df["brt_bill_ft"], 100)
    plt.xlim(x_lo, x_hi)

    plt.xlabel("Éves bruttó villanyszámla [Ft/év]")
    plt.ylabel("Háztartások aránya [%]")
    plt.title("Háztartások villanyszámla szerinti eloszlása")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_case / "household_bill_percentiles_by_group_global_scurve.png", dpi=200)
    plt.close()


def _run_case_a(case_name: str, sim_yaml: str, profiles_csv: str, dhw_profiles_csv: str, out_dir: str,
        max_users: int | None,
        include_bess: bool, include_boiler: bool, include_heat_pump: bool = False, heat_pump_share_pct: float = 0.0,
        bess_share_pct: float = 100.0, boiler_tariff: str = "B", pv_ratio: float = 1.0, ) -> dict:
    out_case = Path(out_dir)
    out_case.mkdir(parents=True, exist_ok=True)

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    inputs = read_simulation_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
        pv_ratio=pv_ratio,
        dt=DT,
    )
    e_pv = inputs.e_pv_kwh
    e_ue = inputs.e_ue_kwh
    e_el_heater = inputs.e_el_heater_kwh
    size_bess = inputs.size_bess
    eta_bess_in_u = inputs.eta_bess_in
    eta_bess_out_u = inputs.eta_bess_out
    eta_bess_stor_u = inputs.eta_bess_stor
    soc_bess_min_u = inputs.soc_bess_min
    soc_bess_max_u = inputs.soc_bess_max
    t_bess_min_u = inputs.t_bess_min
    user_names = inputs.user_names

    U = len(user_names)
    T = e_ue.shape[0]

    e_boiler = e_el_heater if include_boiler else np.zeros_like(e_el_heater)
    rng = np.random.default_rng()

    hp_enabled_arr = np.zeros(U, dtype=bool)
    hp_user_meta: list[dict] = [dict() for _ in range(U)]
    ts_e_hp = np.zeros((T, U), dtype=float)

    if include_heat_pump and heat_pump_share_pct > 0.0:
        try:
            idx_hp, T_env_hp, I_by_dir_hp = load_hp_weather()
            print(f"[INFO] HP weather inputs betöltve: {len(idx_hp)} időlépés")
        except Exception as exc:
            print(f"[WARN] HP weather inputs nem tölthetők be, HP kikapcsolva: {exc}")
            include_heat_pump = False
        else:
            hp_share = _clamp(float(heat_pump_share_pct), 0.0, 100.0)
            n_hp_users = int(round(U * hp_share / 100.0))
            n_hp_users = max(0, min(n_hp_users, U))
            if n_hp_users > 0:
                hp_idx = np.sort(rng.choice(U, size=n_hp_users, replace=False))
                hp_enabled_arr[hp_idx] = True
                print(f"[INFO] HP háztartások száma: {n_hp_users}/{U} ({hp_share:.1f}%)")
                print("[INFO] HP háztartások kiválasztása véletlenszerűen történt.")
            else:
                print(f"[INFO] HP arány {hp_share:.1f}%, ezért nincs HP háztartás.")

            for u in np.where(hp_enabled_arr)[0]:
                print(f"[INFO] HP szimuláció indul: {user_names[u]} ({u + 1}/{U})")
                e_hp_hourly, meta = _simulate_heat_pump_hourly_load(idx_hp, T_env_hp, I_by_dir_hp, rng)
                if len(e_hp_hourly) != T:
                    raise RuntimeError(f"A HP profil hossza {len(e_hp_hourly)}, de a fogyasztási profil hossza {T}.")
                ts_e_hp[:, u] = e_hp_hourly
                hp_user_meta[u] = meta
                print(f"[INFO] HP kész: {user_names[u]} | "
                      f"ház={meta['hp_house_key']} | E_hp={meta['hp_energy_kwh']:.1f} kWh | "
                      f"P_csúcs={meta['hp_peak_kw']:.2f} kW")

    # Keep the optional loads selected above: disabled boilers remain zero,
    # while generated heat-pump profiles contribute to the total load.
    e_total_load = e_ue + e_boiler + ts_e_hp
    e_base_load = e_ue

    rows = []

    ts_e_load = np.zeros((T, U), dtype=float)
    ts_e_pv = np.zeros((T, U), dtype=float)
    ts_e_pv_to_load = np.zeros((T, U), dtype=float)
    ts_e_pv_to_bess = np.zeros((T, U), dtype=float)
    ts_e_bess_to_load = np.zeros((T, U), dtype=float)
    ts_e_grid_to_load = np.zeros((T, U), dtype=float)
    ts_e_grid_to_base = np.zeros((T, U), dtype=float)
    ts_e_grid_to_bess = np.zeros((T, U), dtype=float)
    ts_e_grid_to_boiler = np.zeros((T, U), dtype=float)
    ts_e_inj = np.zeros((T, U), dtype=float)
    ts_e_bess = np.zeros((T, U), dtype=float)
    ts_d_bess_ch = np.zeros((T, U), dtype=float)
    ts_d_bess_dis = np.zeros((T, U), dtype=float)
    ts_e_base_load = np.zeros((T, U), dtype=float)
    ts_e_boiler = np.zeros((T, U), dtype=float)

    pv_annual_kwh = e_pv.sum(axis=0)
    has_pv_arr = pv_annual_kwh > 1e-9

    bess_enabled_arr = np.zeros(len(user_names), dtype=bool)

    if include_bess:
        pv_user_idx = np.where(has_pv_arr)[0]
        n_pv_users = len(pv_user_idx)

        n_bess_users = int(round(n_pv_users * bess_share_pct / 100.0))
        n_bess_users = max(0, min(n_bess_users, n_pv_users))

        # egyszerű, determinisztikus megoldás:
        # az első n darab PV-s háztartás kap BESS-t
        bess_enabled_arr[pv_user_idx[:n_bess_users]] = True

        print(f"[INFO] PV-s háztartások száma: {n_pv_users}")
        print(f"[INFO] BESS arány: {bess_share_pct}%")
        print(f"[INFO] BESS-t kapó PV-s háztartások száma: {n_bess_users}")

    for u in range(U):
        sim = simulate_one_user_greedy(e_load_base=e_ue[:, u], e_boiler=e_boiler[:, u], e_hp=ts_e_hp[:, u],
            e_pv=e_pv[:, u], e_boiler_load=e_boiler[:, u], use_bess=bool(bess_enabled_arr[u]),
            bess_size_kwh=float(size_bess[u]), eta_bess_in=float(eta_bess_in_u[u]),
            eta_bess_out=float(eta_bess_out_u[u]), eta_bess_stor=float(eta_bess_stor_u[u]),
            soc_bess_min=float(soc_bess_min_u[u]), soc_bess_max=float(soc_bess_max_u[u]),
            t_bess_min_h=float(t_bess_min_u[u]), boiler_tariff=boiler_tariff, )

        annual = sim["annual"]
        times = sim["timeseries"]

        base_load_kwh = e_ue[:, u].sum()
        boiler_kwh = e_boiler[:, u].sum()
        hp_kwh = ts_e_hp[:, u].sum()
        pmax_kw = float(np.max(e_total_load[:, u] / DT))

        rows.append({"user_name": user_names[u], "has_pv": bool(e_pv[:, u].sum() > 1e-9),
            "has_bess": bool(bess_enabled_arr[u] and size_bess[u] > 1e-9), "has_heat_pump": bool(hp_enabled_arr[u]),
            "has_boiler": bool(include_boiler and e_boiler[:, u].sum() > 1e-9),
            "group_label": _group_label(has_pv=bool(has_pv_arr[u]),
                                        has_bess=bool(bess_enabled_arr[u] and size_bess[u] > 1e-9),
                                        has_heat_pump=bool(hp_enabled_arr[u]),
                                        has_boiler=bool(include_boiler and e_boiler[:, u].sum() > 1e-9)),
            "base_load_kwh": base_load_kwh, "boiler_kwh": boiler_kwh, "heat_pump_kwh": hp_kwh,
            "heat_pump_peak_kw": float(hp_user_meta[u].get("hp_peak_kw", 0.0)),
            "heat_pump_house_key": hp_user_meta[u].get("hp_house_key", ""),
            "heat_pump_setpoint_offset_c": float(hp_user_meta[u].get("hp_setpoint_offset_c", 0.0)),
            "heat_pump_night_delta_c": float(hp_user_meta[u].get("hp_night_delta_c", 0.0)),
            "heat_pump_deadband_c": float(hp_user_meta[u].get("hp_deadband_c", 0.0)),
            "heat_pump_min_on_min": float(hp_user_meta[u].get("hp_min_on_min", 0.0)),
            "heat_pump_min_off_min": float(hp_user_meta[u].get("hp_min_off_min", 0.0)),
            "heat_pump_ramp_kW_per_step": float(hp_user_meta[u].get("hp_ramp_kW_per_step", 0.0)),
            "heat_pump_p_th_min_kW": float(hp_user_meta[u].get("hp_p_th_min_kW", 0.0)),
            "total_load_kwh": annual["load_kwh"], "pv_kwh": annual["pv_kwh"], "Pmax_kw": pmax_kw,
            "pv_to_load_kwh": annual["pv_to_load_kwh"], "pv_to_bess_kwh": annual["pv_to_bess_kwh"],
            "bess_to_load_kwh": annual["bess_to_load_kwh"], "grid_to_bess_kwh": annual["grid_to_bess_kwh"],
            "grid_import_kwh": annual["grid_import_kwh"], "grid_import_a_kwh": annual["grid_import_a_kwh"],
            "grid_import_b_kwh": annual["grid_import_b_kwh"], "grid_import_low_kwh": annual["grid_import_low_kwh"],
            "grid_import_high_kwh": annual["grid_import_high_kwh"],
            "grid_import_a_low_kwh": annual["grid_import_a_low_kwh"],
            "grid_import_a_high_kwh": annual["grid_import_a_high_kwh"],
            "grid_import_b_low_kwh": annual["grid_import_b_low_kwh"],
            "grid_import_b_high_kwh": annual["grid_import_b_high_kwh"], "injection_kwh": annual["injection_kwh"],
            "self_consumed_pv_kwh": annual["self_consumed_pv_kwh"],
            "locally_supplied_load_kwh": annual["locally_supplied_load_kwh"],
            "self_consumption_ratio": annual["self_consumption_ratio"],
            "self_sufficiency_ratio": annual["self_sufficiency_ratio"], "SCI": annual["self_consumption_ratio"],
            "SSI": annual["self_sufficiency_ratio"], "import_cost_a_ft": annual["import_cost_a_ft"],
            "import_cost_b_ft": annual["import_cost_b_ft"], "import_cost_ft": annual["import_cost_ft"],
            "export_revenue_ft": annual["export_revenue_ft"], "brt_bill_ft": annual["brt_bill_ft"], })

        ts_e_load[:, u] = times["e_load"]
        ts_e_pv[:, u] = times["e_pv"]
        ts_e_pv_to_load[:, u] = times["e_pv_to_load"]
        ts_e_pv_to_bess[:, u] = times["e_pv_to_bess"]
        ts_e_bess_to_load[:, u] = times["e_bess_to_load"]
        ts_e_grid_to_load[:, u] = times["e_grid_to_load"]
        ts_e_grid_to_base[:, u] = times["e_grid_to_base"]
        ts_e_grid_to_bess[:, u] = times["e_grid_to_bess"]
        ts_e_grid_to_boiler[:, u] = times["e_grid_to_boiler"]
        ts_e_inj[:, u] = times["e_inj"]
        ts_e_bess[:, u] = times["e_bess"]
        ts_d_bess_ch[:, u] = times["d_bess_ch"]
        ts_d_bess_dis[:, u] = times["d_bess_dis"]
        ts_e_base_load[:, u] = e_ue[:, u]
        ts_e_boiler[:, u] = e_boiler[:, u]

    per_user_df = pd.DataFrame(rows)
    per_user_df.to_csv(out_case / "per_user_summary.csv", index=False)

    plot_household_percentiles_by_group(per_user_df, out_case)
    plot_household_percentiles_by_group_with_global_scurve(per_user_df, out_case)

    total = {"case_name": case_name, "boiler_tariff": boiler_tariff, "n_users": int(U),
        "base_load_kwh": float(per_user_df["base_load_kwh"].sum()),
        "boiler_kwh": float(per_user_df["boiler_kwh"].sum()),
        "heat_pump_kwh": float(per_user_df["heat_pump_kwh"].sum()),
        "total_load_kwh": float(per_user_df["total_load_kwh"].sum()), "pv_kwh": float(per_user_df["pv_kwh"].sum()),
        "pv_to_load_kwh": float(per_user_df["pv_to_load_kwh"].sum()),
        "pv_to_bess_kwh": float(per_user_df["pv_to_bess_kwh"].sum()),
        "bess_to_load_kwh": float(per_user_df["bess_to_load_kwh"].sum()),
        "locally_supplied_load_kwh": float(per_user_df["locally_supplied_load_kwh"].sum()),
        "grid_import_kwh": float(per_user_df["grid_import_kwh"].sum()),
        "grid_import_a_kwh": float(per_user_df["grid_import_a_kwh"].sum()),
        "grid_import_b_kwh": float(per_user_df["grid_import_b_kwh"].sum()),
        "grid_import_low_kwh": float(per_user_df["grid_import_low_kwh"].sum()),
        "grid_import_high_kwh": float(per_user_df["grid_import_high_kwh"].sum()),
        "grid_import_a_low_kwh": float(per_user_df["grid_import_a_low_kwh"].sum()),
        "grid_import_a_high_kwh": float(per_user_df["grid_import_a_high_kwh"].sum()),
        "grid_import_b_low_kwh": float(per_user_df["grid_import_b_low_kwh"].sum()),
        "grid_import_b_high_kwh": float(per_user_df["grid_import_b_high_kwh"].sum()),
        "grid_to_bess_kwh": float(per_user_df["grid_to_bess_kwh"].sum()),
        "injection_kwh": float(per_user_df["injection_kwh"].sum()),
        "import_cost_a_ft": float(per_user_df["import_cost_a_ft"].sum()),
        "import_cost_b_ft": float(per_user_df["import_cost_b_ft"].sum()),
        "import_cost_ft": float(per_user_df["import_cost_ft"].sum()),
        "export_revenue_ft": float(per_user_df["export_revenue_ft"].sum()),
        "brt_bill_ft": float(per_user_df["brt_bill_ft"].sum()), "bess_share_pct": float(bess_share_pct),
        "heat_pump_share_pct": float(heat_pump_share_pct), "pv_user_count": int(has_pv_arr.sum()),
        "bess_enabled_user_count": int(bess_enabled_arr.sum()),
        "heat_pump_enabled_user_count": int(hp_enabled_arr.sum()), }

    total["self_consumed_pv_kwh"] = (total["pv_to_load_kwh"] + total["pv_to_bess_kwh"])

    total["self_consumption_ratio"] = (
        total["self_consumed_pv_kwh"] / total["pv_kwh"] if total["pv_kwh"] > 1e-12 else 0.0)

    total["self_consumption_ratio"] = min(max(total["self_consumption_ratio"], 0.0), 1.0)
    total["self_sufficiency_ratio"] = (
        total["locally_supplied_load_kwh"] / total["total_load_kwh"] if total["total_load_kwh"] > 1e-12 else 0.0)

    total["heat_pump_share_realized_pct"] = (
        100.0 * total["heat_pump_enabled_user_count"] / total["n_users"] if total["n_users"] > 0 else 0.0)

    total["self_sufficiency_ratio"] = min(max(total["self_sufficiency_ratio"], 0.0), 1.0)
    total["SCI"] = total["self_consumption_ratio"]
    total["SSI"] = total["self_sufficiency_ratio"]

    (out_case / "summary.json").write_text(json.dumps(total, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_ts(arr: np.ndarray, filename: str):
        pd.DataFrame(arr, columns=user_names).to_csv(out_case / filename, index=False)

    save_ts(ts_e_load, "e_load.csv")
    save_ts(ts_e_pv, "e_pv.csv")
    save_ts(ts_e_hp, "e_hp.csv")
    save_ts(ts_e_pv_to_load, "e_pv_to_load.csv")
    save_ts(ts_e_pv_to_bess, "e_pv_to_bess.csv")
    save_ts(ts_e_bess_to_load, "e_bess_to_load.csv")
    save_ts(ts_e_grid_to_load, "e_grid_to_load.csv")
    save_ts(ts_e_grid_to_base, "e_grid_to_base.csv")
    save_ts(ts_e_grid_to_bess, "e_grid_to_bess.csv")
    save_ts(ts_e_grid_to_boiler, "e_grid_to_boiler.csv")
    save_ts(ts_e_inj, "e_inj.csv")
    save_ts(ts_e_bess, "e_bess.csv")
    save_ts(ts_d_bess_ch, "d_bess_ch.csv")
    save_ts(ts_d_bess_dis, "d_bess_dis.csv")
    save_ts(ts_e_base_load, "e_base_load.csv")
    save_ts(ts_e_boiler, "e_boiler.csv")

    community_ts = pd.DataFrame(
        {"e_load_total": ts_e_load.sum(axis=1), "e_load_base_total": (e_ue + e_boiler).sum(axis=1),
            "e_ue_total": e_ue.sum(axis=1), "e_boiler_total": e_boiler.sum(axis=1), "e_hp_total": ts_e_hp.sum(axis=1),
            "e_pv_total": ts_e_pv.sum(axis=1), "e_pv_to_load_total": ts_e_pv_to_load.sum(axis=1),
            "e_pv_to_bess_total": ts_e_pv_to_bess.sum(axis=1), "e_bess_to_load_total": ts_e_bess_to_load.sum(axis=1),
            "e_grid_to_load_total": ts_e_grid_to_load.sum(axis=1),
            "e_grid_to_base_total": ts_e_grid_to_base.sum(axis=1),
            "e_grid_to_boiler_total": ts_e_grid_to_boiler.sum(axis=1), "e_inj_total": ts_e_inj.sum(axis=1),
            "e_bess_total": ts_e_bess.sum(axis=1), "e_grid_to_bess_total": ts_e_grid_to_bess.sum(axis=1), })
    community_ts.to_csv(out_case / "community_timeseries.csv", index=False)

    plot_community_load_breakdown(community_ts, out_case)

    plot_community_energy_balance(community_ts, out_case, per_user_df=per_user_df,
        household_ts={"e_grid_to_load": ts_e_grid_to_load, "e_inj": ts_e_inj, }, user_names=user_names, )

    print(f"[INFO] Kész: {case_name}")
    print(f"[INFO] Output: {out_case}")
    if include_heat_pump:
        print(f"[INFO] HP háztartások: {int(hp_enabled_arr.sum())}/{U}")
        print(f"[INFO] HP összes energia: {float(per_user_df['heat_pump_kwh'].sum()):.1f} kWh")
    print(json.dumps(total, indent=2, ensure_ascii=False))
    return total


def run_case(
        case_name: str,
        sim_yaml: str,
        profiles_csv: str,
        dhw_profiles_csv: str,
        out_dir: str,
        max_users: int | None,
        include_bess: bool,
        include_boiler: bool,
        include_heat_pump: bool = False,
        heat_pump_share_pct: float = 0.0,
        bess_share_pct: float = 100.0,
        boiler_tariff: str = "B",
        pv_ratio: float = 1.0,
        scenario: str = "a",
        sharing_mode: str = "proportional",
) -> dict:
    """Run non-optimized scenario ``a`` (individual) or ``b`` (community sharing).

    Scenario ``b`` calculates the virtual community flows and the seller-buyer
    settlement. Heat-pump profiles are currently available only in scenario ``a``.
    """
    scenario = str(scenario).lower().strip()
    if scenario not in {"a", "b"}:
        raise ValueError("scenario csak 'a' vagy 'b' lehet")

    if scenario == "a":
        return _run_case_a(
            case_name=case_name,
            sim_yaml=sim_yaml,
            profiles_csv=profiles_csv,
            dhw_profiles_csv=dhw_profiles_csv,
            out_dir=out_dir,
            max_users=max_users,
            include_bess=include_bess,
            include_boiler=include_boiler,
            include_heat_pump=include_heat_pump,
            heat_pump_share_pct=heat_pump_share_pct,
            bess_share_pct=bess_share_pct,
            boiler_tariff=boiler_tariff,
            pv_ratio=pv_ratio,
        )

    if include_heat_pump:
        raise ValueError("A hőszivattyús profil jelenleg csak az 'a' szcenárióban támogatott.")

    from NotOptimizedScenarios.noopt_community_1b import run_case_disaggregated_nonopt_shared

    result = run_case_disaggregated_nonopt_shared(
        case_name=case_name,
        sim_yaml=sim_yaml,
        profiles_csv=profiles_csv,
        dhw_profiles_csv=dhw_profiles_csv,
        out_dir=out_dir,
        max_users=max_users,
        include_bess=include_bess,
        include_boiler=include_boiler,
        bess_share_pct=bess_share_pct,
        sharing_mode=sharing_mode,
        boiler_tariff=boiler_tariff,
        pv_ratio=pv_ratio,
    )
    return result["summary"]
