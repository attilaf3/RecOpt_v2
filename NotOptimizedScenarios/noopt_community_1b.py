from __future__ import annotations

"""
Disaggregált, NEM optimalizált energiaközösségi szimuláció.

Logika háztartásonként, 15 perces energia-idősorokkal [kWh / lépés]:
    1) saját PV -> saját fogyasztás
    2) saját PV-többlet -> saját BESS töltés, ha van BESS
    3) saját BESS -> saját fogyasztási hiány, ha van BESS
    4) maradék PV-többlet -> energiaközösségi megosztás
    5) maradék hiány -> energiaközösségi megosztásból vett energia
    6) maradék hiány/többlet -> külső hálózat

A BESS vezérlése itt nem optimalizált, hanem greedy/prioritásos. A BESS külső
hálózatból is tölthet, de csak minimum SOC-védelem miatt.

Megosztási elszámolás:
    - a 2523 kWh/év kedvezményes sáv a vevő éves A-tarifás/
      megosztott energiavásárlási keretéhez tartozik, nem az eladóhoz;
    - ha a vevő még a kedvezményes sávban van, a megosztott energia
      energiadíja 5 Ft/kWh, plusz 31 Ft/kWh RHD;
    - ha a vevő már a kedvezményes sáv felett van, a megosztott energia
      energiadíja 21 Ft/kWh, plusz 31 Ft/kWh RHD;
    - a megosztott energia energiadíját (5 vagy 21 Ft/kWh) a PV-s eladó kapja;
    - a hálózatba menő PV-export átvételi díja külön 5 Ft/kWh.

A fájlt tedd a nonopt_common.py mellé, vagy olyan helyre, ahonnan a
nonopt_common importálható.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import yaml

from nonopt_common import (
    DT,
    EXPORT_FT_PER_KWH,
    HIGH_TARIFF_FT_PER_KWH,
    LOW_TARIFF_FT_PER_KWH,
    LOW_TARIFF_LIMIT_KWH,
    _group_label,
    plot_community_energy_balance,
    plot_household_percentiles_by_group,
    plot_household_percentiles_by_group_with_global_scurve,
)


C_HSS = 0.00116667  # kWh/(liter*K)


def _as_15min_energy(profile: np.ndarray) -> np.ndarray:
    """15 perces energia-idősor ellenőrzése [kWh/lépés]."""
    p = np.asarray(profile, dtype=float).ravel()
    if p.size != 35040:
        raise ValueError(f"A profil hossza {p.size}, de 35040 kell.")
    return np.maximum(p, 0.0)


def _find_user_yaml(
    roots: Iterable[os.PathLike],
    name: str,
) -> Optional[Path]:
    for root in roots:
        for candidate in (name, f"{name}.yaml"):
            path = Path(root) / candidate
            if path.exists():
                return path
    return None


def _dhw_liter_to_thermal_power_kw(
    dhw: pd.DataFrame,
    hss: dict,
    dt: float = DT,
) -> np.ndarray:
    """Liter/lépés DHW-profil átalakítása kW hőigénnyé."""
    profile = hss.get("profile")
    if profile is None or str(profile) not in dhw.columns:
        return np.zeros(35040, dtype=float)

    liters = np.maximum(
        np.asarray(dhw[str(profile)].to_numpy(), dtype=float).ravel(),
        0.0,
    )
    if liters.size != 35040:
        raise ValueError(f"A DHW profil hossza {liters.size}, de 35040 kell.")

    T_in = float(hss.get("T_in", 10.0))
    T_out = float(hss.get("T_out", 40.0))
    return liters * C_HSS * max(T_out - T_in, 0.0) / float(dt)


def build_inputs(
    sim_yaml_path: os.PathLike,
    profiles_csv_path: os.PathLike,
    dhw_profile_path: os.PathLike,
    max_users: int = 10,
    search_roots: Iterable[os.PathLike] | None = None,
):
    """
    Közös bemenetolvasó.

    A fix UT-profil minden felhasználónál megmarad. A HSS/DHW paramétereket
    azért is beolvassa, hogy PV-s háztartásnál dinamikus bojlermodell
    válthassa fel a fix profilt.
    """
    sim_yaml_path = Path(sim_yaml_path)
    profiles_csv_path = Path(profiles_csv_path)
    dhw_profile_path = Path(dhw_profile_path)

    if search_roots is None:
        search_roots = [
            sim_yaml_path.parent / "Users_v2",
            sim_yaml_path.parent,
        ]

    sim = yaml.safe_load(sim_yaml_path.read_text(encoding="utf-8")) or {}
    users_list = list(sim.get("users_list", []))[: int(max_users)]

    exclude = {"battery", "bess", "community"}
    users_list = [
        user for user in users_list
        if str(user).strip().lower() not in exclude
    ]
    if not users_list:
        raise RuntimeError(
            "A simulation YAML nem tartalmaz users_list-et vagy max_users=0."
        )

    profiles = pd.read_csv(profiles_csv_path, index_col=0)
    profiles.columns = [str(column) for column in profiles.columns]
    profiles = profiles[
        [column for column in profiles.columns if column.lower() not in exclude]
    ]

    dhw = pd.read_csv(dhw_profile_path, index_col=0)
    dhw.columns = [str(column) for column in dhw.columns]

    e_pv_cols = []
    e_ue_cols = []
    e_el_heater_fixed_cols = []
    p_dhw_cols = []
    user_names = []

    size_bess = []
    eta_bess_in = []
    eta_bess_out = []
    eta_bess_stor = []
    soc_bess_min = []
    soc_bess_max = []
    t_bess_min = []

    size_elh = []
    vol_hss_water = []
    T_env = []
    T_max = []
    T_min = []
    T_in = []
    T_set = []
    a_hss = []
    eta_elh = []

    for user_key in users_list:
        yaml_path = _find_user_yaml(search_roots, str(user_key))
        if yaml_path is None:
            print(f"[WARN] YAML nem található: {user_key} — kihagyom.")
            continue

        user_yaml = yaml.safe_load(
            yaml_path.read_text(encoding="utf-8")
        ) or {}
        units = user_yaml.get("units") or {}

        user_names.append(
            str((units.get("name") or {}).get("name", user_key))
        )

        ue = units.get("ue") or {}
        pv = units.get("pv") or {}
        ut = units.get("ut") or {}
        hss = units.get("hss") or {}
        bess = units.get("bess") or {}

        ue_profile = ue.get("profile")
        pv_profile = pv.get("profile")
        ut_profile = ut.get("profile")

        e_ue_cols.append(
            _as_15min_energy(profiles[str(ue_profile)].to_numpy())
            if ue_profile is not None and str(ue_profile) in profiles.columns
            else np.zeros(35040, dtype=float)
        )
        e_pv_cols.append(
            _as_15min_energy(profiles[str(pv_profile)].to_numpy())
            if pv_profile is not None and str(pv_profile) in profiles.columns
            else np.zeros(35040, dtype=float)
        )
        e_el_heater_fixed_cols.append(
            _as_15min_energy(profiles[str(ut_profile)].to_numpy())
            if ut_profile is not None and str(ut_profile) in profiles.columns
            else np.zeros(35040, dtype=float)
        )
        p_dhw_cols.append(
            _dhw_liter_to_thermal_power_kw(dhw, hss, dt=DT)
        )

        size_bess.append(float(bess.get("bess_size", 0.0)))
        eta_bess_in.append(float(bess.get("eta_bess_in", 0.98)))
        eta_bess_out.append(float(bess.get("eta_bess_out", 0.96)))
        eta_bess_stor.append(float(bess.get("eta_bess_stor", 0.995)))
        soc_bess_min.append(float(bess.get("soc_bess_min", 0.10)))
        soc_bess_max.append(float(bess.get("soc_bess_max", 0.90)))
        t_bess_min.append(float(bess.get("t_bess_min", 2.0)))

        size_elh.append(float(hss.get("size_elh", 0.0)))
        vol_hss_water.append(float(hss.get("vol_hss_water", 0.0)))
        T_env.append(float(hss.get("T_env", 20.0)))
        T_max.append(float(hss.get("T_max", 65.0)))
        T_min.append(float(hss.get("T_min", 38.0)))
        T_in.append(float(hss.get("T_in", 10.0)))
        T_set.append(
            float(hss.get("T_set", hss.get("T_setpoint", 50.0)))
        )
        a_hss.append(float(hss.get("a_hss", 0.01275)))
        eta_elh.append(float(hss.get("eta_elh", 0.95)))

    if not user_names:
        raise RuntimeError("Nincs érvényes felhasználó.")

    return (
        np.column_stack(e_pv_cols).astype(float),
        np.column_stack(e_ue_cols).astype(float),
        np.column_stack(e_el_heater_fixed_cols).astype(float),
        np.column_stack(p_dhw_cols).astype(float),
        np.asarray(size_bess, dtype=float),
        np.asarray(eta_bess_in, dtype=float),
        np.asarray(eta_bess_out, dtype=float),
        np.asarray(eta_bess_stor, dtype=float),
        np.asarray(soc_bess_min, dtype=float),
        np.asarray(soc_bess_max, dtype=float),
        np.asarray(t_bess_min, dtype=float),
        np.asarray(size_elh, dtype=float),
        np.asarray(vol_hss_water, dtype=float),
        np.asarray(T_env, dtype=float),
        np.asarray(T_max, dtype=float),
        np.asarray(T_min, dtype=float),
        np.asarray(T_in, dtype=float),
        np.asarray(T_set, dtype=float),
        np.asarray(a_hss, dtype=float),
        np.asarray(eta_elh, dtype=float),
        user_names,
    )


def _b_tariff_available(step: int, dt: float = DT) -> bool:
    """B tarifa engedélyezése: 22–04 és 10–16."""
    hour = (step * dt) % 24.0
    return (
        hour < 4.0
        or 10.0 <= hour < 16.0
        or hour >= 22.0
    )


def simulate_rule_based_hss(
    p_dhw_kw: np.ndarray,
    *,
    size_elh_kw: float,
    vol_hss_water_l: float,
    T_env: float,
    T_max: float,
    T_min: float,
    T_in: float,
    T_set: float,
    a_hss_kw_per_k: float,
    eta_elh: float,
    boiler_tariff: str,
    dt: float = DT,
) -> dict:
    """Termosztatikus, nem optimalizált HSS-modell."""
    p_dhw_kw = np.maximum(
        np.asarray(p_dhw_kw, dtype=float).ravel(),
        0.0,
    )
    T = len(p_dhw_kw)

    tariff = str(boiler_tariff).upper().strip()
    if tariff not in {"A", "B"}:
        raise ValueError("boiler_tariff csak 'A' vagy 'B' lehet.")

    result = {
        "e_boiler": np.zeros(T, dtype=float),
        "p_elh_in": np.zeros(T, dtype=float),
        "p_hss_in": np.zeros(T, dtype=float),
        "p_hss_out": p_dhw_kw.copy(),
        "t_hss": np.zeros(T, dtype=float),
        "d_boiler_available": np.zeros(T, dtype=float),
        "d_boiler_on": np.zeros(T, dtype=float),
        "temperature_violation": np.zeros(T, dtype=float),
    }

    if size_elh_kw <= 1e-9 or vol_hss_water_l <= 1e-9:
        return result

    thermal_capacity = vol_hss_water_l * C_HSS
    temperature = float(np.clip(T_set, T_min, T_max))

    for t in range(T):
        available = tariff == "A" or _b_tariff_available(t, dt)
        result["d_boiler_available"][t] = float(available)
        result["t_hss"][t] = temperature

        required_thermal_power = (
            p_dhw_kw[t]
            + a_hss_kw_per_k * (temperature - T_env)
            + thermal_capacity * max(T_set - temperature, 0.0) / dt
        )

        if available:
            p_elh = min(
                size_elh_kw,
                max(required_thermal_power, 0.0) / max(eta_elh, 1e-9),
            )
        else:
            p_elh = 0.0

        result["p_elh_in"][t] = p_elh
        result["p_hss_in"][t] = eta_elh * p_elh
        result["e_boiler"][t] = p_elh * dt
        result["d_boiler_on"][t] = float(p_elh > 1e-9)

        next_temperature = temperature + dt * (
            eta_elh * p_elh
            - p_dhw_kw[t]
            - a_hss_kw_per_k * (temperature - T_env)
        ) / thermal_capacity

        if (
            next_temperature < T_min - 1e-6
            or next_temperature > T_max + 1e-6
        ):
            result["temperature_violation"][t] = 1.0

        temperature = float(
            np.clip(next_temperature, T_in, T_max)
        )

    return result


def build_effective_boiler_profiles(
    *,
    e_pv: np.ndarray,
    e_el_heater_fixed: np.ndarray,
    p_dhw_kw: np.ndarray,
    size_elh: np.ndarray,
    vol_hss_water: np.ndarray,
    T_env: np.ndarray,
    T_max: np.ndarray,
    T_min: np.ndarray,
    T_in: np.ndarray,
    T_set: np.ndarray,
    a_hss: np.ndarray,
    eta_elh: np.ndarray,
    boiler_tariff: str,
    use_boiler_model: bool = True,
) -> dict:
    """
    PV-s és megfelelő HSS-adattal rendelkező háztartásnál dinamikus modellt
    használ. PV nélkül, illetve hiányos HSS-adatnál a fix UT-profil marad.
    """
    T, U = e_pv.shape
    e_boiler = np.asarray(e_el_heater_fixed, dtype=float).copy()

    dynamic_boiler = np.zeros(U, dtype=bool)
    t_hss = np.zeros((T, U), dtype=float)
    p_elh_in = np.zeros((T, U), dtype=float)
    p_hss_in = np.zeros((T, U), dtype=float)
    p_hss_out = np.zeros((T, U), dtype=float)
    d_boiler_available = np.zeros((T, U), dtype=float)
    d_boiler_on = np.zeros((T, U), dtype=float)
    temperature_violation = np.zeros((T, U), dtype=float)

    has_pv = np.asarray(e_pv, dtype=float).sum(axis=0) > 1e-12

    for u in range(U):
        valid_hss = (
            float(size_elh[u]) > 1e-9
            and float(vol_hss_water[u]) > 1e-9
            and float(np.sum(p_dhw_kw[:, u])) > 1e-9
        )

        if not (bool(use_boiler_model) and has_pv[u] and valid_hss):
            continue

        model = simulate_rule_based_hss(
            p_dhw_kw[:, u],
            size_elh_kw=float(size_elh[u]),
            vol_hss_water_l=float(vol_hss_water[u]),
            T_env=float(T_env[u]),
            T_max=float(T_max[u]),
            T_min=float(T_min[u]),
            T_in=float(T_in[u]),
            T_set=float(T_set[u]),
            a_hss_kw_per_k=float(a_hss[u]),
            eta_elh=float(eta_elh[u]),
            boiler_tariff=boiler_tariff,
            dt=DT,
        )

        e_boiler[:, u] = model["e_boiler"]
        t_hss[:, u] = model["t_hss"]
        p_elh_in[:, u] = model["p_elh_in"]
        p_hss_in[:, u] = model["p_hss_in"]
        p_hss_out[:, u] = model["p_hss_out"]
        d_boiler_available[:, u] = model["d_boiler_available"]
        d_boiler_on[:, u] = model["d_boiler_on"]
        temperature_violation[:, u] = model["temperature_violation"]
        dynamic_boiler[u] = True

    return {
        "e_boiler": e_boiler,
        "dynamic_boiler": dynamic_boiler,
        "t_hss": t_hss,
        "p_dhw_kw": p_dhw_kw,
        "p_elh_in_kw": p_elh_in,
        "p_hss_in_kw": p_hss_in,
        "p_hss_out_kw": p_hss_out,
        "d_boiler_available": d_boiler_available,
        "d_boiler_on": d_boiler_on,
        "temperature_violation": temperature_violation,
    }


SHARED_BUYER_LOW_LIMIT_KWH = 2523.0
SHARED_BUYER_LOW_FT_PER_KWH = 5.0
SHARED_BUYER_HIGH_FT_PER_KWH = 21.0
SHARED_RHD_FT_PER_KWH = 31.0

EPS = 1e-12

SharingMode = Literal["proportional", "equal"]
BoilerTariff = Literal["A", "B"]

# B tarifa a bojler hálózati importjára.
# Az A tarifa értékei a nonopt_common importból jönnek.
B_LOW_TARIFF_LIMIT_KWH = 2523.0
B_LOW_TARIFF_FT_PER_KWH = 23.0
B_HIGH_TARIFF_FT_PER_KWH = 61.0


def _as_nonnegative_2d(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} csak 2D tömb lehet, kapott dimenzió: {arr.ndim}")
    return np.maximum(arr, 0.0)


def _two_tier_cost_steps(
    e_steps: np.ndarray,
    low_limit_kwh: float,
    low_rate_ft_per_kwh: float,
    high_rate_ft_per_kwh: float,
) -> tuple[float, float, float]:
    """Éves, időrendben haladó kétlépcsős díjszámítás."""
    remaining_low = float(low_limit_kwh)
    low_kwh = 0.0
    high_kwh = 0.0
    cost_ft = 0.0

    for e in np.maximum(np.asarray(e_steps, dtype=float).ravel(), 0.0):
        e = float(e)
        low_part = min(e, max(remaining_low, 0.0))
        high_part = max(e - low_part, 0.0)

        low_kwh += low_part
        high_kwh += high_part
        cost_ft += low_part * float(low_rate_ft_per_kwh)
        cost_ft += high_part * float(high_rate_ft_per_kwh)
        remaining_low -= low_part

    return low_kwh, high_kwh, cost_ft


def _split_two_streams_by_shared_low_cap(
    e_grid_a_step: float,
    e_shared_in_step: float,
    remaining_low_kwh: float,
) -> tuple[float, float, float, float, float]:
    """
    Egy vevő adott időlépésbeli A-tarifás hálózati importját és
    megosztott energiavásárlását ugyanarra a 2523 kWh/év-es kedvezményes
    vevői keretre könyveli.

    Visszatér:
        grid_a_low, grid_a_high, shared_low, shared_high, new_remaining_low

    Sorrend: először a közösségből vásárolt energia fogyasztja a még
    rendelkezésre álló kedvezményes vevői sávot, és csak a maradék
    kedvezményes keret jut az A-tarifás hálózati importra.
    """
    e_grid = max(float(e_grid_a_step), 0.0)
    e_shared = max(float(e_shared_in_step), 0.0)
    remaining = max(float(remaining_low_kwh), 0.0)

    if e_grid + e_shared <= EPS:
        return 0.0, 0.0, 0.0, 0.0, remaining

    shared_low = min(e_shared, remaining)
    shared_high = max(e_shared - shared_low, 0.0)
    remaining_after_shared = remaining - shared_low

    grid_low = min(e_grid, remaining_after_shared)
    grid_high = max(e_grid - grid_low, 0.0)
    new_remaining = remaining_after_shared - grid_low

    return grid_low, grid_high, shared_low, shared_high, new_remaining


def settle_shared_payments_buyer_tiered(
    e_shared_in: np.ndarray,
    e_shared_out: np.ndarray,
    e_grid_import_a: np.ndarray,
) -> dict:
    """
    Megosztási elszámolás vevői 2523 kWh-os sáv alapján.

    A vevőnél ugyanaz a kedvezményes éves keret fogy az A-tarifás hálózati
    importból és a közösségből vásárolt energiából. A megosztott energia
    energiadíja vevői sáv szerint 5 vagy 21 Ft/kWh, ezt kapja meg a PV-s eladó.
    Az RHD mindig 31 Ft/kWh a teljes shared_in mennyiségre.
    """
    e_shared_in = _as_nonnegative_2d(e_shared_in, "e_shared_in")
    e_shared_out = _as_nonnegative_2d(e_shared_out, "e_shared_out")
    e_grid_import_a = _as_nonnegative_2d(e_grid_import_a, "e_grid_import_a")

    if e_shared_in.shape != e_shared_out.shape or e_shared_in.shape != e_grid_import_a.shape:
        raise ValueError("e_shared_in, e_shared_out és e_grid_import_a alakja nem egyezik")

    T, U = e_shared_in.shape
    pair_kwh = np.zeros((U, U), dtype=float)
    pair_energy_payment_ft = np.zeros((U, U), dtype=float)

    buyer_low_remaining = np.full(U, SHARED_BUYER_LOW_LIMIT_KWH, dtype=float)
    grid_a_low_kwh = np.zeros(U, dtype=float)
    grid_a_high_kwh = np.zeros(U, dtype=float)
    shared_buyer_low_kwh = np.zeros(U, dtype=float)
    shared_buyer_high_kwh = np.zeros(U, dtype=float)
    buyer_energy_cost_ft = np.zeros(U, dtype=float)
    buyer_rhd_ft = e_shared_in.sum(axis=0) * SHARED_RHD_FT_PER_KWH
    seller_revenue_ft = np.zeros(U, dtype=float)

    for t in range(T):
        shared_low_t = np.zeros(U, dtype=float)
        shared_high_t = np.zeros(U, dtype=float)

        for b in range(U):
            g_low, g_high, s_low, s_high, new_remaining = _split_two_streams_by_shared_low_cap(
                e_grid_a_step=float(e_grid_import_a[t, b]),
                e_shared_in_step=float(e_shared_in[t, b]),
                remaining_low_kwh=float(buyer_low_remaining[b]),
            )
            buyer_low_remaining[b] = new_remaining
            grid_a_low_kwh[b] += g_low
            grid_a_high_kwh[b] += g_high
            shared_buyer_low_kwh[b] += s_low
            shared_buyer_high_kwh[b] += s_high
            shared_low_t[b] = s_low
            shared_high_t[b] = s_high

        shared_energy_cost_t = (
            shared_low_t * SHARED_BUYER_LOW_FT_PER_KWH
            + shared_high_t * SHARED_BUYER_HIGH_FT_PER_KWH
        )
        buyer_energy_cost_ft += shared_energy_cost_t

        sin = e_shared_in[t, :]
        sout = e_shared_out[t, :]
        total_share = min(float(sin.sum()), float(sout.sum()))
        if total_share <= EPS:
            continue

        sellers = np.flatnonzero(sout > EPS)
        buyers = np.flatnonzero(sin > EPS)
        if len(sellers) == 0 or len(buyers) == 0:
            continue

        buyer_avg_rates = np.divide(
            shared_energy_cost_t,
            sin,
            out=np.zeros(U, dtype=float),
            where=sin > EPS,
        )

        # Virtuális párosítás: az eladók a shared_out arányában, a vevők
        # a kiosztott shared_in szerint kapcsolódnak össze. Az egységárat
        # mindig a vevő aktuális éves sávja adja.
        buyer_weights = sin[buyers] / total_share
        pair_kwh_t = np.outer(sout[sellers], buyer_weights)
        pair_payment_t = pair_kwh_t * buyer_avg_rates[buyers][None, :]

        idx = np.ix_(sellers, buyers)
        pair_kwh[idx] += pair_kwh_t
        pair_energy_payment_ft[idx] += pair_payment_t
        seller_revenue_ft[sellers] += pair_payment_t.sum(axis=1)

    return {
        "pair_kwh": pair_kwh,
        "pair_energy_payment_ft": pair_energy_payment_ft,
        "grid_a_low_kwh": grid_a_low_kwh,
        "grid_a_high_kwh": grid_a_high_kwh,
        "shared_buyer_low_kwh": shared_buyer_low_kwh,
        "shared_buyer_high_kwh": shared_buyer_high_kwh,
        "buyer_energy_cost_ft": buyer_energy_cost_ft,
        "buyer_rhd_ft": buyer_rhd_ft,
        "buyer_total_shared_cost_ft": buyer_energy_cost_ft + buyer_rhd_ft,
        "seller_revenue_ft": seller_revenue_ft,
    }

def allocate_shared_in(deficit: np.ndarray, total_share_kwh: float, mode: SharingMode) -> np.ndarray:
    """
    Megosztott energia vevői oldali kiosztása.

    proportional:
        a hiány arányában osztja ki.

    equal:
        azonos kWh-kvótát próbál adni minden aktív hiányos háztartásnak;
        aki ennél kevesebbet tud felvenni, annál a maradék újraosztódik.
    """
    d = np.maximum(np.asarray(deficit, dtype=float), 0.0)
    total_share = min(max(float(total_share_kwh), 0.0), float(d.sum()))
    out = np.zeros_like(d)

    if total_share <= EPS or d.sum() <= EPS:
        return out

    if mode == "proportional":
        return d * (total_share / d.sum())

    if mode != "equal":
        raise ValueError("sharing_mode csak 'proportional' vagy 'equal' lehet")

    remaining = total_share
    while remaining > EPS:
        room = d - out
        active = room > EPS
        n_active = int(active.sum())
        if n_active == 0:
            break

        quota = remaining / n_active
        take = np.minimum(room[active], quota)
        out[active] += take
        remaining -= float(take.sum())

    return out


def _select_bess_users(
    has_pv_arr: np.ndarray,
    size_bess: np.ndarray,
    include_bess: bool,
    bess_share_pct: float,
) -> np.ndarray:
    """Deterministikus BESS-kiosztás: az első kiválasztott háztartások kapják."""
    U = len(has_pv_arr)
    enabled = np.zeros(U, dtype=bool)
    if not include_bess:
        return enabled

    candidates = np.where(has_pv_arr)[0]
    candidates = np.array(
        [u for u in candidates if float(size_bess[u]) > EPS],
        dtype=int,
    )

    n = int(round(len(candidates) * float(bess_share_pct) / 100.0))
    n = max(0, min(n, len(candidates)))
    enabled[candidates[:n]] = True
    return enabled


def simulate_nonopt_disaggregated_shared(
    e_pv: np.ndarray,
    e_ue: np.ndarray,
    e_boiler: np.ndarray | None,
    size_bess: np.ndarray,
    eta_bess_in: np.ndarray,
    eta_bess_out: np.ndarray,
    eta_bess_stor: np.ndarray,
    soc_bess_min: np.ndarray,
    soc_bess_max: np.ndarray,
    t_bess_min_h: np.ndarray,
    user_names: list[str],
    *,
    include_bess: bool = True,
    bess_share_pct: float = 100.0,
    sharing_mode: SharingMode = "proportional",
    boiler_tariff: BoilerTariff = "B",
    initial_soc_fraction: float = 0.5,
    allow_grid_charge_for_min_soc: bool = True,
) -> dict:
    """
    Disaggregált, prioritásos BESS + energiaközösségi megosztás.

    Minden idősor kWh / 15 perc egységű.
    """
    e_pv = _as_nonnegative_2d(e_pv, "e_pv")
    e_ue = _as_nonnegative_2d(e_ue, "e_ue")
    T, U = e_ue.shape

    if e_pv.shape != (T, U):
        raise ValueError(f"e_pv alakja {e_pv.shape}, de {(T, U)} kell")

    if e_boiler is None:
        e_boiler = np.zeros_like(e_ue)
    else:
        e_boiler = _as_nonnegative_2d(e_boiler, "e_boiler")
        if e_boiler.shape != (T, U):
            raise ValueError(f"e_boiler alakja {e_boiler.shape}, de {(T, U)} kell")

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    user_names = list(user_names)
    if len(user_names) != U:
        user_names = [f"user_{u + 1:03d}" for u in range(U)]

    size_bess = np.asarray(size_bess, dtype=float).ravel()
    eta_bess_in = np.asarray(eta_bess_in, dtype=float).ravel()
    eta_bess_out = np.asarray(eta_bess_out, dtype=float).ravel()
    eta_bess_stor = np.asarray(eta_bess_stor, dtype=float).ravel()
    soc_bess_min = np.asarray(soc_bess_min, dtype=float).ravel()
    soc_bess_max = np.asarray(soc_bess_max, dtype=float).ravel()
    t_bess_min_h = np.asarray(t_bess_min_h, dtype=float).ravel()

    if any(len(x) != U for x in [size_bess, eta_bess_in, eta_bess_out, eta_bess_stor, soc_bess_min, soc_bess_max, t_bess_min_h]):
        raise ValueError("A BESS paramétervektorok hossza nem egyezik a felhasználók számával")

    # Teljes fogyasztás statisztikához/SSI-hez.
    # A saját PV/BESS dispatch-load ettől eltérhet B tarifa esetén.
    e_load = e_ue + e_boiler
    has_pv_arr = e_pv.sum(axis=0) > EPS
    bess_enabled = _select_bess_users(
        has_pv_arr=has_pv_arr,
        size_bess=size_bess,
        include_bess=include_bess,
        bess_share_pct=bess_share_pct,
    )

    # --- Idősorok ---
    ts_e_pv_to_load = np.zeros((T, U), dtype=float)
    ts_e_pv_to_bess = np.zeros((T, U), dtype=float)
    ts_e_bess_to_load = np.zeros((T, U), dtype=float)
    ts_e_bess_local_to_load = np.zeros((T, U), dtype=float)
    ts_e_grid_to_bess = np.zeros((T, U), dtype=float)

    # Komponensenkénti követés.
    # B tarifa esetén a saját PV/BESS és a közösségi megosztás is csak
    # az alapfogyasztást látja; a bojler kizárólag B tarifás hálózatból vételez.
    ts_e_local_to_base = np.zeros((T, U), dtype=float)
    ts_e_local_to_boiler = np.zeros((T, U), dtype=float)
    ts_e_base_deficit_before_share = np.zeros((T, U), dtype=float)
    ts_e_boiler_deficit_before_share = np.zeros((T, U), dtype=float)
    ts_e_shared_to_base = np.zeros((T, U), dtype=float)
    ts_e_shared_to_boiler = np.zeros((T, U), dtype=float)
    ts_e_grid_to_base = np.zeros((T, U), dtype=float)
    ts_e_grid_to_boiler = np.zeros((T, U), dtype=float)

    ts_e_shared_in = np.zeros((T, U), dtype=float)
    ts_e_shared_out = np.zeros((T, U), dtype=float)
    ts_e_grid_to_load = np.zeros((T, U), dtype=float)
    ts_e_grid_export = np.zeros((T, U), dtype=float)
    ts_e_bess = np.zeros((T, U), dtype=float)
    ts_d_bess_ch = np.zeros((T, U), dtype=float)
    ts_d_bess_dis = np.zeros((T, U), dtype=float)

    # Megosztás előtti maradék pozíciók
    ts_e_deficit_before_share = np.zeros((T, U), dtype=float)
    ts_e_surplus_before_share = np.zeros((T, U), dtype=float)

    # Éves elszámolási mátrixok: seller x buyer
    shared_pair_kwh = np.zeros((U, U), dtype=float)
    shared_pair_energy_payment_ft = np.zeros((U, U), dtype=float)

    # A megosztási pénzügyi elszámolás az időlépéses fizikai allokáció után,
    # a vevői 2523 kWh-os éves sáv figyelembevételével készül.

    # BESS állapotok.
    soc_min_kwh = np.maximum(soc_bess_min, 0.0) * np.maximum(size_bess, 0.0)
    soc_max_kwh = np.maximum(soc_min_kwh, soc_bess_max * np.maximum(size_bess, 0.0))
    soc = np.clip(float(initial_soc_fraction) * np.maximum(size_bess, 0.0), soc_min_kwh, soc_max_kwh)

    # Eredetkövetés SSI-hez: a gridből töltött BESS-kisütést ne számítsuk helyi ellátásnak.
    soc_local = soc.copy()
    soc_grid = np.zeros(U, dtype=float)

    # BESS teljes töltési/kisütési időből számolt maximális 15 perces energia.
    e_bess_max_step = np.full(U, np.inf, dtype=float)
    valid_t = np.asarray(t_bess_min_h, dtype=float) > EPS
    e_bess_max_step[valid_t] = np.maximum(size_bess[valid_t], 0.0) / t_bess_min_h[valid_t] * DT

    for t in range(T):
        # --- 1) Háztartásonként saját PV + saját BESS prioritás ---
        for u in range(U):
            use_bess_u = bool(bess_enabled[u] and size_bess[u] > EPS)
            base_load_t = float(e_ue[t, u])
            boiler_load_t = float(e_boiler[t, u])

            if boiler_tariff == "A":
                # A tarifás bojler ugyanazon a körön van: saját PV/BESS is kiszolgálhatja.
                dispatch_load_t = base_load_t + boiler_load_t
            else:
                # B tarifás bojler külön körön van: saját PV/BESS nem szolgálhatja ki.
                dispatch_load_t = base_load_t

            load_t = dispatch_load_t
            pv_t = float(e_pv[t, u])

            if use_bess_u:
                # Önkisülés.
                eta_stor = float(eta_bess_stor[u])
                soc[u] *= eta_stor
                soc_local[u] *= eta_stor
                soc_grid[u] *= eta_stor

                # Minimum SOC-védelem: külső hálózati töltés megengedett.
                if allow_grid_charge_for_min_soc and soc[u] < soc_min_kwh[u] - EPS:
                    need_soc = soc_min_kwh[u] - soc[u]
                    grid_charge = need_soc / max(float(eta_bess_in[u]), EPS)
                    soc[u] += grid_charge * float(eta_bess_in[u])
                    soc_grid[u] += grid_charge * float(eta_bess_in[u])
                    ts_e_grid_to_bess[t, u] += grid_charge
                    ts_d_bess_ch[t, u] = 1.0

            # Saját PV közvetlenül saját fogyasztásra.
            pv_to_load = min(pv_t, load_t)
            remaining_load = max(load_t - pv_to_load, 0.0)
            remaining_pv = max(pv_t - pv_to_load, 0.0)

            ts_e_pv_to_load[t, u] = pv_to_load

            charge = 0.0
            discharge = 0.0
            discharge_local_output = 0.0

            if use_bess_u and remaining_pv > EPS:
                room_kwh = max(soc_max_kwh[u] - soc[u], 0.0)
                charge_by_soc = room_kwh / max(float(eta_bess_in[u]), EPS)
                charge = min(remaining_pv, e_bess_max_step[u], charge_by_soc)
                charge = max(charge, 0.0)

                soc_add = charge * float(eta_bess_in[u])
                soc[u] += soc_add
                soc_local[u] += soc_add
                remaining_pv -= charge

                ts_e_pv_to_bess[t, u] = charge
                if charge > EPS:
                    ts_d_bess_ch[t, u] = 1.0

            elif use_bess_u and remaining_load > EPS:
                available_soc = max(soc[u] - soc_min_kwh[u], 0.0)
                discharge_by_soc = available_soc * max(float(eta_bess_out[u]), EPS)
                discharge = min(remaining_load, e_bess_max_step[u], discharge_by_soc)
                discharge = max(discharge, 0.0)

                if discharge > EPS:
                    soc_need = discharge / max(float(eta_bess_out[u]), EPS)

                    # Eredetkövetés: először arányosan vesszük ki a local/grid SOC-ból.
                    total_origin = soc_local[u] + soc_grid[u]
                    if total_origin > EPS:
                        local_soc_part = min(soc_local[u], soc_need * (soc_local[u] / total_origin))
                    else:
                        local_soc_part = 0.0
                    grid_soc_part = min(soc_grid[u], soc_need - local_soc_part)

                    # Kerekítési korrekció: ha maradt hiány, először localból, aztán gridből.
                    rem_soc = soc_need - local_soc_part - grid_soc_part
                    if rem_soc > EPS:
                        add_local = min(max(soc_local[u] - local_soc_part, 0.0), rem_soc)
                        local_soc_part += add_local
                        rem_soc -= add_local
                    if rem_soc > EPS:
                        add_grid = min(max(soc_grid[u] - grid_soc_part, 0.0), rem_soc)
                        grid_soc_part += add_grid
                        rem_soc -= add_grid

                    soc_local[u] = max(soc_local[u] - local_soc_part, 0.0)
                    soc_grid[u] = max(soc_grid[u] - grid_soc_part, 0.0)
                    soc[u] = max(soc[u] - soc_need, 0.0)

                    discharge_local_output = local_soc_part * max(float(eta_bess_out[u]), EPS)
                    remaining_load -= discharge
                    ts_d_bess_dis[t, u] = 1.0

                ts_e_bess_to_load[t, u] = discharge
                ts_e_bess_local_to_load[t, u] = discharge_local_output

            # Saját PV/BESS által ellátott energia komponensbontása.
            # A bontás elszámolási célú; a fizikai prioritás: alapfogyasztás előbb, bojler utána.
            local_supply = pv_to_load + discharge
            local_to_base = min(base_load_t, local_supply)
            local_to_boiler = 0.0
            if boiler_tariff == "A":
                local_to_boiler = min(boiler_load_t, max(local_supply - local_to_base, 0.0))

            base_deficit = max(base_load_t - local_to_base, 0.0)
            if boiler_tariff == "A":
                boiler_deficit = max(boiler_load_t - local_to_boiler, 0.0)
            else:
                # B tarifás bojler: külön körön maradó teljes bojlerigény;
                # megosztásra nem jogosult, csak B tarifás hálózat látja el.
                boiler_deficit = boiler_load_t

            ts_e_local_to_base[t, u] = local_to_base
            ts_e_local_to_boiler[t, u] = local_to_boiler
            ts_e_base_deficit_before_share[t, u] = base_deficit
            ts_e_boiler_deficit_before_share[t, u] = boiler_deficit

            # BESS után még megmaradó pozíciók. BESS nem exportál közösségbe,
            # csak a saját PV-felesleg mehet megosztásba.
            ts_e_surplus_before_share[t, u] = max(remaining_pv, 0.0)

            # Megosztásra jogosult hiány:
            # - A tarifán: alapfogyasztás + bojler;
            # - B tarifán: kizárólag az A-körös alapfogyasztás.
            # A B tarifás bojler külön elszámolási kör, ezért közösségi
            # megosztott energia sem fizikailag, sem elszámolásban nem jut rá.
            if boiler_tariff == "A":
                share_eligible_deficit = base_deficit + boiler_deficit
            else:
                share_eligible_deficit = base_deficit

            ts_e_deficit_before_share[t, u] = max(
                share_eligible_deficit,
                0.0,
            )

            if use_bess_u:
                # Az eredetkövetés és SOC összegének kerekítési rendezése.
                origin_sum = soc_local[u] + soc_grid[u]
                if origin_sum > EPS and abs(origin_sum - soc[u]) > 1e-9:
                    scale = soc[u] / origin_sum
                    soc_local[u] *= scale
                    soc_grid[u] *= scale
                elif soc[u] <= EPS:
                    soc_local[u] = 0.0
                    soc_grid[u] = 0.0

                ts_e_bess[t, u] = soc[u]

        # --- 2) Közösségi megosztás ---
        surplus = ts_e_surplus_before_share[t, :]
        deficit = ts_e_deficit_before_share[t, :]
        S = float(surplus.sum())
        D = float(deficit.sum())
        total_share = min(S, D)

        if total_share > EPS:
            shared_in = allocate_shared_in(deficit, total_share, sharing_mode)
            shared_out = surplus * (total_share / S) if S > EPS else np.zeros(U, dtype=float)
        else:
            shared_in = np.zeros(U, dtype=float)
            shared_out = np.zeros(U, dtype=float)

        ts_e_shared_in[t, :] = shared_in
        ts_e_shared_out[t, :] = shared_out

        # A kapott közösségi energia komponensbontása.
        base_def = ts_e_base_deficit_before_share[t, :]
        boiler_def = ts_e_boiler_deficit_before_share[t, :]

        if boiler_tariff == "B":
            # B tarifa: a megosztott energia kizárólag az alapfogyasztásra
            # számolható el. A bojler teljes igénye a B tarifás hálózatra kerül.
            shared_to_base = np.minimum(base_def, shared_in)
            shared_to_boiler = np.zeros(U, dtype=float)
        else:
            # A tarifa: a közös kör teljes hiánya jogosult megosztásra.
            # A háztartáson belül először az alapfogyasztás, majd a bojler kap.
            shared_to_base = np.minimum(base_def, shared_in)
            shared_to_boiler = np.minimum(
                boiler_def,
                np.maximum(shared_in - shared_to_base, 0.0),
            )

        ts_e_shared_to_base[t, :] = shared_to_base
        ts_e_shared_to_boiler[t, :] = shared_to_boiler
        ts_e_grid_to_base[t, :] = np.maximum(
            base_def - shared_to_base,
            0.0,
        )
        ts_e_grid_to_boiler[t, :] = np.maximum(
            boiler_def - shared_to_boiler,
            0.0,
        )

        ts_e_grid_to_load[t, :] = ts_e_grid_to_base[t, :] + ts_e_grid_to_boiler[t, :]
        ts_e_grid_export[t, :] = np.maximum(surplus - shared_out, 0.0)

        # A pénzügyi megosztási elszámolás később készül, mert a vevői
        # 2523 kWh-os sávot az A-tarifás hálózati importtal együtt kell vezetni.

    # Fizikai/elszámolási ellenőrzések.
    if boiler_tariff == "B":
        if float(np.max(np.abs(ts_e_local_to_boiler))) > 1e-9:
            raise RuntimeError(
                "B tarifán saját PV/BESS energia jutott a bojlerre."
            )
        if float(np.max(np.abs(ts_e_shared_to_boiler))) > 1e-9:
            raise RuntimeError(
                "B tarifán megosztott energia jutott a bojlerre."
            )
        if not np.allclose(
            ts_e_grid_to_boiler,
            e_boiler,
            atol=1e-9,
        ):
            raise RuntimeError(
                "B tarifán a bojlerigény nem teljes egészében "
                "B tarifás hálózati importként jelent meg."
            )

    if not np.allclose(
        ts_e_shared_in.sum(axis=1),
        ts_e_shared_out.sum(axis=1),
        atol=1e-9,
    ):
        raise RuntimeError(
            "A megosztott be- és kimenő energia időlépésenként nem egyezik."
        )

    # --- Éves per-user elszámolás ---
    rows = []

    # A tarifás hálózati import: alapfogyasztás hálózati része + gridből történő BESS min-SOC töltés.
    # Ha a bojler A tarifás, akkor a bojler hálózati része is A tarifa.
    # Ha a bojler B tarifás, akkor a bojler hálózati része külön B tarifa.
    if boiler_tariff == "A":
        grid_import_a = ts_e_grid_to_base + ts_e_grid_to_boiler + ts_e_grid_to_bess
        grid_import_b = np.zeros_like(grid_import_a)
    else:
        grid_import_a = ts_e_grid_to_base + ts_e_grid_to_bess
        grid_import_b = ts_e_grid_to_boiler

    grid_import_total = grid_import_a + grid_import_b
    grid_export_revenue_ft = ts_e_grid_export.sum(axis=0) * EXPORT_FT_PER_KWH

    settlement = settle_shared_payments_buyer_tiered(
        e_shared_in=ts_e_shared_in,
        e_shared_out=ts_e_shared_out,
        e_grid_import_a=grid_import_a,
    )
    shared_pair_kwh = settlement["pair_kwh"]
    shared_pair_energy_payment_ft = settlement["pair_energy_payment_ft"]
    grid_import_a_low_kwh_arr = settlement["grid_a_low_kwh"]
    grid_import_a_high_kwh_arr = settlement["grid_a_high_kwh"]
    shared_buyer_low_kwh = settlement["shared_buyer_low_kwh"]
    shared_buyer_high_kwh = settlement["shared_buyer_high_kwh"]
    shared_purchase_energy_cost_ft = settlement["buyer_energy_cost_ft"]
    shared_purchase_rhd_ft = settlement["buyer_rhd_ft"]
    shared_revenue_ft = settlement["seller_revenue_ft"]

    for u in range(U):
        grid_a_low_kwh = float(grid_import_a_low_kwh_arr[u])
        grid_a_high_kwh = float(grid_import_a_high_kwh_arr[u])
        grid_import_a_cost_ft = (
            grid_a_low_kwh * LOW_TARIFF_FT_PER_KWH
            + grid_a_high_kwh * HIGH_TARIFF_FT_PER_KWH
        )
        grid_b_low_kwh, grid_b_high_kwh, grid_import_b_cost_ft = _two_tier_cost_steps(
            grid_import_b[:, u],
            low_limit_kwh=B_LOW_TARIFF_LIMIT_KWH,
            low_rate_ft_per_kwh=B_LOW_TARIFF_FT_PER_KWH,
            high_rate_ft_per_kwh=B_HIGH_TARIFF_FT_PER_KWH,
        )
        grid_import_cost_ft = grid_import_a_cost_ft + grid_import_b_cost_ft

        shared_purchase_cost_ft = shared_purchase_energy_cost_ft[u] + shared_purchase_rhd_ft[u]
        total_cost_ft = grid_import_cost_ft + shared_purchase_cost_ft
        total_revenue_ft = grid_export_revenue_ft[u] + shared_revenue_ft[u]
        brt_bill_ft = total_cost_ft - total_revenue_ft

        pv_kwh = float(e_pv[:, u].sum())
        load_kwh = float(e_load[:, u].sum())
        self_consumed_pv_kwh = float(ts_e_pv_to_load[:, u].sum() + ts_e_pv_to_bess[:, u].sum() + ts_e_shared_out[:, u].sum())
        locally_supplied_load_kwh = float(ts_e_pv_to_load[:, u].sum() + ts_e_bess_local_to_load[:, u].sum() + ts_e_shared_in[:, u].sum())

        sci = self_consumed_pv_kwh / pv_kwh if pv_kwh > EPS else 0.0
        ssi = locally_supplied_load_kwh / load_kwh if load_kwh > EPS else 0.0
        sci = float(np.clip(sci, 0.0, 1.0))
        ssi = float(np.clip(ssi, 0.0, 1.0))

        rows.append(
            {
                "user_name": user_names[u],
                "has_pv": bool(has_pv_arr[u]),
                "has_bess": bool(bess_enabled[u] and size_bess[u] > EPS),
                "has_heat_pump": False,
                "has_boiler": bool(e_boiler[:, u].sum() > EPS),
                "group_label": _group_label(
                    has_pv=bool(has_pv_arr[u]),
                    has_bess=bool(bess_enabled[u] and size_bess[u] > EPS),
                    has_heat_pump=False,
                    has_boiler=bool(e_boiler[:, u].sum() > EPS),
                ),
                "base_load_kwh": float(e_ue[:, u].sum()),
                "boiler_kwh": float(e_boiler[:, u].sum()),
                "heat_pump_kwh": 0.0,
                "total_load_kwh": load_kwh,
                "pv_kwh": pv_kwh,
                "Pmax_kw": float(np.max(e_load[:, u] / DT)) if T else 0.0,
                "pv_to_load_kwh": float(ts_e_pv_to_load[:, u].sum()),
                "pv_to_bess_kwh": float(ts_e_pv_to_bess[:, u].sum()),
                "bess_to_load_kwh": float(ts_e_bess_to_load[:, u].sum()),
                "bess_local_to_load_kwh": float(ts_e_bess_local_to_load[:, u].sum()),
                "grid_to_bess_kwh": float(ts_e_grid_to_bess[:, u].sum()),
                "local_to_base_kwh": float(ts_e_local_to_base[:, u].sum()),
                "local_to_boiler_kwh": float(ts_e_local_to_boiler[:, u].sum()),
                "base_deficit_before_share_kwh": float(ts_e_base_deficit_before_share[:, u].sum()),
                "boiler_deficit_before_share_kwh": float(ts_e_boiler_deficit_before_share[:, u].sum()),
                "shared_in_kwh": float(ts_e_shared_in[:, u].sum()),
                "shared_to_base_kwh": float(ts_e_shared_to_base[:, u].sum()),
                "shared_to_boiler_kwh": float(ts_e_shared_to_boiler[:, u].sum()),
                "shared_out_kwh": float(ts_e_shared_out[:, u].sum()),
                "grid_import_to_load_kwh": float(ts_e_grid_to_load[:, u].sum()),
                "grid_to_base_kwh": float(ts_e_grid_to_base[:, u].sum()),
                "grid_to_boiler_kwh": float(ts_e_grid_to_boiler[:, u].sum()),
                "grid_import_kwh": float(grid_import_total[:, u].sum()),
                "grid_import_a_kwh": float(grid_import_a[:, u].sum()),
                "grid_import_b_kwh": float(grid_import_b[:, u].sum()),
                "grid_import_a_low_kwh": float(grid_a_low_kwh),
                "grid_import_a_high_kwh": float(grid_a_high_kwh),
                "grid_import_b_low_kwh": float(grid_b_low_kwh),
                "grid_import_b_high_kwh": float(grid_b_high_kwh),
                "grid_export_kwh": float(ts_e_grid_export[:, u].sum()),
                "injection_kwh": float(ts_e_grid_export[:, u].sum()),
                "self_consumed_pv_kwh": self_consumed_pv_kwh,
                "locally_supplied_load_kwh": locally_supplied_load_kwh,
                "self_consumption_ratio": sci,
                "self_sufficiency_ratio": ssi,
                "SCI": sci,
                "SSI": ssi,
                "grid_import_a_cost_ft": float(grid_import_a_cost_ft),
                "grid_import_b_cost_ft": float(grid_import_b_cost_ft),
                "grid_import_cost_ft": float(grid_import_cost_ft),
                "shared_purchase_energy_cost_ft": float(shared_purchase_energy_cost_ft[u]),
                "shared_purchase_rhd_ft": float(shared_purchase_rhd_ft[u]),
                "shared_purchase_cost_ft": float(shared_purchase_cost_ft),
                "shared_buyer_low_kwh": float(shared_buyer_low_kwh[u]),
                "shared_buyer_high_kwh": float(shared_buyer_high_kwh[u]),
                "shared_revenue_ft": float(shared_revenue_ft[u]),
                "grid_export_revenue_ft": float(grid_export_revenue_ft[u]),
                "export_revenue_ft": float(grid_export_revenue_ft[u]),
                "cash_out_ft": float(total_cost_ft),
                "cash_in_ft": float(total_revenue_ft),
                "brt_bill_ft": float(brt_bill_ft),
            }
        )

    per_user_df = pd.DataFrame(rows)

    total_load_kwh = float(e_load.sum())
    pv_kwh_total = float(e_pv.sum())
    self_consumed_pv_total = float(ts_e_pv_to_load.sum() + ts_e_pv_to_bess.sum() + ts_e_shared_out.sum())
    locally_supplied_total = float(ts_e_pv_to_load.sum() + ts_e_bess_local_to_load.sum() + ts_e_shared_in.sum())

    total = {
        "n_users": int(U),
        "sharing_mode": sharing_mode,
        "boiler_tariff": boiler_tariff,
        "include_bess": bool(include_bess),
        "bess_share_pct": float(bess_share_pct),
        "pv_user_count": int(has_pv_arr.sum()),
        "bess_enabled_user_count": int(bess_enabled.sum()),
        "base_load_kwh": float(e_ue.sum()),
        "boiler_kwh": float(e_boiler.sum()),
        "heat_pump_kwh": 0.0,
        "total_load_kwh": total_load_kwh,
        "pv_kwh": pv_kwh_total,
        "pv_to_load_kwh": float(ts_e_pv_to_load.sum()),
        "pv_to_bess_kwh": float(ts_e_pv_to_bess.sum()),
        "bess_to_load_kwh": float(ts_e_bess_to_load.sum()),
        "bess_local_to_load_kwh": float(ts_e_bess_local_to_load.sum()),
        "grid_to_bess_kwh": float(ts_e_grid_to_bess.sum()),
        "local_to_base_kwh": float(ts_e_local_to_base.sum()),
        "local_to_boiler_kwh": float(ts_e_local_to_boiler.sum()),
        "base_deficit_before_share_kwh": float(ts_e_base_deficit_before_share.sum()),
        "boiler_deficit_before_share_kwh": float(ts_e_boiler_deficit_before_share.sum()),
        "shared_in_kwh": float(ts_e_shared_in.sum()),
        "shared_to_base_kwh": float(ts_e_shared_to_base.sum()),
        "shared_to_boiler_kwh": float(ts_e_shared_to_boiler.sum()),
        "shared_out_kwh": float(ts_e_shared_out.sum()),
        "grid_import_to_load_kwh": float(ts_e_grid_to_load.sum()),
        "grid_to_base_kwh": float(ts_e_grid_to_base.sum()),
        "grid_to_boiler_kwh": float(ts_e_grid_to_boiler.sum()),
        "grid_import_kwh": float(grid_import_total.sum()),
        "grid_import_a_kwh": float(grid_import_a.sum()),
        "grid_import_b_kwh": float(grid_import_b.sum()),
        "grid_export_kwh": float(ts_e_grid_export.sum()),
        "injection_kwh": float(ts_e_grid_export.sum()),
        "self_consumed_pv_kwh": self_consumed_pv_total,
        "locally_supplied_load_kwh": locally_supplied_total,
        "self_consumption_ratio": float(self_consumed_pv_total / pv_kwh_total) if pv_kwh_total > EPS else 0.0,
        "self_sufficiency_ratio": float(locally_supplied_total / total_load_kwh) if total_load_kwh > EPS else 0.0,
        "grid_import_a_cost_ft": float(per_user_df["grid_import_a_cost_ft"].sum()),
        "grid_import_b_cost_ft": float(per_user_df["grid_import_b_cost_ft"].sum()),
        "grid_import_cost_ft": float(per_user_df["grid_import_cost_ft"].sum()),
        "shared_purchase_energy_cost_ft": float(per_user_df["shared_purchase_energy_cost_ft"].sum()),
        "shared_purchase_rhd_ft": float(per_user_df["shared_purchase_rhd_ft"].sum()),
        "shared_purchase_cost_ft": float(per_user_df["shared_purchase_cost_ft"].sum()),
        "shared_revenue_ft": float(per_user_df["shared_revenue_ft"].sum()),
        "grid_export_revenue_ft": float(per_user_df["grid_export_revenue_ft"].sum()),
        "export_revenue_ft": float(per_user_df["grid_export_revenue_ft"].sum()),
        "cash_out_ft": float(per_user_df["cash_out_ft"].sum()),
        "cash_in_ft": float(per_user_df["cash_in_ft"].sum()),
        "brt_bill_ft": float(per_user_df["brt_bill_ft"].sum()),
        "shared_buyer_low_limit_kwh": SHARED_BUYER_LOW_LIMIT_KWH,
        "shared_buyer_low_ft_per_kwh": SHARED_BUYER_LOW_FT_PER_KWH,
        "shared_buyer_high_ft_per_kwh": SHARED_BUYER_HIGH_FT_PER_KWH,
        "shared_rhd_ft_per_kwh": SHARED_RHD_FT_PER_KWH,
        "b_low_tariff_limit_kwh": B_LOW_TARIFF_LIMIT_KWH,
        "b_low_tariff_ft_per_kwh": B_LOW_TARIFF_FT_PER_KWH,
        "b_high_tariff_ft_per_kwh": B_HIGH_TARIFF_FT_PER_KWH,
    }
    total["SCI"] = float(np.clip(total["self_consumption_ratio"], 0.0, 1.0))
    total["SSI"] = float(np.clip(total["self_sufficiency_ratio"], 0.0, 1.0))
    total["self_consumption_ratio"] = total["SCI"]
    total["self_sufficiency_ratio"] = total["SSI"]

    timeseries = {
        "e_load": e_load,
        "e_base_load": e_ue,
        "e_boiler": e_boiler,
        "e_pv": e_pv,
        "e_pv_to_load": ts_e_pv_to_load,
        "e_pv_to_bess": ts_e_pv_to_bess,
        "e_bess_to_load": ts_e_bess_to_load,
        "e_bess_local_to_load": ts_e_bess_local_to_load,
        "e_grid_to_bess": ts_e_grid_to_bess,
        "e_local_to_base": ts_e_local_to_base,
        "e_local_to_boiler": ts_e_local_to_boiler,
        "e_base_deficit_before_share": ts_e_base_deficit_before_share,
        "e_boiler_deficit_before_share": ts_e_boiler_deficit_before_share,
        "e_deficit_before_share": ts_e_deficit_before_share,
        "e_surplus_before_share": ts_e_surplus_before_share,
        "e_shared_in": ts_e_shared_in,
        "e_shared_to_base": ts_e_shared_to_base,
        "e_shared_to_boiler": ts_e_shared_to_boiler,
        "e_shared_out": ts_e_shared_out,
        "e_grid_to_load": ts_e_grid_to_load,
        "e_grid_to_base": ts_e_grid_to_base,
        "e_grid_to_boiler": ts_e_grid_to_boiler,
        "e_grid_import_a": grid_import_a,
        "e_grid_import_b": grid_import_b,
        "e_grid_import_total": grid_import_total,
        "e_grid_export": ts_e_grid_export,
        "e_inj": ts_e_grid_export,
        "e_bess": ts_e_bess,
        "d_bess_ch": ts_d_bess_ch,
        "d_bess_dis": ts_d_bess_dis,
    }

    return {
        "per_user_df": per_user_df,
        "summary": total,
        "timeseries": timeseries,
        "shared_pair_kwh": shared_pair_kwh,
        "shared_pair_energy_payment_ft": shared_pair_energy_payment_ft,
        "bess_enabled": bess_enabled,
        "has_pv": has_pv_arr,
        "user_names": user_names,
    }


def run_case_disaggregated_nonopt_shared(
    case_name: str,
    sim_yaml: str,
    profiles_csv: str,
    dhw_profiles_csv: str,
    out_dir: str,
    max_users: int,
    *,
    include_bess: bool = True,
    bess_share_pct: float = 100.0,
    sharing_mode: SharingMode = "proportional",
    boiler_tariff: BoilerTariff = "B",
    use_boiler_model: bool = True,
) -> dict:
    out_case = Path(out_dir)
    out_case.mkdir(parents=True, exist_ok=True)

    (
        e_pv,
        e_ue,
        e_el_heater_fixed,
        p_dhw_kw,
        size_bess,
        eta_bess_in,
        eta_bess_out,
        eta_bess_stor,
        soc_bess_min,
        soc_bess_max,
        t_bess_min,
        size_elh,
        vol_hss_water,
        T_env,
        T_max,
        T_min,
        T_in,
        T_set,
        a_hss,
        eta_elh,
        user_names,
    ) = build_inputs(
        sim_yaml_path=sim_yaml,
        profiles_csv_path=profiles_csv,
        dhw_profile_path=dhw_profiles_csv,
        max_users=max_users,
    )

    boiler_profiles = build_effective_boiler_profiles(
        e_pv=e_pv,
        e_el_heater_fixed=e_el_heater_fixed,
        p_dhw_kw=p_dhw_kw,
        size_elh=size_elh,
        vol_hss_water=vol_hss_water,
        T_env=T_env,
        T_max=T_max,
        T_min=T_min,
        T_in=T_in,
        T_set=T_set,
        a_hss=a_hss,
        eta_elh=eta_elh,
        boiler_tariff=boiler_tariff,
        use_boiler_model=use_boiler_model,
    )
    e_boiler = boiler_profiles["e_boiler"]

    result = simulate_nonopt_disaggregated_shared(
        e_pv=e_pv,
        e_ue=e_ue,
        e_boiler=e_boiler,
        size_bess=size_bess,
        eta_bess_in=eta_bess_in,
        eta_bess_out=eta_bess_out,
        eta_bess_stor=eta_bess_stor,
        soc_bess_min=soc_bess_min,
        soc_bess_max=soc_bess_max,
        t_bess_min_h=t_bess_min,
        user_names=user_names,
        include_bess=include_bess,
        bess_share_pct=bess_share_pct,
        sharing_mode=sharing_mode,
        boiler_tariff=boiler_tariff,
    )

    per_user_df: pd.DataFrame = result["per_user_df"]
    total: dict = result["summary"]
    timeseries: dict[str, np.ndarray] = result["timeseries"]
    user_names = result["user_names"]

    total["case_name"] = case_name
    total["use_boiler_model"] = bool(use_boiler_model)
    total["dynamic_boiler_user_count"] = int(
        boiler_profiles["dynamic_boiler"].sum()
    )
    total["fixed_boiler_user_count"] = int(
        (
            (~boiler_profiles["dynamic_boiler"])
            & (e_el_heater_fixed.sum(axis=0) > EPS)
        ).sum()
    )

    per_user_df["dynamic_boiler"] = boiler_profiles["dynamic_boiler"]
    per_user_df["fixed_boiler_profile"] = (
        (~boiler_profiles["dynamic_boiler"])
        & (e_el_heater_fixed.sum(axis=0) > EPS)
    )
    per_user_df["boiler_temperature_violation_steps"] = (
        boiler_profiles["temperature_violation"].sum(axis=0).astype(int)
    )
    per_user_df["boiler_min_temperature_c"] = np.where(
        boiler_profiles["dynamic_boiler"],
        np.min(boiler_profiles["t_hss"], axis=0),
        np.nan,
    )

    per_user_df.to_csv(out_case / "per_user_summary.csv", index=False)
    (out_case / "summary.json").write_text(json.dumps(total, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_ts(key: str, filename: str | None = None) -> None:
        arr = timeseries[key]
        pd.DataFrame(arr, columns=user_names).to_csv(out_case / (filename or f"{key}.csv"), index=False)

    for key in [
        "e_load",
        "e_base_load",
        "e_boiler",
        "e_pv",
        "e_pv_to_load",
        "e_pv_to_bess",
        "e_bess_to_load",
        "e_bess_local_to_load",
        "e_grid_to_bess",
        "e_local_to_base",
        "e_local_to_boiler",
        "e_base_deficit_before_share",
        "e_boiler_deficit_before_share",
        "e_deficit_before_share",
        "e_surplus_before_share",
        "e_shared_in",
        "e_shared_to_base",
        "e_shared_to_boiler",
        "e_shared_out",
        "e_grid_to_load",
        "e_grid_to_base",
        "e_grid_to_boiler",
        "e_grid_import_a",
        "e_grid_import_b",
        "e_grid_import_total",
        "e_grid_export",
        "e_inj",
        "e_bess",
        "d_bess_ch",
        "d_bess_dis",
    ]:
        save_ts(key)

    boiler_diagnostics = {
        "e_boiler_fixed_input": e_el_heater_fixed,
        "t_hss": boiler_profiles["t_hss"],
        "p_dhw_kw": boiler_profiles["p_dhw_kw"],
        "p_elh_in_kw": boiler_profiles["p_elh_in_kw"],
        "p_hss_in_kw": boiler_profiles["p_hss_in_kw"],
        "p_hss_out_kw": boiler_profiles["p_hss_out_kw"],
        "d_boiler_available": boiler_profiles["d_boiler_available"],
        "d_boiler_on": boiler_profiles["d_boiler_on"],
        "boiler_temperature_violation": boiler_profiles[
            "temperature_violation"
        ],
    }
    for key, array in boiler_diagnostics.items():
        pd.DataFrame(array, columns=user_names).to_csv(
            out_case / f"{key}.csv",
            index=False,
        )

    pd.DataFrame(result["shared_pair_kwh"], index=user_names, columns=user_names).to_csv(
        out_case / "shared_pair_kwh_seller_x_buyer.csv"
    )
    pd.DataFrame(result["shared_pair_energy_payment_ft"], index=user_names, columns=user_names).to_csv(
        out_case / "shared_pair_energy_payment_ft_seller_x_buyer.csv"
    )

    community_ts = pd.DataFrame(
        {
            "e_load_total": timeseries["e_load"].sum(axis=1),
            "e_load_base_total": timeseries["e_base_load"].sum(axis=1) + timeseries["e_boiler"].sum(axis=1),
            "e_ue_total": timeseries["e_base_load"].sum(axis=1),
            "e_boiler_total": timeseries["e_boiler"].sum(axis=1),
            "e_hp_total": np.zeros(timeseries["e_load"].shape[0]),
            "e_pv_total": timeseries["e_pv"].sum(axis=1),
            "e_pv_to_load_total": timeseries["e_pv_to_load"].sum(axis=1),
            "e_pv_to_bess_total": timeseries["e_pv_to_bess"].sum(axis=1),
            "e_bess_to_load_total": timeseries["e_bess_to_load"].sum(axis=1),
            "e_bess_local_to_load_total": timeseries["e_bess_local_to_load"].sum(axis=1),
            "e_shared_in_total": timeseries["e_shared_in"].sum(axis=1),
            "e_shared_to_base_total": timeseries["e_shared_to_base"].sum(axis=1),
            "e_shared_to_boiler_total": timeseries["e_shared_to_boiler"].sum(axis=1),
            "e_shared_out_total": timeseries["e_shared_out"].sum(axis=1),
            "e_grid_to_load_total": timeseries["e_grid_to_load"].sum(axis=1),
            "e_grid_to_base_total": timeseries["e_grid_to_base"].sum(axis=1),
            "e_grid_to_boiler_total": timeseries["e_grid_to_boiler"].sum(axis=1),
            "e_grid_to_bess_total": timeseries["e_grid_to_bess"].sum(axis=1),
            "e_grid_import_a_total": timeseries["e_grid_import_a"].sum(axis=1),
            "e_grid_import_b_total": timeseries["e_grid_import_b"].sum(axis=1),
            "e_grid_import_total": timeseries["e_grid_import_total"].sum(axis=1),
            "e_inj_total": timeseries["e_grid_export"].sum(axis=1),
            "e_bess_total": timeseries["e_bess"].sum(axis=1),
        }
    )
    community_ts.to_csv(out_case / "community_timeseries.csv", index=False)

    # A nonopt_common ábrái továbbra is használhatók; ha valami hiányzik, ne álljon meg a futás.
    try:
        plot_household_percentiles_by_group(per_user_df, out_case)
        plot_household_percentiles_by_group_with_global_scurve(per_user_df, out_case)
        plot_community_energy_balance(
            community_ts,
            out_case,
            per_user_df=per_user_df,
            household_ts={"e_grid_to_load": timeseries["e_grid_import_total"], "e_inj": timeseries["e_grid_export"]},
            user_names=user_names,
        )
    except Exception as exc:
        print(f"[WARN] Ábrakészítés kihagyva vagy részben sikertelen: {exc}")

    print(
        f"[INFO] Bojlerprofil mód: "
        f"{'dinamikus modell PV-seknél' if use_boiler_model else 'fix profil mindenkinél'}"
    )
    print(f"[INFO] Kész: {case_name}")
    print(f"[INFO] Output: {out_case}")
    print(json.dumps(total, indent=2, ensure_ascii=False))
    return result


SIM_YAML = "../Input/simulation_config_disaggregated_with_userlist.yaml"
PROFILES_CSV = "../Input/measurements_disaggregated_v2.csv"
DHW_PROFILES_CSV = "../Input/dhw_v2.csv"

MAX_USERS = 105

INCLUDE_BESS = True
BESS_SHARE_PCT = 40.0

# "A": bojler ugyanazon a körön van,
#      saját PV/BESS is kiszolgálhatja.
# "B": bojler külön körön van,
#      saját PV/BESS és közösségi megosztás sem szolgálhatja ki;
#      kizárólag a vezérlőjel szerinti B tarifás hálózati import látja el.
BOILER_TARIFF: BoilerTariff = "B"

# True:
#   PV-s és megfelelő HSS/DHW-adattal rendelkező háztartásnál
#   dinamikus bojlermodell; a többieknél fix profil.
# False:
#   minden háztartásnál a mért fix UT-profil.
USE_BOILER_MODEL = True

# "proportional": fogyasztásarányos megosztás
# "equal": egyenlő kvótás megosztás
SHARING_MODE: SharingMode = "proportional"

BOILER_PROFILE_MODE = (
    "boiler_model" if USE_BOILER_MODEL else "fixed_boiler"
)

CASE_NAME = (
    f"nonopt_community_{BOILER_TARIFF}_"
    f"{SHARING_MODE}_{BOILER_PROFILE_MODE}_{BESS_SHARE_PCT}%bess"
)
OUT_DIR = (
    f"results_nonopt_community_{BOILER_TARIFF}_"
    f"{SHARING_MODE}_{BOILER_PROFILE_MODE}_{BESS_SHARE_PCT}%bess"
)
if BESS_SHARE_PCT == 0:
    OUT_DIR = (
        f"results_nonopt_community_basecase_"
        f"{SHARING_MODE}_{BOILER_PROFILE_MODE}"
    )
if __name__ == "__main__":
    run_case_disaggregated_nonopt_shared(
        case_name=CASE_NAME,
        sim_yaml=SIM_YAML,
        profiles_csv=PROFILES_CSV,
        dhw_profiles_csv=DHW_PROFILES_CSV,
        out_dir=OUT_DIR,
        max_users=MAX_USERS,
        include_bess=INCLUDE_BESS,
        bess_share_pct=BESS_SHARE_PCT,
        sharing_mode=SHARING_MODE,
        boiler_tariff=BOILER_TARIFF,
        use_boiler_model=USE_BOILER_MODEL,
    )