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
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from InputReading import read_simulation_inputs
from Utility.bess_periodic import periodic_simulation
from Utility.bess_dispatch import pv_charge, load_discharge, minimum_grid_charge
from Utility.result_schema import add_bess_states, add_bess_flows
from Utility.energy_allocation import allocate_pool, match_energy
from functools import wraps
from inspect import signature
from Utility.configuration import config

try:
    from NotOptimizedScenarios.nonopt_common import (
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
except ImportError:  # direct script execution from this directory
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

SHARED_BUYER_LOW_LIMIT_KWH = config.getfloat("tariffs", "shared_buyer_low_limit_kwh")
SHARED_BUYER_LOW_FT_PER_KWH = config.getfloat("tariffs", "shared_buyer_low_ft_per_kwh")
SHARED_BUYER_HIGH_FT_PER_KWH = config.getfloat("tariffs", "shared_buyer_high_ft_per_kwh")
SHARED_RHD_FT_PER_KWH = config.getfloat("tariffs", "shared_rhd_ft_per_kwh")

EPS = 1e-12

SharingMode = Literal["proportional", "equal"]
BoilerTariff = Literal["A", "B"]

# B tarifa a bojler hálózati importjára.
# Az A tarifa értékei a nonopt_common importból jönnek.
B_LOW_TARIFF_LIMIT_KWH = config.getfloat("tariffs", "grid_b_low_limit_kwh")
B_LOW_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_b_low_ft_per_kwh")
B_HIGH_TARIFF_FT_PER_KWH = config.getfloat("tariffs", "grid_b_high_ft_per_kwh")


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


def settle_shared_payments_buyer_tiered(e_shared_in, e_shared_out, e_grid_import_a):
    """kWh compatibility adapter; preserve proportional tariff-block allocation."""
    from dataclasses import replace
    from Economics.calculate_economics import DEFAULT_TARIFFS, settle_shared_payments as settle
    tariffs = replace(DEFAULT_TARIFFS,
        shared_buyer_low_limit_kwh=SHARED_BUYER_LOW_LIMIT_KWH,
        shared_buyer_low_ft_per_kwh=SHARED_BUYER_LOW_FT_PER_KWH,
        shared_buyer_high_ft_per_kwh=SHARED_BUYER_HIGH_FT_PER_KWH,
        shared_rhd_ft_per_kwh=SHARED_RHD_FT_PER_KWH)
    result = settle(e_shared_in, e_shared_out, e_grid_import_a,
        tariffs=tariffs, pairing_mode="proportional", shared_low_cap_mode="proportional")
    return {**result,
        "grid_a_low_kwh": result["buyer_grid_a_low_kwh"],
        "grid_a_high_kwh": result["buyer_grid_a_high_kwh"],
        "shared_buyer_low_kwh": result["buyer_shared_low_kwh"],
        "shared_buyer_high_kwh": result["buyer_shared_high_kwh"]}

def allocate_shared_in(deficit: np.ndarray, total_share_kwh: float, mode: SharingMode) -> np.ndarray:
    """
    Megosztott energia vevői oldali kiosztása.

    proportional:
        a hiány arányában osztja ki.

    equal:
        azonos kWh-kvótát próbál adni minden aktív hiányos háztartásnak;
        aki ennél kevesebbet tud felvenni, annál a maradék újraosztódik.
    """
    return allocate_pool(deficit, total_share_kwh, mode)


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


def _simulate_nonopt_disaggregated_shared(
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
    allow_grid_charge_for_min_soc: bool = False,
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
    # B tarifa esetén a saját PV/BESS csak az alapfogyasztást látja,
    # a bojler csak a közösségi megosztásnál és a hálózatnál jelenik meg.
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
    soc = np.clip(np.asarray(initial_soc_fraction) * np.maximum(size_bess, 0.0), soc_min_kwh, soc_max_kwh)

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

            # Saját PV közvetlenül saját fogyasztásra.
            pv_to_load = min(pv_t, load_t)
            remaining_load = max(load_t - pv_to_load, 0.0)
            remaining_pv = max(pv_t - pv_to_load, 0.0)

            ts_e_pv_to_load[t, u] = pv_to_load

            charge = 0.0
            discharge = 0.0
            discharge_local_output = 0.0

            if use_bess_u and remaining_pv > EPS:
                charge = pv_charge(soc[u], remaining_pv, soc_max_kwh[u],
                                   eta_bess_in[u], e_bess_max_step[u])

                soc_add = charge * float(eta_bess_in[u])
                soc[u] += soc_add
                soc_local[u] += soc_add
                remaining_pv -= charge

                ts_e_pv_to_bess[t, u] = charge
                if charge > EPS:
                    ts_d_bess_ch[t, u] = 1.0

            elif use_bess_u and remaining_load > EPS:
                discharge = load_discharge(soc[u], remaining_load, soc_min_kwh[u],
                                           eta_bess_out[u], e_bess_max_step[u])

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

            if use_bess_u:
                grid_charge = minimum_grid_charge(soc[u], soc_min_kwh[u],
                    eta_bess_in[u], e_bess_max_step[u], charge,
                    enabled=allow_grid_charge_for_min_soc, allowed=discharge <= EPS)
                soc[u] += grid_charge * eta_bess_in[u]
                soc_grid[u] += grid_charge * eta_bess_in[u]
                ts_e_grid_to_bess[t, u] = grid_charge
                if grid_charge > EPS:
                    ts_d_bess_ch[t, u] = 1.0

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
                # B tarifás bojler: saját PV/BESS után is teljes egészében
                # megosztásra/hálózatra váró külön bojlerigény.
                boiler_deficit = boiler_load_t

            ts_e_local_to_base[t, u] = local_to_base
            ts_e_local_to_boiler[t, u] = local_to_boiler
            ts_e_base_deficit_before_share[t, u] = base_deficit
            ts_e_boiler_deficit_before_share[t, u] = boiler_deficit

            # BESS után még megmaradó pozíciók. BESS nem exportál közösségbe,
            # csak a saját PV-felesleg mehet megosztásba.
            ts_e_surplus_before_share[t, u] = max(remaining_pv, 0.0)
            ts_e_deficit_before_share[t, u] = max(base_deficit + boiler_deficit, 0.0)

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
        shared_in, shared_out = match_energy(deficit, surplus, sharing_mode, "proportional")

        ts_e_shared_in[t, :] = shared_in
        ts_e_shared_out[t, :] = shared_out

        # A kapott közösségi energiát komponensenként is bontjuk.
        # Itt arányos belső bontást használunk a háztartáson belüli
        # alapfogyasztási és bojlerhiány között. Így B tarifánál egyértelműen
        # látszik: e_shared_to_boiler és e_grid_to_boiler.
        base_def = ts_e_base_deficit_before_share[t, :]
        boiler_def = ts_e_boiler_deficit_before_share[t, :]
        total_def = np.maximum(base_def + boiler_def, 0.0)
        boiler_ratio = np.divide(boiler_def, total_def, out=np.zeros_like(boiler_def), where=total_def > EPS)

        shared_to_boiler = np.minimum(boiler_def, shared_in * boiler_ratio)
        shared_to_base = np.minimum(base_def, shared_in - shared_to_boiler)

        # Kerekítési korrekció: ha a proportional belső bontás miatt marad pár Wh,
        # először az alapfogyasztás, utána a bojler kapja.
        rem_shared = np.maximum(shared_in - shared_to_base - shared_to_boiler, 0.0)
        add_base = np.minimum(np.maximum(base_def - shared_to_base, 0.0), rem_shared)
        shared_to_base += add_base
        rem_shared = np.maximum(rem_shared - add_base, 0.0)
        add_boiler = np.minimum(np.maximum(boiler_def - shared_to_boiler, 0.0), rem_shared)
        shared_to_boiler += add_boiler

        ts_e_shared_to_base[t, :] = shared_to_base
        ts_e_shared_to_boiler[t, :] = shared_to_boiler
        ts_e_grid_to_base[t, :] = np.maximum(base_def - shared_to_base, 0.0)
        ts_e_grid_to_boiler[t, :] = np.maximum(boiler_def - shared_to_boiler, 0.0)

        ts_e_grid_to_load[t, :] = ts_e_grid_to_base[t, :] + ts_e_grid_to_boiler[t, :]
        ts_e_grid_export[t, :] = np.maximum(surplus - shared_out, 0.0)

        # A pénzügyi megosztási elszámolás később készül, mert a vevői
        # 2523 kWh-os sávot az A-tarifás hálózati importtal együtt kell vezetni.

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


@wraps(_simulate_nonopt_disaggregated_shared)
def simulate_nonopt_disaggregated_shared(*args, **kwargs):
    bound = signature(_simulate_nonopt_disaggregated_shared).bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    size = np.asarray(values["size_bess"], dtype=float)
    enabled = _select_bess_users(np.asarray(values["e_pv"]).sum(axis=0)>EPS,
        size, values["include_bess"], values["bess_share_pct"])
    if not np.any(enabled):
        result = _simulate_nonopt_disaggregated_shared(**values)
        shape = result["timeseries"]["e_bess"].shape
        add_bess_states(result["timeseries"], np.zeros((shape[0]+1, shape[1])))
        add_bess_flows(result["timeseries"], DT)
        return result
    lower = np.where(enabled, size*np.asarray(values["soc_bess_min"]), 0)
    upper = np.where(enabled, size*np.asarray(values["soc_bess_max"]), 0)
    initial = np.clip(np.asarray(values["initial_soc_fraction"])*size, lower, upper)
    def simulate(state):
        fractions = np.divide(state, size, out=np.zeros_like(size), where=size>0)
        result = _simulate_nonopt_disaggregated_shared(**{**values, "initial_soc_fraction": fractions})
        return result, result["timeseries"]["e_bess"][-1], lower, upper
    return periodic_simulation(simulate, initial, dt=DT)


def run_case_disaggregated_nonopt_shared(
    case_name: str,
    sim_yaml: str,
    profiles_csv: str,
    dhw_profiles_csv: str,
    out_dir: str,
    max_users: int | None,
    *,
    include_bess: bool = True,
    include_boiler: bool = True,
    bess_share_pct: float = 100.0,
    sharing_mode: SharingMode = "proportional",
    boiler_tariff: BoilerTariff = "B",
    pv_ratio: float = 1.0,
    allow_grid_charge_for_min_soc: bool = False,
) -> dict:
    out_case = Path(out_dir)
    out_case.mkdir(parents=True, exist_ok=True)

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
    eta_bess_in = inputs.eta_bess_in
    eta_bess_out = inputs.eta_bess_out
    eta_bess_stor = inputs.eta_bess_stor
    soc_bess_min = inputs.soc_bess_min
    soc_bess_max = inputs.soc_bess_max
    t_bess_min = inputs.t_bess_min
    user_names = inputs.user_names

    if not include_boiler:
        e_el_heater = np.zeros_like(e_el_heater)

    result = simulate_nonopt_disaggregated_shared(
        allow_grid_charge_for_min_soc=allow_grid_charge_for_min_soc,
        e_pv=e_pv,
        e_ue=e_ue,
        e_boiler=e_el_heater,
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
    from Utility.result_schema import save_bess_result
    save_bess_result(timeseries, out_case, DT, user_names)

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
        "e_bess", "e_bess_start", "e_bess_end", "e_bess_boundary",
        "d_bess_ch",
        "d_bess_dis",
    ]:
        save_ts(key)

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

    print(f"[INFO] Kész: {case_name}")
    print(f"[INFO] Output: {out_case}")
    print(json.dumps(total, indent=2, ensure_ascii=False))
    return result


SIM_YAML = config.getpath("paths", "simulation_yaml")
PROFILES_CSV = config.getpath("paths", "profiles_csv")
DHW_PROFILES_CSV = config.getpath("paths", "dhw_profiles_csv")

MAX_USERS = 105

INCLUDE_BESS = True
BESS_SHARE_PCT = 0

# "A": bojler ugyanazon a körön van,
#      saját PV/BESS is kiszolgálhatja.
# "B": bojler külön körön van,
#      saját PV/BESS nem szolgálhatja ki,
#      de közösségi megosztott energiát kaphat.
BOILER_TARIFF: BoilerTariff = "B"

# "proportional": fogyasztásarányos megosztás
# "equal": egyenlő kvótás megosztás
SHARING_MODE: SharingMode = "proportional"

CASE_NAME = f"nonopt_community_{BOILER_TARIFF}_{SHARING_MODE}"
OUT_DIR = config.getpath("paths", "noopt_output") / CASE_NAME
if BESS_SHARE_PCT == 0:
    OUT_DIR = config.getpath("paths", "noopt_output") / "results_nonopt_community_basecase"
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
    )
