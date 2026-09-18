from __future__ import annotations

"""
Disaggregált, OPTIMALIZÁLT BESS-es energiaközösségi modell megosztással.

A modell az individual_opt_bess.py logikáját emeli közösségi szintre:
    - PV először fixen saját fogyasztásra megy,
    - a maradék PV-többlet döntési változóként mehet saját BESS-be,
      energiaközösségi megosztásba vagy külső hálózatba,
    - a PV után maradó saját hiányt döntési változóként fedezheti saját BESS,
      energiaközösségi megosztás vagy külső hálózat,
    - a BESS töltés/kisütés és SOC optimalizált,
    - van töltési/kisütési bináris, egyszerre nem tölthet és süthet ki,
    - van minimum üzemmódhossz-korlát,
    - a külső hálózati BESS-töltés csak minimum SOC-védőpótlásra engedett,
    - A/B bojler tarifalogika az individual modellhez igazodik.

Fontos: a megosztás fizikailag optimalizált allokációként szerepel a MILP-ben.
Az eladó-vevő páros pénzügyi elszámolás a megoldott p_shared_in / p_shared_out
idősorokból készül.
"""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pulp
from Optimization import extract_vector, variable_value, extract_matrix, solve_problem
from Utility.configuration import config
from Optimization.bess_constraints import build_bess

DT_DEFAULT = config.getfloat("simulation", "dt_hours")
EPS = 1e-9

# Alap A/B és PV árak, az individual modellhez igazítva.
LOW_TARIFF_LIMIT_KWH = config.getfloat("tariffs", "grid_a_low_limit_kwh")
PRICE_GRID_A_LOW = config.getfloat("tariffs", "grid_a_low_ft_per_kwh")
PRICE_GRID_A_HIGH = config.getfloat("tariffs", "grid_a_high_ft_per_kwh")
PRICE_GRID_B_LOW = config.getfloat("tariffs", "grid_b_low_ft_per_kwh")
PRICE_GRID_B_HIGH = config.getfloat("tariffs", "grid_b_high_ft_per_kwh")
PRICE_PV_GRID = config.getfloat("tariffs", "pv_export_ft_per_kwh")

# Energiaközösségi megosztás elszámolása.
# Nincs külön eladói megosztási keret: a megosztott energia a vevő
# ugyanazon éves A-sávját fogyasztja, mint a hálózati A-vételezés.
SHARED_BUYER_LOW_FT_PER_KWH = config.getfloat("tariffs", "shared_buyer_low_ft_per_kwh")
SHARED_BUYER_HIGH_FT_PER_KWH = config.getfloat("tariffs", "shared_buyer_high_ft_per_kwh")
SHARED_RHD_FT_PER_KWH = config.getfloat("tariffs", "shared_rhd_ft_per_kwh")

Objective = Literal["bill", "grid"]
SharingMode = Literal["proportional", "equal"]
PairingMode = SharingMode


def _as_2d_power(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} csak 2D tömb lehet, kapott dimenzió: {arr.ndim}")
    return np.maximum(arr, 0.0)


def _var_matrix(name: str, T: int, U: int, low: float = 0.0) -> list[list[pulp.LpVariable]]:
    return [[pulp.LpVariable(f"{name}_{t}_{u}", lowBound=low) for u in range(U)] for t in range(T)]


def _bin_matrix(name: str, T: int, U: int, run_lp: bool) -> list[list[pulp.LpVariable | float]]:
    if run_lp:
        return [[0.0 for _ in range(U)] for _ in range(T)]
    return [[pulp.LpVariable(f"{name}_{t}_{u}", cat=pulp.LpBinary) for u in range(U)] for t in range(T)]


def settle_shared_payments(p_shared_in, p_shared_out, p_grid_import_a,
        dt=DT_DEFAULT, pairing_mode="proportional", buyer_low_limit_kwh=LOW_TARIFF_LIMIT_KWH):
    """kW compatibility adapter; preserve grid-first tariff-block allocation."""
    from dataclasses import replace
    from Economics.calculate_economics import DEFAULT_TARIFFS, settle_shared_payments as settle
    tariffs = replace(DEFAULT_TARIFFS, shared_buyer_low_limit_kwh=buyer_low_limit_kwh,
        shared_buyer_low_ft_per_kwh=SHARED_BUYER_LOW_FT_PER_KWH,
        shared_buyer_high_ft_per_kwh=SHARED_BUYER_HIGH_FT_PER_KWH,
        shared_rhd_ft_per_kwh=SHARED_RHD_FT_PER_KWH)
    return settle(np.asarray(p_shared_in)*dt, np.asarray(p_shared_out)*dt,
        np.asarray(p_grid_import_a)*dt, tariffs=tariffs,
        pairing_mode=pairing_mode, shared_low_cap_mode="grid_first")

def disaggregated_opt_bess_shared(
    p_pv: np.ndarray,
    p_ue: np.ndarray,
    p_el_heater: np.ndarray | None,
    *,
    dt: float = DT_DEFAULT,
    user_names: list[str] | None = None,
    bess_enabled: np.ndarray | None = None,
    size_bess: np.ndarray | None = None,
    eta_bess_in: np.ndarray | None = None,
    eta_bess_out: np.ndarray | None = None,
    eta_bess_stor: np.ndarray | None = None,
    soc_bess_min: np.ndarray | None = None,
    soc_bess_max: np.ndarray | None = None,
    soc_bess_init: float = 0.5,
    t_bess_min: np.ndarray | None = None,
    price_grid_a_low: float = PRICE_GRID_A_LOW,
    price_grid_a_high: float = PRICE_GRID_A_HIGH,
    price_grid_b_low: float = PRICE_GRID_B_LOW,
    price_grid_b_high: float = PRICE_GRID_B_HIGH,
    price_pv_grid: float = PRICE_PV_GRID,
    grid_a_low_cap_kwh: float = LOW_TARIFF_LIMIT_KWH,
    grid_b_low_cap_kwh: float = LOW_TARIFF_LIMIT_KWH,
    shared_rhd_ft_per_kwh: float = SHARED_RHD_FT_PER_KWH,
    boiler_tariff: str = "B",
    objective: Objective = "bill",
    run_lp: bool = False,
    msg: bool = False,
    gapRel: float | None = 0.005,
    timeLimit: float | None = None,
    bess_min_mode_steps: int = 4,
    allow_grid_charge_for_min_soc: bool = False,
    sharing_mode: SharingMode = "proportional",
    pairing_mode: PairingMode | None = None,
) -> dict:
    p_pv = _as_2d_power(p_pv, "p_pv")
    p_ue = _as_2d_power(p_ue, "p_ue")
    if p_pv.shape != p_ue.shape:
        raise ValueError(f"p_pv alakja {p_pv.shape}, p_ue alakja {p_ue.shape}")
    T, U = p_ue.shape

    if p_el_heater is None:
        p_el_heater = np.zeros_like(p_ue)
    else:
        p_el_heater = _as_2d_power(p_el_heater, "p_el_heater")
        if p_el_heater.shape != (T, U):
            raise ValueError(f"p_el_heater alakja {p_el_heater.shape}, de {(T, U)} kell")

    if user_names is None or len(user_names) != U:
        user_names = [f"user_{u + 1:03d}" for u in range(U)]

    def arr_or(default: float, x: np.ndarray | None) -> np.ndarray:
        if x is None:
            return np.full(U, default, dtype=float)
        a = np.asarray(x, dtype=float).ravel()
        if len(a) != U:
            raise ValueError("BESS paramétervektor hossza nem egyezik U-val")
        return a

    size_bess = arr_or(0.0, size_bess)
    eta_bess_in = arr_or(0.98, eta_bess_in)
    eta_bess_out = arr_or(0.96, eta_bess_out)
    eta_bess_stor = arr_or(0.995, eta_bess_stor)
    soc_bess_min = arr_or(0.10, soc_bess_min)
    soc_bess_max = arr_or(0.90, soc_bess_max)
    t_bess_min = arr_or(2.0, t_bess_min)

    if bess_enabled is None:
        bess_enabled = size_bess > EPS
    else:
        bess_enabled = np.asarray(bess_enabled, dtype=bool).ravel()
        if len(bess_enabled) != U:
            raise ValueError("bess_enabled hossza nem egyezik U-val")

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError("boiler_tariff csak 'A' vagy 'B' lehet")

    sharing_mode = str(sharing_mode).lower().strip()
    if sharing_mode not in {"proportional", "equal"}:
        raise ValueError("sharing_mode csak 'proportional' vagy 'equal' lehet")
    if pairing_mode is None:
        pairing_mode = sharing_mode
    pairing_mode = str(pairing_mode).lower().strip()
    if pairing_mode not in {"proportional", "equal"}:
        raise ValueError("pairing_mode csak 'proportional' vagy 'equal' lehet")

    total_load = p_ue + p_el_heater

    # Individual model szerinti fix saját PV -> saját fogyasztás prioritás.
    if boiler_tariff == "A":
        p_dispatch_load = total_load.copy()
    else:
        p_dispatch_load = p_ue.copy()

    p_pv_to_load_fixed = np.minimum(p_pv, p_dispatch_load)
    p_surplus = np.maximum(p_pv - p_pv_to_load_fixed, 0.0)
    p_deficit_dispatch = np.maximum(p_dispatch_load - p_pv_to_load_fixed, 0.0)

    p_pv_ue_fixed = np.minimum(p_ue, p_pv_to_load_fixed)
    if boiler_tariff == "A":
        p_pv_elh_fixed = np.maximum(p_pv_to_load_fixed - p_pv_ue_fixed, 0.0)
        p_deficit_ue = np.maximum(p_ue - p_pv_ue_fixed, 0.0)
        p_deficit_elh = np.maximum(p_el_heater - p_pv_elh_fixed, 0.0)
    else:
        p_pv_elh_fixed = np.zeros((T, U), dtype=float)
        p_deficit_ue = p_deficit_dispatch.copy()
        p_deficit_elh = p_el_heater.copy()

    battery_power = np.zeros(U, dtype=float)
    for u in range(U):
        if bess_enabled[u] and size_bess[u] > EPS:
            battery_power[u] = float(size_bess[u]) / max(float(t_bess_min[u]), EPS)

    prob = pulp.LpProblem("disaggregated_opt_bess_shared", pulp.LpMinimize)
    time_set = range(T)
    user_set = range(U)

    # --- Döntési változók ---
    p_pv_shared = _var_matrix("p_pv_shared", T, U)
    p_pv_grid = _var_matrix("p_pv_grid", T, U)

    p_shared_ue = _var_matrix("p_shared_ue", T, U)
    p_shared_elh = _var_matrix("p_shared_elh", T, U)
    p_grid_ue = _var_matrix("p_grid_ue", T, U)
    p_grid_elh = _var_matrix("p_grid_elh", T, U)


    p_grid_import = _var_matrix("p_grid_import", T, U)
    p_grid_export = _var_matrix("p_grid_export", T, U)
    d_grid = _bin_matrix("d_grid", T, U, run_lp)

    batteries = [build_bess(prob, T, dt=dt,
        size=float(size_bess[u]) if bess_enabled[u] else 0.0,
        eta_in=float(eta_bess_in[u]), eta_out=float(eta_bess_out[u]),
        retention=float(eta_bess_stor[u]), soc_min=float(soc_bess_min[u]),
        soc_max=float(soc_bess_max[u]), soc_init=float(soc_bess_init),
        min_hours=float(t_bess_min[u]), run_lp=run_lp,
        min_mode_steps=bess_min_mode_steps,
        allow_grid_charge_for_min_soc=allow_grid_charge_for_min_soc,
        prefix=f"user_{u}_") for u in user_set]
    def battery_matrix(attribute):
        return [[getattr(batteries[u], attribute)[t] for u in user_set] for t in time_set]
    p_pv_bess = battery_matrix("pv")
    p_bess_ue = battery_matrix("base")
    p_bess_elh = battery_matrix("boiler")
    p_bess_in = battery_matrix("charge")
    p_bess_out = battery_matrix("discharge")
    p_grid_bess = battery_matrix("grid")
    d_bess_ch = battery_matrix("charge_on")
    d_bess_dis = battery_matrix("discharge_on")
    e_bess = battery_matrix("state")
    for u in user_set:
        for t in time_set:
            prob += batteries[u].fixed[t] == 0, f"unused_fixed_battery_load_{t}_{u}"

    e_grid_a_low_step = _var_matrix("e_grid_a_low_step", T, U)
    e_grid_a_high_step = _var_matrix("e_grid_a_high_step", T, U)
    e_shared_a_low_step = _var_matrix("e_shared_a_low_step", T, U)
    e_shared_a_high_step = _var_matrix("e_shared_a_high_step", T, U)
    rem_a_low = [[pulp.LpVariable(f"rem_a_low_{t}_{u}", lowBound=0, upBound=grid_a_low_cap_kwh) for u in user_set] for t in time_set]

    e_grid_b_low_step = _var_matrix("e_grid_b_low_step", T, U)
    e_grid_b_high_step = _var_matrix("e_grid_b_high_step", T, U)
    rem_b_low = [[pulp.LpVariable(f"rem_b_low_{t}_{u}", lowBound=0, upBound=grid_b_low_cap_kwh) for u in user_set] for t in time_set]

    # Fizikai megosztási mód.
    # proportional: a megosztott energia a PV utáni eredeti hiány arányában oszlik meg.
    # equal: egy közös felső kvótát ad minden aktív hiányos felhasználóra.
    # Megjegyzés: a BESS döntési változó, ezért a "BESS utáni" pontos proportional/equal
    # kiosztás bilineáris lenne; ez a lineáris MILP-kompatibilis változat.
    share_alpha = [pulp.LpVariable(f"share_alpha_{t}", lowBound=0, upBound=1) for t in time_set]
    share_quota = [pulp.LpVariable(f"share_quota_{t}", lowBound=0) for t in time_set]

    M_pv_u = np.maximum(np.max(p_pv, axis=0) + np.max(battery_power) + 1.0, 1.0)
    M_grid_u = np.maximum(np.max(total_load, axis=0) + battery_power + 1.0, 1.0)

    # --- Korlátok ---
    for t in time_set:
        # Közösségi megosztási egyenleg: amit eladnak, azt ugyanabban a lépésben megveszik.
        prob += (
            pulp.lpSum(p_pv_shared[t][u] for u in user_set)
            == pulp.lpSum(p_shared_ue[t][u] + p_shared_elh[t][u] for u in user_set)
        ), f"community_share_balance_{t}"

        for u in user_set:
            # PV-többlet szétosztása: BESS / közösségi megosztás / külső hálózat.
            prob += (
                p_pv_bess[t][u] + p_pv_shared[t][u] + p_pv_grid[t][u]
                == float(p_surplus[t, u])
            ), f"surplus_split_{t}_{u}"

            # PV után maradó saját hiány szétosztása: BESS / megosztás / külső hálózat.
            prob += (
                p_bess_ue[t][u] + p_shared_ue[t][u] + p_grid_ue[t][u]
                == float(p_deficit_ue[t, u])
            ), f"ue_deficit_split_{t}_{u}"

            prob += (
                p_bess_elh[t][u] + p_shared_elh[t][u] + p_grid_elh[t][u]
                == float(p_deficit_elh[t, u])
            ), f"boiler_deficit_split_{t}_{u}"

            if boiler_tariff == "B":
                # B tarifás bojler külön kör: saját PV-t és saját BESS-t nem kap,
                # de energiaközösségi megosztott energiát kaphat.
                prob += p_bess_elh[t][u] == 0, f"no_bess_to_boiler_B_{t}_{u}"

            p_shared_total_tu = p_shared_ue[t][u] + p_shared_elh[t][u]
            deficit_total_const = float(p_deficit_ue[t, u] + p_deficit_elh[t, u])
            if deficit_total_const <= EPS:
                prob += p_shared_total_tu == 0, f"no_shared_without_deficit_{t}_{u}"
            elif sharing_mode == "proportional":
                prob += p_shared_total_tu == share_alpha[t] * deficit_total_const, f"shared_proportional_{t}_{u}"
                # A megosztott energia alap/bojler bontása is a hiány bontását követi,
                # ezért B tarifás bojlernél pontosan mérhető a shared_to_boiler.
                prob += p_shared_ue[t][u] == share_alpha[t] * float(p_deficit_ue[t, u]), f"shared_ue_prop_{t}_{u}"
                prob += p_shared_elh[t][u] == share_alpha[t] * float(p_deficit_elh[t, u]), f"shared_elh_prop_{t}_{u}"
            else:
                # Equal mód: minden aktív hiányos felhasználó ugyanakkora kvótáig kaphat.
                # A kvóta feletti hiányt BESS vagy hálózat fedezheti.
                prob += p_shared_total_tu <= share_quota[t], f"shared_equal_quota_{t}_{u}"
                if deficit_total_const > EPS:
                    # A kapott megosztott energia alap/bojler bontása a hiány bontását követi.
                    prob += p_shared_ue[t][u] == p_shared_total_tu * float(p_deficit_ue[t, u] / deficit_total_const), f"shared_ue_equal_split_{t}_{u}"
                    prob += p_shared_elh[t][u] == p_shared_total_tu * float(p_deficit_elh[t, u] / deficit_total_const), f"shared_elh_equal_split_{t}_{u}"

            prob += p_grid_export[t][u] == p_pv_grid[t][u], f"grid_export_def_{t}_{u}"
            prob += (
                p_grid_import[t][u] == p_grid_ue[t][u] + p_grid_elh[t][u] + p_grid_bess[t][u]
            ), f"grid_import_def_{t}_{u}"

            if not run_lp:
                if boiler_tariff == "A":
                    p_grid_import_a_t = p_grid_ue[t][u] + p_grid_elh[t][u] + p_grid_bess[t][u]
                else:
                    p_grid_import_a_t = p_grid_ue[t][u] + p_grid_bess[t][u]
                prob += p_grid_import_a_t <= float(M_grid_u[u]) * d_grid[t][u], f"grid_import_a_gate_{t}_{u}"
                prob += p_grid_export[t][u] <= float(M_pv_u[u]) * (1 - d_grid[t][u]), f"grid_export_gate_{t}_{u}"

            # A/B tarifa energia-lépcsők.
            if boiler_tariff == "A":
                e_imp_a_t = dt * (p_grid_ue[t][u] + p_grid_elh[t][u] + p_grid_bess[t][u])
                e_imp_b_t = 0.0
            else:
                e_imp_a_t = dt * (p_grid_ue[t][u] + p_grid_bess[t][u])
                e_imp_b_t = dt * p_grid_elh[t][u]

            e_shared_a_t = dt * (p_shared_ue[t][u] + p_shared_elh[t][u])

            # A vevő 2523 kWh-os A-sávját az A-tarifás hálózati import és a
            # közösségi megosztás közösen fogyasztja. Nincs külön shared/seller keret.
            prob += e_shared_a_low_step[t][u] + e_shared_a_high_step[t][u] == e_shared_a_t, f"shared_a_tier_split_{t}_{u}"
            prob += e_grid_a_low_step[t][u] + e_grid_a_high_step[t][u] == e_imp_a_t, f"a_tier_split_{t}_{u}"
            prob += e_grid_b_low_step[t][u] + e_grid_b_high_step[t][u] == e_imp_b_t, f"b_tier_split_{t}_{u}"

            if t == 0:
                prob += rem_a_low[t][u] == float(grid_a_low_cap_kwh) - e_shared_a_low_step[t][u] - e_grid_a_low_step[t][u], f"rem_a_init_{u}"
                prob += e_shared_a_low_step[t][u] <= float(grid_a_low_cap_kwh), f"shared_a_low_cap_init_{u}"
                prob += e_grid_a_low_step[t][u] <= float(grid_a_low_cap_kwh) - e_shared_a_low_step[t][u], f"a_low_cap_init_{u}"
                prob += rem_b_low[t][u] == float(grid_b_low_cap_kwh) - e_grid_b_low_step[t][u], f"rem_b_init_{u}"
                prob += e_grid_b_low_step[t][u] <= float(grid_b_low_cap_kwh), f"b_low_cap_init_{u}"
            else:
                prob += rem_a_low[t][u] == rem_a_low[t - 1][u] - e_shared_a_low_step[t][u] - e_grid_a_low_step[t][u], f"rem_a_dyn_{t}_{u}"
                prob += e_shared_a_low_step[t][u] <= rem_a_low[t - 1][u], f"shared_a_low_remaining_{t}_{u}"
                prob += e_grid_a_low_step[t][u] <= rem_a_low[t - 1][u] - e_shared_a_low_step[t][u], f"a_low_remaining_{t}_{u}"
                prob += rem_b_low[t][u] == rem_b_low[t - 1][u] - e_grid_b_low_step[t][u], f"rem_b_dyn_{t}_{u}"
                prob += e_grid_b_low_step[t][u] <= rem_b_low[t - 1][u], f"b_low_remaining_{t}_{u}"

    # --- Célfüggvény ---
    if objective == "grid":
        prob += pulp.lpSum(
            dt * (p_grid_import[t][u] + p_grid_export[t][u])
            for t in time_set for u in user_set
        )
    elif objective == "bill":
        # Közösségi összes nettó számla: hálózati importköltség - hálózati exportbevétel
        # + megosztási RHD. A megosztott energiadíj a vevők és PV-s eladók között
        # belső transzferként kiesik a közösségi összegből, de utólag per-user elszámoljuk.
        prob += pulp.lpSum(
            price_grid_a_low * e_grid_a_low_step[t][u]
            + price_grid_a_high * e_grid_a_high_step[t][u]
            + price_grid_b_low * e_grid_b_low_step[t][u]
            + price_grid_b_high * e_grid_b_high_step[t][u]
            - price_pv_grid * dt * p_grid_export[t][u]
            + shared_rhd_ft_per_kwh * dt * (p_shared_ue[t][u] + p_shared_elh[t][u])
            + 1e-6 * e_shared_a_high_step[t][u]
            for t in time_set for u in user_set
        )
    else:
        raise ValueError("objective csak 'bill' vagy 'grid' lehet")

    solve_result = solve_problem(
        prob, msg=msg, gap_rel=gapRel, time_limit=timeLimit
    )
    status_str = solve_result.status

    # --- Eredmények kinyerése ---
    p_pv_bess_v = extract_matrix(p_pv_bess)
    p_pv_shared_v = extract_matrix(p_pv_shared)
    p_pv_grid_v = extract_matrix(p_pv_grid)
    p_bess_ue_v = extract_matrix(p_bess_ue)
    p_bess_elh_v = extract_matrix(p_bess_elh)
    p_shared_ue_v = extract_matrix(p_shared_ue)
    p_shared_elh_v = extract_matrix(p_shared_elh)
    p_grid_ue_v = extract_matrix(p_grid_ue)
    p_grid_elh_v = extract_matrix(p_grid_elh)
    p_bess_in_v = extract_matrix(p_bess_in)
    p_bess_out_v = extract_matrix(p_bess_out)
    p_grid_bess_v = extract_matrix(p_grid_bess)
    p_grid_import_v = extract_matrix(p_grid_import)
    p_grid_export_v = extract_matrix(p_grid_export)
    d_bess_ch_v = extract_matrix(d_bess_ch) if not run_lp else np.zeros((T, U))
    d_bess_dis_v = extract_matrix(d_bess_dis) if not run_lp else np.zeros((T, U))
    d_grid_v = extract_matrix(d_grid) if not run_lp else np.zeros((T, U))
    e_bess_v = extract_matrix(e_bess)

    e_grid_a_low_step_v = extract_matrix(e_grid_a_low_step)
    e_grid_a_high_step_v = extract_matrix(e_grid_a_high_step)
    e_shared_a_low_step_v = extract_matrix(e_shared_a_low_step)
    e_shared_a_high_step_v = extract_matrix(e_shared_a_high_step)
    e_grid_b_low_step_v = extract_matrix(e_grid_b_low_step)
    e_grid_b_high_step_v = extract_matrix(e_grid_b_high_step)

    p_pv_load_v = p_pv_ue_fixed + p_pv_elh_fixed
    p_bess_load_v = p_bess_ue_v + p_bess_elh_v
    p_shared_in_v = p_shared_ue_v + p_shared_elh_v
    p_grid_load_v = p_grid_ue_v + p_grid_elh_v
    if boiler_tariff == "A":
        p_grid_import_a_v = p_grid_ue_v + p_grid_elh_v + p_grid_bess_v
    else:
        p_grid_import_a_v = p_grid_ue_v + p_grid_bess_v

    settlement = settle_shared_payments(
        p_shared_in=p_shared_in_v,
        p_shared_out=p_pv_shared_v,
        p_grid_import_a=p_grid_import_a_v,
        dt=dt,
        pairing_mode=pairing_mode,
        buyer_low_limit_kwh=grid_a_low_cap_kwh,
    )

    rows = []
    for u in user_set:
        # A-sáv: a post-process elszámolásban a shared és grid ugyanazt a vevői
        # kedvezményes keretet fogyasztja. A hálózati import költségét ezért
        # a settlement szerinti grid A low/high bontásból vesszük.
        a_low = float(settlement["buyer_grid_a_low_kwh"][u])
        a_high = float(settlement["buyer_grid_a_high_kwh"][u])
        b_low = float(e_grid_b_low_step_v[:, u].sum())
        b_high = float(e_grid_b_high_step_v[:, u].sum())
        import_cost_a = price_grid_a_low * a_low + price_grid_a_high * a_high
        import_cost_b = price_grid_b_low * b_low + price_grid_b_high * b_high
        import_cost = import_cost_a + import_cost_b
        grid_export_revenue = float(p_grid_export_v[:, u].sum() * dt * price_pv_grid)
        shared_purchase_cost = float(settlement["buyer_total_shared_cost_ft"][u])
        shared_revenue = float(settlement["seller_revenue_ft"][u])
        brt_bill = import_cost + shared_purchase_cost - grid_export_revenue - shared_revenue

        pv_kwh = float(p_pv[:, u].sum() * dt)
        load_kwh = float(total_load[:, u].sum() * dt)
        self_consumed_pv = float((p_pv_load_v[:, u] + p_pv_bess_v[:, u] + p_pv_shared_v[:, u]).sum() * dt)
        locally_supplied = float((p_pv_load_v[:, u] + p_bess_load_v[:, u] + p_shared_in_v[:, u]).sum() * dt)
        sci = float(np.clip(self_consumed_pv / pv_kwh, 0.0, 1.0)) if pv_kwh > EPS else 0.0
        ssi = float(np.clip(locally_supplied / load_kwh, 0.0, 1.0)) if load_kwh > EPS else 0.0

        rows.append({
            "household": user_names[u],
            "user_name": user_names[u],
            "boiler_tariff": boiler_tariff,
            "has_pv": bool(pv_kwh > EPS),
            "has_bess": bool(bess_enabled[u] and size_bess[u] > EPS),
            "Pmax_kw": float(np.max(total_load[:, u])) if T else 0.0,
            "base_load_kwh": float(p_ue[:, u].sum() * dt),
            "boiler_kwh": float(p_el_heater[:, u].sum() * dt),
            "total_load_kwh": load_kwh,
            "pv_gen_kwh": pv_kwh,
            "pv_kwh": pv_kwh,
            "pv_to_load_kwh": float(p_pv_load_v[:, u].sum() * dt),
            "pv_to_base_kwh": float(p_pv_ue_fixed[:, u].sum() * dt),
            "pv_to_boiler_kwh": float(p_pv_elh_fixed[:, u].sum() * dt),
            "pv_to_bess_kwh": float(p_pv_bess_v[:, u].sum() * dt),
            "shared_out_kwh": float(p_pv_shared_v[:, u].sum() * dt),
            "pv_to_grid_kwh": float(p_pv_grid_v[:, u].sum() * dt),
            "grid_export_kwh": float(p_grid_export_v[:, u].sum() * dt),
            "bess_to_load_kwh": float(p_bess_load_v[:, u].sum() * dt),
            "bess_to_base_kwh": float(p_bess_ue_v[:, u].sum() * dt),
            "bess_to_boiler_kwh": float(p_bess_elh_v[:, u].sum() * dt),
            "shared_in_kwh": float(p_shared_in_v[:, u].sum() * dt),
            "shared_to_base_kwh": float(p_shared_ue_v[:, u].sum() * dt),
            "shared_to_boiler_kwh": float(p_shared_elh_v[:, u].sum() * dt),
            "grid_to_load_kwh": float(p_grid_load_v[:, u].sum() * dt),
            "grid_to_base_kwh": float(p_grid_ue_v[:, u].sum() * dt),
            "grid_to_boiler_kwh": float(p_grid_elh_v[:, u].sum() * dt),
            "grid_to_bess_kwh": float(p_grid_bess_v[:, u].sum() * dt),
            "grid_import_total_kwh": float(p_grid_import_v[:, u].sum() * dt),
            "grid_import_a_low_kwh": a_low,
            "grid_import_a_high_kwh": a_high,
            "grid_import_b_low_kwh": b_low,
            "grid_import_b_high_kwh": b_high,
            "grid_import_low_kwh": a_low + b_low,
            "grid_import_high_kwh": a_high + b_high,
            "final_bess_energy_kwh": float(variable_value(batteries[u].state[T])),
            "self_consumed_pv_kwh": self_consumed_pv,
            "locally_supplied_load_kwh": locally_supplied,
            "self_consumption_ratio": sci,
            "self_sufficiency_ratio": ssi,
            "SCI": sci,
            "SSI": ssi,
            "import_cost_a_ft": float(import_cost_a),
            "import_cost_b_ft": float(import_cost_b),
            "import_cost_ft": float(import_cost),
            "grid_export_revenue_ft": float(grid_export_revenue),
            "export_revenue_ft": float(grid_export_revenue),
            "shared_purchase_energy_cost_ft": float(settlement["buyer_energy_cost_ft"][u]),
            "shared_purchase_rhd_ft": float(settlement["buyer_rhd_ft"][u]),
            "shared_purchase_cost_ft": float(shared_purchase_cost),
            "shared_buyer_low_kwh": float(settlement["buyer_shared_low_kwh"][u]),
            "shared_buyer_high_kwh": float(settlement["buyer_shared_high_kwh"][u]),
            "shared_revenue_ft": float(shared_revenue),
            "cash_out_ft": float(import_cost + shared_purchase_cost),
            "cash_in_ft": float(grid_export_revenue + shared_revenue),
            "brt_bill_ft": float(brt_bill),
            "status": status_str,
        })

    per_user_df = pd.DataFrame(rows)
    total_load_kwh = float(total_load.sum() * dt)
    pv_total_kwh = float(p_pv.sum() * dt)
    self_consumed_total = float((p_pv_load_v + p_pv_bess_v + p_pv_shared_v).sum() * dt)
    supplied_total = float((p_pv_load_v + p_bess_load_v + p_shared_in_v).sum() * dt)

    summary = {
        "status": status_str,
        "solver": solve_result.solver,
        "objective": objective,
        "objective_value": variable_value(prob.objective),
        "n_users": int(U),
        "T": int(T),
        "dt": float(dt),
        "boiler_tariff": boiler_tariff,
        "sharing_mode": sharing_mode,
        "pairing_mode": pairing_mode,
        "bess_min_mode_steps": int(bess_min_mode_steps),
        "bess_terminal": "cyclic",
        "allow_grid_charge_for_min_soc": allow_grid_charge_for_min_soc,
        "bess_min_mode_minutes": float(bess_min_mode_steps) * dt * 60.0,
        "pv_user_count": int((p_pv.sum(axis=0) * dt > EPS).sum()),
        "bess_enabled_user_count": int(np.asarray(bess_enabled, dtype=bool).sum()),
        "total_base_load_kwh": float(p_ue.sum() * dt),
        "total_boiler_kwh": float(p_el_heater.sum() * dt),
        "total_load_kwh": total_load_kwh,
        "total_pv_gen_kwh": pv_total_kwh,
        "total_pv_to_load_kwh": float(p_pv_load_v.sum() * dt),
        "total_pv_to_base_kwh": float(p_pv_ue_fixed.sum() * dt),
        "total_pv_to_boiler_kwh": float(p_pv_elh_fixed.sum() * dt),
        "total_pv_to_bess_kwh": float(p_pv_bess_v.sum() * dt),
        "total_shared_out_kwh": float(p_pv_shared_v.sum() * dt),
        "total_pv_to_grid_kwh": float(p_pv_grid_v.sum() * dt),
        "total_grid_export_kwh": float(p_grid_export_v.sum() * dt),
        "total_bess_to_load_kwh": float(p_bess_load_v.sum() * dt),
        "total_bess_to_base_kwh": float(p_bess_ue_v.sum() * dt),
        "total_bess_to_boiler_kwh": float(p_bess_elh_v.sum() * dt),
        "total_shared_in_kwh": float(p_shared_in_v.sum() * dt),
        "total_shared_to_base_kwh": float(p_shared_ue_v.sum() * dt),
        "total_shared_to_boiler_kwh": float(p_shared_elh_v.sum() * dt),
        "total_grid_to_load_kwh": float(p_grid_load_v.sum() * dt),
        "total_grid_to_base_kwh": float(p_grid_ue_v.sum() * dt),
        "total_grid_to_boiler_kwh": float(p_grid_elh_v.sum() * dt),
        "total_grid_to_bess_kwh": float(p_grid_bess_v.sum() * dt),
        "total_grid_import_kwh": float(p_grid_import_v.sum() * dt),
        "total_import_cost_a_ft": float(per_user_df["import_cost_a_ft"].sum()),
        "total_import_cost_b_ft": float(per_user_df["import_cost_b_ft"].sum()),
        "total_import_cost_ft": float(per_user_df["import_cost_ft"].sum()),
        "total_shared_purchase_energy_cost_ft": float(per_user_df["shared_purchase_energy_cost_ft"].sum()),
        "total_shared_purchase_rhd_ft": float(per_user_df["shared_purchase_rhd_ft"].sum()),
        "total_shared_purchase_cost_ft": float(per_user_df["shared_purchase_cost_ft"].sum()),
        "total_shared_revenue_ft": float(per_user_df["shared_revenue_ft"].sum()),
        "total_export_revenue_ft": float(per_user_df["export_revenue_ft"].sum()),
        "total_brt_bill_ft": float(per_user_df["brt_bill_ft"].sum()),
        "self_consumed_pv_kwh": self_consumed_total,
        "locally_supplied_load_kwh": supplied_total,
        "self_consumption_ratio": float(np.clip(self_consumed_total / pv_total_kwh, 0.0, 1.0)) if pv_total_kwh > EPS else 0.0,
        "self_sufficiency_ratio": float(np.clip(supplied_total / total_load_kwh, 0.0, 1.0)) if total_load_kwh > EPS else 0.0,
        "shared_buyer_low_ft_per_kwh": SHARED_BUYER_LOW_FT_PER_KWH,
        "shared_buyer_high_ft_per_kwh": SHARED_BUYER_HIGH_FT_PER_KWH,
        "shared_rhd_ft_per_kwh": SHARED_RHD_FT_PER_KWH,
    }
    summary["SCI"] = summary["self_consumption_ratio"]
    summary["SSI"] = summary["self_sufficiency_ratio"]

    timeseries = {
        "p_total_load": total_load,
        "p_ue": p_ue,
        "p_el_heater": p_el_heater,
        "p_pv": p_pv,
        "p_pv_load": p_pv_load_v,
        "p_pv_ue": p_pv_ue_fixed,
        "p_pv_elh": p_pv_elh_fixed,
        "p_pv_bess": p_pv_bess_v,
        "p_pv_shared": p_pv_shared_v,
        "p_pv_grid": p_pv_grid_v,
        "p_bess_load": p_bess_load_v,
        "p_bess_ue": p_bess_ue_v,
        "p_bess_elh": p_bess_elh_v,
        "p_shared_in": p_shared_in_v,
        "p_shared_ue": p_shared_ue_v,
        "p_shared_elh": p_shared_elh_v,
        "p_shared_out": p_pv_shared_v,
        "p_grid_load": p_grid_load_v,
        "p_grid_ue": p_grid_ue_v,
        "p_grid_elh": p_grid_elh_v,
        "p_grid_bess": p_grid_bess_v,
        "p_bess_in": p_bess_in_v,
        "p_bess_out": p_bess_out_v,
        "p_grid_import": p_grid_import_v,
        "p_grid_export": p_grid_export_v,
        "e_bess": e_bess_v,
        "e_bess_boundary": np.column_stack([extract_vector(b.state) for b in batteries]),
        "d_bess_ch": d_bess_ch_v,
        "d_bess_dis": d_bess_dis_v,
        "d_grid": d_grid_v,
    }

    from Utility.result_schema import add_bess_states, add_bess_flows
    add_bess_states(timeseries, timeseries["e_bess_boundary"])
    add_bess_flows(timeseries, dt)
    return {
        "summary": summary,
        "per_user_df": per_user_df,
        "timeseries": timeseries,
        "shared_pair_kwh": settlement["pair_kwh"],
        "shared_pair_energy_payment_ft": settlement["pair_energy_payment_ft"],
        "user_names": list(user_names),
        "bess_enabled": np.asarray(bess_enabled, dtype=bool),
        "status": status_str,
        "solver": solve_result.solver,
        "dt_hours": float(dt),
    }


def save_disaggregated_opt_results(result: dict, out_dir: str | Path, *, save_user_timeseries: bool = True) -> None:
    from Utility.result_schema import save_bess_result
    save_bess_result(result["timeseries"], out_dir, result["dt_hours"], result["user_names"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    user_names = result["user_names"]
    per_user_df: pd.DataFrame = result["per_user_df"]
    summary: dict = result["summary"]
    ts: dict[str, np.ndarray] = result["timeseries"]

    per_user_df.to_csv(out / "household_summary.csv", index=False)
    per_user_df.to_csv(out / "per_user_summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def save_matrix(key: str, filename: str | None = None) -> None:
        pd.DataFrame(ts[key], columns=user_names).to_csv(out / (filename or f"{key}.csv"), index=False)

    for key in [
        "p_total_load", "p_ue", "p_el_heater", "p_pv", "p_pv_load", "p_pv_ue", "p_pv_elh",
        "p_pv_bess", "p_pv_shared", "p_pv_grid", "p_bess_load", "p_bess_ue", "p_bess_elh",
        "p_shared_in", "p_shared_ue", "p_shared_elh", "p_shared_out", "p_grid_load", "p_grid_ue",
        "p_grid_elh", "p_grid_bess", "p_grid_import", "p_grid_export", "e_bess", "d_bess_ch",
        "d_bess_dis", "d_grid", "p_bess_in", "p_bess_out", "e_bess_boundary",
        "e_bess_start", "e_bess_end",
    ]:
        save_matrix(key)

    for key in (
        "p_individual_settlement_grid_import_a",
        "p_individual_settlement_grid_import_b",
        "p_individual_settlement_grid_export",
    ):
        if key in ts:
            save_matrix(key)

    pd.DataFrame(result["shared_pair_kwh"], index=user_names, columns=user_names).to_csv(
        out / "shared_pair_kwh_seller_x_buyer.csv"
    )
    pd.DataFrame(result["shared_pair_energy_payment_ft"], index=user_names, columns=user_names).to_csv(
        out / "shared_pair_energy_payment_ft_seller_x_buyer.csv"
    )

    community_ts = pd.DataFrame({
        "p_total_load_sum": ts["p_total_load"].sum(axis=1),
        "p_ue_sum": ts["p_ue"].sum(axis=1),
        "p_el_heater_sum": ts["p_el_heater"].sum(axis=1),
        "p_pv_sum": ts["p_pv"].sum(axis=1),
        "p_pv_load_sum": ts["p_pv_load"].sum(axis=1),
        "p_pv_bess_sum": ts["p_pv_bess"].sum(axis=1),
        "p_pv_shared_sum": ts["p_pv_shared"].sum(axis=1),
        "p_pv_grid_sum": ts["p_pv_grid"].sum(axis=1),
        "p_bess_load_sum": ts["p_bess_load"].sum(axis=1),
        "p_shared_in_sum": ts["p_shared_in"].sum(axis=1),
        "p_shared_out_sum": ts["p_shared_out"].sum(axis=1),
        "p_grid_load_sum": ts["p_grid_load"].sum(axis=1),
        "p_grid_bess_sum": ts["p_grid_bess"].sum(axis=1),
        "p_grid_import_sum": ts["p_grid_import"].sum(axis=1),
        "p_grid_export_sum": ts["p_grid_export"].sum(axis=1),
        "e_bess_sum": ts["e_bess"].sum(axis=1),
    })
    if "p_individual_settlement_grid_import_a" in ts:
        community_ts["p_individual_settlement_grid_import_a_sum"] = ts[
            "p_individual_settlement_grid_import_a"
        ].sum(axis=1)
        community_ts["p_individual_settlement_grid_import_b_sum"] = ts[
            "p_individual_settlement_grid_import_b"
        ].sum(axis=1)
        community_ts["p_individual_settlement_grid_export_sum"] = ts[
            "p_individual_settlement_grid_export"
        ].sum(axis=1)
    community_ts.to_csv(out / "community_timeseries.csv", index=False)

    if save_user_timeseries:
        for i, name in enumerate(user_names):
            safe_name = str(name).replace("/", "_").replace("\\", "_")
            user_df = pd.DataFrame({key: arr[:, i] for key, arr in ts.items()
                                    if arr.ndim == 2 and key != "e_bess_boundary"})
            user_df.to_csv(out / f"timeseries_{safe_name}.csv", index=False)
