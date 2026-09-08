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

DT_DEFAULT = 0.25
EPS = 1e-9

# Alap A/B és PV árak, az individual modellhez igazítva.
LOW_TARIFF_LIMIT_KWH = 2523.0
PRICE_GRID_A_LOW = 36.0
PRICE_GRID_A_HIGH = 71.0
PRICE_GRID_B_LOW = 23.0
PRICE_GRID_B_HIGH = 61.0
PRICE_PV_GRID = 5.0

# Energiaközösségi megosztás elszámolása.
# Nincs külön eladói megosztási keret: a megosztott energia a vevő
# ugyanazon éves A-sávját fogyasztja, mint a hálózati A-vételezés.
SHARED_BUYER_LOW_FT_PER_KWH = 5.0
SHARED_BUYER_HIGH_FT_PER_KWH = 21.0
SHARED_RHD_FT_PER_KWH = 31.0

Objective = Literal["bill", "grid"]
SharingMode = Literal["proportional", "equal"]
PairingMode = SharingMode


def _as_2d_power(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} csak 2D tömb lehet, kapott dimenzió: {arr.ndim}")
    return np.maximum(arr, 0.0)


def _val(x) -> float:
    v = pulp.value(x)
    return 0.0 if v is None else float(v)


def _var_matrix(name: str, T: int, U: int, low: float = 0.0) -> list[list[pulp.LpVariable]]:
    return [[pulp.LpVariable(f"{name}_{t}_{u}", lowBound=low) for u in range(U)] for t in range(T)]


def _bin_matrix(name: str, T: int, U: int, run_lp: bool) -> list[list[pulp.LpVariable | float]]:
    if run_lp:
        return [[0.0 for _ in range(U)] for _ in range(T)]
    return [[pulp.LpVariable(f"{name}_{t}_{u}", cat=pulp.LpBinary) for u in range(U)] for t in range(T)]


def _extract_matrix(var: list[list[pulp.LpVariable | float]], T: int, U: int) -> np.ndarray:
    out = np.zeros((T, U), dtype=float)
    for t in range(T):
        for u in range(U):
            out[t, u] = _val(var[t][u]) if hasattr(var[t][u], "name") else float(var[t][u])
    return out


def _split_low_high_step(
    e_kwh: float,
    remaining_low_kwh: float,
) -> tuple[float, float, float]:
    """Egy időlépés energiájának bontása a vevő megmaradó kedvezményes sávja szerint."""
    e = max(float(e_kwh), 0.0)
    low = min(e, max(float(remaining_low_kwh), 0.0))
    high = max(e - low, 0.0)
    return low, high, max(float(remaining_low_kwh) - low, 0.0)

def _allocate_equal_no_redistribution(demand: np.ndarray, total_share: float) -> np.ndarray:
    """
    Equal megosztás újraosztás nélkül.

    Minden aktív vevő egyszeri azonos jogosultsági kvótát kap:
        quota = total_share / n_active.
    Ha egy vevő ennél kevesebbet tud felvenni, a maradék nem kerül át
    másik vevőhöz, hanem nem lesz shared energia.
    """
    d = np.maximum(np.asarray(demand, dtype=float), 0.0)
    total_share = min(max(float(total_share), 0.0), float(d.sum()))
    out = np.zeros_like(d)

    if total_share <= EPS or d.sum() <= EPS:
        return out

    active = d > EPS
    n = int(active.sum())
    if n == 0:
        return out

    quota = total_share / n
    out[active] = np.minimum(d[active], quota)
    return out



def _deterministic_shared_from_residuals(
    residual_surplus: np.ndarray,
    residual_deficit_ue: np.ndarray,
    residual_deficit_elh: np.ndarray,
    *,
    boiler_tariff: str,
    sharing_mode: SharingMode,
) -> dict:
    """
    Noopt-tal egyező determinisztikus közösségi megosztás a BESS utáni
    maradék PV-többlet és maradék, sharedre jogosult hiány alapján.

    Fontos:
      - B tarifán a bojler nem jogosult sharedre, ezért residual_deficit_elh
        teljesen grid_elh marad.
      - A kapott shared energiát háztartáson belül először az alapfogyasztásra,
        majd A tarifán a bojlerre könyveli, ugyanúgy, mint a nonopt modell.
    """
    surplus = np.maximum(np.asarray(residual_surplus, dtype=float), 0.0)
    def_ue = np.maximum(np.asarray(residual_deficit_ue, dtype=float), 0.0)
    def_elh_raw = np.maximum(np.asarray(residual_deficit_elh, dtype=float), 0.0)

    T, U = surplus.shape
    if def_ue.shape != (T, U) or def_elh_raw.shape != (T, U):
        raise ValueError("A determinisztikus shared bemeneti mátrixainak alakja nem egyezik")

    tariff = str(boiler_tariff).upper().strip()
    if tariff == "A":
        eligible_elh = def_elh_raw.copy()
    elif tariff == "B":
        eligible_elh = np.zeros_like(def_elh_raw)
    else:
        raise ValueError("boiler_tariff csak 'A' vagy 'B' lehet")

    eligible_def = def_ue + eligible_elh

    p_shared_total = np.zeros((T, U), dtype=float)
    p_shared_ue = np.zeros((T, U), dtype=float)
    p_shared_elh = np.zeros((T, U), dtype=float)
    p_shared_out = np.zeros((T, U), dtype=float)
    p_pv_grid = np.zeros((T, U), dtype=float)
    p_grid_ue = np.zeros((T, U), dtype=float)
    p_grid_elh = np.zeros((T, U), dtype=float)

    for t in range(T):
        S = float(surplus[t, :].sum())
        D = float(eligible_def[t, :].sum())
        total_share = min(S, D)

        if total_share > EPS:
            if sharing_mode == "proportional":
                shared_in_t = eligible_def[t, :] * (total_share / D) if D > EPS else np.zeros(U, dtype=float)
            elif sharing_mode == "equal":
                shared_in_t = _allocate_equal_no_redistribution(eligible_def[t, :], total_share)
            else:
                raise ValueError("sharing_mode csak 'proportional' vagy 'equal' lehet")
            actual_share = float(shared_in_t.sum())
            # Equal módban a kvótából fel nem vett rész nem osztódik újra,
            # ezért csak a ténylegesen felvett shared_in mennyiség kerül
            # eladói shared_out-ba; a maradék PV-többlet grid export lesz.
            shared_out_t = surplus[t, :] * (actual_share / S) if S > EPS and actual_share > EPS else np.zeros(U, dtype=float)
        else:
            shared_in_t = np.zeros(U, dtype=float)
            shared_out_t = np.zeros(U, dtype=float)

        # Háztartáson belüli sorrend: alapfogyasztás előbb, bojler utána.
        shared_ue_t = np.minimum(def_ue[t, :], shared_in_t)
        shared_elh_t = np.minimum(eligible_elh[t, :], np.maximum(shared_in_t - shared_ue_t, 0.0))

        p_shared_total[t, :] = shared_in_t
        p_shared_ue[t, :] = shared_ue_t
        p_shared_elh[t, :] = shared_elh_t
        p_shared_out[t, :] = shared_out_t
        p_pv_grid[t, :] = np.maximum(surplus[t, :] - shared_out_t, 0.0)
        p_grid_ue[t, :] = np.maximum(def_ue[t, :] - shared_ue_t, 0.0)
        if tariff == "A":
            p_grid_elh[t, :] = np.maximum(def_elh_raw[t, :] - shared_elh_t, 0.0)
        else:
            p_grid_elh[t, :] = def_elh_raw[t, :]

    return {
        "p_shared_total": p_shared_total,
        "p_shared_ue": p_shared_ue,
        "p_shared_elh": p_shared_elh,
        "p_shared_out": p_shared_out,
        "p_pv_grid": p_pv_grid,
        "p_grid_ue": p_grid_ue,
        "p_grid_elh": p_grid_elh,
    }


def _allocate_seller_to_buyers(
    seller_energy_kwh: float,
    remaining_buyer_need_kwh: np.ndarray,
    pairing_mode: PairingMode,
) -> np.ndarray:
    """Egy eladó energiájának párosítása a vevőkkel úgy, hogy a vevői oszlopösszegek megmaradjanak."""
    need = np.maximum(np.asarray(remaining_buyer_need_kwh, dtype=float), 0.0)
    e = min(max(float(seller_energy_kwh), 0.0), float(need.sum()))
    out = np.zeros_like(need)
    if e <= EPS or need.sum() <= EPS:
        return out

    if pairing_mode == "proportional":
        return need * (e / need.sum())

    if pairing_mode == "equal":
        # Azonos kWh-kvótát próbál adni az aktív vevőknek, de sosem lépi túl
        # az adott vevő még hátralévő shared_in mennyiségét.
        remaining = e
        while remaining > EPS:
            room = need - out
            active = room > EPS
            n = int(active.sum())
            if n == 0:
                break
            quota = remaining / n
            take = np.minimum(room[active], quota)
            out[active] += take
            remaining -= float(take.sum())
        return out

    raise ValueError("pairing_mode csak 'proportional' vagy 'equal' lehet")


def settle_shared_payments(
    p_shared_in: np.ndarray,
    p_shared_out: np.ndarray,
    p_grid_import_a: np.ndarray,
    dt: float = DT_DEFAULT,
    pairing_mode: PairingMode = "proportional",
    buyer_low_limit_kwh: float = LOW_TARIFF_LIMIT_KWH,
) -> dict:
    """
    Eladó-vevő pénzügyi elszámolás a megoldott idősorok alapján.

    Fontos: a 2523 kWh-os kedvezményes sáv a VEVŐ éves sávja.
    A megosztott energia nem kap külön eladói keretet, hanem ugyanazt a
    buyer_low_remaining keretet fogyasztja, mint az A-tarifás hálózati import.

    Időlépésen belül elszámolási sorrend:
        1) közösségből vett energia,
        2) A-tarifás hálózati import.

    Tehát a shared_in fogyasztja először a vevő 2523 kWh-os
    kedvezményes A-sávját, és csak a maradék keret jut grid_import_a-ra.

    Ez biztosítja, hogy a megosztott energia ára:
        low sávban:  5 + 31 Ft/kWh,
        high sávban: 21 + 31 Ft/kWh,
    az eladó pedig ugyanebből az energiadíjból kap 5 vagy 21 Ft/kWh-t.
    """
    p_shared_in = _as_2d_power(p_shared_in, "p_shared_in")
    p_shared_out = _as_2d_power(p_shared_out, "p_shared_out")
    p_grid_import_a = _as_2d_power(p_grid_import_a, "p_grid_import_a")
    if p_shared_in.shape != p_shared_out.shape:
        raise ValueError("p_shared_in és p_shared_out alakja nem egyezik")
    if p_shared_in.shape != p_grid_import_a.shape:
        raise ValueError("p_grid_import_a alakja nem egyezik a megosztási idősorokkal")

    T, U = p_shared_in.shape
    e_shared_in = p_shared_in * float(dt)
    e_shared_out = p_shared_out * float(dt)
    e_grid_import_a = p_grid_import_a * float(dt)

    pair_kwh = np.zeros((U, U), dtype=float)  # seller x buyer
    pair_energy_payment_ft = np.zeros((U, U), dtype=float)

    buyer_low_remaining = np.full(U, float(buyer_low_limit_kwh), dtype=float)
    buyer_shared_low_kwh = np.zeros(U, dtype=float)
    buyer_shared_high_kwh = np.zeros(U, dtype=float)
    buyer_grid_a_low_kwh = np.zeros(U, dtype=float)
    buyer_grid_a_high_kwh = np.zeros(U, dtype=float)

    seller_revenue_ft = np.zeros(U, dtype=float)
    buyer_energy_cost_ft = np.zeros(U, dtype=float)
    buyer_rhd_ft = e_shared_in.sum(axis=0) * SHARED_RHD_FT_PER_KWH

    for t in range(T):
        sin = e_shared_in[t, :]
        sout = e_shared_out[t, :]
        total_share = min(float(sin.sum()), float(sout.sum()))

        # Vevői közös sáv fogyasztása: először shared, utána A hálózati import.
        # Így a shared_in kap elsőbbséget a 2523 kWh-os kedvezményes A-sávban.
        buyer_shared_low_t = np.zeros(U, dtype=float)
        buyer_shared_high_t = np.zeros(U, dtype=float)
        for b in range(U):
            low_s, high_s, rem = _split_low_high_step(sin[b], buyer_low_remaining[b])
            buyer_shared_low_t[b] = low_s
            buyer_shared_high_t[b] = high_s
            buyer_shared_low_kwh[b] += low_s
            buyer_shared_high_kwh[b] += high_s

            low_g, high_g, rem = _split_low_high_step(e_grid_import_a[t, b], rem)
            buyer_grid_a_low_kwh[b] += low_g
            buyer_grid_a_high_kwh[b] += high_g
            buyer_low_remaining[b] = rem

        if total_share <= EPS:
            continue

        sellers = np.flatnonzero(sout > EPS)
        buyers = np.flatnonzero(sin > EPS)
        if len(sellers) == 0 or len(buyers) == 0:
            continue

        # A vevő által fizetendő és az eladónak továbbadandó energiadíj
        # a vevői sáv szerint: 5 Ft/kWh low, 21 Ft/kWh high.
        buyer_energy_cost_ft[buyers] += (
            buyer_shared_low_t[buyers] * SHARED_BUYER_LOW_FT_PER_KWH
            + buyer_shared_high_t[buyers] * SHARED_BUYER_HIGH_FT_PER_KWH
        )
        buyer_avg_energy_rate = np.divide(
            buyer_shared_low_t * SHARED_BUYER_LOW_FT_PER_KWH
            + buyer_shared_high_t * SHARED_BUYER_HIGH_FT_PER_KWH,
            sin,
            out=np.zeros(U, dtype=float),
            where=sin > EPS,
        )

        remaining_buyers = sin[buyers].copy()
        remaining_buyer_rates = buyer_avg_energy_rate[buyers].copy()
        for s in sellers:
            alloc_to_buyers = _allocate_seller_to_buyers(
                seller_energy_kwh=float(sout[s]),
                remaining_buyer_need_kwh=remaining_buyers,
                pairing_mode=pairing_mode,
            )
            if alloc_to_buyers.sum() <= EPS:
                continue
            pair_kwh[s, buyers] += alloc_to_buyers
            pair_payment = alloc_to_buyers * remaining_buyer_rates
            pair_energy_payment_ft[s, buyers] += pair_payment
            seller_revenue_ft[s] += float(pair_payment.sum())
            remaining_buyers = np.maximum(remaining_buyers - alloc_to_buyers, 0.0)

    return {
        "pair_kwh": pair_kwh,
        "pair_energy_payment_ft": pair_energy_payment_ft,
        "buyer_shared_low_kwh": buyer_shared_low_kwh,
        "buyer_shared_high_kwh": buyer_shared_high_kwh,
        "buyer_grid_a_low_kwh": buyer_grid_a_low_kwh,
        "buyer_grid_a_high_kwh": buyer_grid_a_high_kwh,
        "seller_revenue_ft": seller_revenue_ft,
        "buyer_energy_cost_ft": buyer_energy_cost_ft,
        "buyer_rhd_ft": buyer_rhd_ft,
        "buyer_total_shared_cost_ft": buyer_energy_cost_ft + buyer_rhd_ft,
    }

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
    p_pv_bess = _var_matrix("p_pv_bess", T, U)
    p_pv_shared = _var_matrix("p_pv_shared", T, U)
    p_pv_grid = _var_matrix("p_pv_grid", T, U)

    p_bess_ue = _var_matrix("p_bess_ue", T, U)
    p_bess_elh = _var_matrix("p_bess_elh", T, U)
    p_shared_ue = _var_matrix("p_shared_ue", T, U)
    p_shared_elh = _var_matrix("p_shared_elh", T, U)
    p_grid_ue = _var_matrix("p_grid_ue", T, U)
    p_grid_elh = _var_matrix("p_grid_elh", T, U)

    p_bess_in = _var_matrix("p_bess_in", T, U)
    p_bess_out = _var_matrix("p_bess_out", T, U)
    p_grid_bess = _var_matrix("p_grid_bess", T, U)

    p_grid_import = _var_matrix("p_grid_import", T, U)
    p_grid_export = _var_matrix("p_grid_export", T, U)
    d_grid = _bin_matrix("d_grid", T, U, run_lp)
    d_bess_ch = _bin_matrix("d_bess_ch", T, U, run_lp)
    d_bess_dis = _bin_matrix("d_bess_dis", T, U, run_lp)
    d_grid_bess_guard = _bin_matrix("d_grid_bess_guard", T, U, run_lp)

    e_bess: list[list[pulp.LpVariable | float]] = [[0.0 for _ in user_set] for _ in time_set]
    e_bess_pre: list[list[pulp.LpVariable | float]] = [[0.0 for _ in user_set] for _ in time_set]
    for t in time_set:
        for u in user_set:
            if bess_enabled[u] and size_bess[u] > EPS:
                e_bess[t][u] = pulp.LpVariable(
                    f"e_bess_{t}_{u}",
                    lowBound=float(size_bess[u]) * float(soc_bess_min[u]),
                    upBound=float(size_bess[u]) * float(soc_bess_max[u]),
                )
                e_bess_pre[t][u] = pulp.LpVariable(
                    f"e_bess_pre_{t}_{u}",
                    lowBound=0,
                    upBound=float(size_bess[u]) * float(soc_bess_max[u]),
                )

    e_grid_a_low_step = _var_matrix("e_grid_a_low_step", T, U)
    e_grid_a_high_step = _var_matrix("e_grid_a_high_step", T, U)
    e_shared_a_low_step = _var_matrix("e_shared_a_low_step", T, U)
    e_shared_a_high_step = _var_matrix("e_shared_a_high_step", T, U)
    rem_a_low = [[pulp.LpVariable(f"rem_a_low_{t}_{u}", lowBound=0, upBound=grid_a_low_cap_kwh) for u in user_set] for t in time_set]
    # Bináris a shared-first A-sávhoz:
    # e_shared_a_low_step = min(e_shared_a_t, előző maradék A-sáv).
    # LP relaxációban ez lazított, MILP-ben pontos.
    d_shared_low_full = _bin_matrix("d_shared_low_full", T, U, run_lp)

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
    no_bess_active = not bool(np.any(np.asarray(bess_enabled, dtype=bool) & (size_bess > EPS)))

    # --- Korlátok ---
    for t in time_set:
        det_shared_ue_const = det_shared_elh_const = None
        det_shared_out_const = det_pv_grid_const = None
        det_grid_ue_const = det_grid_elh_const = None
        if no_bess_active:
            # BESS nélküli esetben a shared kiosztás teljesen determinisztikus,
            # ezért az optimalizált modell pontosan a nonopt megosztási szabályát kapja.
            det = _deterministic_shared_from_residuals(
                residual_surplus=p_surplus[t:t + 1, :],
                residual_deficit_ue=p_deficit_ue[t:t + 1, :],
                residual_deficit_elh=p_deficit_elh[t:t + 1, :],
                boiler_tariff=boiler_tariff,
                sharing_mode=sharing_mode,
            )
            det_shared_ue_const = det["p_shared_ue"][0, :]
            det_shared_elh_const = det["p_shared_elh"][0, :]
            det_shared_out_const = det["p_shared_out"][0, :]
            det_pv_grid_const = det["p_pv_grid"][0, :]
            det_grid_ue_const = det["p_grid_ue"][0, :]
            det_grid_elh_const = det["p_grid_elh"][0, :]

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
                # B tarifás bojler külön kör: saját PV-t, saját BESS-t és
                # közösségi megosztott energiát sem kap. A teljes bojlerhiány
                # B tarifás hálózati importként jelenik meg.
                prob += p_bess_elh[t][u] == 0, f"no_bess_to_boiler_B_{t}_{u}"
                prob += p_shared_elh[t][u] == 0, f"no_shared_to_boiler_B_{t}_{u}"

            p_shared_total_tu = p_shared_ue[t][u] + p_shared_elh[t][u]

            if no_bess_active:
                # Exact nonopt shared BESS nélkül: a megosztás mennyisége,
                # eladói oldala és vevői kiosztása is determinisztikus.
                prob += p_pv_shared[t][u] == float(det_shared_out_const[u]), f"det_pv_shared_no_bess_{t}_{u}"
                prob += p_pv_grid[t][u] == float(det_pv_grid_const[u]), f"det_pv_grid_no_bess_{t}_{u}"
                prob += p_shared_ue[t][u] == float(det_shared_ue_const[u]), f"det_shared_ue_no_bess_{t}_{u}"
                prob += p_shared_elh[t][u] == float(det_shared_elh_const[u]), f"det_shared_elh_no_bess_{t}_{u}"
                prob += p_grid_ue[t][u] == float(det_grid_ue_const[u]), f"det_grid_ue_no_bess_{t}_{u}"
                prob += p_grid_elh[t][u] == float(det_grid_elh_const[u]), f"det_grid_elh_no_bess_{t}_{u}"
            else:
                # BESS-es esetben a BESS változtatja meg a shared előtti maradék
                # többletet/hiányt. Az exact BESS utáni proportional/equal kiosztás
                # bilineáris lenne, ezért a MILP-ben lineáris reprezentáció marad,
                # majd az eredményeket a végén determinisztikusan újraszámoljuk.
                if boiler_tariff == "A":
                    eligible_deficit_ue = float(p_deficit_ue[t, u])
                    eligible_deficit_elh = float(p_deficit_elh[t, u])
                else:
                    eligible_deficit_ue = float(p_deficit_ue[t, u])
                    eligible_deficit_elh = 0.0

                deficit_total_const = eligible_deficit_ue + eligible_deficit_elh
                if deficit_total_const <= EPS:
                    prob += p_shared_total_tu == 0, f"no_shared_without_deficit_{t}_{u}"
                elif sharing_mode == "proportional":
                    prob += p_shared_total_tu == share_alpha[t] * deficit_total_const, f"shared_proportional_{t}_{u}"
                    prob += p_shared_ue[t][u] == share_alpha[t] * eligible_deficit_ue, f"shared_ue_prop_{t}_{u}"
                    prob += p_shared_elh[t][u] == share_alpha[t] * eligible_deficit_elh, f"shared_elh_prop_{t}_{u}"
                else:
                    # Equal mód: minden aktív, sharedre jogosult hiányos felhasználó
                    # ugyanakkora kvótáig kaphat. B tarifás bojler nem jogosult.
                    prob += p_shared_total_tu <= share_quota[t], f"shared_equal_quota_{t}_{u}"
                    prob += p_shared_total_tu <= deficit_total_const, f"shared_equal_deficit_cap_{t}_{u}"
                    prob += p_shared_ue[t][u] == p_shared_total_tu * float(eligible_deficit_ue / deficit_total_const), f"shared_ue_equal_split_{t}_{u}"
                    prob += p_shared_elh[t][u] == p_shared_total_tu * float(eligible_deficit_elh / deficit_total_const), f"shared_elh_equal_split_{t}_{u}"

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

            if not (bess_enabled[u] and size_bess[u] > EPS):
                prob += p_pv_bess[t][u] == 0, f"no_bess_pv_bess_{t}_{u}"
                prob += p_bess_ue[t][u] == 0, f"no_bess_ue_{t}_{u}"
                prob += p_bess_elh[t][u] == 0, f"no_bess_elh_{t}_{u}"
                prob += p_bess_in[t][u] == 0, f"no_bess_in_{t}_{u}"
                prob += p_bess_out[t][u] == 0, f"no_bess_out_{t}_{u}"
                prob += p_grid_bess[t][u] == 0, f"no_grid_bess_{t}_{u}"
            else:
                prob += (
                    p_bess_in[t][u] == p_pv_bess[t][u] + p_grid_bess[t][u]
                ), f"bess_in_def_{t}_{u}"
                prob += (
                    p_bess_out[t][u] == p_bess_ue[t][u] + p_bess_elh[t][u]
                ), f"bess_out_def_{t}_{u}"

                if t < T - 1:
                    soc_min_abs = float(size_bess[u]) * float(soc_bess_min[u])
                    M_soc = max(float(size_bess[u]), 1.0)
                    prob += e_bess_pre[t + 1][u] == (
                        e_bess[t][u] * float(eta_bess_stor[u])
                        + dt * (
                            p_pv_bess[t][u] * float(eta_bess_in[u])
                            - p_bess_out[t][u] / max(float(eta_bess_out[u]), EPS)
                        )
                    ), f"bess_pre_dyn_{t}_{u}"

                    prob += e_bess[t + 1][u] == (
                        e_bess_pre[t + 1][u]
                        + dt * float(eta_bess_in[u]) * p_grid_bess[t][u]
                    ), f"bess_dyn_with_grid_guard_{t}_{u}"

                    if not run_lp:
                        grid_added = dt * float(eta_bess_in[u]) * p_grid_bess[t][u]
                        deficit_to_min = soc_min_abs - e_bess_pre[t + 1][u]
                        prob += grid_added >= deficit_to_min, f"grid_bess_guard_lb_{t}_{u}"
                        prob += grid_added <= deficit_to_min + M_soc * (1 - d_grid_bess_guard[t][u]), f"grid_bess_guard_exact_{t}_{u}"
                        prob += grid_added <= M_soc * d_grid_bess_guard[t][u], f"grid_bess_guard_ub_{t}_{u}"
                    else:
                        prob += dt * float(eta_bess_in[u]) * p_grid_bess[t][u] >= soc_min_abs - e_bess_pre[t + 1][u], f"grid_bess_guard_lp_{t}_{u}"
                else:
                    prob += p_grid_bess[t][u] == 0, f"no_grid_bess_last_{t}_{u}"

                if run_lp:
                    prob += p_bess_in[t][u] <= float(battery_power[u]), f"bess_in_cap_{t}_{u}"
                    prob += p_bess_out[t][u] <= float(battery_power[u]), f"bess_out_cap_{t}_{u}"
                else:
                    prob += d_bess_ch[t][u] + d_bess_dis[t][u] <= 1, f"bess_no_simultaneous_{t}_{u}"
                    prob += p_bess_in[t][u] <= d_bess_ch[t][u] * float(battery_power[u]), f"bess_in_gate_{t}_{u}"
                    prob += p_bess_out[t][u] <= d_bess_dis[t][u] * float(battery_power[u]), f"bess_out_gate_{t}_{u}"

            # A/B tarifa energia-lépcsők.
            if boiler_tariff == "A":
                e_imp_a_t = dt * (p_grid_ue[t][u] + p_grid_elh[t][u] + p_grid_bess[t][u])
                e_imp_b_t = 0.0
            else:
                e_imp_a_t = dt * (p_grid_ue[t][u] + p_grid_bess[t][u])
                e_imp_b_t = dt * p_grid_elh[t][u]

            e_shared_a_t = dt * (p_shared_ue[t][u] + p_shared_elh[t][u])

            # A vevő 2523 kWh-os A-sávját a közösségi megosztás és
            # az A-tarifás hálózati import közösen fogyasztja.
            # Sorrend: először shared_in, utána a maradék A-sávból grid_import_a.
            prob += e_shared_a_low_step[t][u] + e_shared_a_high_step[t][u] == e_shared_a_t, f"shared_a_tier_split_{t}_{u}"
            prob += e_grid_a_low_step[t][u] + e_grid_a_high_step[t][u] == e_imp_a_t, f"a_tier_split_{t}_{u}"
            prob += e_grid_b_low_step[t][u] + e_grid_b_high_step[t][u] == e_imp_b_t, f"b_tier_split_{t}_{u}"

            if t == 0:
                rem_a_prev = float(grid_a_low_cap_kwh)
                rem_b_prev = float(grid_b_low_cap_kwh)
            else:
                rem_a_prev = rem_a_low[t - 1][u]
                rem_b_prev = rem_b_low[t - 1][u]

            # Shared-first: e_shared_low = min(e_shared_a_t, rem_a_prev).
            prob += e_shared_a_low_step[t][u] <= e_shared_a_t, f"shared_a_low_le_shared_{t}_{u}"
            prob += e_shared_a_low_step[t][u] <= rem_a_prev, f"shared_a_low_le_remaining_{t}_{u}"

            if not run_lp:
                m_shared_first = float(grid_a_low_cap_kwh) + float(np.max(total_load[:, u]) * dt) + 1.0
                prob += e_shared_a_low_step[t][u] >= e_shared_a_t - m_shared_first * (1 - d_shared_low_full[t][u]), f"shared_a_low_min_shared_{t}_{u}"
                prob += e_shared_a_low_step[t][u] >= rem_a_prev - m_shared_first * d_shared_low_full[t][u], f"shared_a_low_min_remaining_{t}_{u}"

            # A grid csak a shared után megmaradó kedvezményes A-sávot használhatja.
            prob += e_grid_a_low_step[t][u] <= rem_a_prev - e_shared_a_low_step[t][u], f"a_low_after_shared_{t}_{u}"

            prob += rem_a_low[t][u] == rem_a_prev - e_shared_a_low_step[t][u] - e_grid_a_low_step[t][u], f"rem_a_dyn_{t}_{u}"

            # B-sáv külön keret: csak B tarifás bojler grid importja fogyasztja.
            prob += e_grid_b_low_step[t][u] <= rem_b_prev, f"b_low_remaining_{t}_{u}"
            prob += rem_b_low[t][u] == rem_b_prev - e_grid_b_low_step[t][u], f"rem_b_dyn_{t}_{u}"

    # Kezdeti SOC.
    for u in user_set:
        if bess_enabled[u] and size_bess[u] > EPS:
            prob += e_bess[0][u] == float(size_bess[u]) * float(soc_bess_init), f"bess_init_{u}"

    # Minimum töltési/kisütési üzemmódhossz.
    if not run_lp:
        min_steps = max(1, int(bess_min_mode_steps))
        if min_steps > 1:
            for u in user_set:
                if not (bess_enabled[u] and size_bess[u] > EPS):
                    continue
                for t in range(1, T - min_steps + 1):
                    start_ch = d_bess_ch[t][u] - d_bess_ch[t - 1][u]
                    start_dis = d_bess_dis[t][u] - d_bess_dis[t - 1][u]
                    prob += (
                        pulp.lpSum(d_bess_ch[tau][u] for tau in range(t, t + min_steps))
                        >= min_steps * start_ch
                    ), f"bess_min_charge_{t}_{u}"
                    prob += (
                        pulp.lpSum(d_bess_dis[tau][u] for tau in range(t, t + min_steps))
                        >= min_steps * start_dis
                    ), f"bess_min_discharge_{t}_{u}"

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

    solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    status = prob.solve(solver)
    status_str = pulp.LpStatus.get(status, str(status))
    if status_str not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"Optimalizálási hiba: {status_str}")

    # --- Eredmények kinyerése ---
    p_pv_bess_v = _extract_matrix(p_pv_bess, T, U)
    p_pv_shared_v = _extract_matrix(p_pv_shared, T, U)
    p_pv_grid_v = _extract_matrix(p_pv_grid, T, U)
    p_bess_ue_v = _extract_matrix(p_bess_ue, T, U)
    p_bess_elh_v = _extract_matrix(p_bess_elh, T, U)
    p_shared_ue_v = _extract_matrix(p_shared_ue, T, U)
    p_shared_elh_v = _extract_matrix(p_shared_elh, T, U)
    p_grid_ue_v = _extract_matrix(p_grid_ue, T, U)
    p_grid_elh_v = _extract_matrix(p_grid_elh, T, U)
    p_bess_in_v = _extract_matrix(p_bess_in, T, U)
    p_bess_out_v = _extract_matrix(p_bess_out, T, U)
    p_grid_bess_v = _extract_matrix(p_grid_bess, T, U)
    p_grid_import_v = _extract_matrix(p_grid_import, T, U)
    p_grid_export_v = _extract_matrix(p_grid_export, T, U)
    d_bess_ch_v = _extract_matrix(d_bess_ch, T, U) if not run_lp else np.zeros((T, U))
    d_bess_dis_v = _extract_matrix(d_bess_dis, T, U) if not run_lp else np.zeros((T, U))
    d_grid_v = _extract_matrix(d_grid, T, U) if not run_lp else np.zeros((T, U))
    e_bess_v = _extract_matrix(e_bess, T, U)

    e_grid_a_low_step_v = _extract_matrix(e_grid_a_low_step, T, U)
    e_grid_a_high_step_v = _extract_matrix(e_grid_a_high_step, T, U)
    e_shared_a_low_step_v = _extract_matrix(e_shared_a_low_step, T, U)
    e_shared_a_high_step_v = _extract_matrix(e_shared_a_high_step, T, U)
    e_grid_b_low_step_v = _extract_matrix(e_grid_b_low_step, T, U)
    e_grid_b_high_step_v = _extract_matrix(e_grid_b_high_step, T, U)

    p_pv_load_v = p_pv_ue_fixed + p_pv_elh_fixed
    p_bess_load_v = p_bess_ue_v + p_bess_elh_v
    p_shared_in_v = p_shared_ue_v + p_shared_elh_v
    p_grid_load_v = p_grid_ue_v + p_grid_elh_v
    if boiler_tariff == "A":
        p_grid_import_a_v = p_grid_ue_v + p_grid_elh_v + p_grid_bess_v
    else:
        p_grid_import_a_v = p_grid_ue_v + p_grid_bess_v

    # Végső, noopt-tal egyező determinisztikus megosztás a BESS utáni
    # maradék pozíciókból. Ez biztosítja, hogy a shared nem önálló
    # optimalizálási döntés: a kiosztást csak a BESS által módosított
    # surplus/deficit tudja befolyásolni.
    det = _deterministic_shared_from_residuals(
        residual_surplus=np.maximum(p_surplus - p_pv_bess_v, 0.0),
        residual_deficit_ue=np.maximum(p_deficit_ue - p_bess_ue_v, 0.0),
        residual_deficit_elh=np.maximum(p_deficit_elh - p_bess_elh_v, 0.0),
        boiler_tariff=boiler_tariff,
        sharing_mode=sharing_mode,
    )
    p_pv_shared_v = det["p_shared_out"]
    p_pv_grid_v = det["p_pv_grid"]
    p_grid_export_v = p_pv_grid_v.copy()
    p_shared_ue_v = det["p_shared_ue"]
    p_shared_elh_v = det["p_shared_elh"]
    p_shared_in_v = det["p_shared_total"]
    p_grid_ue_v = det["p_grid_ue"]
    p_grid_elh_v = det["p_grid_elh"]
    p_grid_load_v = p_grid_ue_v + p_grid_elh_v
    if boiler_tariff == "A":
        p_grid_import_a_v = p_grid_ue_v + p_grid_elh_v + p_grid_bess_v
    else:
        p_grid_import_a_v = p_grid_ue_v + p_grid_bess_v
    p_grid_import_v = p_grid_ue_v + p_grid_elh_v + p_grid_bess_v

    if boiler_tariff == "B":
        if float(np.max(np.abs(p_pv_elh_fixed))) > 1e-7:
            raise RuntimeError("B tarifán saját PV energia jutott a bojlerre.")
        if float(np.max(np.abs(p_bess_elh_v))) > 1e-7:
            raise RuntimeError("B tarifán saját BESS energia jutott a bojlerre.")
        if float(np.max(np.abs(p_shared_elh_v))) > 1e-7:
            raise RuntimeError("B tarifán shared energia jutott a bojlerre.")
        if not np.allclose(p_grid_elh_v, p_el_heater, atol=1e-7):
            raise RuntimeError("B tarifán a bojlerigény nem teljes egészében B tarifás grid importként jelent meg.")

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
            "final_bess_energy_kwh": float(e_bess_v[-1, u]) if T else 0.0,
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
        "objective": objective,
        "objective_value": float(pulp.value(prob.objective)),
        "n_users": int(U),
        "T": int(T),
        "dt": float(dt),
        "boiler_tariff": boiler_tariff,
        "sharing_mode": sharing_mode,
        "pairing_mode": pairing_mode,
        "deterministic_sharing_after_bess": True,
        "exact_no_bess_matches_nonopt": bool(no_bess_active),
        "bess_min_mode_steps": int(bess_min_mode_steps),
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
        "p_grid_import": p_grid_import_v,
        "p_grid_export": p_grid_export_v,
        "e_bess": e_bess_v,
        "d_bess_ch": d_bess_ch_v,
        "d_bess_dis": d_bess_dis_v,
        "d_grid": d_grid_v,
    }

    return {
        "summary": summary,
        "per_user_df": per_user_df,
        "timeseries": timeseries,
        "shared_pair_kwh": settlement["pair_kwh"],
        "shared_pair_energy_payment_ft": settlement["pair_energy_payment_ft"],
        "user_names": list(user_names),
        "bess_enabled": np.asarray(bess_enabled, dtype=bool),
        "status": status_str,
    }


def save_disaggregated_opt_results(result: dict, out_dir: str | Path, *, save_user_timeseries: bool = True) -> None:
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
        "d_bess_dis", "d_grid",
    ]:
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
    community_ts.to_csv(out / "community_timeseries.csv", index=False)

    if save_user_timeseries:
        for i, name in enumerate(user_names):
            safe_name = str(name).replace("/", "_").replace("\\", "_")
            user_df = pd.DataFrame({key: arr[:, i] for key, arr in ts.items() if arr.ndim == 2})
            user_df.to_csv(out / f"timeseries_{safe_name}.csv", index=False)