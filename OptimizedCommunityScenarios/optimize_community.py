"""4d-K: közösségi PV–bojler–BESS optimalizálás determinisztikus megosztással."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pulp

OPTIMIZER_API_VERSION = 2
EPS = 1e-9
LOW_TARIFF_LIMIT_KWH = 2523.0
PRICE_GRID_A_LOW = 36.0
PRICE_GRID_A_HIGH = 71.0
PRICE_GRID_B_LOW = 23.0
PRICE_GRID_B_HIGH = 61.0
PRICE_PV_GRID = 5.0
SHARED_BUYER_LOW = 5.0
SHARED_BUYER_HIGH = 21.0
SHARED_RHD = 31.0


def _progress(callback, message):
    if callback is not None:
        callback(message)


def _matrix(a, name):
    value = np.maximum(np.asarray(a, dtype=float), 0.0)
    if value.ndim != 2:
        raise ValueError(f"{name} kétdimenziós (idő × háztartás) tömb legyen.")
    return value


def _vector(a, users, name, default=None):
    if a is None:
        if default is None:
            raise ValueError(f"Hiányzó paraméter: {name}")
        return np.full(users, float(default))
    value = np.asarray(a, dtype=float).ravel()
    if value.size == 1:
        return np.full(users, float(value[0]))
    if value.size != users:
        raise ValueError(f"{name} hossza {value.size}, de {users} szükséges.")
    return value


def _values(items):
    return np.asarray([float(pulp.value(item) or 0.0) for item in items], dtype=float)


def _allocate_sharing(surplus, need_base, need_boiler, mode, allocation_need=None):
    """Igényalapú megosztás; a súly nem függ a BESS menetrendjétől."""
    surplus = _matrix(surplus, "surplus")
    need_base = _matrix(need_base, "need_base")
    need_boiler = _matrix(need_boiler, "need_boiler")
    if not (surplus.shape == need_base.shape == need_boiler.shape):
        raise ValueError("A megosztási tömbök alakja nem egyezik.")
    if allocation_need is None:
        allocation_need = need_base + need_boiler
    else:
        allocation_need = _matrix(allocation_need, "allocation_need")
        if allocation_need.shape != surplus.shape:
            raise ValueError("Az allocation_need tömb alakja nem egyezik.")
    if mode not in {"proportional", "equal"}:
        raise ValueError("sharing_mode csak 'proportional' vagy 'equal' lehet.")

    steps, users = surplus.shape
    shared_in = np.zeros_like(surplus)
    shared_out = np.zeros_like(surplus)
    total_need = need_base + need_boiler
    for t in range(steps):
        supply = float(surplus[t].sum())
        demand = float(total_need[t].sum())
        available = min(supply, demand)
        if available <= EPS:
            continue
        active = total_need[t] > EPS
        if mode == "proportional":
            # A kiosztási kulcs előre meghatározott fogyasztás-PV hiány. A
            # tényleges maradékigény csak fizikai felső korlát; így a BESS nem
            # kerül a proporcionális súlyba, és nincs bilineáris szorzat.
            remaining = available
            open_mask = active.copy()
            while remaining > EPS and np.any(open_mask):
                weights = np.where(open_mask, allocation_need[t], 0.0)
                if float(weights.sum()) <= EPS:
                    break
                proposal = remaining * weights / float(weights.sum())
                capacity = np.maximum(total_need[t] - shared_in[t], 0.0)
                addition = np.minimum(proposal, capacity)
                shared_in[t] += addition
                used_now = float(addition.sum())
                if used_now <= EPS:
                    break
                remaining -= used_now
                open_mask &= capacity - addition > EPS
        else:
            quota = available / int(active.sum())
            shared_in[t, active] = np.minimum(total_need[t, active], quota)
        used = float(shared_in[t].sum())
        if used > EPS:
            shared_out[t] = used * surplus[t] / supply

    shared_base = np.minimum(need_base, shared_in)
    shared_boiler = np.minimum(need_boiler, np.maximum(shared_in - shared_base, 0.0))
    return {
        "p_shared_in": shared_in,
        "p_shared_out": shared_out,
        "p_shared_base": shared_base,
        "p_shared_boiler": shared_boiler,
        "p_grid_base": np.maximum(need_base - shared_base, 0.0),
        "p_grid_boiler_a": np.maximum(need_boiler - shared_boiler, 0.0),
        "p_pv_export": np.maximum(surplus - shared_out, 0.0),
    }


def _settle_shared(
    p_shared_in, p_shared_out, p_grid_a, dt, mode,
    a_low_remaining_kwh=None,
):
    """Shared-first A-sáv és eladó–vevő pénzügyi párosítás."""
    steps, users = p_shared_in.shape
    remaining = (
        np.full(users, LOW_TARIFF_LIMIT_KWH)
        if a_low_remaining_kwh is None
        else np.maximum(np.asarray(a_low_remaining_kwh, dtype=float).ravel(), 0.0).copy()
    )
    if remaining.size != users:
        raise ValueError("Az A tarifakeret-maradék hossza hibás.")
    shared_low = np.zeros(users)
    shared_high = np.zeros(users)
    grid_low = np.zeros(users)
    grid_high = np.zeros(users)
    pair_kwh = np.zeros((users, users))
    pair_payment = np.zeros((users, users))

    for t in range(steps):
        e_in = p_shared_in[t] * dt
        e_out = p_shared_out[t] * dt
        low_t = np.minimum(e_in, remaining)
        high_t = e_in - low_t
        remaining -= low_t
        shared_low += low_t
        shared_high += high_t

        e_grid = p_grid_a[t] * dt
        grid_low_t = np.minimum(e_grid, remaining)
        grid_high_t = e_grid - grid_low_t
        remaining -= grid_low_t
        grid_low += grid_low_t
        grid_high += grid_high_t

        sellers = np.flatnonzero(e_out > EPS)
        buyers = np.flatnonzero(e_in > EPS)
        if not len(sellers) or not len(buyers):
            continue
        buyer_value = low_t * SHARED_BUYER_LOW + high_t * SHARED_BUYER_HIGH
        if mode == "proportional":
            for seller in sellers:
                share = e_out[seller] / e_out[sellers].sum()
                pair_kwh[seller, buyers] += share * e_in[buyers]
                pair_payment[seller, buyers] += share * buyer_value[buyers]
        else:
            # Az energiamennyiség kiosztása már equal/no-redistribution. Az
            # eladói oldal tényleges PV-hozzájárulás szerint kap bevételt.
            for seller in sellers:
                share = e_out[seller] / e_out[sellers].sum()
                pair_kwh[seller, buyers] += share * e_in[buyers]
                pair_payment[seller, buyers] += share * buyer_value[buyers]

    buyer_energy = shared_low * SHARED_BUYER_LOW + shared_high * SHARED_BUYER_HIGH
    buyer_rhd = p_shared_in.sum(axis=0) * dt * SHARED_RHD
    return {
        "shared_low_kwh": shared_low,
        "shared_high_kwh": shared_high,
        "grid_a_low_kwh": grid_low,
        "grid_a_high_kwh": grid_high,
        "buyer_energy_ft": buyer_energy,
        "buyer_rhd_ft": buyer_rhd,
        "seller_revenue_ft": pair_payment.sum(axis=1),
        "pair_kwh": pair_kwh,
        "pair_payment_ft": pair_payment,
        "a_low_remaining_kwh": remaining,
    }


def optimize_community(
    p_pv,
    p_base,
    p_dhw,
    p_boiler_fixed,
    *,
    user_names,
    hss_enabled,
    bess_enabled,
    size_elh,
    vol_hss_water,
    size_bess,
    dt=0.25,
    T_env=None,
    T_max=None,
    T_abs_min=None,
    T_in=None,
    T_comfort=None,
    T_setpoint_min=None,
    T_initial=None,
    morning_block=(6.0, 9.0),
    evening_block=(17.0, 22.0),
    a_hss=None,
    eta_elh=None,
    eta_bess_in=None,
    eta_bess_out=None,
    eta_bess_stor=None,
    soc_bess_min=None,
    soc_bess_max=None,
    soc_bess_init=None,
    t_bess_min=None,
    initial_hss_energy_kwhth=None,
    initial_bess_energy_kwh=None,
    a_low_remaining_kwh=None,
    b_low_remaining_kwh=None,
    enforce_terminal_state=False,
    thermal_shortfall_penalty_ft_per_kwh=1_000_000.0,
    objective="bill",
    sharing_mode="proportional",
    run_lp=False,
    solver="gurobi",
    msg=False,
    aggregate_passive=True,
    progress_callback: Callable[[str], None] | None = None,
):
    """Közösségi koordináció vegyes, háztartásonként automatikus bojlerkörrel.

    `hss_enabled=True`: a vízigényes tartálymodell A tarifán optimalizált.
    Minden más mért bojlerprofil változatlan, kizárólag B tarifás import.
    """
    _progress(progress_callback, "Bemeneti tömbök ellenőrzése")
    p_pv = _matrix(p_pv, "p_pv")
    p_base = _matrix(p_base, "p_base")
    p_dhw = _matrix(p_dhw, "p_dhw")
    p_boiler_fixed = _matrix(p_boiler_fixed, "p_boiler_fixed")
    if not (p_pv.shape == p_base.shape == p_dhw.shape == p_boiler_fixed.shape):
        raise ValueError("A teljesítmény-idősorok alakja nem egyezik.")
    steps_count, users_count = p_pv.shape
    if len(user_names) != users_count or dt <= 0:
        raise ValueError("Hibás user_names vagy dt.")
    if objective not in {"bill", "grid"}:
        raise ValueError("objective csak 'bill' vagy 'grid' lehet.")

    hss_enabled = np.asarray(hss_enabled, dtype=bool).ravel()
    bess_enabled = np.asarray(bess_enabled, dtype=bool).ravel()
    if hss_enabled.size != users_count or bess_enabled.size != users_count:
        raise ValueError("A hss_enabled/bess_enabled hossza hibás.")
    size_elh = _vector(size_elh, users_count, "size_elh")
    volume = _vector(vol_hss_water, users_count, "vol_hss_water")
    size_bess = _vector(size_bess, users_count, "size_bess")
    T_env = _vector(T_env, users_count, "T_env", 20)
    T_max = _vector(T_max, users_count, "T_max", 65)
    T_abs_min = _vector(T_abs_min, users_count, "T_abs_min", 10)
    T_in = _vector(T_in, users_count, "T_in", 10)
    T_comfort = _vector(T_comfort, users_count, "T_comfort", 40)
    T_setpoint_min = _vector(T_setpoint_min, users_count, "T_setpoint_min", 50)
    T_initial = _vector(T_initial, users_count, "T_initial", 50)
    a_hss = _vector(a_hss, users_count, "a_hss", 0.01275)
    eta_elh = _vector(eta_elh, users_count, "eta_elh", 0.95)
    eta_in = _vector(eta_bess_in, users_count, "eta_bess_in", 0.98)
    eta_out = _vector(eta_bess_out, users_count, "eta_bess_out", 0.96)
    eta_stor = _vector(eta_bess_stor, users_count, "eta_bess_stor", 0.995)
    soc_min = _vector(soc_bess_min, users_count, "soc_bess_min", 0.10)
    soc_max = _vector(soc_bess_max, users_count, "soc_bess_max", 0.90)
    soc_init = _vector(soc_bess_init, users_count, "soc_bess_init", 0.50)
    t_bess_min = _vector(t_bess_min, users_count, "t_bess_min", 2.0)
    a_low_remaining = _vector(
        a_low_remaining_kwh, users_count, "a_low_remaining_kwh",
        LOW_TARIFF_LIMIT_KWH,
    )
    b_low_remaining = _vector(
        b_low_remaining_kwh, users_count, "b_low_remaining_kwh",
        LOW_TARIFF_LIMIT_KWH,
    )
    thermal_capacity = volume * 0.00116667
    default_hss_initial = thermal_capacity * (T_initial - T_in)
    hss_initial_energy = (
        default_hss_initial
        if initial_hss_energy_kwhth is None
        else _vector(initial_hss_energy_kwhth, users_count, "initial_hss_energy_kwhth")
    )
    default_bess_initial = size_bess * soc_init
    bess_initial_energy = (
        default_bess_initial
        if initial_bess_energy_kwh is None
        else _vector(initial_bess_energy_kwh, users_count, "initial_bess_energy_kwh")
    )
    hss_enabled &= (size_elh > EPS) & (volume > EPS)
    bess_enabled &= size_bess > EPS
    if np.any(
        hss_enabled
        & ~(
            (T_in <= T_abs_min) & (T_abs_min <= T_comfort)
            & (T_comfort <= T_setpoint_min) & (T_setpoint_min <= T_max)
        )
    ):
        raise ValueError(
            "Aktív HSS-nél T_in <= T_abs_min <= T_comfort <= "
            "T_setpoint_min <= T_max szükséges."
        )
    hss_floor = thermal_capacity * (T_abs_min - T_in)
    hss_ceiling = thermal_capacity * (T_max - T_in)
    hss_initial_energy = np.where(
        hss_enabled,
        np.clip(hss_initial_energy, hss_floor, hss_ceiling),
        0.0,
    )
    bess_initial_energy = np.where(
        bess_enabled,
        np.clip(bess_initial_energy, size_bess * soc_min, size_bess * soc_max),
        0.0,
    )
    a_low_remaining = np.clip(a_low_remaining, 0.0, LOW_TARIFF_LIMIT_KWH)
    b_low_remaining = np.clip(b_low_remaining, 0.0, LOW_TARIFF_LIMIT_KWH)
    p_boiler_fixed[:, hss_enabled] = 0.0

    # Az eredeti, háztartásonkénti adatok az eredmény-visszaosztáshoz és az
    # egyedi tarifakeretekhez mindig megmaradnak. A solver csak a teljesen
    # passzív (PV/BESS/HSS/mért bojler nélküli) fogyasztókat vonja össze.
    original = {
        "p_pv": p_pv.copy(), "p_base": p_base.copy(), "p_dhw": p_dhw.copy(),
        "p_boiler_fixed": p_boiler_fixed.copy(), "user_names": list(user_names),
        "hss_enabled": hss_enabled.copy(), "bess_enabled": bess_enabled.copy(),
        "a_low_remaining": a_low_remaining.copy(),
        "b_low_remaining": b_low_remaining.copy(),
        "hss_initial_energy": hss_initial_energy.copy(),
        "bess_initial_energy": bess_initial_energy.copy(),
    }
    original_users_count = users_count
    passive_mask = (
        (p_pv.sum(axis=0) <= EPS)
        & ~hss_enabled
        & ~bess_enabled
        & (p_boiler_fixed.sum(axis=0) <= EPS)
    )
    passive_indices = np.flatnonzero(passive_mask)
    active_indices = np.flatnonzero(~passive_mask)
    aggregation_used = bool(
        aggregate_passive and sharing_mode == "proportional" and passive_indices.size >= 2
    )
    aggregate_model_index = None

    if aggregate_passive and sharing_mode != "proportional":
        _progress(progress_callback, "Passzív aggregálás kihagyva: equal megosztásnál nem egzakt")

    if aggregation_used:
        _progress(
            progress_callback,
            f"Passzív aggregálás: {passive_indices.size} háztartás -> 1 modellfelhasználó",
        )

        def collapse_matrix(value):
            return np.column_stack((value[:, active_indices], value[:, passive_indices].sum(axis=1)))

        def collapse_vector(value, aggregate_value=0.0):
            return np.concatenate((np.asarray(value)[active_indices], [aggregate_value]))

        p_pv = collapse_matrix(p_pv)
        p_base = collapse_matrix(p_base)
        p_dhw = collapse_matrix(p_dhw)
        p_boiler_fixed = collapse_matrix(p_boiler_fixed)
        user_names = [original["user_names"][u] for u in active_indices] + ["__PASSIVE_AGGREGATE__"]
        hss_enabled = collapse_vector(hss_enabled, False).astype(bool)
        bess_enabled = collapse_vector(bess_enabled, False).astype(bool)
        size_elh = collapse_vector(size_elh)
        volume = collapse_vector(volume)
        size_bess = collapse_vector(size_bess)
        T_env = collapse_vector(T_env, 20.0)
        T_max = collapse_vector(T_max, 65.0)
        T_abs_min = collapse_vector(T_abs_min, 10.0)
        T_in = collapse_vector(T_in, 10.0)
        T_comfort = collapse_vector(T_comfort, 40.0)
        T_setpoint_min = collapse_vector(T_setpoint_min, 50.0)
        T_initial = collapse_vector(T_initial, 50.0)
        a_hss = collapse_vector(a_hss, 0.01275)
        eta_elh = collapse_vector(eta_elh, 0.95)
        eta_in = collapse_vector(eta_in, 0.98)
        eta_out = collapse_vector(eta_out, 0.96)
        eta_stor = collapse_vector(eta_stor, 0.995)
        soc_min = collapse_vector(soc_min, 0.10)
        soc_max = collapse_vector(soc_max, 0.90)
        soc_init = collapse_vector(soc_init, 0.50)
        t_bess_min = collapse_vector(t_bess_min, 2.0)
        hss_initial_energy = collapse_vector(hss_initial_energy)
        bess_initial_energy = collapse_vector(bess_initial_energy)
        a_low_remaining = collapse_vector(a_low_remaining)
        b_low_remaining = collapse_vector(b_low_remaining)
        users_count = len(user_names)
        aggregate_model_index = users_count - 1
    else:
        active_indices = np.arange(original_users_count)
        passive_indices = np.asarray([], dtype=int)

    _progress(
        progress_callback,
        f"Modellméret: {original_users_count} eredeti, {users_count} solver-felhasználó, {steps_count} időlépés",
    )

    _progress(progress_callback, "PuLP-modell létrehozása")
    model = pulp.LpProblem("community_opt", pulp.LpMinimize)
    binary = pulp.LpContinuous if run_lp else pulp.LpBinary
    times = range(steps_count)
    users = range(users_count)

    def vars_for(prefix, user, upper=None, count=None):
        n = steps_count if count is None else count
        return [pulp.LpVariable(f"{prefix}_{t}_{user}", 0, upper) for t in range(n)]

    def in_window(hour, window):
        start, stop = map(float, window)
        return start <= hour < stop if start <= stop else hour >= start or hour < stop

    day_steps = int(round(24.0 / dt))
    if abs(day_steps * dt - 24.0) > 1e-9:
        raise ValueError("A dt-nek maradék nélkül kell osztania a 24 órát.")

    # Minden usernél csak a szükséges komponensek; ez lényegesen kisebb,
    # mint a korábbi teljes T×U változómátrix minden részfolyamra.
    v = []
    hss_state = {}
    bess_state = {}
    for u in users:
        d = {
            "pv_base": vars_for("pv_base", u), "pv_boiler": vars_for("pv_boiler", u),
            "pv_bess": vars_for("pv_bess", u), "pv_shared": vars_for("pv_shared", u),
            "pv_export": vars_for("pv_export", u), "grid_base": vars_for("grid_base", u),
            "grid_boiler": vars_for("grid_boiler", u), "grid_bess": vars_for("grid_bess", u),
            "bess_base": vars_for("bess_base", u), "bess_boiler": vars_for("bess_boiler", u),
            "shared_base": vars_for("shared_base", u), "shared_boiler": vars_for("shared_boiler", u),
        }
        v.append(d)
        if hss_enabled[u]:
            capacity_u = volume[u] * 0.00116667
            e_abs_min_u = capacity_u * (T_abs_min[u] - T_in[u])
            e_max_u = capacity_u * (T_max[u] - T_in[u])
            setpoints = {}
            setpoint_shortfalls = {}
            for day_start in range(0, steps_count, day_steps):
                for label, window in (("morning", morning_block), ("evening", evening_block)):
                    start_offset = int(round(float(window[0]) / dt))
                    start_step = day_start + start_offset
                    if start_step < steps_count:
                        setpoints[start_step] = pulp.LpVariable(
                            f"boiler_setpoint_{label}_{day_start // day_steps}_{u}",
                            T_setpoint_min[u], T_max[u],
                        )
                        setpoint_shortfalls[start_step] = pulp.LpVariable(
                            f"setpoint_shortfall_{label}_{day_start // day_steps}_{u}",
                            0, e_max_u - e_abs_min_u,
                        )
            hss_state[u] = {
                "p_elh": vars_for("p_elh", u, size_elh[u]),
                "energy": [
                    pulp.LpVariable(f"ehss_{t}_{u}", e_abs_min_u, e_max_u)
                    for t in range(steps_count + 1)
                ],
                "dhw_out": vars_for("p_dhw_out", u),
                "dhw_shortfall": [
                    pulp.LpVariable(
                        f"p_dhw_shortfall_{t}_{u}", 0, float(p_dhw[t, u])
                    ) if p_dhw[t, u] > EPS else 0.0
                    for t in times
                ],
                "setpoint": setpoints,
                "setpoint_shortfall": setpoint_shortfalls,
                "terminal_shortfall": pulp.LpVariable(
                    f"terminal_setpoint_shortfall_{u}", 0,
                    e_max_u - e_abs_min_u,
                ) if enforce_terminal_state else 0.0,
            }
        if bess_enabled[u]:
            bess_state[u] = {
                "energy": [pulp.LpVariable(f"eb_{t}_{u}", size_bess[u] * soc_min[u], size_bess[u] * soc_max[u]) for t in range(steps_count + 1)],
                "pre": vars_for("eb_pre", u, size_bess[u] * soc_max[u]),
                "charge": [pulp.LpVariable(f"bch_{t}_{u}", 0, 1, cat=binary) for t in times],
                "discharge": [pulp.LpVariable(f"bdis_{t}_{u}", 0, 1, cat=binary) for t in times],
                "charge_start": [pulp.LpVariable(f"bch_start_{t}_{u}", 0, 1, cat=binary) for t in times],
                "discharge_start": [pulp.LpVariable(f"bdis_start_{t}_{u}", 0, 1, cat=binary) for t in times],
                "guard": [pulp.LpVariable(f"bguard_{t}_{u}", 0, 1, cat=binary) for t in times],
            }
        if (u + 1) % 5 == 0 or u + 1 == users_count:
            _progress(progress_callback, f"Változók létrehozva: {u + 1}/{users_count} solver-felhasználó")

    def add_minimum(signal, starts, hours, prefix):
        count = max(1, int(round(hours / dt)))
        model.addConstraint(starts[0] == signal[0], f"{prefix}_start_0")
        for t in range(1, steps_count):
            model.addConstraint(starts[t] >= signal[t] - signal[t - 1], f"{prefix}_start_lo_{t}")
            model.addConstraint(starts[t] <= signal[t], f"{prefix}_start_on_{t}")
            model.addConstraint(starts[t] <= 1 - signal[t - 1], f"{prefix}_start_prev_{t}")
        for t in times:
            if t + count <= steps_count:
                model.addConstraint(pulp.lpSum(signal[k] for k in range(t, t + count)) >= count * starts[t], f"{prefix}_min_{t}")
            else:
                model.addConstraint(starts[t] == 0, f"{prefix}_late_{t}")

    for u in users:
        d = v[u]
        for t in times:
            model += d["pv_base"][t] + d["pv_boiler"][t] + d["pv_bess"][t] + d["pv_shared"][t] + d["pv_export"][t] == p_pv[t, u]
            model += d["pv_base"][t] + d["bess_base"][t] + d["shared_base"][t] + d["grid_base"][t] == p_base[t, u]

            if hss_enabled[u]:
                hs = hss_state[u]
                model += d["pv_boiler"][t] + d["bess_boiler"][t] + d["shared_boiler"][t] + d["grid_boiler"][t] == hs["p_elh"][t]
                c = volume[u] * 0.00116667
                e_abs_min_u = c * (T_abs_min[u] - T_in[u])
                e_max_u = c * (T_max[u] - T_in[u])
                equivalent_temperature = T_in[u] + hs["energy"][t] / c
                model += hs["dhw_out"][t] + hs["dhw_shortfall"][t] == p_dhw[t, u]
                model += hs["energy"][t + 1] == hs["energy"][t] + dt * (
                    eta_elh[u] * hs["p_elh"][t]
                    - hs["dhw_out"][t]
                    - a_hss[u] * (equivalent_temperature - T_env[u])
                )
                model += dt * hs["dhw_out"][t] <= hs["energy"][t] - e_abs_min_u
                model += dt * eta_elh[u] * hs["p_elh"][t] <= e_max_u - hs["energy"][t]

                hour = (t * dt) % 24.0
                blocked = in_window(hour, morning_block) or in_window(hour, evening_block)
                if blocked:
                    # Teljes bojler-tiltás: hálózat, saját PV, BESS és
                    # közösségi megosztás egyaránt nulla.
                    model += hs["p_elh"][t] == 0
                    model += d["grid_boiler"][t] == 0
                    model += d["pv_boiler"][t] == 0
                    model += d["bess_boiler"][t] == 0
                    model += d["shared_boiler"][t] == 0

                if t in hs["setpoint"]:
                    model += hs["energy"][t] + hs["setpoint_shortfall"][t] == c * (
                        hs["setpoint"][t] - T_in[u]
                    )
            else:
                # Mért bojler: külön B-kör, PV/BESS/shared nem láthatja el.
                model += d["pv_boiler"][t] == 0
                model += d["bess_boiler"][t] == 0
                model += d["shared_boiler"][t] == 0
                model += d["grid_boiler"][t] == p_boiler_fixed[t, u]

            if bess_enabled[u]:
                bs = bess_state[u]
                power = size_bess[u] / t_bess_min[u]
                charge = d["pv_bess"][t] + d["grid_bess"][t]
                discharge = d["bess_base"][t] + d["bess_boiler"][t]
                model += charge <= power * bs["charge"][t]
                model += discharge <= power * bs["discharge"][t]
                model += bs["charge"][t] + bs["discharge"][t] <= 1
                model += bs["pre"][t] == bs["energy"][t] * eta_stor[u] + dt * (d["pv_bess"][t] * eta_in[u] - discharge / eta_out[u])
                model += bs["energy"][t + 1] == bs["pre"][t] + dt * eta_in[u] * d["grid_bess"][t]
                minimum = size_bess[u] * soc_min[u]
                big_m = max(size_bess[u], 1.0)
                model += bs["pre"][t] >= minimum - big_m * bs["guard"][t]
                model += bs["pre"][t] <= minimum + big_m * (1 - bs["guard"][t])
                refill = dt * eta_in[u] * d["grid_bess"][t]
                model += refill >= minimum - bs["pre"][t]
                model += refill <= minimum - bs["pre"][t] + big_m * (1 - bs["guard"][t])
                model += d["grid_bess"][t] <= power * bs["guard"][t]
            else:
                for key in ("pv_bess", "grid_bess", "bess_base", "bess_boiler"):
                    model += d[key][t] == 0

        if hss_enabled[u]:
            hs = hss_state[u]
            model += hs["energy"][0] == hss_initial_energy[u]
            if enforce_terminal_state:
                terminal_min = volume[u] * 0.00116667 * (
                    T_setpoint_min[u] - T_in[u]
                )
                model += hs["energy"][steps_count] + hs["terminal_shortfall"] >= terminal_min
        if bess_enabled[u]:
            bs = bess_state[u]
            model += bs["energy"][0] == bess_initial_energy[u]
            add_minimum(bs["charge"], bs["charge_start"], 1.0, f"bch_{u}")
            add_minimum(bs["discharge"], bs["discharge_start"], 1.0, f"bdis_{u}")
        if (u + 1) % 5 == 0 or u + 1 == users_count:
            _progress(progress_callback, f"Eszköz- és energiamérleg-korlátok: {u + 1}/{users_count}")

    _progress(progress_callback, "Közösségi energiamérleg-korlátok létrehozása")
    for t in times:
        model += pulp.lpSum(v[u]["pv_shared"][t] for u in users) == pulp.lpSum(
            v[u]["shared_base"][t] + v[u]["shared_boiler"][t] for u in users
        )

    _progress(progress_callback, "Egyedi A/B tarifakeretek létrehozása")
    bill_terms = []
    for u in users:
        # Az aggregált passzív oszlophoz az eredeti háztartások éves
        # tarifakereteit külön-külön tartjuk meg. Csak a T×U flow-változók
        # tűnnek el; a számlázás nem válik egyetlen mérőponttá.
        if aggregation_used and u == aggregate_model_index:
            aggregate_load = p_base[:, u]
            for original_u in passive_indices:
                a_cap = float(original["a_low_remaining"][original_u])
                weights = np.divide(
                    original["p_base"][:, original_u], aggregate_load,
                    out=np.zeros(steps_count, dtype=float), where=aggregate_load > EPS,
                )
                tag = f"passive_{original_u}"
                al = pulp.LpVariable(f"a_low_{tag}", 0, a_cap)
                ah = pulp.LpVariable(f"a_high_{tag}", 0)
                sl = pulp.LpVariable(f"shared_low_{tag}", 0, a_cap)
                sh = pulp.LpVariable(f"shared_high_{tag}", 0)
                tier_switch = pulp.LpVariable(f"shared_tier_switch_{tag}", 0, 1, cat=binary)
                grid_annual = dt * pulp.lpSum(weights[t] * v[u]["grid_base"][t] for t in times)
                shared_annual = dt * pulp.lpSum(weights[t] * v[u]["shared_base"][t] for t in times)
                model += al + ah == grid_annual
                model += sl + sh == shared_annual
                model += al <= a_cap - sl
                tier_m = a_cap + float(original["p_base"][:, original_u].sum() * dt) + 1.0
                model += sl <= shared_annual
                model += sl >= shared_annual - tier_m * (1 - tier_switch)
                model += sl >= a_cap - tier_m * tier_switch
                bill_terms.append(PRICE_GRID_A_LOW * al + PRICE_GRID_A_HIGH * ah)
            continue

        a_cap = float(a_low_remaining[u])
        b_cap = float(b_low_remaining[u])
        al = pulp.LpVariable(f"a_low_{u}", 0, a_cap)
        ah = pulp.LpVariable(f"a_high_{u}", 0)
        sl = pulp.LpVariable(f"shared_low_{u}", 0, a_cap)
        sh = pulp.LpVariable(f"shared_high_{u}", 0)
        tier_switch = pulp.LpVariable(f"shared_tier_switch_{u}", 0, 1, cat=binary)
        bl = pulp.LpVariable(f"b_low_{u}", 0, b_cap)
        bh = pulp.LpVariable(f"b_high_{u}", 0)
        model += al + ah == dt * pulp.lpSum(
            v[u]["grid_base"][t] + v[u]["grid_bess"][t]
            + (v[u]["grid_boiler"][t] if hss_enabled[u] else 0)
            for t in times
        )
        shared_annual = dt * pulp.lpSum(v[u]["shared_base"][t] + v[u]["shared_boiler"][t] for t in times)
        model += sl + sh == shared_annual
        model += al <= a_cap - sl
        # A megosztott energia fogyasztja először az A kedvezményes sávját.
        tier_m = a_cap + float((p_base[:, u].sum() + size_elh[u] * steps_count) * dt) + 1.0
        model += sl <= shared_annual
        model += sl >= shared_annual - tier_m * (1 - tier_switch)
        model += sl >= a_cap - tier_m * tier_switch
        model += bl + bh == float(p_boiler_fixed[:, u].sum() * dt)
        bill_terms.append(
            PRICE_GRID_A_LOW * al + PRICE_GRID_A_HIGH * ah
            + PRICE_GRID_B_LOW * bl + PRICE_GRID_B_HIGH * bh
        )

    external_import = pulp.lpSum(dt * (v[u]["grid_base"][t] + v[u]["grid_boiler"][t] + v[u]["grid_bess"][t]) for u in users for t in times)
    external_export = pulp.lpSum(dt * v[u]["pv_export"][t] for u in users for t in times)
    shared_energy = pulp.lpSum(dt * (v[u]["shared_base"][t] + v[u]["shared_boiler"][t]) for u in users for t in times)
    thermal_shortfall_energy = pulp.lpSum(
        dt * pulp.lpSum(hss_state[u]["dhw_shortfall"])
        + pulp.lpSum(hss_state[u]["setpoint_shortfall"].values())
        + hss_state[u]["terminal_shortfall"]
        for u in hss_state
    )
    if objective == "grid":
        model += (
            external_import + external_export
            + thermal_shortfall_penalty_ft_per_kwh * thermal_shortfall_energy
        )
    else:
        model += (
            pulp.lpSum(bill_terms) - PRICE_PV_GRID * external_export
            + SHARED_RHD * shared_energy
            + thermal_shortfall_penalty_ft_per_kwh * thermal_shortfall_energy
        )

    _progress(progress_callback, f"Solver indítása: {solver}; változók={len(model.variables())}; korlátok={len(model.constraints)}")
    if solver == "gurobi":
        # A Linuxos környezetben ellenőrzött gurobipy interfész.
        # Nem igényli a külön gurobi_cl parancssori programot.
        command = pulp.GUROBI(msg=msg, gapRel=0.05)
    elif solver == "cbc":
        command = pulp.PULP_CBC_CMD(msg=msg)
    else:
        raise ValueError("solver csak 'gurobi' vagy 'cbc' lehet.")
    status = model.solve(command)
    status_text = pulp.LpStatus.get(status, str(status))
    _progress(progress_callback, f"Solver befejeződött: {status_text}")
    if status_text not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(f"Az optimalizálás nem adott érvényes megoldást: {status_text}")

    def extract(key):
        return np.column_stack([_values(v[u][key]) for u in users])

    _progress(progress_callback, "Solvereredmények kiolvasása")
    out_model = {key: extract(key) for key in v[0]}
    p_elh = np.zeros_like(p_pv)
    temp = np.full_like(p_pv, np.nan)
    e_hss = np.zeros_like(p_pv)
    p_dhw_out = np.zeros_like(p_pv)
    p_dhw_shortfall = np.zeros_like(p_pv)
    setpoint_shortfall = np.zeros_like(p_pv)
    boiler_target = np.full_like(p_pv, np.nan)
    d_cl = np.zeros_like(p_pv)
    e_bess = np.zeros_like(p_pv)
    d_charge = np.zeros_like(p_pv)
    d_discharge = np.zeros_like(p_pv)
    final_hss_energy_model = np.asarray(hss_initial_energy, dtype=float).copy()
    final_bess_energy_model = np.asarray(bess_initial_energy, dtype=float).copy()
    terminal_shortfall_model = np.zeros(users_count, dtype=float)
    for u in users:
        if hss_enabled[u]:
            hs = hss_state[u]
            p_elh[:, u] = _values(hs["p_elh"])
            energy_values = _values(hs["energy"])
            e_hss[:, u] = energy_values[:steps_count]
            final_hss_energy_model[u] = energy_values[-1]
            temp[:, u] = T_in[u] + e_hss[:, u] / (volume[u] * 0.00116667)
            p_dhw_out[:, u] = _values(hs["dhw_out"])
            p_dhw_shortfall[:, u] = _values(hs["dhw_shortfall"])
            d_cl[:, u] = (p_elh[:, u] > 1e-6).astype(float)
            boiler_target[:, u] = T_setpoint_min[u]
            for start_step, setpoint_var in hs["setpoint"].items():
                target_value = float(pulp.value(setpoint_var) or T_setpoint_min[u])
                shortfall_value = float(
                    pulp.value(hs["setpoint_shortfall"][start_step]) or 0.0
                )
                setpoint_shortfall[start_step, u] = shortfall_value
                day_start = (start_step // day_steps) * day_steps
                window = morning_block if (start_step - day_start) * dt < 12 else evening_block
                for k in range(day_start, min(day_start + day_steps, steps_count)):
                    if in_window((k * dt) % 24.0, window):
                        boiler_target[k, u] = target_value
            if enforce_terminal_state:
                terminal_shortfall_model[u] = float(
                    pulp.value(hs["terminal_shortfall"]) or 0.0
                )
        if bess_enabled[u]:
            energy_values = _values(bess_state[u]["energy"])
            e_bess[:, u] = energy_values[:steps_count]
            final_bess_energy_model[u] = energy_values[-1]
            d_charge[:, u] = _values(bess_state[u]["charge"])
            d_discharge[:, u] = _values(bess_state[u]["discharge"])

    if aggregation_used:
        def expand_model_matrix(value, fill_value=0.0):
            expanded = np.full((steps_count, original_users_count), fill_value, dtype=float)
            for model_u, original_u in enumerate(active_indices):
                expanded[:, original_u] = value[:, model_u]
            return expanded

        out = {key: expand_model_matrix(value) for key, value in out_model.items()}
        p_elh = expand_model_matrix(p_elh)
        temp = expand_model_matrix(temp, np.nan)
        e_hss = expand_model_matrix(e_hss)
        p_dhw_out = expand_model_matrix(p_dhw_out)
        p_dhw_shortfall = expand_model_matrix(p_dhw_shortfall)
        setpoint_shortfall = expand_model_matrix(setpoint_shortfall)
        boiler_target = expand_model_matrix(boiler_target, np.nan)
        d_cl = expand_model_matrix(d_cl)
        e_bess = expand_model_matrix(e_bess)
        d_charge = expand_model_matrix(d_charge)
        d_discharge = expand_model_matrix(d_discharge)
        final_hss_energy = original["hss_initial_energy"].copy()
        final_bess_energy = original["bess_initial_energy"].copy()
        terminal_shortfall = np.zeros(original_users_count, dtype=float)
        for model_u, original_u in enumerate(active_indices):
            final_hss_energy[original_u] = final_hss_energy_model[model_u]
            final_bess_energy[original_u] = final_bess_energy_model[model_u]
            terminal_shortfall[original_u] = terminal_shortfall_model[model_u]
    else:
        out = out_model
        final_hss_energy = final_hss_energy_model
        final_bess_energy = final_bess_energy_model
        terminal_shortfall = terminal_shortfall_model

    # Eredmény- és elszámolási oldalon ismét minden eredeti háztartás külön
    # szerepel. A proportional kiosztás így közvetlenül a 15 perces eredeti
    # hiányokra fut, vagyis nincs éves arányú közelítés.
    p_pv = original["p_pv"]
    p_base = original["p_base"]
    p_dhw = original["p_dhw"]
    p_boiler_fixed = original["p_boiler_fixed"]
    user_names = original["user_names"]
    hss_enabled = original["hss_enabled"]
    bess_enabled = original["bess_enabled"]
    a_low_remaining_initial = original["a_low_remaining"]
    b_low_remaining_initial = original["b_low_remaining"]
    users_count = original_users_count
    users = range(users_count)

    _progress(progress_callback, "15 perces determinisztikus megosztás és visszaosztás")
    # A proporcionális kiosztási kulcs pontosan max(A-tarifás fogyasztás - PV, 0).
    # A BESS nincs benne a kulcsban. A lenti need_* tömbökben csak azért marad
    # meg, mert ezek a már elkészült menetrend fizikai maradékigényét korlátozzák.
    # A mért, B tarifás bojler nincs az A-tarifás fogyasztásban, ezért sem saját
    # PV-t, sem megosztott energiát nem kaphat.
    allocation_need = np.maximum(p_base + p_elh - p_pv, 0.0)
    need_base = np.maximum(p_base - out["pv_base"] - out["bess_base"], 0.0)
    need_boiler = np.maximum(p_elh - out["pv_boiler"] - out["bess_boiler"], 0.0)
    surplus = np.maximum(p_pv - out["pv_base"] - out["pv_boiler"] - out["pv_bess"], 0.0)
    shared = _allocate_sharing(
        surplus, need_base, need_boiler, sharing_mode,
        allocation_need=allocation_need,
    )
    grid_a = shared["p_grid_base"] + shared["p_grid_boiler_a"] + out["grid_bess"]
    grid_b = p_boiler_fixed
    settlement = _settle_shared(
        shared["p_shared_in"], shared["p_shared_out"], grid_a, dt,
        sharing_mode, a_low_remaining_kwh=a_low_remaining_initial,
    )

    rows = []
    final_b_low_remaining = np.zeros(users_count, dtype=float)
    for u in users:
        import_a_cost = PRICE_GRID_A_LOW * settlement["grid_a_low_kwh"][u] + PRICE_GRID_A_HIGH * settlement["grid_a_high_kwh"][u]
        b_energy = float(grid_b[:, u].sum() * dt)
        b_low_u = min(b_energy, float(b_low_remaining_initial[u]))
        b_high_u = max(b_energy - b_low_u, 0.0)
        final_b_low_remaining[u] = max(float(b_low_remaining_initial[u]) - b_low_u, 0.0)
        import_b_cost = PRICE_GRID_B_LOW * b_low_u + PRICE_GRID_B_HIGH * b_high_u
        shared_cost = settlement["buyer_energy_ft"][u] + settlement["buyer_rhd_ft"][u]
        export_revenue = float(shared["p_pv_export"][:, u].sum() * dt * PRICE_PV_GRID)
        bill = import_a_cost + import_b_cost + shared_cost - export_revenue - settlement["seller_revenue_ft"][u]
        pv_energy = float(p_pv[:, u].sum() * dt)
        total_load = float((p_base[:, u] + p_elh[:, u] + grid_b[:, u]).sum() * dt)
        used_pv = float((out["pv_base"][:, u] + out["pv_boiler"][:, u] + out["pv_bess"][:, u] + shared["p_shared_out"][:, u]).sum() * dt)
        local_supply = float((out["pv_base"][:, u] + out["pv_boiler"][:, u] + out["bess_base"][:, u] + out["bess_boiler"][:, u] + shared["p_shared_in"][:, u]).sum() * dt)
        dhw_shortfall_u = float(p_dhw_shortfall[:, u].sum() * dt)
        setpoint_shortfall_u = float(
            setpoint_shortfall[:, u].sum() + terminal_shortfall[u]
        )
        if dhw_shortfall_u > 1e-6 and setpoint_shortfall_u > 1e-6:
            boiler_status = "FAIL_DHW_AND_SETPOINT"
        elif dhw_shortfall_u > 1e-6:
            boiler_status = "FAIL_DHW"
        elif setpoint_shortfall_u > 1e-6:
            boiler_status = "FAIL_SETPOINT"
        else:
            boiler_status = "PASS"
        rows.append({
            "household": user_names[u], "has_pv": int(p_pv[:, u].sum() > EPS),
            "has_bess": int(bess_enabled[u]), "hss_optimized": int(hss_enabled[u]),
            "boiler_tariff": "A" if hss_enabled[u] else ("B" if b_energy > EPS else ""),
            "pv_generation_kwh": pv_energy,
            "base_load_kwh": float(p_base[:, u].sum() * dt),
            "boiler_electric_kwh": float((p_elh[:, u] + grid_b[:, u]).sum() * dt),
            "total_load_kwh": total_load, "used_pv_kwh": used_pv,
            "local_supply_kwh": local_supply,
            "shared_in_kwh": float(shared["p_shared_in"][:, u].sum() * dt),
            "shared_out_kwh": float(shared["p_shared_out"][:, u].sum() * dt),
            "shared_to_boiler_kwh": float(shared["p_shared_boiler"][:, u].sum() * dt),
            "grid_import_a_kwh": float(grid_a[:, u].sum() * dt),
            "grid_import_b_kwh": b_energy,
            "grid_import_kwh": float((grid_a[:, u] + grid_b[:, u]).sum() * dt),
            "grid_export_kwh": float(shared["p_pv_export"][:, u].sum() * dt),
            "self_consumption_ratio": used_pv / pv_energy if pv_energy > EPS else 0.0,
            "self_sufficiency_ratio": local_supply / total_load if total_load > EPS else 0.0,
            "import_cost_a_ft": float(import_a_cost), "import_cost_b_ft": float(import_b_cost),
            "shared_purchase_cost_ft": float(shared_cost),
            "shared_revenue_ft": float(settlement["seller_revenue_ft"][u]),
            "export_revenue_ft": export_revenue, "bill_ft": float(bill),
            "dhw_energy_shortfall_kwhth": dhw_shortfall_u,
            "setpoint_shortfall_kwhth": setpoint_shortfall_u,
            "boiler_control_status": boiler_status,
            "a_low_remaining_kwh": float(settlement["a_low_remaining_kwh"][u]),
            "b_low_remaining_kwh": float(final_b_low_remaining[u]),
            "status": status_text,
        })
    household_summary = pd.DataFrame(rows)
    planned_shared = sum(float(out["pv_shared"][:, u].sum() * dt) for u in users)
    settled_shared = float(shared["p_shared_in"].sum() * dt)
    eligible_need = (need_base + need_boiler) * (allocation_need > EPS)
    maximum_share = float(np.minimum(surplus.sum(axis=1), eligible_need.sum(axis=1)).sum() * dt)
    summary = {
        "case": "4d-K-E" if objective == "grid" else "4d-K-C", "status": status_text,
        "objective": objective, "objective_value_solver": float(pulp.value(model.objective)),
        "n_households": users_count, "dt": dt, "sharing_mode": sharing_mode,
        "sharing_need_policy": "max(base_load + optimized_A_boiler - PV, 0); BESS_excluded",
        "passive_aggregation_enabled": bool(aggregate_passive),
        "passive_aggregation_used": aggregation_used,
        "passive_household_count": int(passive_indices.size),
        "solver_user_count": int(users_count - passive_indices.size + 1) if aggregation_used else int(users_count),
        "boiler_tariff_policy": "A_dynamic_PV_HSS__B_measured",
        "hss_optimized_count": int(hss_enabled.sum()), "bess_enabled_count": int(bess_enabled.sum()),
        "optimizer_shared_kwh": planned_shared, "maximum_share_before_equal_quotas_kwh": maximum_share,
        "settled_shared_kwh": settled_shared,
        "settlement_differs_from_optimizer": bool(abs(planned_shared - settled_shared) > 1e-6),
        "equal_unused_quota_kwh": max(maximum_share - settled_shared, 0.0) if sharing_mode == "equal" else 0.0,
        "total_grid_import_a_kwh": float(grid_a.sum() * dt), "total_grid_import_b_kwh": float(grid_b.sum() * dt),
        "total_grid_export_kwh": float(shared["p_pv_export"].sum() * dt),
        "total_shared_to_boiler_kwh": float(shared["p_shared_boiler"].sum() * dt),
        "total_bill_ft": float(household_summary.bill_ft.sum()),
        "total_dhw_energy_shortfall_kwhth": float(
            household_summary.dhw_energy_shortfall_kwhth.sum()
        ),
        "total_setpoint_shortfall_kwhth": float(
            household_summary.setpoint_shortfall_kwhth.sum()
        ),
        "boiler_block_policy": "no_source_06-09_and_17-22",
        "boiler_setpoint_range_c": [
            float(np.min(T_setpoint_min[hss_enabled])) if hss_enabled.any() else None,
            float(np.max(T_max[hss_enabled])) if hss_enabled.any() else None,
        ],
    }
    summary["final_grid_interaction_kwh"] = (
        summary["total_grid_import_a_kwh"] + summary["total_grid_import_b_kwh"]
        + summary["total_grid_export_kwh"]
    )
    pv_total = float(p_pv.sum() * dt)
    load_total = float((p_base + p_elh + grid_b).sum() * dt)
    used_pv_total = float((out["pv_base"] + out["pv_boiler"] + out["pv_bess"] + shared["p_shared_out"]).sum() * dt)
    local_total = float((out["pv_base"] + out["pv_boiler"] + out["bess_base"] + out["bess_boiler"] + shared["p_shared_in"]).sum() * dt)
    summary["SCI"] = used_pv_total / pv_total if pv_total > EPS else 0.0
    summary["SSI"] = local_total / load_total if load_total > EPS else 0.0
    timeseries = {
        "p_pv": p_pv, "p_base": p_base, "p_dhw": p_dhw, "p_boiler_fixed": grid_b,
        "p_elh": p_elh, "p_pv_base": out["pv_base"], "p_pv_boiler": out["pv_boiler"],
        "p_pv_bess": out["pv_bess"], "p_shared_out": shared["p_shared_out"],
        "p_pv_export": shared["p_pv_export"], "p_bess_base": out["bess_base"],
        "p_bess_boiler": out["bess_boiler"], "p_grid_bess": out["grid_bess"],
        "p_shared_in": shared["p_shared_in"], "p_shared_base": shared["p_shared_base"],
        "p_shared_boiler": shared["p_shared_boiler"], "p_grid_base": shared["p_grid_base"],
        "p_grid_boiler_a": shared["p_grid_boiler_a"], "p_grid_import_a": grid_a,
        "p_grid_import_b": grid_b, "e_bess": e_bess, "d_bess_charge": d_charge,
        "d_bess_discharge": d_discharge, "e_hss": e_hss,
        "t_hss_equivalent": temp, "p_dhw_out": p_dhw_out,
        "p_dhw_shortfall": p_dhw_shortfall,
        "setpoint_shortfall_kwhth": setpoint_shortfall,
        "boiler_target_c": boiler_target, "d_cl": d_cl,
    }
    _progress(progress_callback, "Egyedi számlák és eredmény-idősorok elkészültek")
    return {"summary": summary, "household_summary": household_summary, "timeseries": timeseries,
            "pair_kwh": settlement["pair_kwh"], "pair_payment_ft": settlement["pair_payment_ft"],
            "user_names": list(user_names),
            "state": {
                "final_hss_energy_kwhth": np.asarray(final_hss_energy, dtype=float),
                "final_bess_energy_kwh": np.asarray(final_bess_energy, dtype=float),
                "a_low_remaining_kwh": np.asarray(
                    settlement["a_low_remaining_kwh"], dtype=float
                ),
                "b_low_remaining_kwh": np.asarray(final_b_low_remaining, dtype=float),
            }}


def save_community_results(result, out_dir, *, save_user_timeseries=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = result["user_names"]
    result["household_summary"].to_csv(out / "household_summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
    if "state" in result:
        pd.DataFrame({
            "household": names,
            **{
                key: np.asarray(value, dtype=float)
                for key, value in result["state"].items()
            },
        }).to_csv(out / "final_state.csv", index=False)
    ts = result["timeseries"]
    # Egyetlen teljes idősorfájl-készlet marad; nincs azonos tartalmú alias.
    for key, array in ts.items():
        pd.DataFrame(array, columns=names).to_csv(out / f"{key}.csv", index=False)
    pd.DataFrame(result["pair_kwh"], index=names, columns=names).to_csv(out / "shared_pair_kwh.csv")
    pd.DataFrame(result["pair_payment_ft"], index=names, columns=names).to_csv(out / "shared_pair_payment_ft.csv")
    community = pd.DataFrame({key: value.sum(axis=1) for key, value in ts.items()})
    community.to_csv(out / "community_timeseries.csv", index=False)
    if save_user_timeseries:
        for u, name in enumerate(names):
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
            pd.DataFrame({key: value[:, u] for key, value in ts.items()}).to_csv(out / f"timeseries_{safe}.csv", index=False)