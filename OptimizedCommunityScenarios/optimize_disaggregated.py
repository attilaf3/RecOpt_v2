import numpy as np
import pulp
from pandas import DataFrame
from pulp import LpStatus, LpStatusOptimal


def _as_vec(x, U, default=0.0):
    """Broadcast scalar or 1D array-like to length-U numpy array."""
    if x is None:
        return np.full(U, default, dtype=float)
    if np.isscalar(x):
        return np.full(U, float(x), dtype=float)
    x = np.asarray(x, dtype=float)
    assert x.shape == (U,), f"Parameter must be scalar or length-U ({U}), got {x.shape}"
    return x


def _as_matrix(x, shape, default=0.0, name="parameter"):
    """Return ndarray with given shape, or default-filled matrix if x is None."""
    if x is None:
        return np.full(shape, default, dtype=float)
    x = np.asarray(x, dtype=float)
    assert x.shape == shape, f"{name} must have shape {shape}, got {x.shape}"
    return x


def optimize_multi_users_economic(
    p_pv,                   # shape (T, U)
    p_ue,                   # shape (T, U)
    p_dhw=None,             # shape (T, U), optional
    p_el_heater=None,       # shape (T, U), only used if hss_flag=False
    dt=1,
    size_elh=None,          # scalar or shape (U,)
    size_bess=None,         # scalar or shape (U,)
    vol_hss_water=None,     # scalar or shape (U,)
    **kwargs
):
    """
    Multi-user economic optimization with:
      - per-user PV
      - per-user BESS
      - optional per-user HSS/ELH
      - no community battery
      - REC direct sharing
      - economic objective

    Main logic:
      PV -> own load / own BESS / REC / external grid
      own BESS -> own load only
      grid -> own load and own BESS
      REC -> own load / own ELH
    """

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------
    p_pv = np.asarray(p_pv, dtype=float)
    p_ue = np.asarray(p_ue, dtype=float)
    assert p_pv.shape == p_ue.shape, f"p_pv and p_ue must have same shape, got {p_pv.shape} vs {p_ue.shape}"

    T, U = p_pv.shape
    shape = (T, U)
    time_set = range(T)
    users = range(U)

    hss_flag = kwargs.get("hss_flag", False)
    run_lp = kwargs.get("run_lp", False)

    p_dhw = _as_matrix(p_dhw, shape, default=0.0, name="p_dhw")
    p_el_heater = _as_matrix(p_el_heater, shape, default=0.0, name="p_el_heater")

    # -------------------------------------------------------------------------
    # Technical params
    # -------------------------------------------------------------------------
    eta_bess_in = _as_vec(kwargs.get("eta_bess_in", 0.96), U)
    eta_bess_out = _as_vec(kwargs.get("eta_bess_out", 0.96), U)
    eta_bess_stor = _as_vec(kwargs.get("eta_bess_stor", 0.995), U)
    t_bess_min = _as_vec(kwargs.get("t_bess_min", 2.0), U)

    soc_bess_min = _as_vec(kwargs.get("soc_bess_min", 0.10), U)
    soc_bess_max = _as_vec(kwargs.get("soc_bess_max", 0.90), U)
    soc_bess_init = _as_vec(kwargs.get("soc_bess_init", soc_bess_min), U)

    size_bess_u = _as_vec(size_bess, U, default=0.0)
    bess_active_u = size_bess_u > 1e-9
    battery_power_u = np.divide(
        size_bess_u,
        np.maximum(t_bess_min, 1e-9),
        out=np.zeros(U, dtype=float),
        where=t_bess_min > 1e-9
    )

    # HSS / ELH params
    size_elh_u = _as_vec(size_elh, U, default=0.0)
    eta_elh_u = _as_vec(kwargs.get("eta_elh", 1.0), U)
    vol_hss_water_u = _as_vec(vol_hss_water, U, default=120.0)
    c_hss = kwargs.get("c_hss", 0.00116667)
    a_hss_u = _as_vec(kwargs.get("a_hss", [0.01275] * U), U)
    T_in_u = _as_vec(kwargs.get("T_in", [10.0] * U), U)
    T_env_u = _as_vec(kwargs.get("T_env", 20.0), U)
    T_min_u = _as_vec(kwargs.get("T_min", T_in_u), U)
    T_max_u = _as_vec(kwargs.get("T_max", [55.0] * U), U)
    t_hss_min_in_u = _as_vec(kwargs.get("t_hss_min_in", [0.0] * U), U)

    hss_active_u = hss_flag & (size_elh_u > 1e-9)

    # Daily binary logic for ELH if HSS is active
    enforce_daily_cl_rules = kwargs.get("enforce_daily_cl_rules", True)

    # -------------------------------------------------------------------------
    # Economic params
    # -------------------------------------------------------------------------
    price_grid_low = kwargs.get("price_grid_low", 36.0)      # Ft/kWh
    price_grid_high = kwargs.get("price_grid_high", 71.0)    # Ft/kWh
    price_pv_grid = kwargs.get("price_pv_grid", 5.0)         # Ft/kWh
    price_rec_sell = kwargs.get("price_rec_sell", 20.0)      # Ft/kWh
    price_rec_buy = kwargs.get("price_rec_buy", 25.0)        # Ft/kWh

    # Annual / horizon low-tariff cap per user [kWh]
    grid_low_cap_kwh = _as_vec(kwargs.get("grid_low_cap_kwh", 2523.0), U)

    # -------------------------------------------------------------------------
    # Solver params
    # -------------------------------------------------------------------------
    msg = kwargs.get("msg", True)
    gapRel = kwargs.get("gapRel", None)
    timeLimit = kwargs.get("timeLimit", None)

    # -------------------------------------------------------------------------
    # Problem
    # -------------------------------------------------------------------------
    prob = pulp.LpProblem("CSCopt_multi_users_economic", pulp.LpMinimize)

    # Big-M caps
    max_total_load = float(np.max(np.sum(p_ue, axis=1))) if T > 0 else 0.0
    max_total_pv = float(np.max(np.sum(p_pv, axis=1))) if T > 0 else 0.0
    max_total_dhw_el = float(np.max(np.sum(p_dhw / np.maximum(eta_elh_u, 1e-9), axis=1))) if T > 0 else 0.0
    M_rec = max_total_pv + float(np.sum(battery_power_u))
    M_grid = max_total_load + max_total_dhw_el + float(np.sum(battery_power_u)) + 1.0

    # -------------------------------------------------------------------------
    # Variables
    # -------------------------------------------------------------------------
    # PV split
    p_pv_load = [[pulp.LpVariable(f"Ppv_load_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_pv_bess = [[pulp.LpVariable(f"Ppv_bess_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_pv_rec = [[pulp.LpVariable(f"Ppv_rec_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_pv_grid = [[pulp.LpVariable(f"Ppv_grid_{t}_{u}", lowBound=0) for u in users] for t in time_set]

    # User load supply split
    p_bess_load = [[pulp.LpVariable(f"Pbess_load_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_rec_load = [[pulp.LpVariable(f"Prec_load_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_grid_load = [[pulp.LpVariable(f"Pgrid_load_{t}_{u}", lowBound=0) for u in users] for t in time_set]

    # Per-user battery
    p_bess_in = [[pulp.LpVariable(f"Pbess_in_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_bess_out = [[pulp.LpVariable(f"Pbess_out_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_grid_bess = [[pulp.LpVariable(f"Pgrid_bess_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    e_bess = [[
        pulp.LpVariable(
            f"Ebess_{t}_{u}",
            lowBound=size_bess_u[u] * soc_bess_min[u],
            upBound=size_bess_u[u] * soc_bess_max[u]
        ) if bess_active_u[u] else 0
        for u in users
    ] for t in time_set]
    d_bess = [[
        pulp.LpVariable(f"Dbess_{t}_{u}", cat=pulp.LpBinary) if (bess_active_u[u] and not run_lp) else 0
        for u in users
    ] for t in time_set]

    # Optional direct grid export/import aggregate
    p_grid_import = [pulp.LpVariable(f"Pgrid_import_{t}", lowBound=0) for t in time_set]
    p_grid_export = [pulp.LpVariable(f"Pgrid_export_{t}", lowBound=0) for t in time_set]
    d_grid = [pulp.LpVariable(f"Dgrid_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    # Community interface summaries
    p_inj_user = [[pulp.LpVariable(f"Pinj_user_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_with_user = [[pulp.LpVariable(f"Pwith_user_{t}_{u}", lowBound=0) for u in users] for t in time_set]
    p_inj_comm = [pulp.LpVariable(f"Pinj_comm_{t}", lowBound=0) for t in time_set]
    p_with_comm = [pulp.LpVariable(f"Pwith_comm_{t}", lowBound=0) for t in time_set]

    # Low/high tariff block variables per user
    e_grid_low = [pulp.LpVariable(f"Egrid_low_{u}", lowBound=0) for u in users]
    e_grid_high = [pulp.LpVariable(f"Egrid_high_{u}", lowBound=0) for u in users]

    # HSS / ELH variables
    if hss_flag:
        p_elh_in = [[pulp.LpVariable(f"Pelh_in_{t}_{u}", lowBound=0) for u in users] for t in time_set]
        p_elh_out = [[pulp.LpVariable(f"Pelh_out_{t}_{u}", lowBound=0) for u in users] for t in time_set]

        # ELH energy source split
        p_pv_elh = [[pulp.LpVariable(f"Ppv_elh_{t}_{u}", lowBound=0) for u in users] for t in time_set]
        p_rec_elh = [[pulp.LpVariable(f"Prec_elh_{t}_{u}", lowBound=0) for u in users] for t in time_set]
        p_grid_elh = [[pulp.LpVariable(f"Pgrid_elh_{t}_{u}", lowBound=0) for u in users] for t in time_set]

        # Optional binary on/off
        d_cl = [[pulp.LpVariable(f"Dcl_{t}_{u}", cat=pulp.LpBinary) if not run_lp else 0 for u in users] for t in time_set]

        p_hss_in = [[pulp.LpVariable(f"Phss_in_{t}_{u}", lowBound=0) for u in users] for t in time_set]
        p_hss_out = [[pulp.LpVariable(f"Phss_out_{t}_{u}", lowBound=0) for u in users] for t in time_set]
        t_hss = [[
            pulp.LpVariable(f"Thss_{t}_{u}", lowBound=T_min_u[u], upBound=T_max_u[u])
            for u in users
        ] for t in time_set]
    else:
        p_elh_in = None
        p_elh_out = None
        p_pv_elh = None
        p_rec_elh = None
        p_grid_elh = None
        d_cl = None
        p_hss_in = None
        p_hss_out = None
        t_hss = None

    # -------------------------------------------------------------------------
    # Constraints
    # -------------------------------------------------------------------------
    for t in time_set:
        k = (t + 1) % T

        # ---------------------------
        # User-level balances
        # ---------------------------
        for u in users:
            # PV split
            prob += (
                    p_pv_load[t][u] + p_pv_bess[t][u] + p_pv_rec[t][u] + p_pv_grid[t][u] + (
                p_pv_elh[t][u] if hss_flag else 0)
                    == p_pv[t, u]
            ), f"{t}_{u}_pv_split"

            # Battery availability
            if not bess_active_u[u]:
                prob += p_pv_bess[t][u] == 0, f"{t}_{u}_no_bess_pv_bess"
                prob += p_bess_load[t][u] == 0, f"{t}_{u}_no_bess_bess_load"
                prob += p_bess_in[t][u] == 0, f"{t}_{u}_no_bess_bess_in"
                prob += p_bess_out[t][u] == 0, f"{t}_{u}_no_bess_bess_out"
                prob += p_grid_bess[t][u] == 0, f"{t}_{u}_no_bess_grid_bess"
            else:
                # Battery charging sources
                prob += p_bess_in[t][u] == p_pv_bess[t][u] + p_grid_bess[t][u], f"{t}_{u}_bess_in_def"

                # Battery discharge destination
                prob += p_bess_out[t][u] == p_bess_load[t][u], f"{t}_{u}_bess_out_def"

                # Battery dynamics
                prob += (
                    e_bess[k][u] ==
                    e_bess[t][u] * eta_bess_stor[u] +
                    (
                        p_bess_in[t][u] * eta_bess_in[u] -
                        p_bess_out[t][u] / max(eta_bess_out[u], 1e-9)
                    ) * dt
                ), f"{t}_{u}_bess_balance"

                # Power limits
                if run_lp:
                    prob += p_bess_in[t][u] <= battery_power_u[u], f"{t}_{u}_bess_in_cap"
                    prob += p_bess_out[t][u] <= battery_power_u[u], f"{t}_{u}_bess_out_cap"
                else:
                    prob += p_bess_in[t][u] <= d_bess[t][u] * battery_power_u[u], f"{t}_{u}_bess_in_gate"
                    prob += p_bess_out[t][u] <= (1 - d_bess[t][u]) * battery_power_u[u], f"{t}_{u}_bess_out_gate"

                # Explicit SOC bounds
                prob += e_bess[t][u] >= size_bess_u[u] * soc_bess_min[u], f"{t}_{u}_bess_soc_min"
                prob += e_bess[t][u] <= size_bess_u[u] * soc_bess_max[u], f"{t}_{u}_bess_soc_max"

            # HSS / ELH case
            if hss_flag:
                if not hss_active_u[u]:
                    prob += p_elh_in[t][u] == 0, f"{t}_{u}_no_hss_elh_in"
                    prob += p_elh_out[t][u] == 0, f"{t}_{u}_no_hss_elh_out"
                    prob += p_pv_elh[t][u] == 0, f"{t}_{u}_no_hss_pv_elh"
                    prob += p_rec_elh[t][u] == 0, f"{t}_{u}_no_hss_rec_elh"
                    prob += p_grid_elh[t][u] == 0, f"{t}_{u}_no_hss_grid_elh"
                    prob += p_hss_in[t][u] == 0, f"{t}_{u}_no_hss_hss_in"
                    prob += p_hss_out[t][u] == 0, f"{t}_{u}_no_hss_hss_out"
                else:
                    # ELH source split
                    prob += (
                        p_elh_in[t][u] == p_pv_elh[t][u] + p_rec_elh[t][u] + p_grid_elh[t][u]
                    ), f"{t}_{u}_elh_source_split"

                    # ELH constitutive
                    prob += p_elh_out[t][u] == p_elh_in[t][u] * eta_elh_u[u], f"{t}_{u}_elh_const"
                    prob += p_elh_in[t][u] <= size_elh_u[u], f"{t}_{u}_elh_cap"

                    # HSS coupling
                    prob += p_hss_in[t][u] == p_elh_out[t][u], f"{t}_{u}_hss_in_def"
                    prob += p_hss_out[t][u] == p_dhw[t, u], f"{t}_{u}_dhw_balance"

                    # HSS thermal dynamics
                    prob += (
                        vol_hss_water_u[u] * c_hss * (t_hss[k][u] - t_hss[t][u]) ==
                        (
                            p_hss_in[t][u] -
                            p_hss_out[t][u] -
                            a_hss_u[u] * (t_hss[t][u] - T_env_u[u])
                        ) * dt
                    ), f"{t}_{u}_hss_balance"

                    # Available thermal output upper bound next step
                    prob += (
                        p_hss_out[k][u] <=
                        vol_hss_water_u[u] * c_hss * (t_hss[t][u] - T_in_u[u]) / dt
                    ), f"{t}_{u}_hss_max_out"

                    # Charging upper bound due to remaining thermal capacity
                    prob += (
                        p_hss_in[t][u] <=
                        vol_hss_water_u[u] * c_hss * (T_max_u[u] - t_hss[t][u]) / dt
                    ), f"{t}_{u}_hss_charge_space"

                    # Optional "min charging time" equivalent max charging power
                    if t_hss_min_in_u[u] > 1e-9:
                        E_hss_cap_u = vol_hss_water_u[u] * c_hss * (T_max_u[u] - T_min_u[u])
                        P_hss_in_max_u = E_hss_cap_u / t_hss_min_in_u[u]
                        prob += p_hss_in[t][u] <= P_hss_in_max_u, f"{t}_{u}_hss_in_cap_by_min_time"

                    # Optional binary gating of ELH operation
                    if not run_lp:
                        prob += p_elh_in[t][u] <= size_elh_u[u] * d_cl[t][u], f"{t}_{u}_elh_onoff"

                # User electric balance with ELH
                prob += (
                    p_pv_load[t][u] + p_bess_load[t][u] + p_rec_load[t][u] + p_grid_load[t][u] == p_ue[t, u]
                ), f"{t}_{u}_ue_balance"

            else:
                # No HSS: direct electric heater can be part of fixed load
                total_user_load = p_ue[t, u] + p_el_heater[t, u]
                prob += (
                    p_pv_load[t][u] + p_bess_load[t][u] + p_rec_load[t][u] + p_grid_load[t][u] == total_user_load
                ), f"{t}_{u}_load_balance"

            # Community interface definitions
            if hss_flag:
                prob += p_inj_user[t][u] == p_pv_rec[t][u], f"{t}_{u}_inj_user_def"
                prob += p_with_user[t][u] == p_rec_load[t][u] + p_rec_elh[t][u], f"{t}_{u}_with_user_def"
            else:
                prob += p_inj_user[t][u] == p_pv_rec[t][u], f"{t}_{u}_inj_user_def"
                prob += p_with_user[t][u] == p_rec_load[t][u], f"{t}_{u}_with_user_def"

        # ---------------------------
        # Community aggregate
        # ---------------------------
        prob += p_inj_comm[t] == pulp.lpSum(p_inj_user[t][u] for u in users), f"{t}_inj_comm_def"
        prob += p_with_comm[t] == pulp.lpSum(p_with_user[t][u] for u in users), f"{t}_with_comm_def"

        # REC direct sharing: surplus goes to grid export, shortage comes from grid import
        prob += p_inj_comm[t] + p_grid_import[t] == p_with_comm[t] + p_grid_export[t], f"{t}_community_grid_balance"

        # Prevent simultaneous grid import/export at community level
        if not run_lp:
            prob += p_grid_import[t] <= M_grid * d_grid[t], f"{t}_grid_import_gate"
            prob += p_grid_export[t] <= M_grid * (1 - d_grid[t]), f"{t}_grid_export_gate"

    # -------------------------------------------------------------------------
    # Daily ELH optional rules (only if explicitly requested)
    # -------------------------------------------------------------------------
    if hss_flag and enforce_daily_cl_rules and not run_lp:
        n_timesteps_in_a_day = round(24 / dt)
        assert abs(n_timesteps_in_a_day * dt - 24) < 1e-9, "dt must divide 24h exactly for daily CL rules"

        y_middle_day = [0] * int(round(10 / dt)) + [1] * int(round(6 / dt)) + [0] * (n_timesteps_in_a_day - int(round(16 / dt)))
        y_middle_day = y_middle_day[:n_timesteps_in_a_day]

        max_on_steps = kwargs.get("cl_max_on_hours_per_day", 12.0) / dt
        min_mid_steps = kwargs.get("cl_min_midday_hours_per_day", 4.0) / dt

        for j in range(0, T, n_timesteps_in_a_day):
            day_idx = range(j, min(j + n_timesteps_in_a_day, T))
            if len(list(day_idx)) < n_timesteps_in_a_day:
                continue

            for u in users:
                if not hss_active_u[u]:
                    continue
                prob += pulp.lpSum(d_cl[t][u] for t in day_idx) <= max_on_steps, f"day_{j}_{u}_cl_maxon"
                prob += pulp.lpSum(d_cl[t][u] * y_middle_day[t - j] for t in day_idx) >= min_mid_steps, f"day_{j}_{u}_cl_midmin"

    # -------------------------------------------------------------------------
    # Tariff block constraints
    # -------------------------------------------------------------------------
    for u in users:
        grid_energy_expr = pulp.lpSum(
            (
                p_grid_load[t][u] +
                p_grid_bess[t][u] +
                (p_grid_elh[t][u] if hss_flag else 0.0)
            )
            for t in time_set
        )

        prob += e_grid_low[u] + e_grid_high[u] == grid_energy_expr, f"{u}_grid_block_sum"
        prob += e_grid_low[u] <= grid_low_cap_kwh[u], f"{u}_grid_low_cap"

    # -------------------------------------------------------------------------
    # Initial SOC anchoring
    # -------------------------------------------------------------------------
    for u in users:
        if bess_active_u[u]:
            prob += e_bess[0][u] == size_bess_u[u] * soc_bess_init[u], f"{u}_bess_init"

    # -------------------------------------------------------------------------
    # Objective: total net community cost [Ft]
    # -------------------------------------------------------------------------
    cost_grid = pulp.lpSum(
        price_grid_low * e_grid_low[u] + price_grid_high * e_grid_high[u]
        for u in users
    )

    cost_rec_buy = pulp.lpSum(
        price_rec_buy * (
            p_rec_load[t][u] + (p_rec_elh[t][u] if hss_flag else 0.0)
        )
        for t in time_set for u in users
    )

    revenue_rec_sell = pulp.lpSum(
        price_rec_sell * p_pv_rec[t][u]
        for t in time_set for u in users
    )

    revenue_grid_export = pulp.lpSum(
        price_pv_grid * p_pv_grid[t][u]
        for t in time_set for u in users
    )


    prob += cost_grid + cost_rec_buy - revenue_rec_sell - revenue_grid_export

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    status = prob.solve(solver)

    # Accept optimal, and also feasible incumbents if solver stops early
    status_str = LpStatus.get(status, str(status))
    if status_str not in {"Optimal", "Not Solved", "Integer Feasible", "Undefined"}:
        raise RuntimeError(f"Unable to solve the problem! Solver status: {status_str}")

    # -------------------------------------------------------------------------
    # Collect results
    # -------------------------------------------------------------------------
    def val2d(x):
        return np.array([[pulp.value(x[t][u]) for u in users] for t in time_set], dtype=float)

    def val1d(x):
        return np.array([pulp.value(x[t]) for t in time_set], dtype=float)

    # Main outputs
    p_pv_load_v = val2d(p_pv_load)
    p_pv_bess_v = val2d(p_pv_bess)
    p_pv_rec_v = val2d(p_pv_rec)
    p_pv_grid_v = val2d(p_pv_grid)

    p_bess_load_v = val2d(p_bess_load)
    p_rec_load_v = val2d(p_rec_load)
    p_grid_load_v = val2d(p_grid_load)

    p_bess_in_v = val2d(p_bess_in)
    p_bess_out_v = val2d(p_bess_out)
    p_grid_bess_v = val2d(p_grid_bess)

    e_bess_v = np.array([
        [pulp.value(e_bess[t][u]) if bess_active_u[u] else 0.0 for u in users]
        for t in time_set
    ], dtype=float)

    d_bess_v = np.array([
        [pulp.value(d_bess[t][u]) if (bess_active_u[u] and not run_lp) else 0.0 for u in users]
        for t in time_set
    ], dtype=float)

    p_inj_user_v = val2d(p_inj_user)
    p_with_user_v = val2d(p_with_user)
    p_inj_comm_v = val1d(p_inj_comm)
    p_with_comm_v = val1d(p_with_comm)

    p_grid_import_v = val1d(p_grid_import)
    p_grid_export_v = val1d(p_grid_export)
    d_grid_v = np.array([pulp.value(d_grid[t]) if not run_lp else 0.0 for t in time_set], dtype=float)

    e_grid_low_v = np.array([pulp.value(v) for v in e_grid_low], dtype=float)
    e_grid_high_v = np.array([pulp.value(v) for v in e_grid_high], dtype=float)

    if hss_flag:
        p_elh_in_v = val2d(p_elh_in)
        p_elh_out_v = val2d(p_elh_out)
        p_pv_elh_v = val2d(p_pv_elh)
        p_rec_elh_v = val2d(p_rec_elh)
        p_grid_elh_v = val2d(p_grid_elh)
        p_hss_in_v = val2d(p_hss_in)
        p_hss_out_v = val2d(p_hss_out)
        t_hss_v = val2d(t_hss)
        d_cl_v = np.array([
            [pulp.value(d_cl[t][u]) if not run_lp else 0.0 for u in users]
            for t in time_set
        ], dtype=float)

        e_hss_stor_v = np.zeros((T, U), dtype=float)
        for u in users:
            if hss_active_u[u]:
                e_hss_stor_v[:, u] = vol_hss_water_u[u] * c_hss * (t_hss_v[:, u] - T_in_u[u])
    else:
        p_elh_in_v = None
        p_elh_out_v = None
        p_pv_elh_v = None
        p_rec_elh_v = None
        p_grid_elh_v = None
        p_hss_in_v = None
        p_hss_out_v = None
        t_hss_v = None
        d_cl_v = None
        e_hss_stor_v = None

    # Derived energies / costs
    e_grid_total_user = e_grid_low_v + e_grid_high_v

    rec_buy_energy_user = np.sum(
        p_rec_load_v + (p_rec_elh_v if hss_flag else 0.0),
        axis=0
    )

    rec_sell_energy_user = np.sum(p_pv_rec_v, axis=0)
    grid_export_energy_user = np.sum(p_pv_grid_v, axis=0)

    grid_cost_user = price_grid_low * e_grid_low_v + price_grid_high * e_grid_high_v
    rec_buy_cost_user = price_rec_buy * rec_buy_energy_user
    rec_sell_revenue_user = price_rec_sell * rec_sell_energy_user
    grid_export_revenue_user = price_pv_grid * grid_export_energy_user

    net_cost_user = (
        grid_cost_user +
        rec_buy_cost_user -
        rec_sell_revenue_user -
        grid_export_revenue_user
    )

    objective = float(pulp.value(prob.objective))

    # Community-level battery series (1D, summed across users)
    p_grid_bess_total_v = np.sum(p_grid_bess_v, axis=1)
    p_bess_in_total_v = np.sum(p_bess_in_v, axis=1)
    p_bess_out_total_v = np.sum(p_bess_out_v, axis=1)
    e_bess_total_v = np.sum(e_bess_v, axis=1)
    d_bess_any_v = (np.sum(d_bess_v, axis=1) > 0).astype(float)

    results = dict(
        # inputs
        p_pv=p_pv,
        p_ue=p_ue,
        p_dhw=p_dhw if hss_flag else None,
        p_el_heater=p_el_heater if not hss_flag else None,
        dt=dt,

        # params
        size_bess=size_bess_u,
        battery_power=battery_power_u,
        soc_bess_min=soc_bess_min,
        soc_bess_max=soc_bess_max,
        soc_bess_init=soc_bess_init,
        bess_active=bess_active_u.astype(int),

        price_grid_low=price_grid_low,
        price_grid_high=price_grid_high,
        price_pv_grid=price_pv_grid,
        price_rec_sell=price_rec_sell,
        price_rec_buy=price_rec_buy,
        grid_low_cap_kwh=grid_low_cap_kwh,

        # PV split
        p_pv_load=p_pv_load_v,
        p_pv_bess=p_pv_bess_v,
        p_pv_rec=p_pv_rec_v,
        p_pv_grid=p_pv_grid_v,

        # load supply split
        p_bess_load=p_bess_load_v,
        p_rec_load=p_rec_load_v,
        p_grid_load=p_grid_load_v,

        # battery
        p_bess_in=p_bess_in_v,
        p_bess_out=p_bess_out_v,
        p_grid_bess=p_grid_bess_v,
        e_bess=e_bess_v,
        d_bess=d_bess_v,

        # community
        p_inj_user=p_inj_user_v,
        p_with_user=p_with_user_v,
        p_inj_comm=p_inj_comm_v,
        p_with_comm=p_with_comm_v,
        p_grid_import=p_grid_import_v,
        p_grid_export=p_grid_export_v,
        d_grid=d_grid_v,

        # tariffs
        e_grid_low=e_grid_low_v,
        e_grid_high=e_grid_high_v,
        e_grid_total=e_grid_total_user,

        # user economics
        rec_buy_energy_user=rec_buy_energy_user,
        rec_sell_energy_user=rec_sell_energy_user,
        grid_export_energy_user=grid_export_energy_user,
        grid_cost_user=grid_cost_user,
        rec_buy_cost_user=rec_buy_cost_user,
        rec_sell_revenue_user=rec_sell_revenue_user,
        grid_export_revenue_user=grid_export_revenue_user,
        net_cost_user=net_cost_user,

        # totals
        total_grid_cost=float(np.sum(grid_cost_user)),
        total_rec_buy_cost=float(np.sum(rec_buy_cost_user)),
        total_rec_sell_revenue=float(np.sum(rec_sell_revenue_user)),
        total_grid_export_revenue=float(np.sum(grid_export_revenue_user)),
        objective=objective,

        # aliases expected by caller
        p_grid_in=p_grid_import_v,
        p_grid_out=p_grid_export_v,

        # community aggregated battery series
        p_grid_bess_total=p_grid_bess_total_v,
        p_bess_in_total=p_bess_in_total_v,
        p_bess_out_total=p_bess_out_total_v,
        e_bess_total=e_bess_total_v,
        d_bess_any=d_bess_any_v,

    )

    if hss_flag:
        results.update(dict(
            size_elh=size_elh_u,
            vol_hss_water=vol_hss_water_u,
            p_elh_in=p_elh_in_v,
            p_elh_out=p_elh_out_v,
            p_pv_elh=p_pv_elh_v,
            p_rec_elh=p_rec_elh_v,
            p_grid_elh=p_grid_elh_v,
            p_hss_in=p_hss_in_v,
            p_hss_out=p_hss_out_v,
            t_hss=t_hss_v,
            e_hss_stor=e_hss_stor_v,
            d_cl=d_cl_v,
        ))

    # Simple diagnostic table
    user_summary = DataFrame({
        "grid_low_kwh": e_grid_low_v,
        "grid_high_kwh": e_grid_high_v,
        "grid_total_kwh": e_grid_total_user,
        "rec_buy_kwh": rec_buy_energy_user,
        "rec_sell_kwh": rec_sell_energy_user,
        "grid_export_kwh": grid_export_energy_user,
        "grid_cost_Ft": grid_cost_user,
        "rec_buy_cost_Ft": rec_buy_cost_user,
        "rec_sell_revenue_Ft": rec_sell_revenue_user,
        "grid_export_revenue_Ft": grid_export_revenue_user,
        "net_cost_Ft": net_cost_user,
    })
    results["user_summary"] = user_summary

    return (
        results,
        status,
        objective,
        prob.numVariables(),
        prob.numConstraints(),
        getattr(prob, "infeasibilityGap", lambda: None)(),
    )