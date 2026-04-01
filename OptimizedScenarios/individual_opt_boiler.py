import numpy as np
import pulp
import pandas as pd


def individual_opt_boiler(
    p_pv,                   # (T,)
    p_ue,                   # (T,)
    p_dhw,                  # (T,) hőigény kW
    dt=0.25,
    size_elh=0.0,           # kW
    vol_hss_water=0.0,      # liter
    T_env=20.0,
    T_max=65.0,
    T_min=10.0,
    T_in=10.0,
    a_hss=0.01275,
    eta_elh=0.95,
    price_grid_low=36.0,
    price_grid_high=71.0,
    price_pv_grid=5.0,
    grid_low_cap_kwh=2523.0,
    run_lp=True,
    msg=True,
    enforce_cl_rules=True,
    cl_max_on_hours_per_day=8.0,
    cl_min_midday_hours_per_day=4.0,
):
    """
    Egy háztartás optimalizálása:
    - PV van
    - BESS nincs
    - energiamegosztás nincs
    - bojler/HSS optimalizált
    - cél: hálózati interakció minimalizálása = import + export minimum
    """

    p_pv = np.asarray(p_pv, dtype=float).ravel()
    p_ue = np.asarray(p_ue, dtype=float).ravel()
    p_dhw = np.asarray(p_dhw, dtype=float).ravel()

    assert p_pv.shape == p_ue.shape == p_dhw.shape
    T = len(p_pv)
    time_set = range(T)

    hss_active = (size_elh > 1e-9) and (vol_hss_water > 1e-9)

    c_hss = 0.00116667  # kWh / (liter*K)

    prob = pulp.LpProblem("individual_boiler_grid_min", pulp.LpMinimize)

    # Villamos energiaáramok
    p_pv_load = [pulp.LpVariable(f"p_pv_load_{t}", lowBound=0) for t in time_set]
    p_pv_elh  = [pulp.LpVariable(f"p_pv_elh_{t}", lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f"p_pv_grid_{t}", lowBound=0) for t in time_set]

    p_grid_load = [pulp.LpVariable(f"p_grid_load_{t}", lowBound=0) for t in time_set]
    p_grid_elh  = [pulp.LpVariable(f"p_grid_elh_{t}", lowBound=0) for t in time_set]

    d_export = [pulp.LpVariable(f"d_export_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    M_grid = max(float(np.max(p_ue + p_dhw / max(eta_elh, 1e-9))) if T > 0 else 0.0, float(size_elh)) + 1.0
    M_pv = float(np.max(p_pv)) + 1.0

    # HSS / ELH
    if hss_active:
        p_elh_in = [pulp.LpVariable(f"p_elh_in_{t}", lowBound=0, upBound=size_elh) for t in time_set]
        p_hss_in = [pulp.LpVariable(f"p_hss_in_{t}", lowBound=0) for t in time_set]
        p_hss_out = [pulp.LpVariable(f"p_hss_out_{t}", lowBound=0) for t in time_set]
        t_hss = [pulp.LpVariable(f"t_hss_{t}", lowBound=T_min, upBound=T_max) for t in time_set]
        d_cl = [pulp.LpVariable(f"d_cl_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    else:
        p_elh_in = [0.0] * T
        p_hss_in = [0.0] * T
        p_hss_out = [0.0] * T
        t_hss = [0.0] * T
        d_cl = [0.0] * T

    # Tarifa blokk
    e_grid_low = pulp.LpVariable("e_grid_low", lowBound=0)
    e_grid_high = pulp.LpVariable("e_grid_high", lowBound=0)

    for t in time_set:
        k = (t + 1) % T

        # PV szétosztás
        prob += p_pv_load[t] + p_pv_elh[t] + p_pv_grid[t] == p_pv[t], f"pv_split_{t}"

        # Hagyományos villamos fogyasztás
        prob += p_pv_load[t] + p_grid_load[t] == p_ue[t], f"ue_balance_{t}"

        if not run_lp:
            prob += p_pv_grid[t] <= M_pv * d_export[t], f"export_gate_{t}"
            prob += p_grid_load[t] + p_grid_elh[t] <= M_grid * (1 - d_export[t]), f"grid_vs_export_{t}"

        if hss_active:
            # ELH betáplálás
            prob += p_elh_in[t] == p_pv_elh[t] + p_grid_elh[t], f"elh_supply_{t}"
            prob += p_hss_in[t] == eta_elh * p_elh_in[t], f"hss_in_def_{t}"

            # DHW igény kiszolgálása
            prob += p_hss_out[t] == p_dhw[t], f"dhw_balance_{t}"

            # Hőtároló dinamika
            prob += (
                vol_hss_water * c_hss * (t_hss[k] - t_hss[t])
                == (p_hss_in[t] - p_hss_out[t] - a_hss * (t_hss[t] - T_env)) * dt
            ), f"hss_balance_{t}"

            # következő lépésben kivehető max hő
            prob += (
                p_hss_out[k] <= vol_hss_water * c_hss * (t_hss[t] - T_in) / dt
            ), f"hss_max_out_{t}"

            # töltési hely
            prob += (
                p_hss_in[t] <= vol_hss_water * c_hss * (T_max - t_hss[t]) / dt
            ), f"hss_charge_space_{t}"

            if not run_lp:
                prob += p_elh_in[t] <= size_elh * d_cl[t], f"elh_onoff_{t}"

            if not run_lp and t != 0 and t != T - 1:
                prob += d_cl[t + 1] >= d_cl[t] - d_cl[t - 1], f"cl_min_on_time_{t}"

        else:
            prob += p_pv_elh[t] == 0, f"no_hss_pv_elh_{t}"
            prob += p_grid_elh[t] == 0, f"no_hss_grid_elh_{t}"


    if hss_active and enforce_cl_rules and not run_lp:
        n_timesteps_in_a_day = round(24 / dt)
        assert abs(n_timesteps_in_a_day * dt - 24) < 1e-9, "dt must divide 24h exactly"

        # 10:00–16:00 közötti "középső" időszak
        start_mid = round(10 / dt)
        mid_len = round(6 / dt)

        y_middle_day = [0] * n_timesteps_in_a_day
        for i in range(start_mid, min(start_mid + mid_len, n_timesteps_in_a_day)):
            y_middle_day[i] = 1

        max_on_steps = round(cl_max_on_hours_per_day / dt)
        min_mid_steps = round(cl_min_midday_hours_per_day / dt)

        for j in range(0, T, n_timesteps_in_a_day):
            day_idx = range(j, min(j + n_timesteps_in_a_day, T))
            if len(list(day_idx)) < n_timesteps_in_a_day:
                continue

            prob += pulp.lpSum(d_cl[t] for t in day_idx) <= max_on_steps, f"day_{j}_cl_maxon"
            prob += pulp.lpSum(d_cl[t] * y_middle_day[t - j] for t in day_idx) >= min_mid_steps, f"day_{j}_cl_midmin"


    # Grid energia blokk
    total_grid_energy = pulp.lpSum((p_grid_load[t] + p_grid_elh[t]) * dt for t in time_set)
    prob += e_grid_low + e_grid_high == total_grid_energy, "grid_block_sum"
    prob += e_grid_low <= grid_low_cap_kwh, "grid_low_cap"

    # Célfüggvény: hálózati interakció minimalizálása
    # = import + export
    prob += pulp.lpSum((p_grid_load[t] + p_grid_elh[t] + p_pv_grid[t]) * dt for t in time_set)

    solver = pulp.GUROBI_CMD(msg=msg)
    status = prob.solve(solver)

    status_str = pulp.LpStatus[status]
    if status_str not in {"Optimal", "Not Solved", "Integer Feasible", "Undefined"}:
        raise RuntimeError(f"Hiba: {status_str}")

    p_pv_load_v = np.array([pulp.value(v) for v in p_pv_load], dtype=float)
    p_pv_elh_v = np.array([pulp.value(v) for v in p_pv_elh], dtype=float)
    p_pv_grid_v = np.array([pulp.value(v) for v in p_pv_grid], dtype=float)
    p_grid_load_v = np.array([pulp.value(v) for v in p_grid_load], dtype=float)
    p_grid_elh_v = np.array([pulp.value(v) for v in p_grid_elh], dtype=float)

    if hss_active:
        p_elh_in_v = np.array([pulp.value(v) for v in p_elh_in], dtype=float)
        p_hss_in_v = np.array([pulp.value(v) for v in p_hss_in], dtype=float)
        p_hss_out_v = np.array([pulp.value(v) for v in p_hss_out], dtype=float)
        t_hss_v = np.array([pulp.value(v) for v in t_hss], dtype=float)
        e_hss_stor_v = vol_hss_water * c_hss * (t_hss_v - T_in)
    else:
        p_elh_in_v = np.zeros(T)
        p_hss_in_v = np.zeros(T)
        p_hss_out_v = np.zeros(T)
        t_hss_v = np.zeros(T)
        e_hss_stor_v = np.zeros(T)

    if hss_active and not run_lp:
        d_cl_v = np.array([pulp.value(v) for v in d_cl], dtype=float)
    else:
        d_cl_v = np.zeros(T)

    e_grid_low_v = float(pulp.value(e_grid_low))
    e_grid_high_v = float(pulp.value(e_grid_high))
    e_grid_total_v = e_grid_low_v + e_grid_high_v
    e_grid_export_v = float(np.sum(p_pv_grid_v) * dt)

    grid_cost = price_grid_low * e_grid_low_v + price_grid_high * e_grid_high_v
    export_revenue = price_pv_grid * e_grid_export_v
    net_cost = grid_cost - export_revenue

    results = {
        "p_pv_load": p_pv_load_v,
        "p_pv_elh": p_pv_elh_v,
        "p_pv_grid": p_pv_grid_v,
        "p_grid_load": p_grid_load_v,
        "p_grid_elh": p_grid_elh_v,
        "p_elh_in": p_elh_in_v,
        "p_hss_in": p_hss_in_v,
        "p_hss_out": p_hss_out_v,
        "t_hss": t_hss_v,
        "e_hss_stor": e_hss_stor_v,
        "e_grid_low": e_grid_low_v,
        "e_grid_high": e_grid_high_v,
        "e_grid_total": e_grid_total_v,
        "e_grid_export": e_grid_export_v,
        "grid_cost_Ft": grid_cost,
        "grid_export_revenue_Ft": export_revenue,
        "net_cost_Ft": net_cost,
        "objective_grid_interaction_kwh": float(pulp.value(prob.objective)),
        "status": status_str,
        "hss_active": int(hss_active),
        "d_cl": d_cl_v,
    }

    return results