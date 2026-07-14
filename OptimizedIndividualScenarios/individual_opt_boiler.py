import numpy as np
import pulp
from Utility.configuration import config


def individual_opt_boiler(
    p_pv,                   # (T,)
    p_ue,                   # (T,)
    p_dhw,                  # (T,) hőigény kW
    dt=0.25,
    size_elh=0.0,           # kW
    vol_hss_water=0.0,      # liter
    T_env=20.0,
    T_max=65.0,
    T_min=38.0,
    T_in=10.0,
    a_hss=0.01275,
    eta_elh=0.95,
    p_el_heater_fixed=None,
    price_grid_a_low=config.getfloat("tariffs", "grid_a_low_ft_per_kwh"),
    price_grid_a_high=config.getfloat("tariffs", "grid_a_high_ft_per_kwh"),
    price_grid_b_low=config.getfloat("tariffs", "grid_b_low_ft_per_kwh"),
    price_grid_b_high=config.getfloat("tariffs", "grid_b_high_ft_per_kwh"),
    price_pv_grid=config.getfloat("tariffs", "pv_export_ft_per_kwh"),
    grid_a_low_cap_kwh=config.getfloat("tariffs", "grid_a_low_limit_kwh"),
    grid_b_low_cap_kwh=config.getfloat("tariffs", "grid_b_low_limit_kwh"),
    run_lp=False,
    msg=True,
    enforce_cl_rules=True,
    cl_max_on_hours_per_day=8.0,
    cl_min_midday_hours_per_day=4.0,
    gapRel=None,
    timeLimit=None,
    boiler_tariff="B",
    objective="bill"
):
    """
    Egy háztartás optimalizálása:
    - PV van
    - BESS nincs
    - energiamegosztás nincs
    - bojler/HSS optimalizált
    - # cél: alapértelmezetten éves nettó villanyszámla minimalizálása;
      # objective="grid" esetén hálózati interakció minimalizálása
    """

    p_pv = np.asarray(p_pv, dtype=float).ravel()
    p_ue = np.asarray(p_ue, dtype=float).ravel()
    p_dhw = np.asarray(p_dhw, dtype=float).ravel()

    assert p_pv.shape == p_ue.shape == p_dhw.shape
    T = len(p_pv)
    time_set = range(T)

    if p_el_heater_fixed is None:
        p_el_heater_fixed = np.zeros_like(p_ue)
    else:
        p_el_heater_fixed = np.asarray(p_el_heater_fixed, dtype=float).ravel()

    assert p_el_heater_fixed.shape == p_ue.shape

    has_pv = np.sum(p_pv) > 1e-9
    hss_active = has_pv and (size_elh > 1e-9) and (vol_hss_water > 1e-9)

    boiler_tariff = str(boiler_tariff).upper().strip()
    if boiler_tariff not in {"A", "B"}:
        raise ValueError(f"boiler_tariff csak 'A' vagy 'B' lehet, nem: {boiler_tariff}")

    c_hss = 0.00116667  # kWh / (liter*K)

    prob = pulp.LpProblem("individual_boiler_grid_min", pulp.LpMinimize)

    # Villamos teljesítményáramok
    p_pv_load = [pulp.LpVariable(f"p_pv_load_{t}", lowBound=0) for t in time_set]
    p_pv_elh  = [pulp.LpVariable(f"p_pv_elh_{t}", lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f"p_pv_grid_{t}", lowBound=0) for t in time_set]

    p_pv_elh_fixed = [pulp.LpVariable(f"p_pv_elh_fixed_{t}", lowBound=0) for t in time_set]
    p_grid_elh_fixed = [pulp.LpVariable(f"p_grid_elh_fixed_{t}", lowBound=0) for t in time_set]

    p_grid_load = [pulp.LpVariable(f"p_grid_load_{t}", lowBound=0) for t in time_set]
    p_grid_elh  = [pulp.LpVariable(f"p_grid_elh_{t}", lowBound=0) for t in time_set]

    d_export = [pulp.LpVariable(f"d_export_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]
    M_grid = max(
        float(np.max(
            p_ue
            + p_dhw / max(eta_elh, 1e-9)
            + p_el_heater_fixed
        )) if T > 0 else 0.0,
        float(size_elh)
    ) + 1.0
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
    # 15 perces bruttó elszámolás
    e_grid_a_low_step = [pulp.LpVariable(f"e_grid_a_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_a_high_step = [pulp.LpVariable(f"e_grid_a_high_step_{t}", lowBound=0) for t in time_set]
    rem_a_low = [pulp.LpVariable(f"rem_a_low_{t}", lowBound=0, upBound=grid_a_low_cap_kwh) for t in time_set]

    e_grid_b_low_step = [pulp.LpVariable(f"e_grid_b_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_b_high_step = [pulp.LpVariable(f"e_grid_b_high_step_{t}", lowBound=0) for t in time_set]
    rem_b_low = [pulp.LpVariable(f"rem_b_low_{t}", lowBound=0, upBound=grid_b_low_cap_kwh) for t in time_set]

    for t in time_set:
        k = (t + 1) % T

        # PV szétosztás
        prob += (
                p_pv_load[t]
                + p_pv_elh[t]
                + p_pv_elh_fixed[t]
                + p_pv_grid[t]
                == p_pv[t]
        ), f"pv_split_{t}"

        if boiler_tariff == "B":
            # B tarifás bojler: külön mérő/kör, nem kaphat PV-ből.
            prob += p_pv_elh[t] == 0, f"no_pv_to_hss_B_{t}"
            prob += p_pv_elh_fixed[t] == 0, f"no_pv_to_fixed_boiler_B_{t}"

        prob += (
                p_pv_elh_fixed[t] + p_grid_elh_fixed[t] == p_el_heater_fixed[t]
        ), f"fixed_boiler_balance_{t}"

        # Hagyományos villamos fogyasztás
        prob += p_pv_load[t] + p_grid_load[t] == p_ue[t], f"ue_balance_{t}"

        if not run_lp:
            if boiler_tariff == "A":
                p_grid_import_a_t = p_grid_load[t] + p_grid_elh[t] + p_grid_elh_fixed[t]
            else:
                p_grid_import_a_t = p_grid_load[t]

            prob += p_pv_grid[t] <= M_pv * d_export[t], f"export_gate_{t}"
            prob += p_grid_import_a_t <= M_grid * (1 - d_export[t]), f"grid_vs_export_{t}"


        if hss_active:
            # ELH betáplálás
            prob += p_elh_in[t] == p_pv_elh[t] + p_grid_elh[t], f"elh_supply_{t}"
            prob += p_hss_in[t] == eta_elh * p_elh_in[t], f"hss_in_def_{t}"

            # DHW igény kiszolgálása
            prob += p_hss_out[t] == p_dhw[t], f"dhw_balance_{t}"

            # Hőtároló dinamika
            prob += (
                    vol_hss_water * c_hss * (t_hss[k] - t_hss[t])
                    == dt * (p_hss_in[t] - p_hss_out[t])
                    - a_hss * (t_hss[t] - T_env) * dt
            ), f"hss_balance_{t}"

            # következő lépésben kivehető max hő
            prob += (
                p_hss_out[k] <= vol_hss_water * c_hss * (t_hss[t] - T_in) / dt
            ), f"hss_max_out_{t}"

            # max betöltés
            prob += (
                p_hss_in[t] <= vol_hss_water * c_hss * (T_max - t_hss[t]) / dt
            ), f"hss_max_in_{t}"

            if not run_lp:
                prob += p_elh_in[t] <= size_elh * d_cl[t], f"elh_onoff_{t}"

            if not run_lp and t != 0 and t != T - 1:
                prob += d_cl[t + 1] >= d_cl[t] - d_cl[t - 1], f"cl_min_on_time_{t}"

        else:
            prob += p_pv_elh[t] == 0, f"no_hss_pv_elh_{t}"
            prob += p_grid_elh[t] == 0, f"no_hss_grid_elh_{t}"

        # 15 perces import felosztása kedvezményes és piaci részre
        if boiler_tariff == "A":
            # A tarifás bojler: UE + bojler hálózati része is A tarifán van.
            e_imp_a_t = dt * (p_grid_load[t] + p_grid_elh[t] + p_grid_elh_fixed[t])
            e_imp_b_t = 0.0
        else:
            # B tarifás bojler: bojler csak hálózatból, B tarifán.
            e_imp_a_t = dt * p_grid_load[t]
            e_imp_b_t = dt * (p_grid_elh[t] + p_grid_elh_fixed[t])

        prob += e_grid_a_low_step[t] + e_grid_a_high_step[t] == e_imp_a_t
        prob += e_grid_b_low_step[t] + e_grid_b_high_step[t] == e_imp_b_t

        if t == 0:
            prob += rem_a_low[t] == grid_a_low_cap_kwh - e_grid_a_low_step[t], "rem_a_low_init"
            prob += e_grid_a_low_step[t] <= grid_a_low_cap_kwh, f"grid_a_low_step_cap_{t}"

            prob += rem_b_low[t] == grid_b_low_cap_kwh - e_grid_b_low_step[t], "rem_b_low_init"
            prob += e_grid_b_low_step[t] <= grid_b_low_cap_kwh, f"grid_b_low_step_cap_{t}"
        else:
            prob += rem_a_low[t] == rem_a_low[t - 1] - e_grid_a_low_step[t], f"rem_a_low_balance_{t}"
            prob += e_grid_a_low_step[t] <= rem_a_low[t - 1], f"grid_a_low_step_cap_{t}"

            prob += rem_b_low[t] == rem_b_low[t - 1] - e_grid_b_low_step[t], f"rem_b_low_balance_{t}"
            prob += e_grid_b_low_step[t] <= rem_b_low[t - 1], f"grid_b_low_step_cap_{t}"

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

    if objective == "bill":
        prob += pulp.lpSum(
            price_grid_a_low * e_grid_a_low_step[t]
            + price_grid_a_high * e_grid_a_high_step[t]
            + price_grid_b_low * e_grid_b_low_step[t]
            + price_grid_b_high * e_grid_b_high_step[t]
            - price_pv_grid * dt * p_pv_grid[t]
            for t in time_set
        )
    else:
        raise ValueError("objective must be 'bill'")

    prob.writeLP("debug.lp")
    solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    status = prob.solve(solver)

    status_str = pulp.LpStatus[status]

    if status_str not in {"Optimal", "Integer Feasible"}:
        raise RuntimeError(
            f"Optimalizálási hiba: {status_str}. "
            "A modell nem adott érvényes megoldást, ezért nem mentek 0-kat."
        )

    def _val(x):
        v = pulp.value(x)
        return 0.0 if v is None else float(v)

    p_pv_load_v = np.array([_val(v) for v in p_pv_load], dtype=float)
    p_pv_elh_v = np.array([_val(v) for v in p_pv_elh], dtype=float)
    p_pv_grid_v = np.array([_val(v) for v in p_pv_grid], dtype=float)
    p_grid_load_v = np.array([_val(v) for v in p_grid_load], dtype=float)
    p_grid_elh_v = np.array([_val(v) for v in p_grid_elh], dtype=float)
    p_pv_elh_fixed_v = np.array([_val(v) for v in p_pv_elh_fixed], dtype=float)
    p_grid_elh_fixed_v = np.array([_val(v) for v in p_grid_elh_fixed], dtype=float)

    # Explicit grid import/export idősorok, hogy ugyanúgy legyen, mint a BESS modellben
    p_grid_import_v = p_grid_load_v + p_grid_elh_v + p_grid_elh_fixed_v
    p_grid_export_v = p_pv_grid_v

    if hss_active:
        p_elh_in_v = np.array([_val(v) for v in p_elh_in], dtype=float)
        p_hss_in_v = np.array([_val(v) for v in p_hss_in], dtype=float)
        p_hss_out_v = np.array([_val(v) for v in p_hss_out], dtype=float)
        t_hss_v = np.array([_val(v) for v in t_hss], dtype=float)
        e_hss_stor_v = vol_hss_water * c_hss * (t_hss_v - T_in)
    else:
        p_elh_in_v = np.zeros(T)
        p_hss_in_v = np.zeros(T)
        p_hss_out_v = np.zeros(T)
        t_hss_v = np.zeros(T)
        e_hss_stor_v = np.zeros(T)

    if hss_active and not run_lp:
        d_cl_v = np.array([_val(v) for v in d_cl], dtype=float)
    else:
        d_cl_v = np.zeros(T)

    if not run_lp:
        d_export_v = np.array([_val(v) for v in d_export], dtype=float)
    else:
        d_export_v = np.zeros(T)

    e_grid_a_low_step_v = np.array([_val(v) for v in e_grid_a_low_step], dtype=float)
    e_grid_a_high_step_v = np.array([_val(v) for v in e_grid_a_high_step], dtype=float)
    e_grid_b_low_step_v = np.array([_val(v) for v in e_grid_b_low_step], dtype=float)
    e_grid_b_high_step_v = np.array([_val(v) for v in e_grid_b_high_step], dtype=float)

    rem_a_low_v = np.array([_val(v) for v in rem_a_low], dtype=float)
    rem_b_low_v = np.array([_val(v) for v in rem_b_low], dtype=float)

    e_grid_a_low_v = float(np.sum(e_grid_a_low_step_v))
    e_grid_a_high_v = float(np.sum(e_grid_a_high_step_v))
    e_grid_b_low_v = float(np.sum(e_grid_b_low_step_v))
    e_grid_b_high_v = float(np.sum(e_grid_b_high_step_v))

    import_cost_a_ft = float(
        price_grid_a_low * e_grid_a_low_v
        + price_grid_a_high * e_grid_a_high_v
    )

    import_cost_b_ft = float(
        price_grid_b_low * e_grid_b_low_v
        + price_grid_b_high * e_grid_b_high_v
    )

    e_grid_total_v = e_grid_a_low_v + e_grid_a_high_v + e_grid_b_low_v + e_grid_b_high_v
    e_grid_export_v = float(np.sum(p_pv_grid_v) * dt)

    export_revenue = float(price_pv_grid * e_grid_export_v)
    net_cost = import_cost_a_ft + import_cost_b_ft - export_revenue

    if boiler_tariff == "A":
        p_grid_to_base = p_grid_load_v + p_grid_elh_v + p_grid_elh_fixed_v
        p_grid_to_boiler = np.zeros(T, dtype=float)
    else:
        p_grid_to_base = p_grid_load_v
        p_grid_to_boiler = p_grid_elh_v + p_grid_elh_fixed_v

    p_el_heater_total = p_elh_in_v + p_el_heater_fixed

    results = {
        "p_pv_load": p_pv_load_v,
        "p_pv_elh": p_pv_elh_v,
        "p_pv_grid": p_pv_grid_v,
        "p_grid_load": p_grid_load_v,
        "p_grid_elh": p_grid_elh_v,
        "p_grid_import": p_grid_import_v,
        "p_grid_export": p_grid_export_v,
        "p_elh_in": p_elh_in_v,
        "p_hss_in": p_hss_in_v,
        "p_hss_out": p_hss_out_v,
        "t_hss": t_hss_v,
        "e_hss_stor": e_hss_stor_v,
        "grid_export_revenue_Ft": export_revenue,
        "net_cost_Ft": net_cost,
        "objective_value": float(_val(prob.objective)),
        "objective_type": objective,
        "status": status_str,
        "hss_active": int(hss_active),
        "d_cl": d_cl_v,
        "d_export": d_export_v,
        "boiler_tariff": boiler_tariff,

        "e_grid_total": e_grid_total_v,
        "e_grid_export": e_grid_export_v,

        "grid_import_a_kwh": float(np.sum(p_grid_load_v) * dt),
        "grid_import_b_kwh": float(np.sum(p_grid_to_boiler) * dt),

        "grid_import_low_kwh": e_grid_a_low_v + e_grid_b_low_v,
        "grid_import_high_kwh": e_grid_a_high_v + e_grid_b_high_v,

        "grid_import_a_low_kwh": e_grid_a_low_v,
        "grid_import_a_high_kwh": e_grid_a_high_v,
        "grid_import_b_low_kwh": e_grid_b_low_v,
        "grid_import_b_high_kwh": e_grid_b_high_v,

        "import_cost_a_ft": import_cost_a_ft,
        "import_cost_b_ft": import_cost_b_ft,
        "import_cost_ft": import_cost_a_ft + import_cost_b_ft,
        "export_revenue_ft": export_revenue,
        "brt_bill_ft": net_cost,

        "e_grid_a_low_step": e_grid_a_low_step_v,
        "e_grid_a_high_step": e_grid_a_high_step_v,
        "e_grid_b_low_step": e_grid_b_low_step_v,
        "e_grid_b_high_step": e_grid_b_high_step_v,
        "remaining_a_low_block_kwh": rem_a_low_v,
        "remaining_b_low_block_kwh": rem_b_low_v,

        "p_pv_elh_fixed": p_pv_elh_fixed_v,
        "p_grid_elh_fixed": p_grid_elh_fixed_v,
        "p_grid_to_base": p_grid_load_v,
        "p_grid_to_boiler": p_grid_elh_v + p_grid_elh_fixed_v,
        "p_el_heater_total": p_elh_in_v + p_el_heater_fixed,
        "p_pv_to_boiler": p_pv_elh_v + p_pv_elh_fixed_v,
    }

    return results
