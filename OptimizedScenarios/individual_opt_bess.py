import numpy as np
import pulp


def individual_opt_bess(
    p_pv,                   # (T,) kW
    p_ue,                   # (T,) kW - alap villamos fogyasztás
    p_el_heater=None,       # (T,) kW - FIX bojler villamos fogyasztás (nem optimalizált)
    dt=0.25,                # óra
    size_bess=0.0,          # kWh
    eta_bess_in=0.98,
    eta_bess_out=0.96,
    eta_bess_stor=0.995,
    soc_bess_min=0.10,
    soc_bess_max=0.90,
    soc_bess_init=None,
    t_bess_min=2.0,         # h -> max teljesítmény = size_bess / t_bess_min
    price_grid_a_low=36.0,
    price_grid_a_high=71.0,
    price_grid_b_low=23.0,
    price_grid_b_high=61.0,
    price_pv_grid=5.0,
    grid_a_low_cap_kwh=2523.0,
    grid_b_low_cap_kwh=2523.0,
    objective="grid",
    run_lp=False,
    msg=False,
    gapRel=None,
    timeLimit=None,
):
    """
    Egy háztartás egyéni gazdasági optimalizálása, ahol:
      - PV van
      - BESS opcionális
      - közösségi megosztás nincs
      - bojler NEM optimalizált, hanem fix villamos terhelésként szerepel
      - cél: éves nettó villanyszámla minimalizálása

    Energiaáramlási logika:
      PV -> saját load / saját BESS / külső hálózati export
      BESS -> saját load
      Grid -> saját load és saját BESS
    """

    p_pv = np.asarray(p_pv, dtype=float).ravel()
    p_ue = np.asarray(p_ue, dtype=float).ravel()

    if p_el_heater is None:
        p_el_heater = np.zeros_like(p_ue)
    else:
        p_el_heater = np.asarray(p_el_heater, dtype=float).ravel()

    assert p_pv.shape == p_ue.shape == p_el_heater.shape, \
        "p_pv, p_ue, p_el_heater hossza nem egyezik."

    T = len(p_pv)
    time_set = range(T)

    total_load = p_ue + p_el_heater

    bess_active = float(size_bess) > 1e-9
    if soc_bess_init is None:
        soc_bess_init = soc_bess_min

    battery_power = float(size_bess) / max(float(t_bess_min), 1e-9) if bess_active else 0.0


    prob = pulp.LpProblem("individual_opt_bess_fixed_boiler", pulp.LpMinimize)

    # Big-M
    M_pv = float(np.max(p_pv)) + 1.0 if T > 0 else 1.0
    M_grid = float(np.max(total_load) + battery_power) + 1.0 if T > 0 else 1.0

    # ------------------------------------------------------------------
    # Változók
    # ------------------------------------------------------------------
    # PV split
    p_pv_ue = [pulp.LpVariable(f"p_pv_ue_{t}", lowBound=0) for t in time_set]
    p_pv_elh = [pulp.LpVariable(f"p_pv_elh_{t}", lowBound=0) for t in time_set]
    p_pv_bess = [pulp.LpVariable(f"p_pv_bess_{t}", lowBound=0) for t in time_set]
    p_pv_grid = [pulp.LpVariable(f"p_pv_grid_{t}", lowBound=0) for t in time_set]

    p_bess_ue = [pulp.LpVariable(f"p_bess_ue_{t}", lowBound=0) for t in time_set]
    p_bess_elh = [pulp.LpVariable(f"p_bess_elh_{t}", lowBound=0) for t in time_set]

    p_grid_ue = [pulp.LpVariable(f"p_grid_ue_{t}", lowBound=0) for t in time_set]
    p_grid_elh = [pulp.LpVariable(f"p_grid_elh_{t}", lowBound=0) for t in time_set]

    # Load supply split
    p_bess_load = [pulp.LpVariable(f"p_bess_load_{t}", lowBound=0) for t in time_set]
    p_grid_load = [pulp.LpVariable(f"p_grid_load_{t}", lowBound=0) for t in time_set]

    # Battery
    p_bess_in = [pulp.LpVariable(f"p_bess_in_{t}", lowBound=0) for t in time_set]
    p_bess_out = [pulp.LpVariable(f"p_bess_out_{t}", lowBound=0) for t in time_set]
    p_grid_bess = [pulp.LpVariable(f"p_grid_bess_{t}", lowBound=0) for t in time_set]

    if bess_active:
        e_bess = [
            pulp.LpVariable(
                f"e_bess_{t}",
                lowBound=float(size_bess) * float(soc_bess_min),
                upBound=float(size_bess) * float(soc_bess_max),
            )
            for t in time_set
        ]
        d_bess = [
            pulp.LpVariable(f"d_bess_{t}", cat=pulp.LpBinary) if not run_lp else 0
            for t in time_set
        ]
    else:
        e_bess = [0.0] * T
        d_bess = [0.0] * T

    # Grid import/export
    p_grid_import = [pulp.LpVariable(f"p_grid_import_{t}", lowBound=0) for t in time_set]
    p_grid_export = [pulp.LpVariable(f"p_grid_export_{t}", lowBound=0) for t in time_set]
    d_grid = [pulp.LpVariable(f"d_grid_{t}", cat=pulp.LpBinary) if not run_lp else 0 for t in time_set]

    # Tariff blocks
    # 15 perces bruttó elszámolás
    e_grid_a_low_step = [pulp.LpVariable(f"e_grid_a_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_a_high_step = [pulp.LpVariable(f"e_grid_a_high_step_{t}", lowBound=0) for t in time_set]
    rem_a_low = [pulp.LpVariable(f"rem_a_low_{t}", lowBound=0, upBound=grid_a_low_cap_kwh) for t in time_set]

    e_grid_b_low_step = [pulp.LpVariable(f"e_grid_b_low_step_{t}", lowBound=0) for t in time_set]
    e_grid_b_high_step = [pulp.LpVariable(f"e_grid_b_high_step_{t}", lowBound=0) for t in time_set]
    rem_b_low = [pulp.LpVariable(f"rem_b_low_{t}", lowBound=0, upBound=grid_b_low_cap_kwh) for t in time_set]

    # ------------------------------------------------------------------
    # Korlátok
    # ------------------------------------------------------------------
    for t in time_set:
        k = (t + 1) % T

        # PV split
        prob += (
                p_pv_ue[t] + p_pv_elh[t] + p_pv_bess[t] + p_pv_grid[t] == p_pv[t]
        ), f"pv_split_{t}"

        # Fogyasztás kiszolgálása
        prob += (
                p_pv_ue[t] + p_bess_ue[t] + p_grid_ue[t] == p_ue[t]
        ), f"ue_balance_{t}"

        prob += (
                p_pv_elh[t] + p_bess_elh[t] + p_grid_elh[t] == p_el_heater[t]
        ), f"fixed_boiler_balance_{t}"

        # Grid mérleg
        prob += (
            p_grid_import[t] == p_grid_load[t] + p_grid_bess[t]
        ), f"grid_import_def_{t}"

        prob += (
            p_grid_export[t] == p_pv_grid[t]
        ), f"grid_export_def_{t}"

        prob += (
                p_grid_import[t] == p_grid_ue[t] + p_grid_elh[t] + p_grid_bess[t]
        ), f"grid_import_def_{t}"

        # Ne legyen egyszerre import és export
        if not run_lp:
            prob += p_grid_import[t] <= M_grid * d_grid[t], f"grid_import_gate_{t}"
            prob += p_grid_export[t] <= M_pv * (1 - d_grid[t]), f"grid_export_gate_{t}"

        # Akku logika
        if not bess_active:
            prob += p_pv_bess[t] == 0, f"no_bess_pv_bess_{t}"
            prob += p_bess_load[t] == 0, f"no_bess_load_{t}"
            prob += p_bess_in[t] == 0, f"no_bess_in_{t}"
            prob += p_bess_out[t] == 0, f"no_bess_out_{t}"
            prob += p_grid_bess[t] == 0, f"no_bess_grid_bess_{t}"
        else:
            # töltés két forrásból
            prob += (
                p_bess_in[t] == p_pv_bess[t] + p_grid_bess[t]
            ), f"bess_in_def_{t}"

            # kisütés csak a loadra
            prob += (
                p_bess_out[t] == p_bess_load[t]
            ), f"bess_out_def_{t}"

            prob += (
                    p_bess_load[t] == p_bess_ue[t] + p_bess_elh[t]
            ), f"bess_load_split_{t}"

            # dinamika
            prob += e_bess[k] == (
                    e_bess[t] * eta_bess_stor
                    + dt * (p_bess_in[t] * eta_bess_in - p_bess_out[t] / max(eta_bess_out, 1e-9))
            )

            # teljesítménykorlát
            if run_lp:
                prob += p_bess_in[t] <= battery_power, f"bess_in_cap_{t}"
                prob += p_bess_out[t] <= battery_power, f"bess_out_cap_{t}"
            else:
                prob += p_bess_in[t] <= d_bess[t] * battery_power, f"bess_in_gate_{t}"
                prob += p_bess_out[t] <= (1 - d_bess[t]) * battery_power, f"bess_out_gate_{t}"

            # SOC korlát külön is
            prob += e_bess[t] >= float(size_bess) * float(soc_bess_min), f"soc_min_{t}"
            prob += e_bess[t] <= float(size_bess) * float(soc_bess_max), f"soc_max_{t}"

        # 15 perces bruttó import felosztása
        e_imp_t = p_grid_import[t] * dt

        e_imp_a_t = dt * (p_grid_ue[t] + p_grid_bess[t])
        e_imp_b_t = dt * p_grid_elh[t]

        prob += e_grid_a_low_step[t] + e_grid_a_high_step[t] == e_imp_a_t
        prob += e_grid_b_low_step[t] + e_grid_b_high_step[t] == e_imp_b_t

        if t == 0:
            prob += rem_a_low[t] == grid_a_low_cap_kwh - e_grid_a_low_step[t]
            prob += e_grid_a_low_step[t] <= grid_a_low_cap_kwh

            prob += rem_b_low[t] == grid_b_low_cap_kwh - e_grid_b_low_step[t]
            prob += e_grid_b_low_step[t] <= grid_b_low_cap_kwh
        else:
            prob += rem_a_low[t] == rem_a_low[t - 1] - e_grid_a_low_step[t]
            prob += e_grid_a_low_step[t] <= rem_a_low[t - 1]

            prob += rem_b_low[t] == rem_b_low[t - 1] - e_grid_b_low_step[t]
            prob += e_grid_b_low_step[t] <= rem_b_low[t - 1]

    # kezdeti SOC
    if bess_active:
        prob += e_bess[0] == float(size_bess) * float(soc_bess_init), "bess_init"

    if bess_active and not run_lp:
        min_on_steps = 4  # 4 * 15 perc = 1 óra
        for t in range(1, T - min_on_steps + 1):
            start_up = d_bess[t] - d_bess[t - 1]
            prob += pulp.lpSum(d_bess[tau] for tau in range(t, t + min_on_steps)) >= min_on_steps * start_up, \
                f"bess_min_on_{t}"

    # ------------------------------------------------------------------
    # Célfüggvény: nettó villanyszámla [Ft]
    if objective == "grid":
        prob += pulp.lpSum(
            dt * (p_grid_import[t] + p_grid_export[t])
            for t in time_set
        )
    else:
        raise ValueError("objective must be 'grid'")
    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    solver = pulp.GUROBI_CMD(msg=msg, gapRel=gapRel, timeLimit=timeLimit)
    status = prob.solve(solver)

    status_str = pulp.LpStatus.get(status, str(status))
    if status_str not in {"Optimal", "Not Solved", "Integer Feasible", "Undefined"}:
        raise RuntimeError(f"Hiba: {status_str}")

    def _val(x):
        v = pulp.value(x)
        return 0.0 if v is None else float(v)

    p_pv_load_v = np.array([_val(v) for v in p_pv_load], dtype=float)
    p_pv_bess_v = np.array([_val(v) for v in p_pv_bess], dtype=float)
    p_pv_grid_v = np.array([_val(v) for v in p_pv_grid], dtype=float)

    p_bess_load_v = np.array([_val(v) for v in p_bess_load], dtype=float)
    p_grid_load_v = np.array([_val(v) for v in p_grid_load], dtype=float)

    p_bess_in_v = np.array([_val(v) for v in p_bess_in], dtype=float)
    p_bess_out_v = np.array([_val(v) for v in p_bess_out], dtype=float)
    p_grid_bess_v = np.array([_val(v) for v in p_grid_bess], dtype=float)

    p_grid_import_v = np.array([_val(v) for v in p_grid_import], dtype=float)
    p_grid_export_v = np.array([_val(v) for v in p_grid_export], dtype=float)

    if bess_active:
        e_bess_v = np.array([_val(v) for v in e_bess], dtype=float)
        d_bess_v = np.array([_val(v) if not run_lp else 0.0 for v in d_bess], dtype=float)
    else:
        e_bess_v = np.zeros(T, dtype=float)
        d_bess_v = np.zeros(T, dtype=float)

    d_grid_v = np.array([_val(v) if not run_lp else 0.0 for v in d_grid], dtype=float)

    e_grid_low_step_v = np.array([_val(v) for v in e_grid_low_step], dtype=float)
    e_grid_high_step_v = np.array([_val(v) for v in e_grid_high_step], dtype=float)
    rem_low_v = np.array([_val(v) for v in rem_low], dtype=float)

    e_grid_low_v = float(np.sum(e_grid_low_step_v))
    e_grid_high_v = float(np.sum(e_grid_high_step_v))
    e_grid_total_v = e_grid_low_v + e_grid_high_v

    grid_cost = float(np.sum(
        price_grid_low * e_grid_low_step_v + price_grid_high * e_grid_high_step_v
    ))
    export_revenue = float(np.sum(p_pv_grid_v) * dt * price_pv_grid)
    net_cost = grid_cost - export_revenue

    results = {
        "p_pv": p_pv,
        "p_ue": p_ue,
        "p_el_heater": p_el_heater,
        "p_total_load": total_load,
        "dt": dt,

        "size_bess": float(size_bess),
        "battery_power": float(battery_power),
        "soc_bess_min": float(soc_bess_min),
        "soc_bess_max": float(soc_bess_max),
        "soc_bess_init": float(soc_bess_init),
        "bess_active": int(bess_active),

        "p_pv_load": p_pv_ue_v + p_pv_elh_v,
        "p_pv_ue": p_pv_ue_v,
        "p_pv_elh": p_pv_elh_v,
        "p_pv_bess": p_pv_bess_v,
        "p_pv_grid": p_pv_grid_v,

        "p_bess_load": p_bess_ue_v + p_bess_elh_v,
        "p_bess_ue": p_bess_ue_v,
        "p_bess_elh": p_bess_elh_v,

        "p_grid_load": p_grid_ue_v + p_grid_elh_v,
        "p_grid_ue": p_grid_ue_v,
        "p_grid_elh": p_grid_elh_v,
        "p_grid_to_base": p_grid_ue_v,
        "p_grid_to_boiler": p_grid_elh_v,

        "p_bess_in": p_bess_in_v,
        "p_bess_out": p_bess_out_v,
        "p_grid_bess": p_grid_bess_v,
        "e_bess": e_bess_v,
        "d_bess": d_bess_v,

        "p_grid_import": p_grid_import_v,
        "p_grid_export": p_grid_export_v,
        "d_grid": d_grid_v,

        "e_grid_low": e_grid_low_v,
        "e_grid_high": e_grid_high_v,
        "e_grid_total": e_grid_total_v,

        "grid_cost_Ft": float(grid_cost),
        "grid_export_revenue_Ft": float(export_revenue),
        "net_cost_Ft": float(net_cost),
        "objective_Ft": float(pulp.value(prob.objective)),
        "status": status_str,

        "e_grid_low_step": e_grid_low_step_v,
        "e_grid_high_step": e_grid_high_step_v,
        "remaining_low_block_kwh": rem_low_v,

        "grid_import_a_kwh": float(np.sum(p_grid_ue_v + p_grid_bess_v) * dt),
        "grid_import_b_kwh": float(np.sum(p_grid_elh_v) * dt),
        "grid_import_a_low_kwh": e_grid_a_low_v,
        "grid_import_a_high_kwh": e_grid_a_high_v,
        "grid_import_b_low_kwh": e_grid_b_low_v,
        "grid_import_b_high_kwh": e_grid_b_high_v,
        "import_cost_a_ft": import_cost_a_ft,
        "import_cost_b_ft": import_cost_b_ft,
        "import_cost_ft": import_cost_a_ft + import_cost_b_ft,
        "export_revenue_ft": export_revenue,
        "brt_bill_ft": net_cost,
    }

    return results